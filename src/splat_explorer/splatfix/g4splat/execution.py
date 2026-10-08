"""G4Splat LRZ transport using the existing queue and exclusive compute lease."""
import json
import shlex
import time
from pathlib import Path
from ..checkpoint import Checkpoint, atomic_json
from ..resolution import prepare_repair_checkpoint
from .inputs import export_inputs
from .repair import DEFAULT_RUNTIME


def execute_remote(executor, run_id, options, root, stop, update):
    from ... import repair_lrz as lrz
    from ...scene_runs.lrz_transport import LrzSceneRunTransport
    cfg = lrz.load_lrz_config()
    root = Path(root)
    saved = executor.store.get_run(run_id).state.details
    remote = saved.get('remote_dir') or str(cfg['workspace']).rstrip('/') + '/splatfix-jobs/' + run_id
    container = '/workspace/splatfix-jobs/' + run_id
    endpoint = f"{cfg['user']}@{cfg['host']}"
    def ssh(command):
        reply = lrz._ssh_run(cfg, command, timeout=30)
        if reply.returncode:
            raise RuntimeError((reply.stderr or reply.stdout or 'Remote command failed').strip())
        return reply.stdout
    def sync(source, destination):
        lrz._mux_run(['rsync', '-az', '-e', lrz.rsync_ssh_cmd(cfg), source, destination])
    def read_status():
        raw = ssh('if [ -f ' + shlex.quote(remote + '/worker-status.json') + ' ]; then cat ' +
                  shlex.quote(remote + '/worker-status.json') + '; fi')
        return json.loads(raw) if raw.strip() else {}
    launched = saved.get('remote_dir') and ssh('test ! -d ' + shlex.quote(remote + '/launched') + ' || printf launched').strip()
    if not launched:
        LrzSceneRunTransport(executor.cfg, run_id, root).validate()
        checkpoint = Checkpoint.load(options['checkpoint'])
        runtime = {**DEFAULT_RUNTIME, **dict(executor.cfg.get('splatfix', {}).get('g4splat_runtime', {}))}
        runtime['resolution_profile'] = options['resolution_profile']
        if options['mode'] == 'edited':
            checkpoint.require_viser_images()
            for view in checkpoint.views:
                checkpoint.image_path(view, repaired=True)
        update(phase='inputs', message='Recapturing G4Splat inputs at the selected resolution')
        # Preparation stays local: only calibrated RGBs go to the GPU. Neither
        # the source splat nor any ArtiFixer trajectories initialize G4Splat.
        import uuid
        preparation = root / ('g4splat-inputs-' + uuid.uuid4().hex[:8])
        prepared = prepare_repair_checkpoint(checkpoint, preparation / 'checkpoint', options['resolution_profile'],
                                             should_stop=stop)
        export_inputs(prepared, preparation / 'inputs', options['mode'])
        if stop():
            raise InterruptedError('Cancelled before G4Splat staging')
        update(phase='staging', message='Staging G4Splat calibrated RGB inputs')
        lrz.sync_code_to_dss(cfg)
        ssh('mkdir -p ' + shlex.quote(remote + '/inputs'))
        sync(str(preparation / 'inputs') + '/', endpoint + ':' + remote + '/inputs/')
        request = {'reconstruction_method': 'g4splat', 'stage': 'repair', 'mode': options['mode'],
                   'inputs': container + '/inputs', 'output': container + '/results/g4splat', 'runtime': runtime,
                   'stop_at_epoch': executor.store.get_run(run_id).state.details.get('effective_deadline')}
        atomic_json(root / 'worker-request.json', request)
        sync(str(root / 'worker-request.json'), endpoint + ':' + remote + '/worker-request.json')
        inner = lrz._remote_pythonpath_exports(cfg) + shlex.join([
            'python', '-u', '-m', 'splat_explorer.splatfix.job_worker', '--run-dir', container])
        command = lrz.container_srun_prefix(cfg) + 'bash -lc ' + shlex.quote(inner)
        wrapped = command + "; printf '%s\\n' $? > " + shlex.quote(remote + '/launcher-exit')
        launcher = ('mkdir ' + shlex.quote(remote + '/launched') + ' && (nohup bash -lc ' + shlex.quote(wrapped)
                    + ' > ' + shlex.quote(remote + '/stage.log') + ' 2>&1 < /dev/null &)')
        update(remote_dir=remote, phase='g4splat', message='Launching official G4Splat')
        ssh(launcher)
    state, last = {}, None
    last_log = 0.
    def mirror_log():
        nonlocal last_log
        if time.monotonic() - last_log < 15:
            return
        last_log = time.monotonic()
        relative = 'results/g4splat/g4splat.log'
        try:
            text = ssh('if [ -f ' + shlex.quote(remote + '/' + relative) + ' ]; then tail -c 65536 ' +
                       shlex.quote(remote + '/' + relative) + '; fi')
            if text:
                destination = root / 'gpu/live-stage.log'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text)
                atomic_json(destination.with_suffix('.json'), {'phase': 'G4Splat', 'relative_path': relative})
        except Exception:
            pass  # Observing logs must never cancel training or release its lease.
    while True:
        deadline = executor.store.get_run(run_id).state.details.get('effective_deadline')
        if deadline and time.time() >= float(deadline) and not stop():
            executor.store.request_stop(run_id)
        if stop():
            ssh('touch ' + shlex.quote(remote + '/STOP'))
        state = read_status()
        mirror_log()
        if state:
            atomic_json(root / 'worker-status.json', state)
        if state and state != last:
            update(phase=state.get('phase', 'g4splat'), message=state.get('message', 'G4Splat running'))
            last = state
        if state.get('status') in ('completed', 'error', 'stopped'):
            break
        exited = ssh('if [ -f ' + shlex.quote(remote + '/launcher-exit') + ' ]; then cat ' +
                     shlex.quote(remote + '/launcher-exit') + '; fi')
        if exited.strip():
            state = read_status()
            if state.get('status') not in ('completed', 'error', 'stopped'):
                state = {'status': 'error', 'message': 'G4Splat launcher exited without terminal worker status: ' + exited.strip()}
            break
        time.sleep(2)
    # Connection/transfer failures keep the remote identity unresolved, allowing
    # reattachment and transfer retry before the shared lease is released.
    sync(endpoint + ':' + remote + '/', str(root / 'gpu') + '/')
    update(remote_finished=True)
    if state['status'] == 'stopped' or stop():
        raise InterruptedError('G4Splat stopped; available artifacts downloaded')
    if state['status'] != 'completed':
        raise RuntimeError(state.get('message', 'G4Splat failed'))
    return {'reconstruction_method': 'g4splat', 'results': str(root / 'gpu/results'),
            'mode': options['mode'], 'checkpoint': options['checkpoint']}
