"""OracleGS LRZ transport using the existing queue and exclusive compute lease."""
import json
import shlex
import time
from pathlib import Path
from ..checkpoint import Checkpoint, atomic_json
from .inputs import export_inputs
from .repair import DEFAULT_RUNTIME

# Only research artifacts: never transfer optimizer checkpoints during training.
PREVIEW_FILTERS = ['- model/', '- combined/', '- seva-padded-inputs/', '+ */',
    '+ *.json', '+ *.txt', '+ *.png', '+ *.jpg', '+ *.mp4', '- *']


def mirror_previews(lrz, cfg, endpoint, remote, root, *, phase=None):
    """One cache writer across executor/temporary observer; failures never stop GPU work."""
    import fcntl
    from ..jobs import read_json
    root = Path(root)
    with (root / '.oraclegs-preview.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        previous = read_json(root / 'oraclegs-preview-sync.json', {})
        if time.time() - previous.get('updated_at_epoch', 0) < 60:
            return False
        if (phase == 'reconstruction' and previous.get('phase') == phase
                and previous.get('status') == 'cached'):
            return False
        destination = root / 'gpu/results/oraclegs'
        destination.mkdir(parents=True, exist_ok=True)
        command = ['rsync', '-az', '--prune-empty-dirs', '-e', lrz.rsync_ssh_cmd(cfg)]
        for rule in PREVIEW_FILTERS:
            command += ['--filter', rule]
        command += [endpoint + ':' + remote + '/results/oraclegs/', str(destination) + '/']
        record = {'phase':phase, 'updated_at_epoch':time.time(), 'interval_seconds':60}
        try:
            reply = lrz._mux_run(command)
            if getattr(reply, 'returncode', 0):
                raise RuntimeError('OracleGS preview transfer failed; will retry')
            atomic_json(root / 'oraclegs-preview-sync.json', {**record, 'status':'cached'})
            return True
        except Exception as exc:
            atomic_json(root / 'oraclegs-preview-sync.json', {**record, 'status':'retrying', 'message':str(exc)})
            raise


def job_runtime(cfg, options):
    runtime = {**DEFAULT_RUNTIME, **dict(cfg.get('splatfix', {}).get('oraclegs_runtime', {}))}
    for key in ('oracle_capture_source', 'oracle_reference_mode', 'oracle_split_mode', 'seva_geometry_guidance', 'oracle_depth_source', 'oracle_initialization', 'oracle_attention_batching', 'seva_resolution', 'seva_fit', 'oracle_adapter', 'oracle_preset', 'oracle_preset_scale', 'oracle_target_frames', 'oracle_path_order'):
        if key in options:
            runtime[key] = options[key]
    return runtime


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
        runtime = job_runtime(executor.cfg, options)
        if options['mode'] == 'edited':
            from .captures import select_checkpoint
            references = select_checkpoint(checkpoint, runtime.get('oracle_capture_source', 'saved'), edited=True)
            references.require_viser_images()
            for view in references.views:
                references.image_path(view, repaired=True)
        update(phase='inputs', message='Staging saved OracleGS reference RGBs and calibrated cameras')
        # Stage original saved references locally, without recapture.
        import uuid
        preparation = root / ('oraclegs-inputs-' + uuid.uuid4().hex[:8])
        atomic_json(root / 'oraclegs-staging.json', {
            'version': 1, 'input_manifest': str((preparation / 'inputs/inputs.json').relative_to(root)),
            'status': 'exporting', 'adapter': runtime['oracle_adapter']})
        export_inputs(checkpoint, preparation / 'inputs', options['mode'], runtime=runtime, should_stop=stop)
        import hashlib
        input_manifest = preparation / 'inputs/inputs.json'
        staging = {'version': 1, 'input_manifest': str(input_manifest.relative_to(root)),
                   'manifest_sha256': hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
                   'status': 'exported', 'remote_inputs': remote + '/inputs',
                   'adapter': runtime['oracle_adapter']}
        atomic_json(root / 'oraclegs-staging.json', staging)
        if stop():
            raise InterruptedError('Cancelled before OracleGS staging')
        update(phase='staging', message='Staging OracleGS calibrated RGB inputs')
        lrz.sync_code_to_dss(cfg)
        ssh('mkdir -p ' + shlex.quote(remote + '/inputs'))
        sync(str(preparation / 'inputs') + '/', endpoint + ':' + remote + '/inputs/')
        atomic_json(root / 'oraclegs-staging.json', {**staging, 'status': 'transferred'})
        request = {'reconstruction_method': 'oraclegs', 'stage': 'repair', 'mode': options['mode'],
                   'inputs': container + '/inputs', 'output': container + '/results/oraclegs', 'runtime': runtime,
                   'stop_at_epoch': executor.store.get_run(run_id).state.details.get('effective_deadline')}
        atomic_json(root / 'worker-request.json', request)
        sync(str(root / 'worker-request.json'), endpoint + ':' + remote + '/worker-request.json')
        inner = lrz._remote_pythonpath_exports(cfg) + shlex.join([
            'python', '-u', '-m', 'splat_explorer.splatfix.job_worker', '--run-dir', container])
        command = lrz.container_srun_prefix(cfg) + 'bash -lc ' + shlex.quote(inner)
        wrapped = command + "; printf '%s\\n' $? > " + shlex.quote(remote + '/launcher-exit')
        launcher = ('mkdir ' + shlex.quote(remote + '/launched') + ' && (nohup bash -lc ' + shlex.quote(wrapped)
                    + ' > ' + shlex.quote(remote + '/stage.log') + ' 2>&1 < /dev/null &)')
        update(remote_dir=remote, phase='oraclegs', message='Launching official OracleGS')
        ssh(launcher)
    state, last = {}, None
    last_log = 0.
    last_preview = 0.
    preview_phase = None
    def mirror_log():
        nonlocal last_log
        if time.monotonic() - last_log < 60:
            return
        last_log = time.monotonic()
        phase = state.get('log_phase', state.get('phase', ''))
        relative = 'results/oraclegs/' + phase + '.log' if phase.isidentifier() else 'stage.log'
        try:
            text = ssh('if [ -f ' + shlex.quote(remote + '/' + relative) + ' ]; then tail -c 65536 ' +
                       shlex.quote(remote + '/' + relative) + '; fi')
            if text:
                destination = root / 'gpu/live-stage.log'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text)
                atomic_json(destination.with_suffix('.json'), {'phase': 'OracleGS', 'relative_path': relative})
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
        # Queue-owned observation, independent of browser tabs. Retry incomplete
        # transfers; rsync publishes each file atomically (no --inplace).
        phase = state.get('phase')
        now = time.monotonic()
        if now - last_preview >= 60 and (phase != preview_phase or phase in
                ('seva_generation', 'vggt_attention') or preview_phase is None):
            last_preview = now
            try:
                transferred = mirror_previews(lrz, cfg, endpoint, remote, root, phase=phase)
                preview_phase = phase if transferred else None
            except Exception:
                preview_phase = None
        if state:
            atomic_json(root / 'worker-status.json', state)
        if state and state != last:
            update(phase=state.get('phase', 'oraclegs'), message=state.get('message', 'OracleGS running'))
            last = state
        if state.get('status') in ('completed', 'error', 'stopped'):
            break
        exited = ssh('if [ -f ' + shlex.quote(remote + '/launcher-exit') + ' ]; then cat ' +
                     shlex.quote(remote + '/launcher-exit') + '; fi')
        if exited.strip():
            state = read_status()
            if state.get('status') not in ('completed', 'error', 'stopped'):
                state = {'status': 'error', 'message': 'OracleGS launcher exited without terminal worker status: ' + exited.strip()}
            break
        # Remain responsive to stop/deadline without repeatedly querying SSH.
        for _ in range(10):
            time.sleep(1)
            if stop():
                break
    # Connection/transfer failures keep the remote identity unresolved, allowing
    # reattachment and transfer retry before the shared lease is released.
    sync(endpoint + ':' + remote + '/', str(root / 'gpu') + '/')
    update(remote_finished=True)
    if state['status'] == 'stopped' or stop():
        raise InterruptedError('OracleGS stopped; available artifacts downloaded')
    if state['status'] != 'completed':
        raise RuntimeError(state.get('message', 'OracleGS failed'))
    return {'reconstruction_method': 'oraclegs', 'results': str(root / 'gpu/results'),
            'mode': options['mode'], 'checkpoint': options['checkpoint']}
