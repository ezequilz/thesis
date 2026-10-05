"""Execute independent dashboard jobs without entering the old exploration loop."""
from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .checkpoint import Checkpoint, atomic_json
from .jobs import read_json


def persist_checkpoint_caches(downloaded, checkpoint_root):
    """Publish reusable GPU caches without dereferencing remote source links."""
    import shutil
    import tempfile
    from .repair import digest_file
    downloaded, checkpoint_root = Path(downloaded), Path(checkpoint_root)
    trajectories = downloaded / 'trajectories'
    if trajectories.is_dir():
        # Metric alignment lives inside each trajectory. Its measurement image
        # links are temporary inputs, not replay data. Omit them rather than
        # dereferencing links to the old container or preserving dangling links.
        shutil.copytree(trajectories, checkpoint_root / 'trajectories',
                        dirs_exist_ok=True, symlinks=True,
                        ignore=lambda directory, names: [name for name in names
                            if (Path(directory) / name).is_symlink()])
    captions = downloaded / 'captions'
    if not captions.is_dir():
        return
    destination = checkpoint_root / 'captions'
    destination.mkdir(exist_ok=True)
    for entry in sorted(captions.iterdir()):
        if entry.name.startswith('.') or entry.is_symlink() or not entry.is_dir():
            continue
        manifest, artifact = entry / 'manifest.json', entry / 'caption.h5'
        if not manifest.is_file() or not artifact.is_file():
            raise ValueError('Committed caption cache is missing its manifest or HDF5')
        if manifest.is_symlink() or artifact.is_symlink():
            raise ValueError('Caption cache manifest and HDF5 must be regular files')
        metadata = json.loads(manifest.read_text())
        if metadata.get('signature') != entry.name or digest_file(artifact) != metadata.get('sha256'):
            raise ValueError('Downloaded caption cache failed signature/checksum validation')
        target = destination / entry.name
        if target.exists():
            if target.is_symlink() or (target / 'caption.h5').is_symlink() or digest_file(target / 'caption.h5') != metadata['sha256']:
                raise ValueError('Existing canonical caption cache conflicts with downloaded artifact')
            existing = json.loads((target / 'manifest.json').read_text())
            if any(existing.get(key) != metadata.get(key) for key in ('signature', 'sha256', 'recipe')):
                raise ValueError('Existing canonical caption manifest conflicts with downloaded artifact')
            continue
        temporary = Path(tempfile.mkdtemp(prefix='.' + entry.name + '.', dir=destination))
        try:
            # source/ contains only temporary input links. The HDF5 and recipe
            # manifest are sufficient for cache replay in a different job.
            shutil.copy2(manifest, temporary / 'manifest.json')
            shutil.copy2(artifact, temporary / 'caption.h5')
            if digest_file(temporary / 'caption.h5') != metadata['sha256']:
                raise ValueError('Caption HDF5 changed during cache publication')
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def terminate(process):
    """Terminate only the stage's own process group, including model children."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_process(argv, log_path, should_stop, on_tick=lambda: None):
    with Path(log_path).open('w') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if should_stop():
                    raise InterruptedError('Cancelled by user')
                on_tick()
                time.sleep(.5)
            if process.returncode:
                tail = Path(log_path).read_text(errors='replace')[-4000:]
                raise RuntimeError(f'Stage exited with code {process.returncode}: {tail}')
        finally:
            terminate(process)


class SplatfixExecutor:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store

    def execute(self, run_id):
        # Keep the shared lease held while a detached GPU job is unresolved.
        # This also blocks manual repairs from racing a reconnecting stage.
        while True:
            result = self._execute_attempt(run_id)
            state = self.store.get_run(run_id).state
            if not (state.status.value == 'queued' and state.details.get('remote_dir')
                    and not state.details.get('remote_finished')):
                return result
            time.sleep(5)

    def _execute_attempt(self, run_id):
        run = self.store.get_run(run_id)
        options = run.config.splatfix
        root = self.store.run_path(run_id)
        stop = lambda: self.store.stop_requested(run_id)
        update = lambda **fields: self.store.update_status(run_id, **fields)
        if stop() and not run.state.details.get('remote_dir'):
            update(status='stopped', message='Cancelled before starting')
            return {}
        update(status='running', phase=options['stage'], message='Starting ' + options['stage'])
        try:
            if options['stage'] in ('repair', 'benchmark'):
                result = self._remote(run_id, options, root, stop, update)
            else:
                result = self._local(run, root, stop, update)
            if stop():
                raise InterruptedError('Cancelled by user')
            update(status='completed', phase='finished', message='Stage completed', result=result)
            return result
        except InterruptedError as exc:
            update(status='stopped', phase='finished', message=str(exc))
            return {}
        except Exception as exc:
            details = self.store.get_run(run_id).state.details
            if details.get('remote_dir') and not details.get('remote_finished'):
                update(status='queued', message='GPU connection interrupted; reattaching before any other job: ' + str(exc), error=str(exc))
            else:
                update(status='error', phase='finished', message=str(exc), error=str(exc))
            return {}

    def _local(self, run, root, stop, update):
        import yaml
        from ..scene.catalog import apply_spec, spec_by_id
        cfg = copy.deepcopy(self.cfg)
        cfg['agent']['vlm_backend'] = 'cli_relay'
        if run.config.splatfix['stage'] == 'select':
            apply_spec(cfg, spec_by_id(cfg, run.config.scene_id))
        # Persist parameters, never resolved environment credentials.
        config_path = root / 'stage-config.yaml'
        config_path.write_text(yaml.safe_dump(dict(cfg)))
        options = run.config.splatfix
        argv = [sys.executable, '-u', '-m', 'splat_explorer.splatfix.local_worker', '--owner-pid', str(os.getpid()), '--config', str(config_path)]
        if options['stage'] == 'select':
            argv += ['select', '--scene', str(Path(cfg.scene.path).resolve()), '--output', str(root / 'checkpoints'),
                     '--views', str(options['views']), '--renderer', options['renderer'], '--select-only']
        else:
            cp = Checkpoint.load(options['checkpoint'])
            if not cp.complete:
                raise ValueError('View selection must finish before GPT-image editing')
            argv += ['edit', str(cp.root)]
        def progress():
            paths = list((root / 'checkpoints').glob('*/checkpoint.json')) if options['stage'] == 'select' else [Path(options['checkpoint']) / 'checkpoint.json']
            if paths:
                manifest = read_json(paths[0], {})
                views = manifest.get('views', [])
                selected = len(views)
                edited = sum(bool(v.get('repaired_rgb')) for v in views)
                previous = getattr(progress, 'counts', None)
                if previous != (selected, edited):
                    progress.counts = selected, edited
                    update(checkpoint=str(paths[0].parent), selected_views=selected, edited_views=edited,
                           target_views=manifest.get('target_views'), message=f'{selected} views saved · {edited} images repaired')
        run_process(argv, root / 'stage.log', stop, progress)
        progress()
        return {'checkpoint': self.store.get_run(run.run_id).state.details.get('checkpoint')}

    def _remote(self, run_id, options, root, stop, update):
        from .. import repair_lrz as lrz
        from ..scene_runs.lrz_transport import LrzSceneRunTransport
        benchmark = options['stage'] == 'benchmark'
        cp = None if benchmark else Checkpoint.load(options['checkpoint'])
        if cp is not None and (not cp.complete or (options['mode'] == 'edited' and any(not v.get('repaired_rgb') for v in cp.views))):
            raise ValueError('Checkpoint is not ready for the requested reconstruction mode')
        cfg = lrz.load_lrz_config()
        saved = self.store.get_run(run_id).state.details
        # Reattachment only needs DSS/SSH, even if the original allocation expired.
        if not saved.get('remote_dir'):
            LrzSceneRunTransport(self.cfg, run_id, root).validate()
        remote_dir = saved.get('remote_dir') or str(cfg['workspace']).rstrip('/') + '/splatfix-jobs/' + run_id
        container_dir = '/workspace/splatfix-jobs/' + run_id
        endpoint = f"{cfg['user']}@{cfg['host']}"
        def ssh(command):
            answer = lrz._ssh_run(cfg, command, timeout=30)
            if answer.returncode:
                raise RuntimeError((answer.stderr or answer.stdout or 'Remote command failed').strip())
            return answer.stdout
        def sync(source, destination):
            lrz._mux_run(['rsync', '-az', '-e', lrz.rsync_ssh_cmd(cfg), source, destination])
        resuming = bool(self.store.get_run(run_id).state.details.get('remote_dir'))
        if resuming:
            resuming = bool(ssh('test ! -d ' + shlex.quote(remote_dir + '/launched') + ' || printf launched').strip())
        if not resuming:
            update(message='Staging checkpoint and source on the configured LRZ allocation', phase='staging')
            lrz.sync_code_to_dss(cfg)
            from .repair import DEFAULT_RUNTIME
            runtime = {**DEFAULT_RUNTIME, **dict(self.cfg.get('splatfix', {}).get('runtime', {}))}
            if benchmark:
                from .jobs import registered_benchmark
                source, _ = registered_benchmark(Path(self.cfg.output.dir) / 'benchmarks', options['source'])
                from ..scene_runs_ext.config import model_runtime
                runtime = model_runtime(options['model'], runtime)
                if options.get('resume_from'):
                    from .resume import resolve_remote_resume
                    runtime['resume_from'] = resolve_remote_resume(self.store, options, cfg, ssh)
                if options.get('preparation_from'):
                    from .resume import resolve_remote_preparation
                    runtime['preparation_from'] = resolve_remote_preparation(self.store, options, cfg, ssh)
                ssh('mkdir -p ' + shlex.quote(remote_dir + '/benchmark-input'))
                sync(str(source) + '/', endpoint + ':' + remote_dir + '/benchmark-input/')
                request = {'stage': 'benchmark', 'source': container_dir + '/benchmark-input',
                           'output': container_dir + '/results', 'runtime': runtime}
            else:
                ssh('mkdir -p ' + shlex.quote(remote_dir + '/checkpoint') + ' ' + shlex.quote(remote_dir + '/source'))
                sync(str(cp.root) + '/', endpoint + ':' + remote_dir + '/checkpoint/')
                source = Path(cp.manifest['scene_path'])
                from ..scene.catalog import openable_scene_path
                source = openable_scene_path(source)
                if not source.exists() and not list((cp.root / 'trajectories').glob('*/trajectory.json')):
                    raise FileNotFoundError('Source scene is unavailable for GPU trajectory preparation')
                directory = source if source.is_dir() else source.parent if source.name in ('meta.json', 'lod-meta.json') else None
                if directory:
                    sync(str(directory) + '/', endpoint + ':' + remote_dir + '/source/')
                    remote_scene = container_dir + '/source/' + (source.name if source.is_file() else '')
                else:
                    if source.exists():
                        sync(str(source), endpoint + ':' + remote_dir + '/source/')
                    remote_scene = container_dir + '/source/' + source.name
                runtime['scene_path'] = remote_scene
                request = {'checkpoint': container_dir + '/checkpoint', 'output': container_dir + '/results',
                           'runtime': runtime, **{k: options[k] for k in ('mode', 'frames', 'span_fraction')}}
            atomic_json(root / 'worker-request.json', request)
            sync(str(root / 'worker-request.json'), endpoint + ':' + remote_dir + '/worker-request.json')
            inner = lrz._remote_pythonpath_exports(cfg) + shlex.join(['python', '-u', '-m', 'splat_explorer.splatfix.job_worker', '--run-dir', container_dir])
            command = lrz.container_srun_prefix(cfg) + 'bash -lc ' + shlex.quote(inner)
            # Stable job directory and exclusive marker prevent a restarted manager from replaying GPU work.
            wrapped = command + "; printf '%s\\n' $? > " + shlex.quote(remote_dir + '/launcher-exit')
            launcher = ('mkdir ' + shlex.quote(remote_dir + '/launched') + ' && (nohup bash -lc ' + shlex.quote(wrapped)
                        + ' > ' + shlex.quote(remote_dir + '/stage.log') + ' 2>&1 < /dev/null &)')
            if stop():
                raise InterruptedError('Cancelled before GPU launch')
            # Save location before launch so a manager restart can reattach safely.
            update(phase='reconstruction', message='Authors’ fresh reconstruction running on LRZ', remote_dir=remote_dir)
            ssh(launcher)
        from .live_logs import PhaseLogMirror
        log_mirror = PhaseLogMirror(root, container_dir, remote_dir, ssh)
        last = None
        started = time.monotonic()
        try:
            while True:
                if stop():
                    ssh('touch ' + shlex.quote(remote_dir + '/STOP'))
                raw = ssh('if [ -f ' + shlex.quote(remote_dir + '/worker-status.json') + ' ]; then cat ' + shlex.quote(remote_dir + '/worker-status.json') + '; fi')
                state = json.loads(raw) if raw.strip() else {}
                log_mirror.poll(state)
                if state != last and state:
                    last = state
                    update(phase=state.get('phase', 'reconstruction'), message=state.get('message') or state.get('phase', 'Running'))
                if state.get('status') in ('completed', 'error', 'stopped'):
                    update(remote_finished=True)
                    break
                exited = ssh('if [ -f ' + shlex.quote(remote_dir + '/launcher-exit') + ' ]; then cat ' + shlex.quote(remote_dir + '/launcher-exit') + '; fi')
                if exited.strip():
                    update(remote_finished=True)
                    raise RuntimeError('GPU launcher exited before stage completion (code ' + exited.strip() + '); inspect the GPU stage log')
                if time.monotonic() - started > 24 * 3600:
                    raise TimeoutError('GPU stage exceeded 24 hours; inspect remote stage.log')
                time.sleep(2)
        except BaseException:
            ssh('touch ' + shlex.quote(remote_dir + '/STOP'))
            raise
        finally:
            # Pull partial results/logs as well; checkpoint trajectories are reusable by both arms.
            try:
                lrz._mux_run(['rsync', '-az', '--exclude=source/', '--exclude=benchmark-input/', '-e', lrz.rsync_ssh_cmd(cfg),
                              endpoint + ':' + remote_dir + '/', str(root / 'gpu') + '/'])
            except lrz.RemoteCommandError as exc:
                # Keep the initial SSH error as well as the rsync tail; the UI
                # summary intentionally truncates diagnostics and can hide it.
                diagnostic = root / 'artifact-transfer.log'
                temporary = diagnostic.with_suffix('.log.tmp')
                temporary.write_text(f'Artifact transfer exited with code {exc.returncode}\n'
                                     f'\nSTDERR\n{exc.stderr}\nSTDOUT\n{exc.stdout}')
                os.replace(temporary, diagnostic)
                raise
            if cp is not None:
                persist_checkpoint_caches(root / 'gpu' / 'checkpoint', cp.root)
        if state['status'] == 'stopped' or stop():
            raise InterruptedError('GPU reconstruction cancelled')
        if state['status'] != 'completed':
            raise RuntimeError(state.get('message', 'GPU reconstruction failed'))
        return {'results': str(root / 'gpu' / 'results'), 'mode': options['mode'],
                **({'source': options['source'], 'model': options['model']} if benchmark else {'checkpoint': str(cp.root)})}
