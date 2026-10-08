"""Execute independent dashboard jobs without entering the old exploration loop."""
from __future__ import annotations

import copy
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .checkpoint import Checkpoint
from .jobs import read_json


def persist_checkpoint_caches(*args, **kwargs):
    """Legacy entry point; cache publication belongs to the backend."""
    from .artifixer.execution import persist_checkpoint_caches as publish
    return publish(*args, **kwargs)


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
        from ..image_edit import overlay_image_edit_cfg
        cfg = copy.deepcopy(self.cfg)
        cfg = overlay_image_edit_cfg(cfg, backend=run.config.image_edit_backend)
        cfg['agent']['vlm_backend'] = 'cli_relay'
        cfg.setdefault('splatfix', {})['resolution_profile'] = run.config.splatfix['resolution_profile']
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
        from .methods import get_method
        return get_method(options.get('reconstruction_method'), stage=options['stage']).execute_remote(
            self, run_id, options, root, stop, update)
