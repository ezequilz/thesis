"""Durable artifact transfers independent of the compute queue and GPU lease."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .checkpoint import atomic_json
from .jobs import read_json

REQUEST = 'artifact-download-request.json'
STATUS = 'artifact-download-status.json'
DONE = {'completed', 'diagnostics_only'}


def status(root):
    return read_json(Path(root) / STATUS, {})


def active(root):
    return (Path(root) / REQUEST).is_file() and status(root).get('status') not in DONE


def require_finished(root):
    if active(root):
        raise ValueError('Results are still downloading in background; wait before reusing or moving these artifacts')


def enqueue(cfg, root, run_id, remote, state, *, excludes=(), checkpoint=None,
            diagnostics_only=False):
    """Persist before returning; transfer failures never change compute status."""
    from .persistence import remote_path
    root = Path(root)
    if remote != remote_path(cfg, run_id):
        raise ValueError('Remote job is outside the configured workspace')
    if not (root / REQUEST).is_file():
        atomic_json(root / STATUS, {'status': 'queued', 'attempt': 0,
                                   'message': 'Results queued for background download'})
        atomic_json(root / REQUEST, {
            'version': 1, 'run_id': run_id, 'remote_dir': remote,
            'transport': {key: cfg[key] for key in ('host', 'user', 'workspace')},
            'worker_status': state, 'excludes': list(excludes),
            'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
            'diagnostics_only': diagnostics_only})
    # The request remains durable even if process launch fails. The manager
    # retries starting the download worker on its next heartbeat/restart.
    try:
        ensure_worker(root.parent)
    except Exception as exc:
        atomic_json(root / STATUS, {**status(root), 'status': 'queued',
                    'message': 'Download worker pending: ' + str(exc)})


def pending(root):
    return [path.parent for path in sorted(Path(root).glob('*/' + REQUEST))
            if status(path.parent).get('status') not in DONE]


def ensure_worker(root):
    """One detached downloader per queue, with an inherited lifetime lock."""
    root = Path(root).resolve()
    if not pending(root):
        return
    with (root / '.artifact-download-worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = src + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        with (root / 'artifact-download-worker.log').open('a') as log:
            subprocess.Popen([sys.executable, '-u', '-m', __name__, '--root', str(root),
                              '--lock-fd', str(lock.fileno())],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, pass_fds=(lock.fileno(),), env=env)


def transfer(root, lrz):
    """One retryable attempt. Original remote artifacts survive failed transfers."""
    from .persistence import RECEIPT, persist_remote_run, cleanup_persisted, wait_for_exit
    root = Path(root)
    request = read_json(root / REQUEST)
    cfg, remote = request['transport'], request['remote_dir']
    run_id = request['run_id']
    previous = status(root)
    attempt = int(previous.get('attempt', 0)) + 1
    atomic_json(root / STATUS, {'status': 'downloading', 'attempt': attempt,
                'message': 'Downloading results in background', 'started_at': time.time()})
    try:
        # A live preview may still be finishing after the GPU job exits. Wait
        # here, outside the compute queue, before either writer publishes files.
        from .persistence import artifact_lock
        with artifact_lock(root):
            if request.get('diagnostics_only'):
                wait_for_exit(lrz, cfg, remote)
                (root / 'gpu').mkdir(exist_ok=True)
                lrz._mux_run(['rsync', '-az', '--timeout=60', '--prune-empty-dirs',
                    '--exclude=checkpoint/', '--exclude=source/', '--exclude=inputs/',
                    '--exclude=benchmark-input/', '--exclude=worldexplorer-source/',
                    '--exclude=corrected-runtime/', '--include=*/', '--include=*.log',
                    '--include=*.json', '--include=launcher-exit', '--exclude=*',
                    '-e', lrz.rsync_ssh_cmd(cfg), f"{cfg['user']}@{cfg['host']}:{remote}/",
                    str(root / 'gpu') + '/'])
                atomic_json(root / RECEIPT, {'status': 'diagnostics_only', 'remote_dir': remote,
                            'remote_cleanup': 'retained', 'reason': 'Failed before generation'})
            else:
                receipt = read_json(root / RECEIPT, {})
                if receipt.get('status') != 'persisted' or receipt.get('remote_dir') != remote:
                    persist_remote_run(lrz, cfg, root, run_id, remote,
                                       excludes=request.get('excludes', ()), cleanup=False)
                if request.get('checkpoint'):
                    from .artifixer.execution import persist_checkpoint_caches
                    from .persistence import checkpoint_cache_lock
                    with checkpoint_cache_lock(request['checkpoint']):
                        persist_checkpoint_caches(root / 'gpu/checkpoint', request['checkpoint'])
                cleanup_persisted(lrz, cfg, root, run_id)
        atomic_json(root / STATUS, {'status': 'diagnostics_only' if request.get('diagnostics_only') else 'completed',
                    'attempt': attempt, 'finished_at': time.time(),
                    'message': ('Failure logs saved; bulk download skipped; remote files retained'
                                if request.get('diagnostics_only') else 'Results downloaded')})
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        (root / 'artifact-transfer.log').write_text(message + '\n\nSTDERR\n' +
            str(getattr(exc, 'stderr', '')) + '\nSTDOUT\n' + str(getattr(exc, 'stdout', '')))
        atomic_json(root / STATUS, {'status': 'retrying', 'attempt': attempt,
                    'message': 'Download failed; retrying in background', 'error': message,
                    'retry_at': time.time() + min(300, 15 * 2 ** min(attempt - 1, 5))})


def work(root):
    from .. import repair_lrz as lrz
    from .persistence import cleanup_persisted
    last_cleanup = 0
    while True:
        jobs = pending(root)
        for job in jobs:
            if status(job).get('retry_at', 0) <= time.time():
                transfer(job, lrz)
        if time.monotonic() - last_cleanup > 300:
            last_cleanup = time.monotonic()
            for path in Path(root).glob('*/' + REQUEST):
                if status(path.parent).get('status') == 'completed':
                    request = read_json(path)
                    cleanup_persisted(lrz, request['transport'], path.parent, request['run_id'])
        time.sleep(5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    with os.fdopen(args.lock_fd, 'a'):
        work(args.root)
