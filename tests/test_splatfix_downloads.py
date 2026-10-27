import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from splat_explorer.splatfix import downloads as d, persistence as p

ensure_worker = d.ensure_worker
CFG = {'workspace': '/dss', 'user': 'u', 'host': 'h'}


def queue(tmp_path, **kwargs):
    root = tmp_path / 'run'
    root.mkdir(exist_ok=True)
    d.enqueue(CFG, root, 'run', '/dss/splatfix-jobs/run', {'status': 'error'}, **kwargs)
    return root


def transport(events, fail=False):
    def ssh(cfg, command, timeout):
        events.append('delete' if 'shutil.rmtree' in command else 'exit')
        return SimpleNamespace(returncode=0, stdout='removed' if 'shutil.rmtree' in command else 'exited', stderr='')
    def transfer(argv):
        events.append(argv)
        if fail:
            raise RuntimeError('network unavailable')
    return SimpleNamespace(_ssh_run=ssh, _mux_run=transfer, rsync_ssh_cmd=lambda cfg: 'ssh')


def test_queue_is_durable_idempotent_and_does_no_network(tmp_path):
    root = queue(tmp_path)
    assert d.status(root)['status'] == 'queued'
    request = (root / d.REQUEST).read_bytes()
    d.enqueue(CFG, root, 'run', '/dss/splatfix-jobs/run', {'status': 'error'})
    assert (root / d.REQUEST).read_bytes() == request
    assert d.pending(tmp_path) == [root]
    assert d.active(root)
    with pytest.raises(ValueError, match='still downloading'):
        d.require_finished(root)


def test_explicit_discard_is_terminal_without_claiming_successful_archive(tmp_path):
    root = queue(tmp_path)
    (root / d.STATUS).write_text(json.dumps({'status': 'discarded'}))
    assert not d.pending(tmp_path)
    assert not d.active(root)
    assert not (root / p.RECEIPT).exists()


def test_failed_download_retries_without_touching_compute_outcome(tmp_path):
    root = queue(tmp_path)
    (root / 'status.json').write_text('{"status":"error","message":"GPU failure"}')
    events = []
    d.transfer(root, transport(events, fail=True))
    assert 'delete' not in events
    assert d.status(root)['status'] == 'retrying'
    assert 'network unavailable' in (root / 'artifact-transfer.log').read_text()
    assert json.loads((root / 'status.json').read_text())['message'] == 'GPU failure'
    d.transfer(root, transport(events))
    assert d.status(root)['status'] == 'completed'
    assert d.status(root)['attempt'] == 2
    assert events[-1] == 'delete'
    assert not d.pending(tmp_path)


def test_diagnostics_only_keeps_remote_inputs_and_skips_large_files(tmp_path):
    root = queue(tmp_path, diagnostics_only=True)
    events = []
    d.transfer(root, transport(events))
    argv = next(value for value in events if isinstance(value, list))
    assert '--include=*.log' in argv and '--include=*.json' in argv
    assert '--exclude=checkpoint/' in argv and '--exclude=*' in argv
    assert 'delete' not in events
    assert d.status(root)['status'] == 'diagnostics_only'
    assert json.loads((root / p.RECEIPT).read_text())['remote_cleanup'] == 'retained'


def test_cache_publication_precedes_remote_deletion_and_can_retry_locally(tmp_path, monkeypatch):
    cp = tmp_path / 'checkpoint'
    cp.mkdir()
    root = queue(tmp_path, checkpoint=cp)
    events = []
    def fail(*args):
        events.append('cache')
        raise ValueError('incomplete cache')
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.execution.persist_checkpoint_caches', fail)
    d.transfer(root, transport(events))
    assert 'delete' not in events
    assert d.status(root)['status'] == 'retrying'
    events.clear()
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.execution.persist_checkpoint_caches', lambda *a: events.append('cache'))
    d.transfer(root, transport(events))
    assert events == ['cache', 'delete']  # Download was already verified.


def test_worker_launch_uses_inherited_lock_and_independent_session(tmp_path, monkeypatch):
    queue(tmp_path)
    calls = []
    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        import fcntl
        with (tmp_path / '.artifact-download-worker.lock').open('a') as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(d.subprocess, 'Popen', popen)
    ensure_worker(tmp_path)
    argv, kwargs = calls[0]
    assert argv[2:4] == ['-m', d.__name__]
    assert kwargs['start_new_session'] and len(kwargs['pass_fds']) == 1
    assert 'src' in kwargs['env']['PYTHONPATH']


def test_worker_launch_failure_keeps_retryable_request(tmp_path, monkeypatch):
    def fail(root):
        raise OSError('launch unavailable')
    monkeypatch.setattr(d, 'ensure_worker', fail)
    root = queue(tmp_path)
    assert (root / d.REQUEST).is_file()
    assert d.status(root)['status'] == 'queued'
