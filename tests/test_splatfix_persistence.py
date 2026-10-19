import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from splat_explorer.splatfix import persistence as p


def test_portable_image_survives_move(tmp_path):
    root = tmp_path / 'run'
    gpu = root / 'gpu'
    (gpu / 'images').mkdir(parents=True)
    (gpu / 'images/frame.png').write_bytes(b'image')
    (gpu / 'pred.png').symlink_to('/workspace/splatfix-jobs/run/images/frame.png')
    result = p.portable_links(root, 'run')
    assert result['repaired_links'] == 1
    assert os.readlink(gpu / 'pred.png') == 'images/frame.png'
    root.rename(tmp_path / 'moved')
    assert (tmp_path / 'moved/gpu/pred.png').read_bytes() == b'image'


def test_nested_checkpoint_and_runtime_links(tmp_path):
    gpu = tmp_path / 'gpu'
    (gpu / 'checkpoint/views').mkdir(parents=True)
    (gpu / 'checkpoint/views/ref.png').write_bytes(b'reference')
    (gpu / 'ref.png').symlink_to('/workspace/splatfix-jobs/run/results/edited/checkpoint/views/ref.png')
    (gpu / 'corrected-runtime').mkdir()
    (gpu / 'corrected-runtime/shared.png').symlink_to('/workspace/third_party/shared.png')
    p.portable_links(tmp_path, 'run')
    assert (gpu / 'ref.png').read_bytes() == b'reference'


def test_missing_image_blocks_persistence(tmp_path):
    (tmp_path / 'gpu').mkdir()
    (tmp_path / 'gpu/missing.png').symlink_to('/workspace/splatfix-jobs/run/missing.png')
    with pytest.raises(ValueError, match='not preserved'):
        p.portable_links(tmp_path, 'run')


def test_external_image_is_not_archived(tmp_path):
    root = tmp_path / 'run'
    (root / 'gpu').mkdir(parents=True)
    (tmp_path / 'external.png').write_bytes(b'external')
    (root / 'gpu/external.png').symlink_to(tmp_path / 'external.png')
    with pytest.raises(ValueError, match='not preserved'):
        p.portable_links(root, 'run')


def transport(tmp_path, *, fail=False, missing=False):
    events = []
    def ssh(cfg, command, timeout):
        events.append('delete' if 'shutil.rmtree' in command else 'exit')
        return SimpleNamespace(returncode=0, stdout='removed' if 'shutil.rmtree' in command else 'exited', stderr='')
    def transfer(argv):
        events.append('download')
        assert '-azc' in argv
        if fail:
            raise RuntimeError('transfer failed')
        if missing:
            (tmp_path / 'gpu/missing.png').symlink_to('/workspace/splatfix-jobs/run/missing.png')
    return SimpleNamespace(_ssh_run=ssh, _mux_run=transfer, rsync_ssh_cmd=lambda cfg: 'ssh'), events


@pytest.mark.parametrize('failure', ['transfer', 'image'])
def test_failed_persistence_never_deletes(tmp_path, failure):
    lrz, events = transport(tmp_path, fail=failure == 'transfer', missing=failure == 'image')
    with pytest.raises((RuntimeError, ValueError)):
        p.persist_remote_run(lrz, {'workspace': '/dss', 'user': 'u', 'host': 'h'}, tmp_path, 'run', '/dss/splatfix-jobs/run')
    assert 'delete' not in events
    assert json.loads((tmp_path / p.RECEIPT).read_text())['status'] == 'downloading'


def test_success_deletes_after_verified_download(tmp_path):
    lrz, events = transport(tmp_path)
    p.persist_remote_run(lrz, {'workspace': '/dss', 'user': 'u', 'host': 'h'}, tmp_path, 'run', '/dss/splatfix-jobs/run')
    assert events == ['exit', 'download', 'delete']
    assert json.loads((tmp_path / p.RECEIPT).read_text())['remote_cleanup'] == 'removed'


def test_cleanup_guard_retains_active_worker(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    result = subprocess.run(['python3', '-c', p._DELETE, str(job)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == 'pending'
    assert job.exists()
    (job / 'launcher-exit').write_text('0')
    subprocess.run(['python3', '-c', p._DELETE, str(job)], check=True, capture_output=True)
    assert not job.exists()


def test_cleanup_failure_keeps_verified_receipt_for_retry(tmp_path):
    lrz, events = transport(tmp_path)
    original = lrz._ssh_run
    def ssh(cfg, command, timeout):
        if 'shutil.rmtree(p)' in command:
            return SimpleNamespace(returncode=255, stdout='', stderr='connection lost')
        return original(cfg, command, timeout)
    lrz._ssh_run = ssh
    cfg = {'workspace': '/dss', 'user': 'u', 'host': 'h'}
    p.persist_remote_run(lrz, cfg, tmp_path, 'run', '/dss/splatfix-jobs/run')
    receipt = json.loads((tmp_path / p.RECEIPT).read_text())
    assert receipt['status'] == 'persisted'
    assert receipt['remote_cleanup'] == 'pending'
    lrz._ssh_run = original
    p.cleanup_persisted(lrz, cfg, tmp_path, 'run')
    assert json.loads((tmp_path / p.RECEIPT).read_text())['remote_cleanup'] == 'removed'


def test_restore_archived_benchmark_before_remote_resume(tmp_path, monkeypatch):
    from splat_explorer.splatfix.artifixer.resume import _resolve_remote_result
    from splat_explorer import repair_lrz as lrz
    (tmp_path / 'gpu').mkdir()
    remote = '/dss/splatfix-jobs/run'
    (tmp_path / p.RECEIPT).write_text(json.dumps({'status': 'persisted', 'remote_dir': remote, 'remote_cleanup': 'removed'}))
    prior = SimpleNamespace(run_id='run', path=tmp_path, state=SimpleNamespace(details={'remote_dir': remote}))
    events = []
    def ssh(command):
        events.append(command)
        return json.dumps({'relative': 'results/benchmark_saved'})
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: events.append(argv))
    result = _resolve_remote_result(prior, {'workspace': '/dss', 'user': 'u', 'host': 'h'}, ssh)
    assert result == '/workspace/splatfix-jobs/run/results/benchmark_saved'
    assert events[0].startswith('mkdir -p ')
    assert events[1][-2:] == [str(tmp_path / 'gpu') + '/', 'u@h:' + remote + '/']
    assert 'benchmark-run.json' in events[2]
    assert json.loads((tmp_path / p.RECEIPT).read_text())['remote_cleanup'] == 'pending'


def test_running_launcher_does_not_download_or_delete(tmp_path):
    lrz, events = transport(tmp_path)
    lrz._ssh_run = lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='pending', stderr='')
    with pytest.raises(RuntimeError, match='not exited'):
        p.persist_remote_run(lrz, {'workspace': '/dss', 'user': 'u', 'host': 'h'}, tmp_path, 'run', '/dss/splatfix-jobs/run')
    assert events == []
