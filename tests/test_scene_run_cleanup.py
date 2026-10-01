"""Remote cleanup must never discard an active or unarchived scene."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from splat_explorer import repair_lrz
from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport


@pytest.fixture
def transport(tmp_path, monkeypatch):
    cfg = dict(workspace=str(tmp_path / 'remote'), user='u', host='h')
    monkeypatch.setattr(repair_lrz, 'load_lrz_config', lambda: cfg)
    t = LrzSceneRunTransport({}, 'run_test', tmp_path / 'local' / 'run_test')
    t.run_dir.mkdir(parents=True)
    monkeypatch.setattr(repair_lrz, '_ssh_run', lambda cfg, cmd, **kw: subprocess.run(
        cmd, shell=True, capture_output=True, text=True))
    return t


def remote(t, worker=None, heartbeat=None):
    from pathlib import Path
    p = Path(t.remote_dir)
    p.mkdir(parents=True)
    if worker:
        (p / 'worker.json').write_text(json.dumps({'status': worker}))
    if heartbeat:
        (p / 'heartbeat.json').write_text(json.dumps({'status': heartbeat}))
    return p


@pytest.mark.parametrize('worker,heartbeat', [('ready','running'), ('stopped','running'), ('error',None)])
def test_unconfirmed_worker_is_retained(transport, worker, heartbeat):
    p = remote(transport, worker, heartbeat)
    assert not transport._archive_and_remove(transport.run_id, transport.run_dir)
    assert p.exists()


def test_failed_archive_retains_scene(transport, monkeypatch):
    p = remote(transport, 'stopped', 'stopped')
    def fail(argv):
        raise RuntimeError('disk full')
    monkeypatch.setattr(repair_lrz, '_mux_run', fail)
    with pytest.raises(RuntimeError, match='disk full'):
        transport._archive_and_remove(transport.run_id, transport.run_dir)
    assert p.exists()


def test_archive_precedes_deletion(transport, monkeypatch):
    p = remote(transport, 'stopped', 'stopped')
    calls = []
    def copy(argv):
        assert p.exists()
        calls.append(argv)
    monkeypatch.setattr(repair_lrz, '_mux_run', copy)
    assert transport._archive_and_remove(transport.run_id, transport.run_dir)
    assert not p.exists()
    assert '-azc' in calls[0]
    assert '/scene.ply' in calls[0]
    assert '/scene_repaired.ply' not in calls[0]


def test_partial_stage_is_cleaned(transport, monkeypatch):
    p = remote(transport)
    transport._staged = True
    monkeypatch.setattr(repair_lrz, '_mux_run', lambda argv: None)
    transport.close()
    assert not p.exists()


def test_symlink_is_not_removed(transport, tmp_path):
    from pathlib import Path
    p = Path(transport.remote_dir)
    p.parent.mkdir(parents=True)
    p.symlink_to(transport.run_dir, target_is_directory=True)
    assert not transport._cleanup_state(transport.run_id, delete=True)
    assert p.is_symlink()


def test_stage_uploads_only_config_and_source(transport, monkeypatch, tmp_path):
    source = tmp_path / 'source.ply'
    source.write_bytes(b'original')
    calls = []
    monkeypatch.setattr(transport, 'cleanup_finished_runs', lambda: None)
    monkeypatch.setattr(transport, '_worker_config', lambda config: {})
    monkeypatch.setattr(repair_lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(repair_lrz, '_mux_run', lambda argv: calls.append(argv))
    transport._stage(source, {})
    uploads = [c for c in calls if c[0] == 'rsync']
    assert len(uploads) == 2
    assert uploads[0][-2] == str(transport.run_dir / 'worker_config.json')
    assert uploads[1][-2] == str(source)


def test_close_timeout_retains_remote_scene(transport, monkeypatch):
    from splat_explorer.scene_runs import lrz_transport
    p = remote(transport, 'ready', 'running')
    transport._staged = True
    transport._launch_attempted = True
    monkeypatch.setattr(transport, 'stop', lambda: None)
    monkeypatch.setattr(transport, '_remote_json', lambda name: {'status': 'running'})
    ticks = iter([0, 31])
    monkeypatch.setattr(lrz_transport.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(RuntimeError, match='retained'):
        transport.close()
    assert p.exists()


def test_retry_only_locally_finished_runs(transport, monkeypatch):
    for name, state in [('run_done', 'completed'), ('run_active', 'running')]:
        p = transport.run_dir.parent / name
        p.mkdir()
        (p / 'status.json').write_text(json.dumps({'status': state}))
    calls = []
    monkeypatch.setattr(transport, '_archive_and_remove', lambda run, path: calls.append(run))
    transport.cleanup_finished_runs()
    assert calls == ['run_done']
