"""Allocation-specific cancellation guards; no LRZ requests are made."""
from types import SimpleNamespace

import pytest

from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.web.repair_studio import RepairStudio


@pytest.mark.parametrize("owner", [
    {"run_id": "gpu-setup", "job_id": "5866526"},
    {"run_id": "gpu-setup", "kind": "setup"},  # Existing setup lease.
    {"run_id": "scene-run", "job_id": "5866526"},
])
def test_cancel_only_blocks_owned_allocation(monkeypatch, owner):
    studio = RepairStudio(SimpleNamespace())
    monkeypatch.setattr(studio, "_scene_run_gpu_owner", lambda: owner)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {"job_id": "5866526"})
    calls = []

    def cancel(job_id, *, confirm):
        calls.append((job_id, confirm))
        return {"message": "Cancelled"}

    monkeypatch.setattr("splat_explorer.repair_lrz.cancel_lrz_job", cancel)
    monkeypatch.setattr(studio, "gpu_snapshot", lambda **kwargs: {})
    assert studio.gpu_cancel_job("5866602", confirm=True)["ok"] is True
    assert calls == [("5866602", True)]
    with pytest.raises(RuntimeError, match="reserved by|setup is still active"):
        studio.gpu_cancel_job("5866526", confirm=True)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="switching LRZ jobs"):
        studio.gpu_use_job("5866602")


def test_setup_lease_records_allocation(tmp_path):
    store = SceneRunStore(tmp_path)
    lease = store.acquire_setup_lease(job_id="5866526")
    assert lease is not None
    try:
        owner = store.gpu_lease_owner()
        assert owner["job_id"] == "5866526"
        assert owner["run_id"] == "gpu-setup"
    finally:
        lease.release()


def test_explicit_lease_allocation_takes_precedence(monkeypatch):
    studio = RepairStudio(SimpleNamespace())
    monkeypatch.setattr(studio, "_scene_run_gpu_owner", lambda: {"run_id": "gpu-setup", "job_id": "42"})
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {"job_id": "99"})
    studio._require_gpu_available("cancelling", job_id="99")
    with pytest.raises(RuntimeError, match="reserved by|setup is still active"):
        studio._require_gpu_available("cancelling", job_id="42")


@pytest.mark.parametrize("kind,alive,local,recovered", [
    ("gpu-setup", False, True, True),
    ("gpu-setup", True, True, False),
    ("gpu-setup", False, False, False),
    ("scene-run", False, True, False),
])
def test_dashboard_recovers_only_dead_local_setup(tmp_path, monkeypatch, kind, alive, local, recovered):
    import json
    import socket
    from splat_explorer.scene_runs.store import GPU_OWNER_NAME

    store = SceneRunStore(tmp_path / "scene-runs")
    lease = store.acquire_setup_lease(job_id="42")
    owner = dict(lease.owner, run_id=kind, hostname=socket.gethostname() if local else "other-host")
    (store.gpu_lease_path / GPU_OWNER_NAME).write_text(json.dumps(owner))
    monkeypatch.setattr("splat_explorer.scene_runs.store._pid_alive", lambda pid: alive)
    studio = RepairStudio(SimpleNamespace(cfg=SimpleNamespace(output=SimpleNamespace(dir=tmp_path))))
    assert (studio._scene_run_gpu_owner() is None) == recovered
    assert store.gpu_lease_path.exists() != recovered
    if recovered:
        retry = store.acquire_setup_lease(job_id="42")
        assert retry is not None
        assert not lease.release()  # Old controller cannot release the new lease.
        retry.release()


@pytest.mark.parametrize("failed", [False, True])
def test_setup_worker_releases_lease_on_success_and_failure(tmp_path, monkeypatch, failed):
    from splat_explorer import repair_lrz as lrz

    store = SceneRunStore(tmp_path)
    lease = store.acquire_setup_lease(job_id="42")
    original = dict(lrz._SETUP)

    def setup(*args, **kwargs):
        if failed:
            raise RuntimeError("setup failed")
        return {"ok": True, "gpu": "test GPU"}

    monkeypatch.setattr(lrz, "setup_lrz_gpu", setup)
    monkeypatch.setattr(lrz, "finalize_setup_log", lambda **kwargs: None)
    try:
        lrz._SETUP["inflight"] = True
        lrz._run_setup_thread({"job_id": "42"}, setup_lease=lease)
        assert store.gpu_lease_owner() is None
        assert lrz._SETUP["inflight"] is False
        assert lrz._SETUP["ok"] is (not failed)
        retry = store.acquire_setup_lease(job_id="42")
        assert retry is not None
        retry.release()
    finally:
        lrz._SETUP.update(original)
