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
    with pytest.raises(RuntimeError, match="reserved by"):
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
    with pytest.raises(RuntimeError, match="reserved by"):
        studio._require_gpu_available("cancelling", job_id="42")
