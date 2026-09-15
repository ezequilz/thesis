from __future__ import annotations

from pathlib import Path

from splat_explorer.config import Config
from splat_explorer.scene_runs.manager import SceneRunManager
from splat_explorer.scene_runs.runner import _triggered
from splat_explorer.agent.actions import Action


class _Store:
    def __init__(self):
        self.rows = [{
            "id": "run_20260915_204300",
            "created_at": 1.0,
            "status": "queued",
        }]
        self.details = {
            self.rows[0]["id"]: {
                "id": self.rows[0]["id"],
                "config": {"duration_seconds": 3600},
                "status": {},
            },
        }
        self.updates = []
        self.released = False

    def list_runs(self):
        return list(self.rows)

    def detail(self, run_id):
        return self.details[run_id]

    def update_status(self, run_id, **fields):
        self.updates.append((run_id, fields))
        self.details[run_id]["status"].update(fields)

    def stop_requested(self, _run_id):
        return False

    def acquire_gpu_lease(self, run_id, **_kwargs):
        return {"owner": run_id}

    def release_gpu_lease(self, _lease):
        self.released = True


def _cfg(tmp_path: Path) -> Config:
    return Config({"output": {"dir": str(tmp_path)}})


def test_trigger_modes():
    move = Action("move", {"direction": "forward", "distance": 1})
    artifact_no = Action("report_artifact", {"regenerate": "no"})
    artifact_yes = Action("report_artifact", {"regenerate": "yes"})
    assert _triggered("every_step", move)
    assert not _triggered("every_artifact", move)
    assert _triggered("every_artifact", artifact_no)
    assert not _triggered("regenerate_yes", artifact_no)
    assert _triggered("regenerate_yes", artifact_yes)


def test_manager_runs_one_ready_item(tmp_path, monkeypatch):
    store = _Store()
    called = []

    class Executor:
        def __init__(self, cfg, passed_store):
            assert passed_store is store

        def execute(self, run_id):
            called.append(run_id)

    manager = SceneRunManager(
        _cfg(tmp_path),
        store=store,
        executor_factory=Executor,
        gpu_probe_seconds=30,
    )
    monkeypatch.setattr(manager, "_gpu_ready", lambda force=False: {
        "ready": True,
        "job_id": "1234",
        "state": "R",
    })
    assert manager.run_once() is True
    assert called == ["run_20260915_204300"]
    assert store.released is True
    assert any(fields.get("gpu", {}).get("job_id") == "1234"
               for _, fields in store.updates)
