from __future__ import annotations

import os
from pathlib import Path

from splat_explorer.config import Config, load_dotenv
from splat_explorer.scene_runs.manager import SceneRunManager
from splat_explorer.scene_runs.runner import _repair_trigger_state, _triggered
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


def test_shared_dotenv_loader_supplies_manager_credentials(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("CLIRELAY_API_KEY=test-key\n")
    monkeypatch.delenv("CLIRELAY_API_KEY", raising=False)
    assert load_dotenv(env_path) == ["CLIRELAY_API_KEY"]
    assert os.environ["CLIRELAY_API_KEY"] == "test-key"


def test_trigger_modes():
    move = Action("move", {"direction": "forward", "distance": 1})
    artifact_no = Action("report_artifact", {"regenerate": "no"})
    artifact_yes = Action("report_artifact", {"regenerate": "yes"})
    assert _triggered("every_step", move)
    assert not _triggered("every_artifact", move)
    assert _triggered("every_artifact", artifact_no)
    assert not _triggered("regenerate_yes", artifact_no)
    assert _triggered("regenerate_yes", artifact_yes)


def test_every_step_repairs_arm_on_first_artifact():
    move = Action("move", {"direction": "forward", "distance": 1})
    rotate = Action("rotate", {"yaw_degrees": 45})
    artifact = Action("report_artifact", {"regenerate": "no"})

    trigger, armed = _repair_trigger_state("every_step", move, False)
    assert (trigger, armed) == (False, False)
    trigger, armed = _repair_trigger_state("every_step", rotate, armed)
    assert (trigger, armed) == (False, False)
    trigger, armed = _repair_trigger_state("every_step", artifact, armed)
    assert (trigger, armed) == (True, True)
    trigger, armed = _repair_trigger_state("every_step", move, armed)
    assert (trigger, armed) == (True, True)


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


def test_manager_requeues_starting_runs_but_errors_running_ones(tmp_path):
    store = _Store()
    store.rows = [
        {"id": "run_start", "created_at": 1.0, "status": "starting"},
        {"id": "run_live", "created_at": 2.0, "status": "running"},
    ]
    store.details = {
        "run_start": {"id": "run_start", "config": {}, "status": {}},
        "run_live": {"id": "run_live", "config": {}, "status": {}},
    }
    manager = SceneRunManager(_cfg(tmp_path), store=store)
    manager._recover_interrupted()
    assert store.updates[0][0] == "run_start"
    assert store.updates[0][1]["status"] == "queued"
    assert store.updates[1][0] == "run_live"
    assert store.updates[1][1]["status"] == "error"
