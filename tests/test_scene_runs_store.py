"""Fast, filesystem-only tests for the scene-run control plane."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone

import pytest

from splat_explorer.agent.actions import Action
from splat_explorer.scene_runs import (
    RepairTrigger,
    RunStatus,
    SceneRunConfig,
    SceneRunStore,
    effective_deadline,
    parse_slurm_duration,
    parse_slurm_end,
    should_trigger_repair,
)
from splat_explorer.scene_runs.store import GPU_OWNER_NAME


NOW = datetime(2026, 9, 15, 18, 30, 45, tzinfo=timezone.utc)


def test_config_defaults_and_validation():
    config = SceneRunConfig()
    assert config.to_dict() == {
        "scene_id": "venetian-balcony",
        "backend": "cli_relay",
            "model": "gpt-5.6-luna",
        "width": 960,
        "height": 720,
        "duration_seconds": 3600,
        "send_map": True,
        "image_edit_backend": "qwen-image-edit",
        "repair_backend": "gsfix-gsplat",
        "repair_trigger": "regenerate_yes",
        "repair_type": "original",
        "repair_seconds": 180,
    }
    assert SceneRunConfig.from_dict({"model": "demo", "width": 640}).width == 640
    with pytest.raises(ValueError, match="send_map is fixed"):
        SceneRunConfig(send_map=False)
    with pytest.raises(ValueError, match="positive integer"):
        SceneRunConfig(duration_seconds=0)
    with pytest.raises(ValueError, match="repair_trigger"):
        SceneRunConfig(repair_trigger="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repair_type"):
        SceneRunConfig(repair_type="neon")  # type: ignore[arg-type]
    assert SceneRunConfig(repair_type="paper").repair_type.value == "original"
    assert SceneRunConfig(repair_type="loop").repair_type.value == "looped"
    assert SceneRunConfig.from_dict({"model": "demo", "future_flag": True}).model == "demo"


def test_create_collision_list_detail_status_events_and_stop(tmp_path):
    store = SceneRunStore(tmp_path, clock=lambda: NOW)
    first = store.create_run({"model": "first"}, now=NOW)
    second = store.create_run({"model": "second"}, now=NOW)

    assert first.run_id == "run_20260915_183045"
    assert second.run_id == "run_20260915_183045_2"
    assert [run.run_id for run in store.list_runs()] == [second.run_id, first.run_id]
    assert store.detail(first.run_id).config.model == "first"
    assert store.detail(first.run_id).state.status is RunStatus.QUEUED

    state = store.update_status(
        first.run_id,
        "running",
        message="worker started",
        details={"pid": 123},
    )
    assert state.status is RunStatus.RUNNING
    assert store.get_run(first.run_id).state.details == {"pid": 123}

    custom = store.append_event(first.run_id, {"event": "frame", "step": 4})
    assert custom["timestamp"].endswith("Z")
    events = store.read_events(first.run_id)
    assert [event["event"] for event in events] == ["created", "status", "frame"]

    marker = store.request_stop(first.run_id)
    assert marker.name == "STOP"
    assert store.request_stop(first.run_id) == marker
    assert store.stop_requested(first.run_id)
    assert store.get_run(first.run_id).stop_requested
    assert [event["event"] for event in store.read_events(first.run_id)].count(
        "stop_requested"
    ) == 1

    # The persisted files are directly consumable by non-Python workers.
    config_json = json.loads((tmp_path / first.run_id / "config.json").read_text())
    status_json = json.loads((tmp_path / first.run_id / "status.json").read_text())
    assert config_json["repair_trigger"] == "regenerate_yes"
    assert status_json["status"] == "running"


@pytest.mark.parametrize(
    "unsafe",
    ["../run_20260915_183045", "run_bad", "/tmp/run_20260915_183045", ""],
)
def test_run_path_rejects_unsafe_ids(tmp_path, unsafe):
    store = SceneRunStore(tmp_path)
    with pytest.raises(ValueError):
        store.run_path(unsafe)


def test_run_path_rejects_symlink_alias(tmp_path):
    store = SceneRunStore(tmp_path)
    run = store.create_run(now=NOW)
    alias = tmp_path / "run_20260915_183046"
    alias.symlink_to(run.path, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe run path"):
        store.get_run(alias.name)


def test_trigger_predicate_accepts_actions_and_dicts():
    report = Action("report_artifact", {"regenerate": "yes"})
    report_no = {"action": "report_artifact", "args": {"regenerate": "no"}}
    move = {"name": "move", "arguments": {"distance": 1}}

    assert should_trigger_repair(RepairTrigger.EVERY_STEP, move)
    assert should_trigger_repair("every_artifact", report)
    assert should_trigger_repair("every_artifact", report_no)
    assert should_trigger_repair("regenerate_yes", report)
    assert not should_trigger_repair("regenerate_yes", report_no)
    assert not should_trigger_repair("regenerate_yes", move)


def test_slurm_parsers_and_effective_deadline():
    assert parse_slurm_duration("2-03:04:05") == timedelta(
        days=2, hours=3, minutes=4, seconds=5
    )
    assert parse_slurm_duration("2-03:04") == timedelta(days=2, hours=3, minutes=4)
    assert parse_slurm_duration("2-03") == timedelta(days=2, hours=3)
    assert parse_slurm_duration("01:30:00") == timedelta(hours=1, minutes=30)
    assert parse_slurm_duration("15:20") == timedelta(minutes=15, seconds=20)
    assert parse_slurm_duration("90") == timedelta(minutes=90)
    assert parse_slurm_duration("UNLIMITED") is None
    with pytest.raises(ValueError):
        parse_slurm_duration("1:99")

    assert parse_slurm_end("Unknown") is None
    gpu_end = parse_slurm_end("2026-09-15T20:00:00Z")
    assert gpu_end == datetime(2026, 9, 15, 20, 0, tzinfo=timezone.utc)
    user_end = datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc)
    assert effective_deadline(user_end, gpu_end, safety_seconds=300) == datetime(
        2026, 9, 15, 19, 55, tzinfo=timezone.utc
    )
    assert effective_deadline(user_end, None) == user_end


def test_gpu_lease_contention_release_and_stale_local_pid_recovery(tmp_path):
    store = SceneRunStore(tmp_path, clock=lambda: NOW)
    first_run = store.create_run(now=NOW)
    second_run = store.create_run(now=NOW)

    first = store.acquire_gpu_lease(first_run.run_id, metadata={"job_id": "42"})
    assert first is not None
    assert first.owner["job_id"] == "42"
    assert store.acquire_gpu_lease(second_run.run_id) is None

    assert first.release()
    assert not first.release()
    second = store.acquire_gpu_lease(second_run.run_id)
    assert second is not None

    # Simulate a process that died while holding the lease on this host.
    owner_path = store.gpu_lease_path / GPU_OWNER_NAME
    stale_owner = dict(second.owner)
    stale_owner["pid"] = 2_000_000_000
    stale_owner["hostname"] = socket.gethostname()
    owner_path.write_text(json.dumps(stale_owner), encoding="utf-8")

    recovered = store.acquire_gpu_lease(first_run.run_id)
    assert recovered is not None
    assert recovered.owner["run_id"] == first_run.run_id
    assert not second.release()  # Its ownership token was replaced.
    assert recovered.release()
