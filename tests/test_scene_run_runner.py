from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.agent.actions import Action
from splat_explorer.config import Config
from splat_explorer.scene.catalog import SceneSpec
from splat_explorer.scene_runs.runner import SceneRunExecutor, _require_vlm_response
from splat_explorer.scene_runs.store import SceneRunStore


class _Scene:
    def copy(self):
        return _Scene()


class _Renderer:
    def render(self, camera):
        return np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

    def render_depth(self, camera):
        return np.ones((camera.height, camera.width), dtype=np.float32)


def test_image_edit_prompt_is_static():
    from splat_explorer.agent.actions import Action
    from splat_explorer.scene_runs.gpu_worker import DEFAULT_PROMPT
    from splat_explorer.scene_runs.runner import _image_edit_prompt

    expected = (
        "Regenerate and fix this image. Repair artifacts, reconstruct plausible "
        "geometry and upscale to higher resolution."
    )
    assert DEFAULT_PROMPT == expected
    generic = _image_edit_prompt(Action("rotate", {"yaw_degrees": 15}), {})
    prompt = _image_edit_prompt(
        Action("report_artifact", {
            "description": "stretched smears on the right windows",
            "image_region": "right third",
            "severity": "high",
        }),
        {"image_edit_prompt": "do not use this override"},
    )
    assert generic == expected
    assert prompt == expected
    assert "stretched smears on the right windows" not in prompt
    policy = SimpleNamespace(last_debug={
        "backend": "cli_relay",
        "fallback": True,
        "attempts": [
            {"error": "AuthenticationError: 401"},
            {"error": "AuthenticationError: 401"},
            {"error": "AuthenticationError: 401"},
        ],
    })
    with pytest.raises(RuntimeError, match="refusing scripted rotation fallback"):
        _require_vlm_response(policy)


def test_scene_run_allows_fallback_after_real_unparseable_replies():
    policy = SimpleNamespace(last_debug={
        "backend": "cli_relay",
        "fallback": True,
        "attempts": [{"error": None, "reply": "not JSON"}],
    })
    _require_vlm_response(policy)


def _run_cfg(tmp_path: Path) -> Config:
    return Config({
        "output": {"dir": str(tmp_path)},
        "renderer": {
            "backend": "cpu_points",
            "width": 16,
            "height": 12,
            "fov_deg": 75.0,
        },
        "camera": {
            "up_axis": "+y",
            "start_yaw_deg": 0.0,
            "start_position": [0.0, 1.0, 0.0],
        },
        "agent": {
            "vlm_backend": "scripted",
            "model": "",
            "max_move_distance": 2.0,
            "max_rotate_degrees": 90.0,
        },
    })


def test_repair_reload_keeps_policy_and_pose_state(tmp_path, monkeypatch):
    now = datetime(2026, 9, 15, 18, 43, tzinfo=timezone.utc)
    store = SceneRunStore(tmp_path / "scene-runs", clock=lambda: now)
    created = store.create_run({
        "width": 16,
        "height": 12,
        "duration_seconds": 30,
        "repair_trigger": "regenerate_yes",
    }, now=now)
    cfg = _run_cfg(tmp_path)
    policy_calls = []
    renderers = []
    published = []

    class Policy:
        last_debug = {"backend": "fake"}

        def decide(self, _rgb, _pose, step, **kwargs):
            policy_calls.append((step, kwargs, self))
            if step == 0:
                return Action("report_artifact", {
                    "description": "floater",
                    "image_region": "center",
                    "severity": "high",
                    "regenerate": "yes",
                })
            store.request_stop(created.run_id)
            return Action("rotate", {"yaw_degrees": 15})

    policy = Policy()

    class Gpu:
        def start(self, **_kwargs):
            return {}

        def repair(self, **kwargs):
            path = store.run_path(created.run_id) / "scene_repaired.ply"
            path.write_bytes(b"repaired")
            return {
                "status": "ok",
                "checkpoint": str(path),
                "metrics": {"l1_before": 0.2, "l1_after": 0.1},
            }

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.save_ply",
        lambda _scene, path: Path(path).write_bytes(b"ply"),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.load_ply", lambda _path: _Scene(),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.make_renderer",
        lambda *_args: renderers.append(_Renderer()) or renderers[-1],
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.make_policy", lambda _cfg: policy,
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner._build_navigation",
        lambda *_args: (None, None),
    )

    class Executor(SceneRunExecutor):
        def _load_source(self, _params):
            return cfg, SceneSpec("venetian-balcony", "Venetian", Path("x")), _Scene()

        def _publish_and_wait(self, _spec, _ply, generation, **_kwargs):
            published.append(generation)

    executor = Executor(
        cfg,
        store,
        gpu_factory=lambda *_args: Gpu(),
    )
    detail = executor.execute(created.run_id)

    assert len(policy_calls) == 2
    assert policy_calls[0][2] is policy_calls[1][2] is policy
    assert len(renderers) == 2  # initial renderer, then repaired-scene renderer
    assert len(published) == 2
    assert policy_calls[0][1]["map_image"] is None
    assert detail["state"]["status"] == "stopped"
    actions = (store.run_path(created.run_id) / "actions.jsonl").read_text()
    assert actions.count('"step":') == 2
    assert not (store.run_path(created.run_id) / "step_00000_repair.png").exists()


@pytest.mark.parametrize("repair_trigger", ["every_step", "regenerate_yes", "every_artifact"])
def test_extended_report_uses_repair_resolution_only_for_the_image_model(tmp_path, monkeypatch, repair_trigger):
    now = datetime(2026, 9, 15, 18, 43, tzinfo=timezone.utc)
    store = SceneRunStore(tmp_path / "scene-runs", clock=lambda: now)
    created = store.create_run({
        "pipeline": "extended",
        "extended": {"repair_limit":1},
        "width": 32,
        "height": 32,
        "repair_width": 64,
        "repair_height": 64,
        "duration_seconds": 30,
        "repair_trigger": repair_trigger,
        "repair_backend": "artifixer-gsplat",
    }, now=now)
    cfg = _run_cfg(tmp_path)
    seen = []
    repairs = []

    class Policy:
        last_debug = {"backend": "fake"}
        _task = None
        _tools = []

        def decide(self, rgb, _pose, step, **_kwargs):
            from splat_explorer.tasks import artifact_hunt_3
            seen.append(rgb.shape)
            if step <= 1:
                assert self._task is artifact_hunt_3
                assert 'jump_to_waypoint' in [t['function']['name'] for t in self._tools]
            else:
                assert self._task is not artifact_hunt_3
                assert 'jump_to_waypoint' not in [t['function']['name'] for t in self._tools]
            if step == 0:
                return Action("rotate", {"yaw_degrees": 15})
            step -= 1
            if step == 0:
                return Action("report_artifact", {
                    "description": "floater",
                    "image_region": "center",
                    "severity": "high",
                    "regenerate": "no",
                })
            if step < 21:
                if step % 2:
                    return Action("move", {"direction":"right", "distance":1})
                return Action("rotate", {"yaw_degrees":-10})
            if step == 21:
                return Action("select_repair_views", {"views":[2,4,6,8,10]})
            store.request_stop(created.run_id)
            return Action("rotate", {"yaw_degrees": 15})

    class Gpu:
        def start(self, **_kwargs):
            return {}

        def render(self, *, step, camera, deadline, should_stop, purpose=""):
            rgb = np.full((camera.height, camera.width, 3), 90 if purpose else 20, np.uint8)
            depth = np.ones((camera.height, camera.width), np.float32)
            return rgb, depth

        def repair(self, **kwargs):
            repairs.append(kwargs)
            path = store.run_path(created.run_id) / "scene_repaired.ply"
            path.write_bytes(b"repaired")
            return {"status": "ok", "checkpoint": str(path)}

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.save_ply",
        lambda _scene, path: Path(path).write_bytes(b"ply"),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.load_ply", lambda _path: _Scene(),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.make_renderer", lambda *_args: _Renderer(),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.make_policy", lambda _cfg: Policy(),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner._build_navigation",
        lambda *_args: (None, None),
    )

    class Executor(SceneRunExecutor):
        def _load_source(self, _params):
            return cfg, SceneSpec("venetian-balcony", "Venetian", Path("x")), _Scene()

        def _publish_and_wait(self, *_args, **_kwargs):
            return None

    executor = Executor(cfg, store, gpu_factory=lambda *_args: Gpu())
    executor.execute(created.run_id)
    run_dir = store.run_path(created.run_id)

    records = [json.loads(line) for line in (run_dir / "actions.jsonl").read_text().splitlines()]
    entry = next(r for r in records if r["step"] == 1)
    returned = next(r for r in records if r["step"] == 22)["motion"]
    assert returned["kind"] == "local_return"
    for key in ("position", "yaw_deg", "pitch_deg"):
        assert returned[key] == entry[key]
    assert seen == [(32, 32, 3)] * 22 + [(128,96,3)]
    assert Image.open(run_dir / "step_00000.png").size == (32, 32)
    assert Image.open(run_dir / "step_00001_repair.png").size == (64, 64)
    assert Image.open(run_dir / "step_00022_repair.png").size == (64, 64)
    assert not (run_dir / "step_00000_repair.png").exists()
    assert repairs[0]["proposal"]["view_steps"] == [10,14,18,22]
    assert [Path(item["rendered_path"]).name for item in repairs] == [
        "step_00022_repair.png",
    ]
    assert all(item["camera"].width == 32 and item["camera"].height == 32 for item in repairs)


    # Tile 2 is the result of the second movement, not the final camera.
    assert np.linalg.norm(repairs[0]["camera"].position - np.array([0.,1.,0.])) == pytest.approx(2 * np.cos(np.radians(5)))

def test_report_artifact_saves_repair_frame_without_calling_the_image_model(tmp_path, monkeypatch):
    now = datetime(2026, 9, 15, 18, 44, tzinfo=timezone.utc)
    store = SceneRunStore(tmp_path / "scene-runs", clock=lambda: now)
    created = store.create_run({
        "pipeline": "extended",
        "width": 32,
        "height": 32,
        "repair_width": 64,
        "repair_height": 64,
        "duration_seconds": 30,
        "repair_trigger": "regenerate_yes",
    }, now=now)
    cfg = _run_cfg(tmp_path)
    repairs = []

    class Policy:
        last_debug = {"backend": "fake"}

        def decide(self, rgb, _pose, step, **_kwargs):
            assert rgb.shape == (32, 32, 3)
            if step == 0:
                return Action("report_artifact", {
                    "description": "smear",
                    "regenerate": "no",
                })
            store.request_stop(created.run_id)
            return Action("rotate", {"yaw_degrees": 10})

    class Gpu:
        def start(self, **_kwargs):
            return {}

        def render(self, *, camera, purpose="", **_kwargs):
            return (
                np.full((camera.height, camera.width, 3), 30, np.uint8),
                np.ones((camera.height, camera.width), np.float32),
            )

        def repair(self, **kwargs):
            repairs.append(kwargs)
            return {"status": "ok"}

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.save_ply",
        lambda _scene, path: Path(path).write_bytes(b"ply"),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.make_renderer", lambda *_args: _Renderer(),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner.make_policy", lambda _cfg: Policy(),
    )
    monkeypatch.setattr(
        "splat_explorer.scene_runs.runner._build_navigation",
        lambda *_args: (None, None),
    )

    class Executor(SceneRunExecutor):
        def _load_source(self, _params):
            return cfg, SceneSpec("venetian-balcony", "Venetian", Path("x")), _Scene()

        def _publish_and_wait(self, *_args, **_kwargs):
            return None

    Executor(cfg, store, gpu_factory=lambda *_args: Gpu()).execute(created.run_id)
    run_dir = store.run_path(created.run_id)
    assert Image.open(run_dir / "step_00000_repair.png").size == (64, 64)
    assert Image.open(run_dir / "step_00001.png").size == (32, 32)
    assert not (run_dir / "step_00001_repair.png").exists()
    assert repairs == []
