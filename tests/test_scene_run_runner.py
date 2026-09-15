from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from splat_explorer.agent.actions import Action
from splat_explorer.config import Config
from splat_explorer.scene.catalog import SceneSpec
from splat_explorer.scene_runs.runner import SceneRunExecutor
from splat_explorer.scene_runs.store import SceneRunStore


class _Scene:
    def copy(self):
        return _Scene()


class _Renderer:
    def render(self, camera):
        return np.zeros((camera.height, camera.width, 3), dtype=np.uint8)

    def render_depth(self, camera):
        return np.ones((camera.height, camera.width), dtype=np.float32)


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
