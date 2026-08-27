"""Depth and coverage maps are computed only when asked for."""

from __future__ import annotations

import json

import numpy as np

from splat_explorer.agent.actions import Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.loop import run_episode
from splat_explorer.navigation import SpawnSelection
from splat_explorer.rendering.base import Camera
from splat_explorer.rendering.birdseye import COVERAGE_PROMPT_MAX_SIDE


def _topdown_camera(width: int = 240, height: int = 180) -> Camera:
    return Camera.look_at(
        np.array([0.0, 12.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        width=width, height=height, fov_deg=60.0,
    )


def _spawn(width: int = 240, height: int = 180) -> SpawnSelection:
    camera = _topdown_camera(width, height)
    base = np.full((height, width, 3), 80, dtype=np.uint8)
    return SpawnSelection(image=base, points=[], base_image=base, camera=camera)


class CountingRenderer:
    def __init__(self):
        self.n_render = 0
        self.n_with_depth = 0
        self.n_depth = 0

    def _rgb(self, camera):
        return np.full((camera.height, camera.width, 3), 30, dtype=np.uint8)

    def _depth(self, camera):
        return np.full((camera.height, camera.width), 1.5, dtype=np.float32)

    def render(self, camera):
        self.n_render += 1
        return self._rgb(camera)

    def render_with_depth(self, camera):
        self.n_with_depth += 1
        return self._rgb(camera), self._depth(camera)

    def render_depth(self, camera):
        self.n_depth += 1
        return self._depth(camera)


class _RotateThenDone:
    last_debug = None
    allow_done = True

    def __init__(self):
        self.depth = []
        self.coverage = []

    def decide(self, observation, pose, step, depth_image=None, map_image=None,
               coverage_image=None):
        self.depth.append(None if depth_image is None else depth_image.shape[:2])
        self.coverage.append(None if coverage_image is None else coverage_image.shape[:2])
        if step >= 1:
            return Action("done", {"summary": "ok"})
        return Action("rotate", {"yaw_degrees": 15.0})


def _records(episode_dir):
    path = episode_dir / "actions.jsonl"
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("step", -1) >= 0:
                out.append(rec)
    return out


def _run(tmp_path, name, policy=None, renderer=None, spawn=None, **kwargs):
    policy = policy or _RotateThenDone()
    renderer = renderer or CountingRenderer()
    episode = run_episode(
        renderer=renderer,
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=policy,
        output_dir=tmp_path / name,
        width=64, height=48, fov_deg=75.0,
        max_steps=2,
        spawn=spawn if spawn is not None else _spawn(),
        **kwargs,
    )
    return episode, policy, renderer


def test_default_skips_depth_and_coverage_pngs(tmp_path):
    episode, policy, renderer = _run(tmp_path, "default")
    assert (episode / "step_000.png").is_file()
    assert (episode / "step_000_map.png").is_file()
    assert not (episode / "step_000_depth.png").is_file()
    assert not (episode / "step_000_coverage.png").is_file()
    assert renderer.n_render == 2
    assert renderer.n_with_depth == 0
    assert renderer.n_depth == 0
    assert all(d is None for d in policy.depth)
    assert all(c is None for c in policy.coverage)
    recs = _records(episode)
    assert all(r.get("depth_frame") is None for r in recs)
    assert all(r.get("coverage_frame") is None for r in recs)
    assert all(r.get("depth_sent") is False for r in recs)
    assert all(r.get("coverage_sent") is False for r in recs)


def test_compute_depth_saves_but_does_not_send(tmp_path):
    episode, policy, renderer = _run(tmp_path, "cdepth", compute_depth=True)
    assert (episode / "step_000_depth.png").is_file()
    assert (episode / "step_001_depth.png").is_file()
    assert not (episode / "step_000_coverage.png").is_file()
    assert renderer.n_with_depth == 2
    assert renderer.n_render == 0
    assert all(d is None for d in policy.depth)
    recs = _records(episode)
    assert all(r["depth_frame"] == f"step_{r['step']:03d}_depth.png" for r in recs)
    assert all(r["depth_sent"] is False for r in recs)


def test_compute_coverage_saves_but_does_not_send(tmp_path):
    episode, policy, renderer = _run(tmp_path, "ccov", compute_coverage=True)
    assert (episode / "step_000_coverage.png").is_file()
    assert not (episode / "step_000_depth.png").is_file()
    assert renderer.n_render == 2
    assert renderer.n_with_depth == 0
    assert all(c is None for c in policy.coverage)
    recs = _records(episode)
    assert all(r["coverage_frame"] == f"step_{r['step']:03d}_coverage.png" for r in recs)
    assert all(r["coverage_sent"] is False for r in recs)


def test_send_depth_implies_compute_and_attaches(tmp_path):
    episode, policy, renderer = _run(tmp_path, "sdepth", send_depth=True)
    assert renderer.n_with_depth == 2
    assert all(d is not None for d in policy.depth)
    recs = _records(episode)
    assert all(r["depth_sent"] is True for r in recs)
    assert (episode / "step_000_depth.png").is_file()


def test_send_coverage_implies_compute_and_attaches_compact(tmp_path):
    episode, policy, renderer = _run(
        tmp_path, "scov", send_coverage=True, spawn=_spawn(400, 300),
    )
    assert all(c is not None for c in policy.coverage)
    assert all(max(c) == COVERAGE_PROMPT_MAX_SIDE for c in policy.coverage)
    assert all(c is not None for c in policy.coverage)
    assert all(max(c) == COVERAGE_PROMPT_MAX_SIDE for c in policy.coverage)
    recs = _records(episode)
    assert all(r["coverage_sent"] is True for r in recs)
    assert (episode / "step_000_coverage.png").is_file()
    assert renderer.n_with_depth == 0


def test_view_depth_computes_only_the_next_step(tmp_path):
    class Policy:
        last_debug = None
        allow_done = True

        def __init__(self):
            self.depth = []

        def decide(self, observation, pose, step, depth_image=None, map_image=None,
                   coverage_image=None):
            self.depth.append(depth_image is not None)
            if step == 0:
                return Action("view_depth", {})
            return Action("done", {"summary": "ok"})

    policy = Policy()
    renderer = CountingRenderer()
    episode, policy, renderer = _run(tmp_path, "vdepth", policy=policy, renderer=renderer)
    assert policy.depth == [False, True]
    assert renderer.n_render == 1
    assert renderer.n_with_depth == 1
    recs = _records(episode)
    assert recs[0]["depth_frame"] is None and recs[0]["depth_sent"] is False
    assert recs[1]["depth_frame"] == "step_001_depth.png" and recs[1]["depth_sent"] is True
    assert not (episode / "step_000_depth.png").is_file()
    assert (episode / "step_001_depth.png").is_file()


def test_view_coverage_computes_only_the_next_step(tmp_path):
    class Policy:
        last_debug = None
        allow_done = True

        def __init__(self):
            self.coverage = []

        def decide(self, observation, pose, step, depth_image=None, map_image=None,
                   coverage_image=None):
            self.coverage.append(coverage_image is not None)
            if step == 0:
                return Action("view_coverage_map", {})
            return Action("done", {"summary": "ok"})

    policy = Policy()
    episode, policy, renderer = _run(tmp_path, "vcov", policy=policy)
    assert policy.coverage == [False, True]
    recs = _records(episode)
    assert recs[0]["coverage_frame"] is None
    assert recs[1]["coverage_frame"] == "step_001_coverage.png"
    assert recs[1]["coverage_sent"] is True
    assert renderer.n_with_depth == 0


def test_move_toward_lazily_renders_depth(tmp_path):
    class Policy:
        last_debug = None
        allow_done = True

        def decide(self, observation, pose, step, depth_image=None, map_image=None,
                   coverage_image=None):
            if step == 0:
                return Action("move_toward", {"pixel_x": 32, "pixel_y": 24, "amount": 0.4})
            return Action("done", {"summary": "ok"})

    renderer = CountingRenderer()
    episode, _, renderer = _run(tmp_path, "toward", policy=Policy(), renderer=renderer)
    assert renderer.n_render == 2
    assert renderer.n_with_depth == 0
    assert renderer.n_depth == 1
    recs = _records(episode)
    assert recs[0].get("depth_frame") is None
    assert recs[0]["action"]["name"] == "move_toward"
    assert recs[0]["motion"]["kind"] == "move_toward"
    assert "error" not in recs[0]["motion"]
