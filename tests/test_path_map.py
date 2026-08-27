"""Path-map overlay: connected trail + per-step camera frustums on a bird's-eye."""

from __future__ import annotations

import numpy as np

from splat_explorer.agent.actions import ACTION_TOOLS, Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.cli_relay import parse_action, render_tool_catalog
from splat_explorer.navigation import ground_basis
from splat_explorer.rendering.annotate import draw_path_map, project_to_pixels
from splat_explorer.rendering.base import Camera, up_vector
from splat_explorer.rendering.birdseye import ExplorationMap
from splat_explorer.tasks.registry import canonical_names, load_prompt


def _topdown_camera(width: int = 240, height: int = 240) -> Camera:
    up = up_vector("+y")
    _, e1 = ground_basis(up)
    return Camera.look_at(
        np.array([0.0, 12.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        up=e1,
        width=width, height=height, fov_deg=60.0,
    )


def _pose(position, heading, step: int) -> dict:
    return {
        "position": np.asarray(position, dtype=np.float64),
        "heading": np.asarray(heading, dtype=np.float64),
        "step": step,
    }


def test_path_connects_poses_and_marks_current():
    camera = _topdown_camera()
    blank = np.full((camera.height, camera.width, 3), 40, dtype=np.uint8)
    a = np.array([-2.0, 1.5, -2.0])
    b = np.array([2.0, 1.5, 2.0])
    poses = [
        _pose(a, [0.0, 0.0, 1.0], 0),
        _pose(b, [1.0, 0.0, 0.0], 1),
    ]
    out = draw_path_map(blank, camera, poses, fov_deg=75.0, up=np.array([0.0, 1.0, 0.0]))
    assert out.shape == blank.shape
    assert not np.array_equal(out, blank)

    uv = project_to_pixels(camera, np.stack([a, b]))
    assert np.all(np.isfinite(uv))
    mid = (uv[0] + uv[1]) / 2.0
    x, y = int(round(mid[0])), int(round(mid[1]))
    patch = out[max(y - 4, 0):y + 5, max(x - 4, 0):x + 5]
    # Connected path is red: some pixel in the neighborhood is clearly reddish.
    reddish = (patch[:, :, 0] > patch[:, :, 1] + 20) & (patch[:, :, 0] > patch[:, :, 2] + 20)
    assert reddish.any(), "expected a red path segment between the two poses"

    # Current pose (last) is highlighted cyan near its projected position.
    bx, by = int(round(uv[1, 0])), int(round(uv[1, 1]))
    around = out[max(by - 8, 0):by + 9, max(bx - 8, 0):bx + 9]
    cyan = (around[:, :, 2] > around[:, :, 0] + 10) & (around[:, :, 1] > around[:, :, 0] - 20)
    assert cyan.any(), "expected a cyan current-pose marker"


def test_exploration_map_does_not_mutate_backdrop():
    camera = _topdown_camera()
    base = np.full((camera.height, camera.width, 3), 80, dtype=np.uint8)
    snap = base.copy()
    emap = ExplorationMap(base, camera, fov_deg=75.0, up=np.array([0.0, 1.0, 0.0]))
    emap.add_pose(np.array([0.0, 1.5, 0.0]), np.array([0.0, 0.0, 1.0]), 0)
    emap.add_pose(np.array([1.5, 1.5, 1.0]), np.array([1.0, 0.0, 0.0]), 1)
    first = emap.render()
    second = emap.render()
    np.testing.assert_array_equal(base, snap)
    assert first.shape == base.shape
    np.testing.assert_array_equal(first, second)
    assert len(emap.poses) == 2


def test_view_map_does_not_move_the_rig():
    eye = np.array([1.0, 1.5, 2.0])
    rig = CameraRig(eye, up_axis="+y", yaw_deg=40.0, pitch_deg=-10.0)
    before = (rig.position.copy(), rig.yaw_deg, rig.pitch_deg)
    for name in ("view_map", "view_coverage_map", "view_depth"):
        rig.position[:] = before[0]
        rig.yaw_deg, rig.pitch_deg = before[1], before[2]
        outcome = rig.apply(Action(name, {}))
        assert outcome == {"kind": name}
        np.testing.assert_array_equal(rig.position, before[0])
        assert rig.yaw_deg == before[1]
        assert rig.pitch_deg == before[2]


def test_parse_action_accepts_view_extras():
    names = {tool["function"]["name"] for tool in ACTION_TOOLS}
    assert "view_map" in names
    assert "view_coverage_map" in names
    assert "view_depth" in names
    for name in ("view_map", "view_coverage_map", "view_depth"):
        action = parse_action(f'{{"action": "{name}", "args": {{}}}}')
        assert action is not None and action.name == name
        action = parse_action(f'{{"action": "{name}"}}')
        assert action is not None and action.name == name


def test_system_prompt_mentions_extras_when_attached():
    for name in canonical_names():
        system_prompt = load_prompt(name).system_prompt
        base = system_prompt(with_depth=False, with_map=False, with_coverage=False)
        with_map = system_prompt(with_depth=False, with_map=True)
        with_depth = system_prompt(with_depth=True, with_map=False)
        with_coverage = system_prompt(with_depth=False, with_coverage=True)
        assert "view_map" in base and "view_depth" in base, name
        assert "view_coverage_map" in base, name
        assert "BIRD'S-EYE MAP" not in base, name
        assert "This turn also includes a COVERAGE MAP" not in base, name
        assert "BIRD'S-EYE MAP" in with_map, name
        assert "DEPTH MAP" in with_depth, name
        assert "This turn also includes a COVERAGE MAP" in with_coverage, name
        assert "RGB view" in with_map and "RGB view" in with_depth, name
        assert "RGB view" in with_coverage, name


def test_prompt_variants_are_distinct_and_v3_is_default():
    from splat_explorer.agent.actions import filter_tools
    from splat_explorer.tasks.registry import DEFAULT_PROMPT

    assert DEFAULT_PROMPT == "v3"
    assert canonical_names() == ["v1", "v2", "v3"]
    v1 = load_prompt("v1").system_prompt(False)
    v2 = load_prompt("v2").system_prompt(False)
    v3 = load_prompt("v3").system_prompt(False)
    assert "quality-inspection agent walking" in v1
    assert "autonomous visual inspector" in v2
    assert "Inspect this 3D Gaussian Splatting indoor scene" in v3
    assert "done" not in v3.lower()
    assert load_prompt().system_prompt(False) == v3
    hidden = getattr(load_prompt("v3"), "HIDDEN_TOOLS", ())
    catalog = render_tool_catalog(filter_tools(hidden))
    assert "done" not in catalog
    assert parse_action('{"action": "done", "args": {"summary": "x"}}',
                        {t["function"]["name"] for t in filter_tools(hidden)}) is None


class _BlankRenderer:
    def render_with_depth(self, camera):
        rgb = np.full((camera.height, camera.width, 3), 30, dtype=np.uint8)
        depth = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
        return rgb, depth


class _RecordMapPolicy:
    def __init__(self):
        self.seen: list[tuple[int, int] | None] = []
        self.last_debug = None
        self.allow_done = True

    def decide(self, observation, pose, step, depth_image=None, map_image=None,
               coverage_image=None):
        self.seen.append(None if map_image is None else map_image.shape[:2])
        if step >= 1:
            return Action("done", {"summary": "ok"})
        return Action("rotate", {"yaw_degrees": 45.0})


def test_loop_send_map_attaches_full_res_every_step(tmp_path):
    """Always-on bird's-eye: full spawn resolution, never the compact coverage size."""
    from splat_explorer.agent.loop import run_episode
    from splat_explorer.navigation import SpawnSelection
    from splat_explorer.rendering.birdseye import COVERAGE_PROMPT_MAX_SIDE

    camera = _topdown_camera(width=400, height=300)
    base = np.full((300, 400, 3), 80, dtype=np.uint8)
    spawn = SpawnSelection(image=base, points=[], base_image=base, camera=camera)
    policy = _RecordMapPolicy()
    run_episode(
        renderer=_BlankRenderer(),
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=policy,
        output_dir=tmp_path / "on",
        width=96, height=72, fov_deg=75.0,
        max_steps=2,
        spawn=spawn,
        send_map=True,
    )
    assert len(policy.seen) == 2
    assert all(shape == (300, 400) for shape in policy.seen)
    assert all(max(shape) > COVERAGE_PROMPT_MAX_SIDE for shape in policy.seen)

    off = _RecordMapPolicy()
    run_episode(
        renderer=_BlankRenderer(),
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=off,
        output_dir=tmp_path / "off",
        width=96, height=72, fov_deg=75.0,
        max_steps=2,
        spawn=spawn,
        send_map=False,
    )
    assert all(shape is None for shape in off.seen)
