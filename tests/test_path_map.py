"""Path-map overlay: connected trail + per-step camera frustums on a bird's-eye."""

from __future__ import annotations

import numpy as np

from splat_explorer.agent.actions import ACTION_TOOLS, Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.cli_relay import parse_action
from splat_explorer.navigation import ground_basis
from splat_explorer.rendering.annotate import draw_path_map, project_to_pixels
from splat_explorer.rendering.base import Camera, up_vector
from splat_explorer.rendering.birdseye import ExplorationMap
from splat_explorer.tasks.artifact_hunt import system_prompt


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
    base = system_prompt(with_depth=False, with_map=False, with_coverage=False)
    with_map = system_prompt(with_depth=False, with_map=True)
    with_depth = system_prompt(with_depth=True, with_map=False)
    with_coverage = system_prompt(with_depth=False, with_coverage=True)
    assert "view_map" in base and "view_depth" in base
    assert "view_coverage_map" in base
    assert "BIRD'S-EYE MAP" not in base
    assert "This turn also includes a COVERAGE MAP" not in base
    assert "BIRD'S-EYE MAP" in with_map
    assert "DEPTH MAP" in with_depth
    assert "This turn also includes a COVERAGE MAP" in with_coverage
    assert "RGB view" in with_map and "RGB view" in with_depth
    assert "RGB view" in with_coverage
