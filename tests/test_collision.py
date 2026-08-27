"""Path collision clamp: full (backup) vs low vs off (free-cam)."""

from __future__ import annotations

import numpy as np
import pytest

from splat_explorer.config import load_config
from splat_explorer.navigation import (
    COLLISION_MODES,
    CollisionWorld,
    resolve_move_toward,
)
from splat_explorer.scene.types import GaussianScene


def _doorway_scene(jamb_x: float = 0.50, z: float = 1.50, scale: float = 0.15) -> GaussianScene:
    """Two vertical jambs with a gap on +z. Mid-gap clearance is ~0.20.

    radius = min(2*0.15, 0.75) = 0.30, so center-gap clearance =
    jamb_x - 0.30 = 0.20. That sits between full keep-out (0.25) and low
    (0.05): full treats the doorway as a wall, low and off walk through.
    """
    ys = np.linspace(0.0, 2.0, 11)
    zs = np.linspace(z - 0.05, z + 0.05, 3)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    n = yy.size
    left = np.stack([np.full(n, -jamb_x), yy.ravel(), zz.ravel()], axis=1)
    right = np.stack([np.full(n, jamb_x), yy.ravel(), zz.ravel()], axis=1)
    means = np.concatenate([left, right]).astype(np.float32)
    n_all = len(means)
    return GaussianScene(
        means=means,
        scales=np.full((n_all, 3), scale, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n_all, 1)),
        opacities=np.ones(n_all, np.float32),
        colors=np.ones((n_all, 3), np.float32),
    )


def _world(mode: str) -> CollisionWorld:
    return CollisionWorld(_doorway_scene(), solid_opacity=0.1, clearance_radius=0.25, collision=mode)


def test_default_config_is_free_cam():
    assert load_config().navigation.collision == "off"


def test_unknown_collision_mode_raises():
    with pytest.raises(ValueError, match="collision"):
        _world("banana")


def test_yaml_boolean_off_is_free_cam():
    """Unquoted YAML `off` loads as False; that must still mean free-cam."""
    world = CollisionWorld(
        _doorway_scene(), solid_opacity=0.1, clearance_radius=0.25, collision=False,
    )
    assert world.collision == "off"


@pytest.mark.parametrize("mode", COLLISION_MODES)
def test_collision_modes_accepted(mode: str):
    world = _world(mode)
    assert world.collision == mode


def test_full_blocks_doorway_low_and_off_pass():
    start = np.array([0.0, 1.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    distance = 2.0
    mid_gap = float(_world("off").clearance(np.array([0.0, 1.0, 1.5])))
    assert 0.05 < mid_gap < 0.25, mid_gap

    full_d, full_blocked = _world("full").clamp_motion(start, direction, distance)
    low_d, low_blocked = _world("low").clamp_motion(start, direction, distance)
    off_d, off_blocked = _world("off").clamp_motion(start, direction, distance)

    assert full_blocked and full_d < 1.5
    assert not low_blocked and low_d == distance
    assert not off_blocked and off_d == distance


def test_move_toward_off_still_stops_short_of_picked_surface():
    """Free-cam skips the path clamp but still does not land inside the target."""
    from splat_explorer.rendering.base import Camera

    eye = np.array([0.0, 1.0, 0.0])
    camera = Camera.look_at(
        eye, eye + np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
        width=32, height=24, fov_deg=75.0,
    )
    depth = np.full((camera.height, camera.width), 4.0, dtype=np.float32)
    world = _world("off")
    result = resolve_move_toward(
        camera, depth, camera.width // 2, camera.height // 2, amount=1.0,
        world=world, up=np.array([0.0, 1.0, 0.0]),
    )
    assert result is not None and not result.get("error")
    assert result["blocked"] is False
    assert result["travelled"] == round(max(result["target_distance"] - world.clearance_radius, 0.0), 3)
