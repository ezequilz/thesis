"""move_toward must walk on the ground plane, not dive into the floor."""

from __future__ import annotations

import numpy as np

from splat_explorer.navigation import CollisionWorld, resolve_move_toward
from splat_explorer.rendering.base import Camera
from splat_explorer.scene.types import GaussianScene


def _floor_scene(z_extent: float = 8.0, y: float = 0.0, scale: float = 0.30) -> GaussianScene:
    """Dense floor of large gaussians at y=`y`, covering x,z in [-2, 2] x [0, z_extent]."""
    xs = np.linspace(-2.0, 2.0, 17)
    zs = np.linspace(0.0, z_extent, 33)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    n = xx.size
    means = np.stack([xx.ravel(), np.full(n, y), zz.ravel()], axis=1).astype(np.float32)
    return GaussianScene(
        means=means,
        scales=np.full((n, 3), scale, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        opacities=np.ones(n, np.float32),
        colors=np.ones((n, 3), np.float32),
    )


def _forward_camera(position, height: int = 72, width: int = 96) -> Camera:
    pos = np.asarray(position, dtype=np.float64)
    return Camera.look_at(
        pos, pos + np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
        width=width, height=height, fov_deg=75.0,
    )


def _depth_to_floor(camera: Camera, pixel_x: int, pixel_y: int, floor_y: float = 0.0) -> np.ndarray:
    """Depth buffer that is inf everywhere except the picked pixel, whose
    camera-z is the ray's intersection with the plane y=floor_y."""
    H, W = camera.height, camera.width
    depth = np.full((H, W), np.inf, dtype=np.float32)
    ray_cam = np.array([
        (pixel_x + 0.5 - W / 2.0) / camera.fx,
        (pixel_y + 0.5 - H / 2.0) / camera.fy,
        1.0,
    ])
    direction = camera.rotation.astype(np.float64) @ ray_cam
    origin = np.asarray(camera.position, dtype=np.float64)
    # origin_y + t * dir_y = floor_y
    if abs(direction[1]) < 1e-9:
        raise AssertionError("ray is parallel to the floor")
    t = (floor_y - origin[1]) / direction[1]
    assert t > 0, "floor intersection is behind the camera"
    depth[pixel_y, pixel_x] = np.float32(t * direction[2])  # camera-z
    return depth


def test_floor_pixel_walks_horizontally_instead_of_diving():
    """Reproduce the step-9 failure: a floor pixel used to 3D-dive into the
    floor splat envelope and get clamped to travelled=0. After the fix it
    walks forward at constant eye height.

    Camera sits exactly at the floor keep-out boundary (clearance ==
    clearance_radius), so any downward component is immediately blocked.
    """
    # radius = min(2*0.4, 0.75) = 0.75; eye at 1.0 => clearance 0.25.
    eye = np.array([0.0, 1.0, 0.0])
    camera = _forward_camera(eye)
    px, py = camera.width // 2, int(camera.height * 0.8)  # well below the horizon
    depth = _depth_to_floor(camera, px, py)
    world = CollisionWorld(
        _floor_scene(scale=0.40), solid_opacity=0.1, clearance_radius=0.25,
    )
    up = np.array([0.0, 1.0, 0.0])
    assert abs(float(world.clearance(eye)) - 0.25) < 0.02

    # Without flattening the 3D path heads into the floor and is blocked.
    diving = resolve_move_toward(camera, depth, px, py, amount=0.6, world=world, up=None)
    assert diving is not None and not diving.get("error")
    assert diving["travelled"] == 0.0, diving
    assert diving["blocked"] is True

    walking = resolve_move_toward(camera, depth, px, py, amount=0.6, world=world, up=up)
    assert walking is not None and not walking.get("error"), walking
    assert walking["travelled"] > 0.5, walking
    # Eye height held.
    np.testing.assert_allclose(walking["new_position"][1], eye[1], atol=1e-6)
    # Moved forward along +z.
    assert walking["new_position"][2] > eye[2] + 0.4


def test_amount_scales_ground_distance():
    eye = np.array([0.0, 1.5, 0.0])
    camera = _forward_camera(eye)
    px, py = camera.width // 2, int(camera.height * 0.75)
    depth = _depth_to_floor(camera, px, py)
    up = np.array([0.0, 1.0, 0.0])

    full = resolve_move_toward(camera, depth, px, py, amount=1.0, world=None, up=up)
    half = resolve_move_toward(camera, depth, px, py, amount=0.5, world=None, up=up)
    assert full is not None and half is not None
    assert abs(half["travelled"] - 0.5 * full["target_distance"]) < 1e-6
    # amount=1 stops a safety margin short of the ground-projected target.
    assert full["travelled"] == round(max(full["target_distance"] - 0.25, 0.0), 3)


def test_empty_depth_returns_none():
    camera = _forward_camera([0.0, 1.0, 0.0])
    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    assert resolve_move_toward(camera, depth, 10, 10, 0.5) is None


def test_camera_rig_keeps_eye_height():
    from splat_explorer.agent.actions import Action
    from splat_explorer.agent.camera_rig import CameraRig
    from splat_explorer.navigation import MotionContext

    eye = np.array([0.0, 1.0, 0.0])
    # CameraRig yaw=0 looks along -Z; 180 looks +Z onto the floor slab.
    rig = CameraRig(eye, up_axis="+y", yaw_deg=180.0, pitch_deg=0.0)
    camera = rig.camera(96, 72, 75.0)
    px, py = 48, 58
    depth = _depth_to_floor(camera, px, py)
    world = CollisionWorld(
        _floor_scene(scale=0.40), solid_opacity=0.1, clearance_radius=0.25,
    )
    outcome = rig.apply(
        Action("move_toward", {"pixel_x": px, "pixel_y": py, "amount": 0.6}),
        MotionContext(world=world, camera=camera, depth=depth),
    )
    assert outcome["travelled"] > 0.5, outcome
    np.testing.assert_allclose(rig.position[1], eye[1], atol=1e-6)
