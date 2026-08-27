"""Waypoint covering + jump_to_waypoint (vantages and past steps)."""

from __future__ import annotations

import numpy as np

from splat_explorer.agent.actions import Action, parse_jump_target
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.navigation import (
    CollisionWorld,
    MotionContext,
    Waypoint,
    _dense_floor_mask,
    _on_reconstructed_interior,
    find_spawn_points,
    find_waypoints,
)
from splat_explorer.rendering.annotate import draw_path_map, project_to_pixels
from splat_explorer.rendering.base import Camera, up_vector
from splat_explorer.navigation import ground_basis
from splat_explorer.scene.types import GaussianScene


def _grid_points(xs, ys, zs) -> np.ndarray:
    xx, yy, zz = np.meshgrid(np.asarray(xs), np.asarray(ys), np.asarray(zs), indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _apartment_scene() -> GaussianScene:
    """Large living room at the origin plus a smaller bedroom to +x.

    Spawn search (centroid pull) prefers the living room; waypoint covering
    should still place at least one vantage in the bedroom.
    """
    chunks = [
        _grid_points(np.linspace(-4.0, 4.0, 28), [0.0], np.linspace(-3.5, 3.5, 24)),
        _grid_points(np.linspace(8.0, 11.5, 14), [0.0], np.linspace(-1.4, 1.4, 12)),
        _grid_points(np.linspace(4.0, 8.0, 16), [0.0], np.linspace(-0.4, 0.4, 6)),
        _grid_points(np.linspace(-4.0, 11.5, 30), [2.6], np.linspace(-3.5, 3.5, 20)),
    ]
    ys = np.linspace(0.05, 2.4, 10)
    chunks += [
        _grid_points([-4.05], ys, np.linspace(-3.5, 3.5, 24)),
        _grid_points(np.linspace(-4.0, 4.0, 28), ys, [3.55]),
        _grid_points(np.linspace(-4.0, 4.0, 28), ys, [-3.55]),
        _grid_points(
            [4.05], ys,
            np.concatenate([np.linspace(-3.5, -0.5, 12), np.linspace(0.5, 3.5, 12)]),
        ),
        _grid_points(np.linspace(4.0, 8.0, 16), ys, [0.5]),
        _grid_points(np.linspace(4.0, 8.0, 16), ys, [-0.5]),
        _grid_points([11.55], ys, np.linspace(-1.4, 1.4, 12)),
        _grid_points(np.linspace(8.0, 11.5, 14), ys, [1.45]),
        _grid_points(np.linspace(8.0, 11.5, 14), ys, [-1.45]),
        _grid_points(
            [7.95], ys,
            np.concatenate([np.linspace(-1.4, -0.5, 6), np.linspace(0.5, 1.4, 6)]),
        ),
        _grid_points(np.linspace(-1.2, 1.2, 8), np.linspace(0.1, 0.6, 4), np.linspace(-0.6, 0.6, 6)),
    ]
    means = np.concatenate(chunks).astype(np.float32)
    n = len(means)
    return GaussianScene(
        means=means,
        scales=np.full((n, 3), 0.06, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        opacities=np.ones(n, np.float32),
        colors=np.ones((n, 3), np.float32),
    )


def _apartment_with_void_outliers() -> GaussianScene:
    """Apartment plus a tiny floor-splat island in the void, AABB-padded so
    the occupancy grid actually covers that island (as real floater clouds do)."""
    scene = _apartment_scene()
    cloud = _grid_points(
        np.linspace(-2.4, -1.6, 6),
        np.linspace(0.8, 1.6, 5),
        np.linspace(5.2, 5.8, 6),
    )
    island = _grid_points(
        np.linspace(-2.05, -1.95, 2),
        [0.0],
        np.linspace(5.45, 5.55, 2),
    )
    big = np.array([[-2.0, 0.2, 5.5]], dtype=np.float32)
    extra = np.concatenate([cloud, island, big]).astype(np.float32)
    means = np.concatenate([scene.means, extra])
    n_extra = len(extra)
    scales = np.concatenate([
        scene.scales,
        np.full((n_extra - 1, 3), 0.06, np.float32),
        np.full((1, 3), 0.50, np.float32),
    ])
    n = len(means)
    return GaussianScene(
        means=means,
        scales=scales,
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        opacities=np.ones(n, np.float32),
        colors=np.ones((n, 3), np.float32),
    )


def _apartment_with_sparse_exterior() -> GaussianScene:
    """Apartment plus a large, sparse reconstruction halo outside the walls.

    Real 3DGS rooms have dense floor/furniture; the outdoor halo is a wide
    but thin cloud of floaters. Size filters keep that halo (it covers
    several m²); density must drop it.
    """
    scene = _apartment_scene()
    indoor_floor = np.concatenate([
        _grid_points(np.linspace(-3.8, 3.8, 48), [0.02], np.linspace(-3.3, 3.3, 42)),
        _grid_points(np.linspace(8.1, 11.4, 22), [0.02], np.linspace(-1.3, 1.3, 18)),
    ])
    exterior = _grid_points(
        np.linspace(-2.2, 2.2, 16),
        [0.0],
        np.linspace(6.4, 10.4, 16),
    )
    extra = np.concatenate([indoor_floor, exterior]).astype(np.float32)
    means = np.concatenate([scene.means, extra])
    n = len(means)
    return GaussianScene(
        means=means,
        scales=np.full((n, 3), 0.06, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        opacities=np.ones(n, np.float32),
        colors=np.ones((n, 3), np.float32),
    )


def _topdown_camera(width: int = 240, height: int = 240) -> Camera:
    up = up_vector("+y")
    _, e1 = ground_basis(up)
    return Camera.look_at(
        np.array([0.0, 12.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        up=e1,
        width=width, height=height, fov_deg=60.0,
    )


def test_parse_jump_target_accepts_step_and_waypoint_forms():
    assert parse_jump_target({"target": "step 3"}) == ("step", 3)
    assert parse_jump_target({"target": "Step 3"}) == ("step", 3)
    assert parse_jump_target({"target": "s-3"}) == ("step", 3)
    assert parse_jump_target({"target": "waypoint 2"}) == ("waypoint", 2)
    assert parse_jump_target({"target": "W2"}) == ("waypoint", 2)
    assert parse_jump_target({"target": "w 2"}) == ("waypoint", 2)
    assert parse_jump_target({"target": "2"}) == ("waypoint", 2)
    assert parse_jump_target({"target": 2}) == ("waypoint", 2)
    assert parse_jump_target({"waypoint": 4}) == ("waypoint", 4)
    assert parse_jump_target({"step": 1}) == ("step", 1)
    assert parse_jump_target({"target": "nowhere"}) is None
    assert parse_jump_target({}) is None


def test_jump_to_waypoint_teleports_and_keeps_heading():
    rig = CameraRig(np.array([0.0, 1.2, 0.0]), up_axis="+y", yaw_deg=40.0, pitch_deg=-8.0)
    dest = np.array([3.0, 1.2, -1.5])
    ctx = MotionContext(waypoints=[
        Waypoint(index=0, position=np.array([1.0, 1.2, 1.0]), clearance=1.0, view_distance=2.0),
        Waypoint(index=2, position=dest, clearance=1.2, view_distance=2.5),
    ])
    outcome = rig.apply(Action("jump_to_waypoint", {"target": "W2"}), ctx)
    assert outcome is not None and not outcome.get("error")
    assert outcome["destination"] == "waypoint 2 (W2)"
    np.testing.assert_allclose(rig.position, dest)
    assert rig.yaw_deg == 40.0
    assert rig.pitch_deg == -8.0


def test_jump_to_past_step_restores_full_pose():
    rig = CameraRig(np.array([5.0, 1.2, 5.0]), up_axis="+y", yaw_deg=10.0, pitch_deg=5.0)
    ctx = MotionContext(pose_history=[
        {"step": 0, "position": np.array([0.0, 1.2, 0.0]), "yaw_deg": 0.0, "pitch_deg": 0.0},
        {"step": 3, "position": np.array([2.0, 1.1, -1.0]), "yaw_deg": 90.0, "pitch_deg": -15.0},
    ])
    outcome = rig.apply(Action("jump_to_waypoint", {"target": "step 3"}), ctx)
    assert outcome is not None and not outcome.get("error")
    np.testing.assert_allclose(rig.position, [2.0, 1.1, -1.0])
    assert rig.yaw_deg == 90.0
    assert rig.pitch_deg == -15.0


def test_jump_rejects_unknown_targets():
    rig = CameraRig(np.array([0.0, 1.2, 0.0]), up_axis="+y")
    start = rig.position.copy()
    ctx = MotionContext(
        waypoints=[Waypoint(index=0, position=np.array([1.0, 1.2, 0.0]),
                            clearance=1.0, view_distance=2.0)],
        pose_history=[{"step": 0, "position": start.copy(), "yaw_deg": 0.0, "pitch_deg": 0.0}],
    )
    bad_wp = rig.apply(Action("jump_to_waypoint", {"target": "waypoint 9"}), ctx)
    assert bad_wp["error"]
    np.testing.assert_array_equal(rig.position, start)
    bad_step = rig.apply(Action("jump_to_waypoint", {"target": "step 12"}), ctx)
    assert bad_step["error"]
    np.testing.assert_array_equal(rig.position, start)


def test_waypoints_cover_both_rooms_and_stay_in_open_space():
    scene = _apartment_scene()
    world = CollisionWorld(scene, solid_opacity=0.1, collision="off")
    waypoints = find_waypoints(
        scene, up_axis="+y", world=world, num_points=6,
        ceiling_percentile=25.0, grid_cell=0.2, spawn_height_fraction=0.5,
    )
    assert len(waypoints) >= 2, f"expected several vantages, got {len(waypoints)}"
    xs = np.array([w.position[0] for w in waypoints])
    assert (xs < 4.0).any(), f"no living-room waypoint: {xs}"
    assert (xs > 7.5).any(), f"no bedroom waypoint: {xs}"
    for w in waypoints:
        assert w.clearance >= 0.35, w
        assert float(world.clearance(w.position)) >= 0.30
    if len(waypoints) >= 2:
        pts = np.stack([w.position for w in waypoints])
        d = np.linalg.norm(pts[:, None, :3] - pts[None, :, :3], axis=-1)
        np.fill_diagonal(d, np.inf)
        assert float(d.min()) > 1.0, f"waypoints clustered, min sep {d.min():.2f}"


def test_dense_floor_mask_drops_tiny_islands():
    hist = np.zeros((40, 40), dtype=np.float64)
    hist[10:26, 10:26] = 4  # real room
    hist[2, 2] = 1          # single outlier cell
    hist[2, 37] = 3
    mask = _dense_floor_mask(hist, cell=0.15)
    assert mask[18, 18]
    assert not mask[2, 2]
    assert not mask[2, 37]


def test_dense_floor_mask_drops_sparse_exterior():
    hist = np.zeros((50, 50), dtype=np.float64)
    hist[18:38, 18:38] = 8   # dense indoor room
    hist[2:14, 2:14] = 1     # large but sparse exterior (well above 0.35 m²)
    mask = _dense_floor_mask(hist, cell=0.15)
    assert mask[28, 28]
    assert not mask[6, 6]
    assert not mask[12, 12]


def test_dense_floor_mask_keeps_hallway_attached_to_rooms():
    hist = np.zeros((50, 50), dtype=np.float64)
    hist[10:30, 10:30] = 8   # room
    hist[18:24, 30:42] = 5   # hallway of similar coverage, slightly less furniture
    mask = _dense_floor_mask(hist, cell=0.15)
    assert mask[20, 20]
    assert mask[21, 36]


def test_waypoints_ignore_void_outlier_islands():
    scene = _apartment_with_void_outliers()
    world = CollisionWorld(scene, solid_opacity=0.1, collision="off")
    waypoints = find_waypoints(
        scene, up_axis="+y", world=world, num_points=6,
        ceiling_percentile=25.0, grid_cell=0.2, spawn_height_fraction=0.5,
    )
    assert waypoints, "expected indoor waypoints"
    floater = np.array([-2.0, 1.2, 5.5])
    for w in waypoints:
        dist = float(np.linalg.norm(w.position - floater))
        assert dist > 2.0, f"waypoint {w.index} landed on the void island at {w.position}"
        assert w.position[2] < 4.2, f"waypoint {w.index} z={w.position[2]:.2f} is outside the rooms"


def test_waypoints_ignore_sparse_exterior_halo():
    scene = _apartment_with_sparse_exterior()
    world = CollisionWorld(scene, solid_opacity=0.1, collision="off")
    waypoints = find_waypoints(
        scene, up_axis="+y", world=world, num_points=6,
        ceiling_percentile=25.0, grid_cell=0.2, spawn_height_fraction=0.5,
    )
    assert waypoints, "expected indoor waypoints"
    xs = np.array([w.position[0] for w in waypoints])
    zs = np.array([w.position[2] for w in waypoints])
    assert (xs < 4.0).any(), f"no living-room waypoint: {xs}"
    for w in waypoints:
        assert w.position[2] < 5.5, (
            f"waypoint {w.index} landed in the sparse exterior halo at {w.position}"
        )
    assert (zs > 6.0).sum() == 0


def test_interior_luma_rejects_black_void():
    camera = _topdown_camera()
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    image[90:150, 90:150] = 140
    inside = np.array([[0.0, 1.5, 0.0]])
    outside = np.array([[-3.5, 1.5, 3.5]])
    assert _on_reconstructed_interior(image, camera, inside)[0]
    assert not _on_reconstructed_interior(image, camera, outside)[0]


def test_interior_luma_rejects_grey_halo():
    camera = _topdown_camera()
    image = np.full((camera.height, camera.width, 3), 32, dtype=np.uint8)
    image[90:150, 90:150] = 140
    inside = np.array([[0.0, 1.5, 0.0]])
    halo = np.array([[-3.5, 1.5, 3.5]])
    assert _on_reconstructed_interior(image, camera, inside)[0]
    assert not _on_reconstructed_interior(image, camera, halo)[0]


def test_waypoints_spread_more_than_spawn_across_rooms():
    scene = _apartment_scene()
    world = CollisionWorld(scene, solid_opacity=0.1, collision="off")
    kwargs = dict(
        up_axis="+y", world=world, ceiling_percentile=25.0,
        grid_cell=0.2, spawn_height_fraction=0.5,
    )
    spawn = find_spawn_points(scene, num_points=5, centroid_pull=0.35, **kwargs)
    waypoints = find_waypoints(scene, num_points=6, **kwargs)
    assert spawn and waypoints
    spawn_span = np.ptp([p.position[0] for p in spawn])
    way_span = np.ptp([w.position[0] for w in waypoints])
    assert way_span > spawn_span + 1.0, (
        f"waypoints should cover more of the floor plan than spawn "
        f"(spawn dx={spawn_span:.2f}, waypoints dx={way_span:.2f})"
    )


def test_waypoints_painted_lightly_on_path_map():
    camera = _topdown_camera()
    blank = np.full((camera.height, camera.width, 3), 50, dtype=np.uint8)
    dest = np.array([[-2.0, 1.5, 2.0]])
    out = draw_path_map(
        blank, camera,
        poses=[{"position": np.array([0.0, 1.5, 0.0]),
                "heading": np.array([0.0, 0.0, 1.0]), "step": 0}],
        fov_deg=75.0, up=np.array([0.0, 1.0, 0.0]),
        waypoints=dest,
    )
    uv = project_to_pixels(camera, dest)[0]
    x, y = int(round(uv[0])), int(round(uv[1]))
    patch = out[max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]
    gold = (patch[:, :, 0] > patch[:, :, 2] + 20) & (patch[:, :, 1] > patch[:, :, 2])
    assert gold.any(), "expected a gold waypoint disk"
    assert int(patch.max()) > 180, "waypoint marker should be clearly visible"


def test_loop_jump_restores_step_zero_pose(tmp_path):
    from splat_explorer.agent.loop import run_episode

    class _Policy:
        allow_done = True
        last_debug = None

        def decide(self, observation, pose, step, depth_image=None, map_image=None,
                   coverage_image=None):
            if step == 0:
                return Action("move", {"direction": "forward", "distance": 1.5})
            if step == 1:
                return Action("jump_to_waypoint", {"target": "step 0"})
            return Action("done", {"summary": "ok"})

    class _Renderer:
        def render(self, camera):
            return np.full((camera.height, camera.width, 3), 30, dtype=np.uint8)

        def render_with_depth(self, camera):
            rgb = self.render(camera)
            depth = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
            return rgb, depth

    start = np.array([0.0, 1.5, 0.0])
    rig = CameraRig(start, up_axis="+y", yaw_deg=0.0)
    run_episode(
        renderer=_Renderer(),
        rig=rig,
        policy=_Policy(),
        output_dir=tmp_path,
        width=64, height=48, fov_deg=75.0,
        max_steps=3,
    )
    np.testing.assert_allclose(rig.position, start, atol=1e-6)
    assert rig.yaw_deg == 0.0
