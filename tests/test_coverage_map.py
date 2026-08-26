"""Coverage-map overlay: distance-faded view cones stacked on a bird's-eye."""

from __future__ import annotations

import numpy as np

from splat_explorer.navigation import ground_basis
from splat_explorer.rendering.annotate import (
    _COVERAGE_DISPLAY_ALPHA,
    overlay_coverage,
    paint_coverage_cone,
    project_to_pixels,
)
from splat_explorer.rendering.base import Camera, up_vector
from splat_explorer.rendering.birdseye import ExplorationMap


def _topdown_camera(width: int = 240, height: int = 240) -> Camera:
    up = up_vector("+y")
    _, e1 = ground_basis(up)
    return Camera.look_at(
        np.array([0.0, 12.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        up=e1,
        width=width, height=height, fov_deg=60.0,
    )


def _sample(coverage: np.ndarray, camera: Camera, point) -> float:
    uv = project_to_pixels(camera, np.asarray(point, dtype=np.float64)[None])[0]
    x, y = int(round(uv[0])), int(round(uv[1]))
    assert 0 <= x < coverage.shape[1] and 0 <= y < coverage.shape[0]
    return float(coverage[y, x])


def test_cone_paints_ahead_not_behind_and_falls_off():
    camera = _topdown_camera()
    coverage = np.zeros((camera.height, camera.width), dtype=np.float32)
    origin = np.array([0.0, 1.5, 0.0])
    heading = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    paint_coverage_cone(coverage, camera, origin, heading, up, fov_deg=75.0)

    near = _sample(coverage, camera, [0.0, 1.5, 0.8])
    curtain = _sample(coverage, camera, [0.0, 1.5, 1.45])
    mid = _sample(coverage, camera, [0.0, 1.5, 3.0])
    far = _sample(coverage, camera, [0.0, 1.5, 5.2])
    behind = _sample(coverage, camera, [0.0, 1.5, -1.5])

    assert near > 0.20, f"close cone should be ~one-view gain, got {near}"
    assert abs(curtain - near) < 0.03, f"full strength should hold to ~1.5m, near={near} curtain={curtain}"
    assert near > mid > far, f"expected distance falloff after ~1.5m, got near={near} mid={mid} far={far}"
    assert far < 0.04, f"far end of the room should be faint, got {far}"
    assert behind < 0.01, f"nothing behind the camera, got {behind}"
    assert float(coverage.max()) <= 1.0 + 1e-6


def test_overlapping_views_stack_until_one():
    camera = _topdown_camera()
    coverage = np.zeros((camera.height, camera.width), dtype=np.float32)
    origin = np.array([0.0, 1.5, 0.0])
    heading = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    paint_coverage_cone(coverage, camera, origin, heading, up, fov_deg=75.0)
    once = coverage.copy()
    paint_coverage_cone(coverage, camera, origin, heading, up, fov_deg=75.0)
    twice = coverage
    overlap = once > 0.01
    assert overlap.any()
    np.testing.assert_array_less(once[overlap], twice[overlap] + 1e-6)
    assert (twice[overlap] > once[overlap] + 1e-4).mean() > 0.9
    assert float(twice.max()) <= 1.0 + 1e-6


def test_overlay_one_view_is_a_wash_four_go_solid_lime():
    from splat_explorer.rendering.annotate import _COVERAGE_GAIN, _COVERAGE_RGB

    base = np.full((40, 40, 3), 80, dtype=np.uint8)
    one = overlay_coverage(base, np.full((40, 40), _COVERAGE_GAIN, dtype=np.float32))
    full = overlay_coverage(base, np.ones((40, 40), dtype=np.float32))
    # One close view: a visible lime wash, map still underneath.
    assert one[20, 20, 1] > one[20, 20, 2]
    assert abs(int(one[20, 20, 1]) - 80) > 20
    assert abs(int(one[20, 20, 1]) - 80) < 80
    # Saturated coverage is solid lime.
    np.testing.assert_allclose(full[20, 20], _COVERAGE_RGB, atol=1)
    max_shift = float(np.abs(full.astype(np.float64) - 80).max())
    assert max_shift <= 255 * _COVERAGE_DISPLAY_ALPHA + 1


def test_exploration_map_coverage_does_not_mutate_backdrop():
    camera = _topdown_camera()
    base = np.full((camera.height, camera.width, 3), 80, dtype=np.uint8)
    snap = base.copy()
    emap = ExplorationMap(base, camera, fov_deg=75.0, up=np.array([0.0, 1.0, 0.0]))
    emap.add_pose(np.array([0.0, 1.5, 0.0]), np.array([0.0, 0.0, 1.0]), 0)
    emap.add_pose(np.array([0.0, 1.5, 0.0]), np.array([1.0, 0.0, 0.0]), 1)
    first = emap.render_coverage()
    second = emap.render_coverage()
    np.testing.assert_array_equal(base, snap)
    assert first.shape == base.shape
    np.testing.assert_array_equal(first, second)
    assert 0.0 < emap.coverage_fraction <= 1.0
    path = emap.render()
    assert not np.array_equal(first, path)
