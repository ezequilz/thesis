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
    mid = _sample(coverage, camera, [0.0, 1.5, 3.0])
    far = _sample(coverage, camera, [0.0, 1.5, 5.2])
    behind = _sample(coverage, camera, [0.0, 1.5, -1.5])

    assert near > 0.08, f"close cone should be painted, got {near}"
    assert near > mid > far, f"expected distance falloff, got near={near} mid={mid} far={far}"
    assert far < 0.03, f"far end of the room should be faint, got {far}"
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


def test_overlay_keeps_map_readable():
    base = np.full((40, 40, 3), 80, dtype=np.uint8)
    full = np.ones((40, 40), dtype=np.float32)
    out = overlay_coverage(base, full)
    # Even at coverage=1 the backdrop must still show through.
    assert np.all(out > 30)
    assert np.all(out < 220)
    # Tint is yellow-green: G high relative to B.
    assert out[20, 20, 1] > out[20, 20, 2]
    max_shift = float(np.abs(out.astype(np.float64) - 80).max())
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
