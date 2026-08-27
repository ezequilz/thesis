"""Bird's-eye framing: indoor percentile path vs outdoor density-core overlay."""

from __future__ import annotations

import numpy as np

from splat_explorer.rendering.birdseye import birdseye_camera
from splat_explorer.scene.types import GaussianScene


def _halo_scene() -> GaussianScene:
    """Dense island near the origin plus a sparse far halo."""
    gx, gz = np.meshgrid(np.linspace(-1.2, 1.2, 18), np.linspace(-1.2, 1.2, 18))
    island = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    halo = np.stack(
        [18.0 * np.cos(angles), np.zeros(40), 18.0 * np.sin(angles)], axis=1,
    )
    means = np.concatenate([island, halo]).astype(np.float32)
    n = len(means)
    return GaussianScene(
        means=means,
        scales=np.full((n, 3), 0.08, np.float32),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        opacities=np.ones(n, np.float32),
        colors=np.ones((n, 3), np.float32),
    )


def test_percentile_frame_is_the_default_and_includes_halo():
    scene = _halo_scene()
    default = birdseye_camera(scene, "+y", 200, 150)
    named = birdseye_camera(scene, "+y", 200, 150, frame="percentile")
    np.testing.assert_allclose(default.position, named.position)
    # Halo at ~18 m: look-down camera must sit high enough to cover it.
    assert default.position[1] > 15.0


def test_core_frame_stays_over_the_dense_island():
    scene = _halo_scene()
    indoor = birdseye_camera(scene, "+y", 200, 150, frame="percentile")
    outdoor = birdseye_camera(scene, "+y", 200, 150, frame="core")
    ground_indoor = np.linalg.norm(indoor.position[[0, 2]])
    ground_outdoor = np.linalg.norm(outdoor.position[[0, 2]])
    assert ground_outdoor < 3.0
    assert outdoor.position[1] < indoor.position[1] - 8.0
    assert ground_outdoor <= ground_indoor + 0.5
