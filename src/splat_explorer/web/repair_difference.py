"""Local review-only difference proxy; no optimizer history is required."""
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from ..scene.types import GaussianScene


def repair_difference(original: GaussianScene, repaired: GaussianScene) -> GaussianScene:
    """Tint repaired colors by nearest-original parameter difference.

    Spatial matching tolerates pruning/reordering, but is an approximation in
    dense regions and cannot identify identical clones or show deleted splats.
    Fixed scales (not scene percentiles) keep small edits faint. Geometry and
    opacity are preserved. Queries are chunked to bound temporary allocations.
    """
    colors = repaired.colors.copy()
    if not original.num_gaussians:
        colors[:] = (1, 0, 0)
        return replace(repaired, colors=colors)
    tree = cKDTree(original.means)
    for start in range(0, repaired.num_gaussians, 65536):
        sl = slice(start, start + 65536)
        distance, idx = tree.query(repaired.means[sl], workers=1)
        scale = np.maximum(original.scales[idx], 1e-6)
        position = distance / np.maximum(np.linalg.norm(scale, axis=1), 1e-6)
        size = np.max(np.abs(repaired.scales[sl] - scale) / scale, axis=1)
        color = np.max(np.abs(repaired.colors[sl] - original.colors[idx]), axis=1)
        opacity = np.abs(repaired.opacities[sl] - original.opacities[idx])
        # q and -q describe the same rotation.
        q0, q1 = original.quats[idx], repaired.quats[sl]
        rotation = np.minimum(np.linalg.norm(q0 - q1, axis=1),
                              np.linalg.norm(q0 + q1, axis=1))
        strength = np.maximum.reduce([position, size, color / .25, opacity / .25, rotation])
        strength = np.where(strength <= 1e-4, 0, np.clip(strength, 0, 1))
        colors[sl] *= 1 - strength[:, None]
        colors[sl, 0] += strength
    return replace(repaired, colors=colors)
