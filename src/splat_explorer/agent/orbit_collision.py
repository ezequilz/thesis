"""Continuous chord checks against finite, oriented Gaussian support.

Gaussians have infinite tails; collision uses their three-sigma ellipsoids
for splats with opacity >= 1/255. A conservative ellipsoid encloses the
Minkowski sum with a small camera sphere, guaranteeing positive clearance.
The index is rebuilt per action so scene repairs cannot leave stale geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..rendering.base import quats_to_covariances

ORBIT_CLEARANCE = .01  # scene units, in addition to the splat's extent


class OrbitCollision:
    def __init__(self, scene):
        self.groups = []
        # Bound temporary covariance memory for large scenes.
        for start in range(0, len(scene.means), 65536):
            sl = slice(start, start + 65536)
            means, scales, quats, opacity = (
                np.asarray(value[sl], dtype=float)
                for value in (scene.means, scene.scales, scene.quats, scene.opacities))
            norms = np.linalg.norm(quats, axis=1)
            valid = (np.isfinite(means).all(axis=1) & np.isfinite(scales).all(axis=1)
                     & (scales > 0).all(axis=1) & np.isfinite(quats).all(axis=1)
                     & (norms > 0) & np.isfinite(opacity) & (opacity >= 1 / 255))
            means, axes = means[valid], 3 * scales[valid]
            if not len(means):
                continue
            quats = quats[valid] / norms[valid, None]
            # For any beta > 0, (1+beta) A + (1+1/beta) r² I encloses
            # ellipsoid(A) + ball(r), by Cauchy-Schwarz on support functions.
            beta = ORBIT_CLEARANCE / axes.max(axis=1)
            axes = np.sqrt((1 + beta[:, None]) * axes**2
                           + (1 + 1 / beta[:, None]) * ORBIT_CLEARANCE**2)
            radii = axes.max(axis=1)
            precision = quats_to_covariances(quats, 1 / axes).astype(float)
            # Size buckets prevent one large splat from expanding every query.
            buckets = np.floor(np.log2(radii))
            for bucket in np.unique(buckets):
                keep = buckets == bucket
                self.groups.append((cKDTree(means[keep]), means[keep],
                                    precision[keep], float(radii[keep].max())))

    def intersects(self, start, end, arc_error=0.):
        """Check the entire chord, conservatively covering its curved arc too.

        arc_error bounds the maximum arc-to-chord distance. Enlarging the
        normalized ellipsoid radius by error/min_axis encloses that tube.
        """
        delta = end - start
        midpoint = (start + end) / 2
        half_length = np.linalg.norm(delta) / 2
        for tree, means, precision, radius in self.groups:
            ids = tree.query_ball_point(midpoint, half_length + radius + arc_error)
            if not ids:
                continue
            p = precision[ids]
            offset = start - means[ids]
            pd = p @ delta
            denominator = pd @ delta
            t = np.clip(-np.einsum('ni,ni->n', offset, pd)
                        / np.maximum(denominator, 1e-300), 0, 1)
            closest = offset + t[:, None] * delta
            d2 = np.einsum('ni,nij,nj->n', closest, p, closest)
            # Frobenius norm bounds the maximum eigenvalue of precision.
            limit = 1 + arc_error * np.sqrt(np.linalg.norm(p, axis=(1, 2)))
            if np.any(d2 <= limit**2):
                return True
        return False
