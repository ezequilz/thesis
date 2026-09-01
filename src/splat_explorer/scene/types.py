from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class GaussianScene:
    """A decoded 3D Gaussian splat scene in scene units.

    Coordinate system follows the source asset. For SOG this is right-handed,
    y-up, z-back (see the SOG spec, section 1.2).
    """

    means: np.ndarray      # (N, 3) float32 — gaussian centers
    scales: np.ndarray     # (N, 3) float32 — per-axis std deviations, linear (exp applied)
    quats: np.ndarray      # (N, 4) float32 — orientation, (w, x, y, z), normalized
    opacities: np.ndarray  # (N,)   float32 — in [0, 1]
    colors: np.ndarray     # (N, 3) float32 — gamma-space RGB in [0, 1], from SH DC term
    # Higher-order SH coefficients are not decoded yet (view-dependent effects).
    # See sog_loader.load_sog for the palette layout when we add them.

    @property
    def num_gaussians(self) -> int:
        return len(self.means)

    def copy(self) -> "GaussianScene":
        """Independent array copy (original file / live renderer stay untouched)."""
        return replace(
            self,
            means=self.means.copy(),
            scales=self.scales.copy(),
            quats=self.quats.copy(),
            opacities=self.opacities.copy(),
            colors=self.colors.copy(),
        )

    def filtered(self, mask: np.ndarray) -> "GaussianScene":
        """New scene keeping only the gaussians where `mask` is True."""
        return replace(
            self,
            means=self.means[mask],
            scales=self.scales[mask],
            quats=self.quats[mask],
            opacities=self.opacities[mask],
            colors=self.colors[mask],
        )

    def filtered_by_opacity(self, min_opacity: float) -> "GaussianScene":
        return self.filtered(self.opacities >= min_opacity)

    def robust_bounds(self, lower_pct: float = 2.0, upper_pct: float = 98.0):
        """Percentile-based AABB, ignoring stray background gaussians.

        Returns (mins, maxs) as (3,) float32 arrays.
        """
        mins = np.percentile(self.means, lower_pct, axis=0).astype(np.float32)
        maxs = np.percentile(self.means, upper_pct, axis=0).astype(np.float32)
        return mins, maxs

    def robust_centroid(self) -> np.ndarray:
        mins, maxs = self.robust_bounds()
        return ((mins + maxs) / 2).astype(np.float32)

    @staticmethod
    def concatenate(scenes: Sequence["GaussianScene"]) -> "GaussianScene":
        """Stack independently decoded chunks into one working set."""
        parts = [s for s in scenes if s is not None and s.num_gaussians]
        if not parts:
            raise ValueError("No gaussians to concatenate")
        if len(parts) == 1:
            return parts[0]
        return GaussianScene(
            means=np.concatenate([s.means for s in parts]),
            scales=np.concatenate([s.scales for s in parts]),
            quats=np.concatenate([s.quats for s in parts]),
            opacities=np.concatenate([s.opacities for s in parts]),
            colors=np.concatenate([s.colors for s in parts]),
        )
