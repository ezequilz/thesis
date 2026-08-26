"""CPU gaussian-splat rasterizer: full anisotropic splats, no GPU required.

Renders actual gaussians (EWA splatting: each 3D covariance is projected to an
anisotropic 2D screen-space footprint) instead of the bare center dots of
cpu_points, so the model and debug outputs show real splat imagery — soft
surfaces, floaters, blur blobs — rather than speckle.

Vectorized numpy pipeline per frame:
  1. Transform + project all gaussians; cull behind-camera and off-screen.
  2. Project covariances (J W Sigma W^T J^T + low-pass) to per-splat conics
     and integer pixel radii (3 sigma, capped at max_splat_radius_px).
  3. Rasterize by radius group in bounded-memory chunks; every splat scatters
     alpha-weighted contributions to the pixels of its footprint (bincount).
  4. Bin splats into log-depth layers and alpha-composite back-to-front
     (same over-operator as viser's WebGL splats). Within a thin layer,
     overlapping splats are still weight-averaged.

The old 2-layer OIT mix of sofa/wall/fog in one band is what produced the
smeared watercolor look. Layered compositing keeps those surfaces apart.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from ..scene import GaussianScene
from .base import Camera, quats_to_covariances

logger = logging.getLogger(__name__)

# Weight below which a contribution is invisible at 8 bits and gets dropped.
_MIN_WEIGHT = 1.0 / 255.0
# Bound on (splats x footprint pixels) processed per chunk, to cap memory.
_CONTRIB_BUDGET = 8_000_000
# Log-depth slices for back-to-front over. More slices = less color bleeding
# between surfaces; 12 is a memory/quality tradeoff at 960x720.
_N_LAYERS = 12


class CpuSplatRenderer:
    last_backend = "cpu_splats"

    def __init__(
        self,
        scene: GaussianScene,
        max_splat_radius_px: int = 120,
        near: float = 0.05,
        background: tuple[int, int, int] = (30, 30, 34),
        # Depth band (relative to the front-surface depth) blended together as
        # one surface; also sets the sharpness of the soft depth buffer.
        depth_band_rel: float = 0.04,
        depth_band_min: float = 0.05,
        # Coverage saturation: alpha = 1 - exp(-boost * sum(weights)). At 1.0 a
        # lone splat renders near its true opacity (1-e^-w ~ w) while dense
        # surfaces (sum w >~ 3) still saturate to solid.
        coverage_boost: float = 1.0,
    ):
        self.scene = scene
        self.max_radius = int(max_splat_radius_px)
        self.near = float(near)
        self.background = np.asarray(background, dtype=np.float32) / 255.0
        self.depth_band_rel = float(depth_band_rel)
        self.depth_band_min = float(depth_band_min)
        self.coverage_boost = float(coverage_boost)

        self._cov3 = quats_to_covariances(scene.quats, scene.scales)  # (N,3,3)
        self._max_scale = scene.scales.max(axis=1)  # cheap pre-cull radius bound
        self._alphas = np.clip(scene.opacities, 0.0, 0.995)

    def render(self, camera: Camera) -> np.ndarray:
        return self.render_with_depth(camera)[0]

    def render_with_depth(self, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        """Render RGB plus a per-pixel depth map.

        Depth is the soft front-surface depth (camera z, scene units) already
        computed for occlusion in pass 1. Pixels with no meaningful coverage
        (background / holes / thin haze) are np.inf.
        """
        t0 = time.perf_counter()
        W, H = camera.width, camera.height
        fx, fy = camera.fx, camera.fy
        w2c = camera.w2c
        R = w2c[:3, :3].astype(np.float32)

        pcam = self.scene.means @ R.T + w2c[:3, 3]
        z = pcam[:, 2]
        keep = z > self.near
        pcam, z = pcam[keep], z[keep]
        idx0 = np.flatnonzero(keep)

        uf = fx * pcam[:, 0] / z + W / 2.0
        vf = fy * pcam[:, 1] / z + H / 2.0

        # Cheap conservative screen cull before the covariance math.
        r_bound = 3.0 * self._max_scale[idx0] * max(fx, fy) / z
        on = (uf > -r_bound) & (uf < W + r_bound) & (vf > -r_bound) & (vf < H + r_bound)
        idx0, pcam, z, uf, vf = idx0[on], pcam[on], z[on], uf[on], vf[on]
        if len(idx0) == 0:
            return self._flat_background(H, W), np.full((H, W), np.inf, dtype=np.float32)

        # EWA: Sigma2D = J (R Sigma R^T) J^T + low-pass, as elementwise math.
        cov = self._cov3[idx0]
        tmp = np.einsum("ij,njk->nik", R, cov)
        covc = np.einsum("nik,lk->nil", tmp, R)
        c00, c01, c02 = covc[:, 0, 0], covc[:, 0, 1], covc[:, 0, 2]
        c11, c12, c22 = covc[:, 1, 1], covc[:, 1, 2], covc[:, 2, 2]
        j00 = fx / z
        j02 = -fx * pcam[:, 0] / (z * z)
        j11 = fy / z
        j12 = -fy * pcam[:, 1] / (z * z)
        A = j00 * j00 * c00 + 2 * j00 * j02 * c02 + j02 * j02 * c22 + 0.3
        B = j00 * (c01 * j11 + c02 * j12) + j02 * (c12 * j11 + c22 * j12)
        C = j11 * j11 * c11 + 2 * j11 * j12 * c12 + j12 * j12 * c22 + 0.3

        det = A * C - B * B
        ok = det > 1e-12
        idx0, z, uf, vf = idx0[ok], z[ok], uf[ok], vf[ok]
        A, B, C, det = A[ok], B[ok], C[ok], det[ok]

        # Conic (inverse covariance) and 3-sigma pixel radius. Splats larger
        # than the radius cap get their falloff shifted to reach zero exactly
        # at the clipped footprint edge (no hard square truncation). Opacity
        # is NOT renormalized when clipped — that used to concentrate mass
        # into bright blobs.
        ia, ib, ic = C / det, -B / det, A / det
        lam_max = 0.5 * (A + C) + np.sqrt(np.square(0.5 * (A - C)) + B * B)
        r_f = np.minimum(3.0 * np.sqrt(lam_max), float(self.max_radius))
        radius = np.maximum(np.ceil(r_f), 1).astype(np.int32)
        w_edge = np.exp(-0.5 * r_f * r_f / lam_max).astype(np.float32)  # ~0.011 unclipped

        alphas = self._alphas[idx0].astype(np.float32)
        colors = self.scene.colors[idx0]
        splats = dict(uf=uf.astype(np.float32), vf=vf.astype(np.float32),
                      ia=ia.astype(np.float32), ib=ib.astype(np.float32),
                      ic=ic.astype(np.float32), alpha=alphas, w_edge=w_edge,
                      radius=radius)

        # Log-depth layers: 0 = nearest, N-1 = farthest. Surfaces at different
        # depths composite with viser's over-operator instead of mixing colors.
        z_lo = max(float(z.min()), self.near)
        z_hi = float(np.percentile(z, 99.5))
        log_lo, log_hi = np.log(z_lo), np.log(max(z_hi, z_lo * 1.01))
        layer = np.clip(
            ((np.log(np.maximum(z, z_lo)) - log_lo) / (log_hi - log_lo) * _N_LAYERS
             ).astype(np.int32),
            0, _N_LAYERS - 1,
        )

        npix = H * W
        nL = _N_LAYERS
        acc = np.zeros((nL, 5, npix))  # w, r, g, b, w*z
        n_contrib = 0
        for pix, w, gi in self._contributions(splats, W, H):
            n_contrib += len(pix)
            lpix = pix + layer[gi].astype(np.int64) * npix
            acc[:, 0] += np.bincount(lpix, weights=w, minlength=nL * npix).reshape(nL, npix)
            acc[:, 4] += np.bincount(lpix, weights=w * z[gi], minlength=nL * npix).reshape(nL, npix)
            for ch in range(3):
                acc[:, ch + 1] += np.bincount(
                    lpix, weights=w * colors[gi, ch], minlength=nL * npix
                ).reshape(nL, npix)

        img = self._composite(acc[:, :4], npix)

        depth = np.full(npix, np.inf)
        for li in range(nL):
            wsum = acc[li, 0]
            hit = (1.0 - np.exp(-self.coverage_boost * wsum) >= 0.15) & ~np.isfinite(depth)
            with np.errstate(divide="ignore", invalid="ignore"):
                depth[hit] = acc[li, 4, hit] / np.maximum(wsum[hit], 1e-12)
        depth = depth.astype(np.float32)

        logger.debug("cpu_splats: %d splats, %d contributions, %d layers -> %dx%d in %.1fs",
                     len(idx0), n_contrib, nL, W, H, time.perf_counter() - t0)
        return img.reshape(H, W, 3), depth.reshape(H, W)

    def _contributions(
        self, s: dict, W: int, H: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (pixel_index, weight, splat_index) chunks for every splat's
        footprint, grouped by radius and chunked to bound memory."""
        u0 = np.round(s["uf"]).astype(np.int32)
        v0 = np.round(s["vf"]).astype(np.int32)
        for r in np.unique(s["radius"]):
            grp = np.flatnonzero(s["radius"] == r).astype(np.int32)
            k_sz = 2 * int(r) + 1
            dv, du = np.mgrid[-r:r + 1, -r:r + 1]
            du = du.ravel().astype(np.int32)
            dv = dv.ravel().astype(np.int32)
            step = max(1, _CONTRIB_BUDGET // (k_sz * k_sz))
            for i in range(0, len(grp), step):
                g = grp[i:i + step]
                px = u0[g][:, None] + du[None, :]          # (m, k^2)
                py = v0[g][:, None] + dv[None, :]
                dx = px.astype(np.float32) - s["uf"][g][:, None]
                dy = py.astype(np.float32) - s["vf"][g][:, None]
                quad = (s["ia"][g][:, None] * dx * dx
                        + 2.0 * s["ib"][g][:, None] * dx * dy
                        + s["ic"][g][:, None] * dy * dy)
                w = s["alpha"][g][:, None] * (np.exp(-0.5 * quad) - s["w_edge"][g][:, None])
                m = (w > _MIN_WEIGHT) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
                gi = np.broadcast_to(g[:, None], m.shape)[m]
                yield (py[m] * W + px[m]).astype(np.int64), w[m], gi

    def _composite(self, acc: np.ndarray, npix: int) -> np.ndarray:
        out = np.empty((npix, 3), dtype=np.float32)
        out[:] = self.background
        for layer in range(acc.shape[0] - 1, -1, -1):  # far to near
            wsum = acc[layer, 0]
            hit = wsum > 0
            rgb = np.zeros((npix, 3), dtype=np.float32)
            # Mixed slice + boolean indexing puts the pixel axis first: (nhit, 3).
            rgb[hit] = (acc[layer, 1:4, hit] / wsum[hit, None]).astype(np.float32)
            a = (1.0 - np.exp(-self.coverage_boost * wsum)).astype(np.float32)[:, None]
            out = a * rgb + (1.0 - a) * out
        return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

    def _flat_background(self, H: int, W: int) -> np.ndarray:
        img = np.empty((H, W, 3), dtype=np.uint8)
        img[:] = (self.background * 255).astype(np.uint8)
        return img
