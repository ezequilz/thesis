"""Loader for SOG v2 Gaussian splat bundles (.sog).

SOG (Spatially Ordered Gaussians) is PlayCanvas' compressed splat container:
a zip of lossless WebP images plus meta.json. Spec:
https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/

Decoding summary (all images are indexed pixel i -> gaussian i, row-major):
  - means_l/means_u: 16-bit per axis, dequantized into log-domain via
    meta.means.mins/maxs, then unlogged: sign(n) * (exp(|n|) - 1)
  - scales: RGB are indices into a 256-entry log-domain codebook (exp to linear)
  - quats: smallest-three in (w,x,y,z) order; RGB are the kept components
    quantized to [-sqrt(2)/2, +sqrt(2)/2]; A - 252 is the omitted component
  - sh0: RGB are codebook indices for the SH DC coefficients; A is opacity
  - shN (optional, NOT decoded yet): palette of higher-order SH coefficients

Higher-order SH is intentionally skipped for now — the harness only needs
base color for navigation-quality rendering. The gsplat renderer can add it
later for view-dependent effects.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .types import GaussianScene

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814  # Y_0^0 = 1 / (2 * sqrt(pi))


def _read_image(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Read a webp from the bundle as a (H, W, C) uint8 array."""
    with zf.open(name) as f:
        img = Image.open(io.BytesIO(f.read()))
        return np.asarray(img)


def _pixels(img: np.ndarray, count: int) -> np.ndarray:
    """Flatten a (H, W, C) image to the first `count` per-gaussian rows (count, C)."""
    return img.reshape(-1, img.shape[-1])[:count]


def load_sog(path: str | Path) -> GaussianScene:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("meta.json"))
        if meta.get("version") != 2:
            raise ValueError(f"Unsupported SOG version: {meta.get('version')}")
        count = meta["count"]
        logger.info("Loading SOG bundle %s (%d gaussians)", path.name, count)

        # --- Positions -----------------------------------------------------
        means_l = _pixels(_read_image(zf, meta["means"]["files"][0]), count)
        means_u = _pixels(_read_image(zf, meta["means"]["files"][1]), count)
        q16 = (means_u[:, :3].astype(np.uint16) << 8) | means_l[:, :3].astype(np.uint16)
        mins = np.asarray(meta["means"]["mins"], dtype=np.float32)
        maxs = np.asarray(meta["means"]["maxs"], dtype=np.float32)
        log_pos = mins + (q16.astype(np.float32) / 65535.0) * (maxs - mins)
        means = np.sign(log_pos) * (np.exp(np.abs(log_pos)) - 1.0)

        # --- Scales --------------------------------------------------------
        scales_img = _pixels(_read_image(zf, meta["scales"]["files"][0]), count)
        scales_codebook = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
        scales = np.exp(scales_codebook[scales_img[:, :3]])

        # --- Orientation (smallest-three, (w,x,y,z) order) ------------------
        quats_img = _pixels(_read_image(zf, meta["quats"]["files"][0]), count)
        kept = (quats_img[:, :3].astype(np.float32) / 255.0 - 0.5) * (2.0 / np.sqrt(2.0))
        omitted = np.sqrt(np.clip(1.0 - np.sum(kept**2, axis=1), 0.0, None))
        mode = quats_img[:, 3].astype(np.int64) - 252  # which of (w,x,y,z) was omitted
        quats = np.empty((count, 4), dtype=np.float32)
        cols = np.arange(4)
        for m in range(4):
            sel = mode == m
            kept_cols = cols[cols != m]
            quats[np.ix_(sel, kept_cols)] = kept[sel]
            quats[sel, m] = omitted[sel]

        # --- Base color + opacity -------------------------------------------
        sh0_img = _pixels(_read_image(zf, meta["sh0"]["files"][0]), count)
        sh0_codebook = np.asarray(meta["sh0"]["codebook"], dtype=np.float32)
        colors = np.clip(0.5 + sh0_codebook[sh0_img[:, :3]] * SH_C0, 0.0, 1.0)
        opacities = sh0_img[:, 3].astype(np.float32) / 255.0

    return GaussianScene(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats,
        opacities=opacities,
        colors=colors.astype(np.float32),
    )
