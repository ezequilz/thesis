"""Loader for SOG v2 Gaussian splat bundles (.sog) and streamed SOG directories.

SOG (Spatially Ordered Gaussians) is PlayCanvas' compressed splat container:
a zip of lossless WebP images plus meta.json, or the same files unpacked in a
folder. Spec:
https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/

Streamed SOG (lod-meta.json + per-chunk folders) is the same decode per chunk,
then one LOD level concatenated with the optional environment splat:
https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/

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
from typing import Protocol

import numpy as np
from PIL import Image

from .types import GaussianScene

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814  # Y_0^0 = 1 / (2 * sqrt(pi))


class _SogBundle(Protocol):
    def read_bytes(self, name: str) -> bytes: ...


class _ZipBundle:
    def __init__(self, zf: zipfile.ZipFile):
        self._zf = zf

    def read_bytes(self, name: str) -> bytes:
        return self._zf.read(name)


class _DirBundle:
    def __init__(self, root: Path):
        self._root = root

    def read_bytes(self, name: str) -> bytes:
        path = self._root / name
        if not path.is_file():
            raise FileNotFoundError(f"SOG file missing: {path}")
        return path.read_bytes()


def _read_image(bundle: _SogBundle, name: str) -> np.ndarray:
    """Read a webp from the bundle as a (H, W, C) uint8 array."""
    img = Image.open(io.BytesIO(bundle.read_bytes(name)))
    return np.asarray(img)


def _pixels(img: np.ndarray, count: int) -> np.ndarray:
    """Flatten a (H, W, C) image to the first `count` per-gaussian rows (count, C)."""
    if img.ndim == 2:
        img = img[..., None]
    return img.reshape(-1, img.shape[-1])[:count]


def _decode_sog(bundle: _SogBundle, label: str) -> GaussianScene:
    meta = json.loads(bundle.read_bytes("meta.json"))
    if meta.get("version") != 2:
        raise ValueError(f"Unsupported SOG version: {meta.get('version')}")
    count = meta["count"]
    logger.info("Loading SOG bundle %s (%d gaussians)", label, count)

    # --- Positions -----------------------------------------------------
    means_l = _pixels(_read_image(bundle, meta["means"]["files"][0]), count)
    means_u = _pixels(_read_image(bundle, meta["means"]["files"][1]), count)
    q16 = (means_u[:, :3].astype(np.uint16) << 8) | means_l[:, :3].astype(np.uint16)
    mins = np.asarray(meta["means"]["mins"], dtype=np.float32)
    maxs = np.asarray(meta["means"]["maxs"], dtype=np.float32)
    log_pos = mins + (q16.astype(np.float32) / 65535.0) * (maxs - mins)
    means = np.sign(log_pos) * (np.exp(np.abs(log_pos)) - 1.0)

    # --- Scales --------------------------------------------------------
    scales_img = _pixels(_read_image(bundle, meta["scales"]["files"][0]), count)
    scales_codebook = np.asarray(meta["scales"]["codebook"], dtype=np.float32)
    scales = np.exp(scales_codebook[scales_img[:, :3]])

    # --- Orientation (smallest-three, (w,x,y,z) order) ------------------
    quats_img = _pixels(_read_image(bundle, meta["quats"]["files"][0]), count)
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
    sh0_img = _pixels(_read_image(bundle, meta["sh0"]["files"][0]), count)
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


def load_sog(path: str | Path) -> GaussianScene:
    """Load a bundled (.sog zip) or unbundled (folder / meta.json) SOG v2 scene."""
    path = Path(path)
    if path.is_file() and path.name == "meta.json":
        path = path.parent
    if path.is_dir():
        if not (path / "meta.json").is_file():
            raise FileNotFoundError(f"No meta.json in SOG directory {path}")
        return _decode_sog(_DirBundle(path), path.name)
    with zipfile.ZipFile(path) as zf:
        return _decode_sog(_ZipBundle(zf), path.name)


def _lod_file_indices(tree: dict, lod_level: int) -> list[int]:
    """Unique chunk-file indices referenced by one LOD level, in first-seen order.

    Each streamed-SOG chunk holds exactly one LOD and is fully covered by the
    leaf ranges that point at it, so loading those files whole is equivalent
    to concatenating the per-leaf [offset, offset+count) slices.
    """
    found: list[int] = []
    seen: set[int] = set()
    stack = [tree]
    key = str(lod_level)
    while stack:
        node = stack.pop()
        spec = (node.get("lods") or {}).get(key)
        if spec is not None:
            idx = int(spec["file"])
            if idx not in seen:
                seen.add(idx)
                found.append(idx)
        stack.extend(reversed(node.get("children") or []))
    return found


def load_sog_lod(
    path: str | Path,
    lod_level: int = 0,
    include_environment: bool = True,
) -> GaussianScene:
    """Load one LOD level of a streamed SOG directory (plus the environment splat).

    LOD 0 is finest. Higher levels are coarser stand-ins for the same volume —
    they must not be concatenated together.
    """
    path = Path(path)
    root = path.parent if path.name == "lod-meta.json" else path
    meta_path = root / "lod-meta.json"
    meta = json.loads(meta_path.read_text())
    version = meta.get("version", 1)
    if version > 1:
        raise ValueError(f"Unsupported streamed SOG version: {version}")

    n_levels = int(meta.get("lodLevels") or len(meta.get("counts") or [1]))
    if n_levels < 1:
        raise ValueError(f"{meta_path} has no LOD levels")
    level = int(lod_level)
    if level < 0 or level >= n_levels:
        clamped = max(0, min(level, n_levels - 1))
        logger.warning("LOD %d out of range 0..%d for %s; using %d",
                       level, n_levels - 1, root.name, clamped)
        level = clamped

    filenames = list(meta.get("filenames") or [])
    file_indices = _lod_file_indices(meta["tree"], level)
    if not file_indices:
        raise ValueError(f"No chunks at LOD {level} in {meta_path}")

    counts = meta.get("counts")
    expected = counts[level] if isinstance(counts, list) and level < len(counts) else None
    logger.info(
        "Loading streamed SOG %s LOD %d (%d chunks%s)",
        root.name, level, len(file_indices),
        f", ~{expected:,} gaussians" if expected else "",
    )

    parts: list[GaussianScene] = []
    for idx in file_indices:
        rel = filenames[idx]
        chunk = (root / rel).parent
        parts.append(load_sog(chunk))

    env = meta.get("environment")
    if include_environment and env:
        env_dir = (root / env).parent
        logger.info("Loading environment splat %s", env_dir.name)
        parts.append(load_sog(env_dir))

    scene = GaussianScene.concatenate(parts)
    if expected is not None and include_environment is False and scene.num_gaussians != expected:
        logger.warning(
            "LOD %d decoded %d gaussians, lod-meta counts[%d]=%d",
            level, scene.num_gaussians, level, expected,
        )
    return scene
