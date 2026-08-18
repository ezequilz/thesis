"""Loader for standard 3D Gaussian Splatting PLY exports (.ply).

The uncompressed format written by the INRIA reference trainer and most
pipelines since (nerfstudio, gsplat, PlayCanvas export): a binary
little-endian PLY whose vertex element carries per-gaussian floats
  x y z, rot_0..3 (quaternion, w-first), scale_0..2 (log-domain),
  opacity (logit), f_dc_0..2 (SH DC term), f_rest_* (higher-order SH).

Activation functions are applied on load (exp for scales, sigmoid for
opacity), matching how trainers interpret these fields. Higher-order SH is
skipped, same as the SOG loader — only the DC term is decoded into base
color. Property order is taken from the header, so field layout variations
across exporters are handled.

The file is memory-mapped and only the needed columns are copied out, so the
f_rest block (45 of the 62 floats per splat here) never occupies RAM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .types import GaussianScene

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814  # Y_0^0 = 1 / (2 * sqrt(pi))

_REQUIRED = (
    "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
    "scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
)

_PLY_DTYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def _parse_header(f) -> tuple[int, list[tuple[str, str]], int]:
    """Return (vertex_count, [(name, numpy_dtype)], data_offset)."""
    if f.readline().strip() != b"ply":
        raise ValueError("Not a PLY file.")
    count = 0
    props: list[tuple[str, str]] = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected end of PLY header.")
        parts = line.decode("ascii").strip().split()
        if not parts:
            continue
        if parts[0] == "format":
            if parts[1] != "binary_little_endian":
                raise ValueError(f"Only binary_little_endian PLY is supported, got {parts[1]}.")
        elif parts[0] == "element":
            if parts[1] != "vertex" and count:
                raise ValueError("Extra PLY elements after vertex are not supported.")
            if parts[1] == "vertex":
                count = int(parts[2])
        elif parts[0] == "property":
            if parts[1] == "list":
                raise ValueError("PLY list properties are not supported for splats.")
            props.append((parts[2], _PLY_DTYPES[parts[1]]))
        elif parts[0] == "end_header":
            return count, props, f.tell()


def load_ply(path: str | Path) -> GaussianScene:
    path = Path(path)
    with open(path, "rb") as f:
        count, props, offset = _parse_header(f)

    names = [n for n, _ in props]
    missing = [n for n in _REQUIRED if n not in names]
    if missing:
        raise ValueError(
            f"{path.name} is not a standard 3DGS PLY — missing properties: {missing}. "
            f"(Compressed/self-organizing PLY variants are not supported; "
            f"export uncompressed or use the .sog bundle.)"
        )
    logger.info("Loading 3DGS PLY %s (%d gaussians, %d properties)", path.name, count, len(props))

    data = np.memmap(path, dtype=np.dtype(props), mode="r", offset=offset, shape=(count,))

    def cols(*fields: str) -> np.ndarray:
        return np.stack([np.asarray(data[f], dtype=np.float32) for f in fields], axis=1)

    means = cols("x", "y", "z")
    scales = np.exp(cols("scale_0", "scale_1", "scale_2"))
    quats = cols("rot_0", "rot_1", "rot_2", "rot_3")  # (w, x, y, z)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(data["opacity"], dtype=np.float32)))
    colors = np.clip(0.5 + cols("f_dc_0", "f_dc_1", "f_dc_2") * SH_C0, 0.0, 1.0)

    return GaussianScene(
        means=means, scales=scales, quats=quats, opacities=opacities, colors=colors
    )
