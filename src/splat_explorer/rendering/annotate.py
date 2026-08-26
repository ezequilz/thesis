"""Image annotation helpers for what the VLM sees.

- depth_to_image: turn a float depth buffer into a labelled grayscale image
  (bright = near) so it can be attached to prompts next to the RGB view.
- draw_spawn_markers: paint numbered high-visibility dots for candidate start
  positions onto the bird's-eye render.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import Camera

_MARKER_FILL = (255, 64, 255)     # bright magenta — absent from indoor scenes
_MARKER_OUTLINE = (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 18) -> None:
    """Text with a dark backing box so it stays readable on any render."""
    font = _font(size)
    box = draw.textbbox(xy, text, font=font)
    pad = 4
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    draw.text(xy, text, fill=(255, 255, 255), font=font)


def depth_to_image(depth: np.ndarray, label: str = "DEPTH MAP  bright = near, dark = far, black = nothing") -> np.ndarray:
    """(H, W) float depth (np.inf = empty) -> labelled (H, W, 3) uint8 image."""
    H, W = depth.shape
    gray = np.zeros((H, W), dtype=np.uint8)
    finite = np.isfinite(depth)
    if finite.any():
        d = depth[finite]
        vmin = float(np.percentile(d, 1.0))
        vmax = float(np.percentile(d, 99.0))
        span = max(vmax - vmin, 1e-6)
        norm = np.clip((vmax - depth[finite]) / span, 0.0, 1.0)  # near -> 1
        gray[finite] = (40 + 215 * norm).astype(np.uint8)

    img = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2))
    _draw_label(ImageDraw.Draw(img), (8, 6), label)
    return np.asarray(img)


def project_to_pixels(camera: Camera, points: np.ndarray) -> np.ndarray:
    """World points (N, 3) -> pixel coords (N, 2) float; NaN if behind camera."""
    w2c = camera.w2c
    pcam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pcam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera.fx * pcam[:, 0] / z + camera.width / 2.0
        v = camera.fy * pcam[:, 1] / z + camera.height / 2.0
    uv = np.stack([u, v], axis=1)
    uv[z <= 0] = np.nan
    return uv


def draw_spawn_markers(
    image: np.ndarray,
    camera: Camera,
    positions: np.ndarray,
    title: str = "BIRD'S-EYE VIEW (ceiling removed) | numbered dots = candidate start points",
) -> np.ndarray:
    """Paint numbered start-point markers onto a rendered view."""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    r = max(6, image.shape[1] // 90)
    font = _font(max(14, 2 * r - 4))

    for i, (u, v) in enumerate(project_to_pixels(camera, np.asarray(positions, dtype=np.float64))):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        draw.ellipse((u - r, v - r, u + r, v + r), fill=_MARKER_FILL,
                     outline=_MARKER_OUTLINE, width=2)
        text = str(i)
        # Number to the right of the dot, black-outlined for contrast.
        tx, ty = u + r + 3, v - r - 2
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), text, fill=(255, 255, 0), font=font)

    _draw_label(draw, (8, 6), title)
    return np.asarray(img)
