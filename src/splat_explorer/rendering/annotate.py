"""Image annotation helpers for what the VLM sees.

- depth_to_image: turn a float depth buffer into a labelled grayscale image
  (bright = near) so it can be attached to prompts next to the RGB view.
- draw_spawn_markers: paint numbered high-visibility dots for candidate start
  positions onto the bird's-eye render.
- draw_path_map: paint the agent's walked path and per-step camera frustums
  onto the same bird's-eye render (used as the on-demand map action).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import Camera

_MARKER_FILL = (255, 64, 255)     # bright magenta — absent from indoor scenes
_MARKER_OUTLINE = (255, 255, 255)
# Match the viser overlay: red trajectory + frustum, cyan for the live pose.
_PATH_COLOR = (255, 80, 80)
_CURRENT_COLOR = (90, 190, 255)


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


def _rotate_around(vec: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation of `vec` around `axis` by `degrees`."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _ground_heading(heading: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    h = np.asarray(heading, dtype=np.float64)
    h = h - up * np.dot(h, up)
    n = np.linalg.norm(h)
    if n < 1e-8:
        return None
    return h / n


def _px(uv) -> tuple[float, float] | None:
    u, v = float(uv[0]), float(uv[1])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return (u, v)


def _frustum_length(camera: Camera, position: np.ndarray, heading: np.ndarray,
                    target_px: float = 40.0) -> float:
    """World-space frustum depth that projects to about `target_px` pixels."""
    p0 = project_to_pixels(camera, position[None])[0]
    p1 = project_to_pixels(camera, (position + heading)[None])[0]
    if _px(p0) is None or _px(p1) is None:
        return 1.5
    px_per_unit = float(np.linalg.norm(p1 - p0))
    if px_per_unit < 1e-3:
        return 1.5
    return float(np.clip(target_px / px_per_unit, 0.4, 8.0))


def draw_path_map(
    image: np.ndarray,
    camera: Camera,
    poses: list[dict],
    fov_deg: float = 75.0,
    up: np.ndarray | None = None,
    title: str = (
        "BIRD'S-EYE MAP (ceiling removed) | red = path | triangles = view | cyan = now"
    ),
) -> np.ndarray:
    """Paint the walked path and a top-down camera frustum at every past pose.

    `poses` are dicts with `position` (3,), `heading` (3, horizontal look
    direction) and `step` (int). The last pose is treated as current.
    The splat backdrop is not re-rendered — only this overlay is cheap to
    refresh after every agent step.
    """
    img = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    up_vec = np.asarray(up if up is not None else (0.0, 1.0, 0.0), dtype=np.float64)
    up_vec = up_vec / max(np.linalg.norm(up_vec), 1e-12)

    pixel_path: list[tuple[float, float] | None] = []
    prepared: list[tuple[dict, tuple[float, float], np.ndarray]] = []
    for pose in poses:
        position = np.asarray(pose["position"], dtype=np.float64)
        heading = _ground_heading(pose["heading"], up_vec)
        uv = _px(project_to_pixels(camera, position[None])[0])
        pixel_path.append(uv)
        if uv is None or heading is None:
            continue
        prepared.append((pose, uv, heading))

    line_w = max(3, image.shape[1] // 220)
    # Connected path: skip zero-length segments (in-place rotates) so the
    # line stays a walk trail rather than a blob of overlapping dots.
    for i in range(1, len(pixel_path)):
        a, b = pixel_path[i - 1], pixel_path[i]
        if a is None or b is None:
            continue
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < 4.0:
            continue
        draw.line([a, b], fill=_PATH_COLOR + (230,), width=line_w)

    n = max(len(prepared), 1)
    half_fov = float(fov_deg) / 2.0
    r = max(4, image.shape[1] // 110)
    for i, (pose, uv, heading) in enumerate(prepared):
        current = i == len(prepared) - 1
        position = np.asarray(pose["position"], dtype=np.float64)
        length = _frustum_length(camera, position, heading)
        left = position + _rotate_around(heading, up_vec, half_fov) * length
        right = position + _rotate_around(heading, up_vec, -half_fov) * length
        lu, ru = _px(project_to_pixels(camera, left[None])[0]), _px(
            project_to_pixels(camera, right[None])[0]
        )
        if current:
            fill, outline = _CURRENT_COLOR + (130,), _CURRENT_COLOR + (255,)
        else:
            fade = int(50 + 90 * (i + 1) / n)
            fill, outline = _PATH_COLOR + (fade,), _PATH_COLOR + (min(255, fade + 80),)
        if lu is not None and ru is not None:
            draw.polygon([uv, lu, ru], fill=fill, outline=outline)
        tip = _px(project_to_pixels(camera, (position + heading * length)[None])[0])
        if tip is not None:
            draw.line([uv, tip], fill=outline, width=2)
        dot = _CURRENT_COLOR if current else _PATH_COLOR
        draw.ellipse((uv[0] - r, uv[1] - r, uv[0] + r, uv[1] + r),
                     fill=dot + (255,), outline=(255, 255, 255, 255), width=2)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    ink = ImageDraw.Draw(img)
    font = _font(max(12, 2 * r - 6))
    for i, (pose, uv, _) in enumerate(prepared):
        current = i == len(prepared) - 1
        text = str(pose.get("step", i))
        tx, ty = uv[0] + r + 3, uv[1] - r - 2
        fill = (180, 230, 255) if current else (255, 255, 0)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            ink.text((tx + dx, ty + dy), text, fill=(0, 0, 0), font=font)
        ink.text((tx, ty), text, fill=fill, font=font)

    _draw_label(ink, (8, 6), title)
    return np.asarray(img)
