"""Image annotation helpers for what the VLM sees.

- depth_to_image: turn a float depth buffer into a labelled grayscale image
  (bright = near) so it can be attached to prompts next to the RGB view.
- draw_spawn_markers: paint numbered high-visibility dots for candidate start
  positions onto the bird's-eye render.
- draw_path_map: paint the agent's walked path and per-step camera frustums
  onto the same bird's-eye render (used as the on-demand map action).
- coverage overlay: accumulate large, distance-faded view cones on a duplicate
  of that bird's-eye so the VLM can see which floor area has been looked at.
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
# Coverage cones: saturated lime. One close view is ~25% opaque (readable on
# parquet); 4 overlapping looks saturate to solid lime. Full strength holds
# out to ~1.5 m (curtain distance), then fades.
_COVERAGE_RGB = np.array([110.0, 220.0, 40.0], dtype=np.float64)
_COVERAGE_GAIN = 0.25          # peak contribution of one close-range view
_COVERAGE_NEAR_M = 1.5         # full strength out to this many scene units
_COVERAGE_FAR_M = 5.5          # fade to zero by this distance (far walls stay faint)
_COVERAGE_DISPLAY_ALPHA = 1.0  # coverage=1 is fully lime; the path map is separate
_SCENE_LUMA_MIN = 16.0         # bird's-eye void is darker than this; rooms are not


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


def scene_mask(image: np.ndarray) -> np.ndarray:
    """Pixels that look like reconstructed interior (not the dark void)."""
    luma = np.asarray(image, dtype=np.float64).mean(axis=-1)
    return luma > _SCENE_LUMA_MIN


def overlay_coverage(image: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Tint `image` with accumulated coverage. One close view is a light lime
    wash; several overlapping views go to solid lime."""
    rgb = np.asarray(image, dtype=np.float64)
    cov = np.clip(np.asarray(coverage, dtype=np.float64), 0.0, 1.0)
    alpha = (cov * _COVERAGE_DISPLAY_ALPHA)[..., None]
    out = rgb * (1.0 - alpha) + _COVERAGE_RGB * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def paint_coverage_cone(
    coverage: np.ndarray,
    camera: Camera,
    position: np.ndarray,
    heading: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    gain: float = _COVERAGE_GAIN,
    near_m: float = _COVERAGE_NEAR_M,
    far_m: float = _COVERAGE_FAR_M,
) -> None:
    """Add one view cone into `coverage` (in-place, clipped to [0, 1]).

    The cone matches the camera's horizontal FOV. Strength stays at `gain` for
    a circular sector out to `near_m` (about 1.5 m / the curtains), then falls
    off with a smooth gradient to zero by `far_m`. Overlapping views stack
    until the per-pixel value saturates at 1 (~4 close looks).
    """
    heading = _ground_heading(heading, up)
    if heading is None:
        return
    position = np.asarray(position, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-12)
    H, W = coverage.shape
    half_fov = float(fov_deg) / 2.0
    box = _cone_bbox(camera, position, heading, up, far_m, half_fov, H, W)
    if box is None:
        return
    y0, y1, x0, x1 = box

    uu, vv = np.meshgrid(
        np.arange(x0, x1, dtype=np.float64) + 0.5,
        np.arange(y0, y1, dtype=np.float64) + 0.5,
    )
    fx, fy = float(camera.fx), float(camera.fy)
    dirs_cam = np.stack([
        (uu - camera.width / 2.0) / fx,
        (vv - camera.height / 2.0) / fy,
        np.ones_like(uu),
    ], axis=-1)
    R = np.asarray(camera.rotation, dtype=np.float64)
    dirs = dirs_cam @ R.T
    origin = np.asarray(camera.position, dtype=np.float64)
    denom = dirs @ up
    plane_h = float(np.dot(position, up))
    origin_h = float(np.dot(origin, up))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_h - origin_h) / denom
    valid = np.isfinite(t) & (t > 0.0)
    pts = origin + t[..., None] * dirs
    delta = pts - position
    delta = delta - (delta @ up)[..., None] * up
    dist = np.linalg.norm(delta, axis=-1)
    along = delta @ heading
    with np.errstate(invalid="ignore"):
        cosang = np.divide(along, dist, out=np.zeros_like(along), where=dist > 1e-6)
    in_cone = valid & (along > 0.05) & (cosang >= np.cos(np.radians(half_fov)))

    span = max(far_m - near_m, 1e-3)
    fade = np.clip((far_m - dist) / span, 0.0, 1.0)
    falloff = fade * fade * (3.0 - 2.0 * fade)  # smoothstep
    add = np.where(in_cone, gain * falloff, 0.0)
    region = coverage[y0:y1, x0:x1]
    np.clip(region + add, 0.0, 1.0, out=region)


def _cone_bbox(
    camera: Camera,
    position: np.ndarray,
    heading: np.ndarray,
    up: np.ndarray,
    far_m: float,
    half_fov_deg: float,
    H: int,
    W: int,
) -> tuple[int, int, int, int] | None:
    """Pixel bbox of the coverage triangle, padded, clipped to the image."""
    left = position + _rotate_around(heading, up, half_fov_deg) * far_m
    right = position + _rotate_around(heading, up, -half_fov_deg) * far_m
    tip = position + heading * far_m
    uv = project_to_pixels(camera, np.stack([position, left, right, tip]))
    ok = np.isfinite(uv).all(axis=1)
    if not ok.any():
        return None
    pts = uv[ok]
    x0 = int(np.floor(pts[:, 0].min())) - 2
    x1 = int(np.ceil(pts[:, 0].max())) + 2
    y0 = int(np.floor(pts[:, 1].min())) - 2
    y1 = int(np.ceil(pts[:, 1].max())) + 2
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def draw_coverage_map(
    image: np.ndarray,
    camera: Camera,
    poses: list[dict],
    coverage: np.ndarray,
    coverage_fraction: float,
    fov_deg: float = 75.0,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Bird's-eye with accumulated view cones plus the usual path/frustum overlay.

    Lime paint marks floor that has been looked at; overlapping views stack
    toward solid lime. The small camera triangles stay on top so heading is
    still readable. `coverage_fraction` is shown in the title (0..1).
    """
    tinted = overlay_coverage(image, coverage)
    pct = int(round(100.0 * float(np.clip(coverage_fraction, 0.0, 1.0))))
    title = (
        f"COVERAGE MAP (viewed area) | lime = seen within ~1.5m, then fades "
        f"| ~4 overlaps = solid | coverage {pct}%"
    )
    return draw_path_map(
        tinted, camera, poses, fov_deg=fov_deg, up=up, title=title,
    )
