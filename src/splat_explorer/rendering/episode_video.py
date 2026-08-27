"""On-demand episode video: RGB | bird's-eye map, one frame per step.

When a step has a gpt-image-2 repair (`step_NNN_regen.png`), that one slide
shows RGB | repaired instead of the map, then the next step returns to
RGB | map.

Knobs live on `EpisodeVideoConfig` so timing, layout, overlay styling, and
output names can be changed without touching the dashboard server. The
encoder prefers ffmpeg (H.264 MP4) and falls back to animated WebP via
Pillow so the dashboard image does not need extra Python packages.

Once a video exists for an episode it is reused, unless the step count,
timing knobs, or layout version no longer match (a live run that grew, a
config tweak, or a composition change such as repaired-image comparison).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class EpisodeVideoConfig:
    """Adjustable video layout / encode settings."""

    seconds_per_step: float = 1.0
    output_mp4: str = "episode_video.mp4"
    output_webp: str = "episode_video.webp"
    meta_name: str = "episode_video.json"
    # None = use the first RGB frame's size for BOTH panes (equal size).
    pane_width: int | None = None
    pane_height: int | None = None
    gap_px: int = 2
    background: tuple[int, int, int] = (10, 13, 18)
    include_choose_start: bool = False
    # Overlay (report_artifact only) — bottom-right of the RGB pane.
    overlay_font_div: int = 48          # pane_height / this ≈ font size
    overlay_min_font: int = 12
    overlay_max_font: int = 18
    overlay_margin: int = 10
    overlay_pad: int = 8
    overlay_max_width_frac: float = 0.46
    overlay_fill: tuple[int, int, int] = (236, 240, 246)
    overlay_box: tuple[int, int, int, int] = (12, 16, 22, 165)
    # Final "Artifact reports" slide (dashboard-style cards, dark background).
    summary_seconds: float = 1.0
    summary_bg: tuple[int, int, int] = (16, 19, 24)       # --bg
    summary_panel: tuple[int, int, int] = (23, 28, 36)    # --panel
    summary_border: tuple[int, int, int] = (230, 180, 80) # --warn
    summary_text: tuple[int, int, int] = (220, 227, 238)  # --text
    summary_dim: tuple[int, int, int] = (139, 151, 168)   # --dim
    ffmpeg_timeout_s: float = 120.0


# Bump when frame composition changes so cached episode videos restitch.
LAYOUT_VERSION = 2


class EpisodeVideoError(Exception):
    """Raised when an episode has nothing to stitch, or encoding fails."""


@dataclass
class VideoResult:
    path: Path
    content_type: str
    cached: bool


def format_artifact_overlay(args: dict) -> str:
    """Parsed report_artifact args as the on-video caption."""
    description = args.get("description", "")
    image_region = args.get("image_region", "")
    severity = args.get("severity", "")
    return (
        "report_artifact{\n"
        f'"description": {json.dumps(description, ensure_ascii=False)},\n'
        f'    "image_region": {json.dumps(image_region, ensure_ascii=False)},\n'
        f'    "severity": {json.dumps(severity, ensure_ascii=False)}\n'
        "}"
    )


def load_episode_steps(episode_dir: Path, cfg: EpisodeVideoConfig) -> list[dict]:
    """actions.jsonl records that become video frames (RGB + optional map)."""
    trace = episode_dir / "actions.jsonl"
    if not trace.is_file():
        raise EpisodeVideoError("Episode has no actions.jsonl.")
    steps = []
    with open(trace) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            name = (rec.get("action") or {}).get("name")
            if name == "choose_start" and not cfg.include_choose_start:
                continue
            frame = rec.get("frame")
            if not frame or not (episode_dir / frame).is_file():
                continue
            steps.append(rec)
    if not steps:
        raise EpisodeVideoError("Episode has no RGB frames to stitch.")
    return steps


def render_episode_video(
    episode_dir: Path,
    cfg: EpisodeVideoConfig | None = None,
    force: bool = False,
) -> VideoResult:
    """Return a video for `episode_dir`, rendering only if nothing reusable exists."""
    cfg = cfg or EpisodeVideoConfig()
    episode_dir = Path(episode_dir)
    steps = load_episode_steps(episode_dir, cfg)

    existing = _existing_output(episode_dir, cfg)
    if existing is not None and not force and _cache_valid(episode_dir, cfg, len(steps)):
        path, content_type = existing
        return VideoResult(path, content_type, cached=True)

    frames = build_video_frames(episode_dir, steps, cfg)
    path, content_type = _encode(frames, episode_dir, cfg)
    _write_meta(episode_dir, cfg, n_steps=len(steps), format=path.suffix.lstrip("."))
    logger.info("Wrote episode video %s (%d steps, %s)", path.name, len(steps), content_type)
    return VideoResult(path, content_type, cached=False)


def compose_episode_frames(
    episode_dir: Path,
    steps: list[dict],
    cfg: EpisodeVideoConfig,
) -> list[Image.Image]:
    """RGB | map composites, with artifact text on report_artifact steps.

    If that step has a repaired PNG, the right pane is the repair instead of
    the map (one comparison slide, then back to the path map).
    """
    pane_w, pane_h = _pane_size(episode_dir, steps, cfg)
    frames = []
    for rec in steps:
        rgb = Image.open(episode_dir / rec["frame"]).convert("RGB")
        repaired = _repaired_image(episode_dir, rec)
        map_img = None
        if repaired is None:
            map_name = rec.get("map_frame")
            if map_name and (episode_dir / map_name).is_file():
                map_img = Image.open(episode_dir / map_name).convert("RGB")
        overlay = None
        action = rec.get("action") or {}
        if action.get("name") == "report_artifact":
            overlay = format_artifact_overlay(action.get("args") or {})
        right = repaired if repaired is not None else map_img
        right_caption = "repaired" if repaired is not None else None
        frames.append(_compose_pair(
            rgb, right, overlay, pane_w, pane_h, cfg,
            right_caption=right_caption,
        ))
    return frames


def artifacts_from_steps(steps: list[dict]) -> list[dict]:
    """report_artifact rows in dashboard-card order: step, severity, region, description."""
    out = []
    for rec in steps:
        action = rec.get("action") or {}
        if action.get("name") != "report_artifact":
            continue
        args = action.get("args") or {}
        out.append({
            "step": rec.get("step"),
            "severity": args.get("severity") or "?",
            "image_region": args.get("image_region") or "",
            "description": args.get("description") or "",
        })
    return out


def summary_hold_count(cfg: EpisodeVideoConfig) -> int:
    """How many constant-fps copies of the summary slide equal `summary_seconds`."""
    return max(1, int(round(cfg.summary_seconds / max(cfg.seconds_per_step, 0.05))))


def build_video_frames(
    episode_dir: Path,
    steps: list[dict],
    cfg: EpisodeVideoConfig,
) -> list[Image.Image]:
    """Step composites plus a held final artifact-report slide."""
    frames = compose_episode_frames(episode_dir, steps, cfg)
    summary = compose_artifact_summary_slide(frames[0].size, artifacts_from_steps(steps), cfg)
    return frames + [summary] * summary_hold_count(cfg)


def compose_artifact_summary_slide(
    size: tuple[int, int],
    artifacts: list[dict],
    cfg: EpisodeVideoConfig,
) -> Image.Image:
    """Dashboard-style artifact list on a dark full-frame slide."""
    w, h = size
    canvas = Image.new("RGB", (w, h), cfg.summary_bg)
    margin = max(20, w // 48)
    card_w = min(w - 2 * margin, max(420, int(w * 0.58)))
    title_size = max(16, min(28, h // 28))
    body_size = max(13, min(18, h // 40))
    gap = max(8, h // 80)
    pad = max(8, h // 70)

    for _ in range(8):
        if _layout_summary_fits(h, margin, title_size, body_size, pad, gap, card_w, artifacts, cfg):
            break
        title_size = max(14, title_size - 1)
        body_size = max(11, body_size - 1)
        pad = max(6, pad - 1)
        gap = max(6, gap - 1)

    draw = ImageDraw.Draw(canvas)
    title_font = _sans_font(title_size, bold=True)
    body_font = _sans_font(body_size, bold=False)
    bold_font = _sans_font(body_size, bold=True)
    y = margin
    draw.text((margin, y), "Artifact reports", fill=cfg.summary_text, font=title_font)
    title_box = draw.textbbox((margin, y), "Artifact reports", font=title_font)
    y = title_box[3] + gap + 4

    if not artifacts:
        draw.text((margin, y), "No artifacts reported.", fill=cfg.summary_dim, font=body_font)
        return canvas

    inner_w = card_w - 2 * pad
    for art in artifacts:
        header_h, desc_lines, desc_h = _measure_card(
            draw, art, inner_w, body_font, bold_font, body_size,
        )
        card_h = pad + header_h + 6 + desc_h + pad
        x0, y0 = margin, y
        x1, y1 = margin + card_w, y + card_h
        draw.rounded_rectangle(
            (x0, y0, x1, y1), radius=6,
            fill=cfg.summary_panel, outline=cfg.summary_border, width=1,
        )
        _draw_card_contents(
            draw, art, x0 + pad, y0 + pad, inner_w,
            body_font, bold_font, body_size, cfg,
            desc_lines,
        )
        y = y1 + gap
        if y >= h - margin:
            break
    return canvas


# --- internals --------------------------------------------------------------

def _pane_size(
    episode_dir: Path, steps: list[dict], cfg: EpisodeVideoConfig,
) -> tuple[int, int]:
    if cfg.pane_width and cfg.pane_height:
        return int(cfg.pane_width), int(cfg.pane_height)
    with Image.open(episode_dir / steps[0]["frame"]) as im:
        return im.size


def _fit(img: Image.Image, pane_w: int, pane_h: int, bg: tuple[int, int, int]) -> Image.Image:
    """Letterbox `img` into a pane, preserving aspect ratio."""
    canvas = Image.new("RGB", (pane_w, pane_h), bg)
    img = img.convert("RGB")
    scale = min(pane_w / max(img.width, 1), pane_h / max(img.height, 1))
    nw = max(1, int(round(img.width * scale)))
    nh = max(1, int(round(img.height * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((pane_w - nw) // 2, (pane_h - nh) // 2))
    return canvas


def _repaired_image(episode_dir: Path, rec: dict) -> Image.Image | None:
    """Load step_NNN_regen.png when the repair actually produced pixels."""
    names: list[str] = []
    step = rec.get("step")
    try:
        step_i = int(step)
    except (TypeError, ValueError):
        step_i = None
    if step_i is not None and step_i >= 0:
        meta_path = episode_dir / f"step_{step_i:03d}_regen.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
            if meta.get("status") not in (None, "ok"):
                if not meta.get("image_name"):
                    return None
            if meta.get("image_name"):
                names.append(meta["image_name"])
        names.append(f"step_{step_i:03d}_regen.png")
    if rec.get("regenerate_frame"):
        names.append(rec["regenerate_frame"])
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        path = episode_dir / name
        if path.is_file():
            return Image.open(path).convert("RGB")
    return None


def _placeholder_map(pane_w: int, pane_h: int, cfg: EpisodeVideoConfig) -> Image.Image:
    img = Image.new("RGB", (pane_w, pane_h), cfg.background)
    draw = ImageDraw.Draw(img)
    font = _font(max(cfg.overlay_min_font, pane_h // 24))
    msg = "no map"
    box = draw.textbbox((0, 0), msg, font=font)
    x = (pane_w - (box[2] - box[0])) // 2
    y = (pane_h - (box[3] - box[1])) // 2
    draw.text((x, y), msg, fill=(90, 100, 112), font=font)
    return img


def _compose_pair(
    rgb: Image.Image,
    map_img: Image.Image | None,
    overlay_text: str | None,
    pane_w: int,
    pane_h: int,
    cfg: EpisodeVideoConfig,
    right_caption: str | None = None,
) -> Image.Image:
    left = _fit(rgb, pane_w, pane_h, cfg.background)
    if overlay_text:
        left = _draw_overlay(left, overlay_text, cfg)
    right = _fit(map_img, pane_w, pane_h, cfg.background) if map_img is not None \
        else _placeholder_map(pane_w, pane_h, cfg)
    if right_caption:
        right = _draw_overlay(right, right_caption, cfg)
    gap = max(0, int(cfg.gap_px))
    canvas = Image.new("RGB", (pane_w * 2 + gap, pane_h), cfg.background)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (pane_w + gap, 0))
    # H.264 wants even dimensions.
    w, h = canvas.size
    if w % 2 or h % 2:
        padded = Image.new("RGB", (w + w % 2, h + h % 2), cfg.background)
        padded.paste(canvas, (0, 0))
        canvas = padded
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    size = max(8, int(size))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Courier New.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _sans_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    size = max(8, int(size))
    paths = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        )
        if bold else
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        )
    )
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    if bold:
        # Helvetica.ttc face 1 is often bold on macOS.
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=size, index=1)
        except OSError:
            pass
    return _font(size)


def _text_w(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap_text(text: str, font: ImageFont.ImageFont, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = (text or "").split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = cur + " " + word
        if _text_w(draw, trial, font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def _measure_card(
    draw: ImageDraw.ImageDraw,
    art: dict,
    inner_w: int,
    body_font: ImageFont.ImageFont,
    bold_font: ImageFont.ImageFont,
    body_size: int,
) -> tuple[int, list[str], int]:
    line_h = body_size + 4
    prefix = f"step {art.get('step')}  ·  "
    mid = str(art.get("severity") or "?")
    region = str(art.get("image_region") or "")
    header_w = (
        _text_w(draw, prefix, body_font)
        + _text_w(draw, mid, bold_font)
        + _text_w(draw, f"  ·  {region}", body_font)
    )
    header_h = line_h * (2 if header_w > inner_w and region else 1)
    desc_lines = _wrap_text(str(art.get("description") or ""), body_font, inner_w, draw)
    desc_h = line_h * max(len(desc_lines), 1)
    return header_h, desc_lines, desc_h


def _layout_summary_fits(
    canvas_h: int,
    margin: int,
    title_size: int,
    body_size: int,
    pad: int,
    gap: int,
    card_w: int,
    artifacts: list[dict],
    cfg: EpisodeVideoConfig,
) -> bool:
    dummy = Image.new("RGB", (8, 8), cfg.summary_bg)
    draw = ImageDraw.Draw(dummy)
    body_font = _sans_font(body_size, bold=False)
    bold_font = _sans_font(body_size, bold=True)
    y = margin + title_size + 8 + gap + 4
    inner_w = max(40, card_w - 2 * pad)
    if not artifacts:
        return y + body_size + margin < canvas_h
    for art in artifacts:
        header_h, _, desc_h = _measure_card(draw, art, inner_w, body_font, bold_font, body_size)
        y += pad + header_h + 6 + desc_h + pad + gap
        if y > canvas_h - margin:
            return False
    return True


def _draw_card_contents(
    draw: ImageDraw.ImageDraw,
    art: dict,
    x: int,
    y: int,
    inner_w: int,
    body_font: ImageFont.ImageFont,
    bold_font: ImageFont.ImageFont,
    body_size: int,
    cfg: EpisodeVideoConfig,
    desc_lines: list[str],
) -> None:
    line_h = body_size + 4
    fill = cfg.summary_text
    prefix = f"step {art.get('step')}  ·  "
    severity = str(art.get("severity") or "?")
    region = str(art.get("image_region") or "")
    draw.text((x, y), prefix, fill=fill, font=body_font)
    sx = x + _text_w(draw, prefix, body_font)
    draw.text((sx, y), severity, fill=fill, font=bold_font)
    sx += _text_w(draw, severity, bold_font)
    tail = f"  ·  {region}"
    if sx + _text_w(draw, tail, body_font) <= x + inner_w:
        draw.text((sx, y), tail, fill=fill, font=body_font)
        y += line_h
    else:
        y += line_h
        for line in _wrap_text(region, body_font, inner_w, draw):
            draw.text((x, y), line, fill=fill, font=body_font)
            y += line_h
    y += 4
    for line in desc_lines:
        draw.text((x, y), line, fill=fill, font=body_font)
        y += line_h


def _draw_overlay(pane: Image.Image, text: str, cfg: EpisodeVideoConfig) -> Image.Image:
    """Semi-transparent caption, bottom-right, wrapping long lines."""
    font_size = min(
        cfg.overlay_max_font,
        max(cfg.overlay_min_font, pane.height // max(cfg.overlay_font_div, 1)),
    )
    font = _font(font_size)
    max_w = int(pane.width * cfg.overlay_max_width_frac)
    avg = max(font_size * 0.55, 6)
    wrap = max(12, int(max_w / avg))
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, width=wrap) or [""])

    tmp = Image.new("RGB", (1, 1))
    measure = ImageDraw.Draw(tmp)
    line_h = font_size + 3
    text_w = 0
    for line in lines:
        box = measure.textbbox((0, 0), line, font=font)
        text_w = max(text_w, box[2] - box[0])
    text_h = line_h * len(lines)
    pad, margin = cfg.overlay_pad, cfg.overlay_margin
    box_w = min(max_w, text_w) + 2 * pad
    box_h = text_h + 2 * pad
    x1 = pane.width - margin
    y1 = pane.height - margin
    x0 = max(margin, x1 - box_w)
    y0 = max(margin, y1 - box_h)

    rgba = pane.convert("RGBA")
    layer = Image.new("RGBA", pane.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=cfg.overlay_box)
    ty = y0 + pad
    for line in lines:
        draw.text((x0 + pad, ty), line, fill=cfg.overlay_fill + (255,), font=font)
        ty += line_h
    return Image.alpha_composite(rgba, layer).convert("RGB")


def _existing_output(
    episode_dir: Path, cfg: EpisodeVideoConfig,
) -> tuple[Path, str] | None:
    mp4 = episode_dir / cfg.output_mp4
    if mp4.is_file() and mp4.stat().st_size > 0:
        return mp4, "video/mp4"
    webp = episode_dir / cfg.output_webp
    if webp.is_file() and webp.stat().st_size > 0:
        converted = _webp_to_mp4(webp, mp4, cfg)
        if converted is not None:
            return converted, "video/mp4"
        if shutil.which("ffmpeg"):
            return None  # restitch once as H.264 rather than keep serving WebP
        return webp, "image/webp"
    return None


def _cache_valid(episode_dir: Path, cfg: EpisodeVideoConfig, n_steps: int) -> bool:
    meta_path = episode_dir / cfg.meta_name
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if int(meta.get("steps", n_steps)) != n_steps:
        return False
    if abs(float(meta.get("seconds_per_step", cfg.seconds_per_step)) - cfg.seconds_per_step) > 1e-6:
        return False
    if not meta.get("summary_slide"):
        return False
    if abs(float(meta.get("summary_seconds", 0)) - cfg.summary_seconds) > 1e-6:
        return False
    if int(meta.get("layout_version", 0)) != LAYOUT_VERSION:
        return False
    return True


def _write_meta(episode_dir: Path, cfg: EpisodeVideoConfig, n_steps: int, format: str) -> None:
    payload = {
        "steps": n_steps,
        "seconds_per_step": cfg.seconds_per_step,
        "summary_seconds": cfg.summary_seconds,
        "summary_slide": True,
        "layout_version": LAYOUT_VERSION,
        "format": format,
    }
    tmp = episode_dir / (cfg.meta_name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(episode_dir / cfg.meta_name)


def _encode(
    frames: list[Image.Image], episode_dir: Path, cfg: EpisodeVideoConfig,
) -> tuple[Path, str]:
    if shutil.which("ffmpeg"):
        dest = episode_dir / cfg.output_mp4
        try:
            _encode_mp4(frames, dest, cfg)
            return dest, "video/mp4"
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("ffmpeg encode failed (%s); falling back to WebP", exc)
    dest = episode_dir / cfg.output_webp
    _encode_webp(frames, dest, cfg)
    converted = _webp_to_mp4(dest, episode_dir / cfg.output_mp4, cfg)
    if converted is not None:
        return converted, "video/mp4"
    return dest, "image/webp"


def _webp_to_mp4(src: Path, dest: Path, cfg: EpisodeVideoConfig) -> Path | None:
    """Remux an already-stitched WebP into H.264 MP4 (no frame restitch)."""
    if not shutil.which("ffmpeg"):
        return None
    tmp_out = dest.with_name(dest.stem + ".encoding.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=cfg.ffmpeg_timeout_s)
        tmp_out.replace(dest)
        return dest
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("webp→mp4 convert failed (%s)", exc)
        if tmp_out.is_file():
            tmp_out.unlink()
        return None


def _encode_mp4(frames: list[Image.Image], dest: Path, cfg: EpisodeVideoConfig) -> None:
    fps = 1.0 / max(cfg.seconds_per_step, 0.05)
    # ffmpeg picks the muxer from the suffix; ".mp4.tmp" is not a valid format.
    tmp_out = dest.with_name(dest.stem + ".encoding.mp4")
    with tempfile.TemporaryDirectory(prefix="episode-video-") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            frame.save(tmp_dir / f"f{i:04d}.png")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", f"{fps:g}",
            "-i", str(tmp_dir / "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(tmp_out),
        ]
        subprocess.run(cmd, check=True, timeout=cfg.ffmpeg_timeout_s)
    tmp_out.replace(dest)


def _encode_webp(frames: list[Image.Image], dest: Path, cfg: EpisodeVideoConfig) -> None:
    duration_ms = max(50, int(round(cfg.seconds_per_step * 1000)))
    tmp_out = dest.with_name(dest.stem + ".encoding.webp")
    frames[0].save(
        tmp_out,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        method=4,
        quality=82,
    )
    tmp_out.replace(dest)
