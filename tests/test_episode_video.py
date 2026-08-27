"""On-demand episode video: side-by-side RGB|map and artifact overlay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.rendering.episode_video import (
    EpisodeVideoConfig,
    artifacts_from_steps,
    build_video_frames,
    compose_artifact_summary_slide,
    compose_episode_frames,
    format_artifact_overlay,
    load_episode_steps,
    render_episode_video,
    summary_hold_count,
)


def _write_png(path: Path, color, size=(48, 36)) -> None:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _episode(tmp: Path) -> Path:
    d = tmp / "20260101_000000"
    d.mkdir()
    _write_png(d / "step_000.png", (180, 40, 40), (64, 48))
    _write_png(d / "step_000_map.png", (40, 180, 80), (40, 40))
    _write_png(d / "step_001.png", (40, 80, 200), (64, 48))
    _write_png(d / "step_001_map.png", (40, 180, 80), (40, 40))
    _write_png(d / "birdseye.png", (10, 10, 10), (80, 80))
    records = [
        {
            "step": -1,
            "action": {"name": "choose_start", "args": {"point": 0}},
            "frame": "birdseye.png",
        },
        {
            "step": 0,
            "action": {"name": "rotate", "args": {"yaw_degrees": 15}},
            "frame": "step_000.png",
            "map_frame": "step_000_map.png",
        },
        {
            "step": 1,
            "action": {
                "name": "report_artifact",
                "args": {
                    "description": "floater near the lamp.",
                    "image_region": "upper-right",
                    "severity": "high",
                },
            },
            "frame": "step_001.png",
            "map_frame": "step_001_map.png",
        },
    ]
    (d / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return d


def test_artifact_overlay_format():
    text = format_artifact_overlay({
        "description": "hole in the wall.",
        "image_region": "center",
        "severity": "medium",
    })
    assert text.startswith("report_artifact{")
    assert '"description": "hole in the wall."' in text
    assert '    "image_region": "center"' in text
    assert '    "severity": "medium"' in text
    assert text.endswith("}")


def test_skips_choose_start_and_composes_equal_panes(tmp_path: Path):
    d = _episode(tmp_path)
    cfg = EpisodeVideoConfig()
    steps = load_episode_steps(d, cfg)
    assert [s["step"] for s in steps] == [0, 1]
    frames = compose_episode_frames(d, steps, cfg)
    assert len(frames) == 2
    # Two equal panes (+ even padding / 2px gap).
    w, h = frames[0].size
    assert h in (48, 50)  # original RGB height, maybe +1 for even
    assert w >= 64 * 2
    # Left pane of step 0 is reddish; right pane is greenish.
    arr = np.asarray(frames[0])
    left = arr[:, :64]
    right = arr[:, -40:]
    assert left[:, :, 0].mean() > left[:, :, 2].mean()
    assert right[:, :, 1].mean() > right[:, :, 0].mean()


def test_artifact_overlay_only_on_report_steps(tmp_path: Path):
    d = _episode(tmp_path)
    cfg = EpisodeVideoConfig()
    steps = load_episode_steps(d, cfg)
    frames = compose_episode_frames(d, steps, cfg)
    a0 = np.asarray(frames[0])
    a1 = np.asarray(frames[1])
    left0, left1 = a0[:, :64], a1[:, :64]
    # Step 0 RGB is untouched; step 1 has a darker caption box in the corner.
    assert np.abs(left0.astype(int) - (180, 40, 40)).max() < 5
    corner = left1[-18:, -36:].astype(int)
    assert (corner.sum(axis=2) < 40 + 80 + 200 - 80).any()


def test_summary_slide_lists_artifacts(tmp_path: Path):
    d = _episode(tmp_path)
    cfg = EpisodeVideoConfig()
    steps = load_episode_steps(d, cfg)
    arts = artifacts_from_steps(steps)
    assert arts == [{
        "step": 1,
        "severity": "high",
        "image_region": "upper-right",
        "description": "floater near the lamp.",
    }]
    assert summary_hold_count(cfg) == 1  # 1.0s summary at 1.0s/frame

    slide = compose_artifact_summary_slide((960, 360), arts, cfg)
    arr = np.asarray(slide)
    # Dark dashboard background, gold card border, light title text.
    assert arr[8, 8].mean() < 40
    gold = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 130) & (arr[:, :, 2] < 130)
    assert gold.any()
    title = arr[16:50, 16:280]
    assert (title.mean(axis=2) > 150).any()

    frames = build_video_frames(d, steps, cfg)
    assert len(frames) == 2 + summary_hold_count(cfg)
    assert frames[-1].size == frames[0].size
    # Last frame is the dark summary, not a step composite.
    assert np.asarray(frames[-1])[8, 8].mean() < 40


def test_empty_summary_slide_still_renders():
    slide = compose_artifact_summary_slide((640, 240), [], EpisodeVideoConfig())
    arr = np.asarray(slide)
    assert arr.shape[0] == 240
    # Title (or empty-state copy) is brighter than the dark fill.
    title = arr[16:50, 16:280]
    assert (title.mean(axis=2) > 150).any()


def test_render_reuses_cached_file(tmp_path: Path):
    d = _episode(tmp_path)
    first = render_episode_video(d)
    assert first.path.is_file()
    assert first.path.stat().st_size > 0
    mtime = first.path.stat().st_mtime
    second = render_episode_video(d)
    assert second.cached is True
    assert second.path == first.path
    assert second.path.stat().st_mtime == mtime


def test_repaired_image_replaces_map_on_that_step_only(tmp_path: Path):
    d = _episode(tmp_path)
    _write_png(d / "step_001_regen.png", (220, 200, 20), (64, 48))
    (d / "step_001_regen.json").write_text(json.dumps({
        "step": 1, "status": "ok", "image_name": "step_001_regen.png",
    }))
    cfg = EpisodeVideoConfig()
    steps = load_episode_steps(d, cfg)
    frames = compose_episode_frames(d, steps, cfg)
    assert len(frames) == 2
    arr0 = np.asarray(frames[0])
    arr1 = np.asarray(frames[1])
    right0 = arr0[:, -40:]
    right1 = arr1[:, -40:]
    # Step 0 still uses the green map; step 1 uses the yellow repair.
    assert right0[:, :, 1].mean() > right0[:, :, 0].mean()
    assert right1[:, :, 0].mean() > right1[:, :, 2].mean()
    assert right1[:, :, 1].mean() > right1[:, :, 2].mean()


def test_disabled_regen_keeps_the_map(tmp_path: Path):
    d = _episode(tmp_path)
    (d / "step_001_regen.json").write_text(json.dumps({
        "step": 1, "status": "disabled", "error": "image regeneration setting off",
    }))
    cfg = EpisodeVideoConfig()
    steps = load_episode_steps(d, cfg)
    frames = compose_episode_frames(d, steps, cfg)
    right1 = np.asarray(frames[1])[:, -40:]
    assert right1[:, :, 1].mean() > right1[:, :, 0].mean()

