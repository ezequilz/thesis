#!/usr/bin/env python3
"""One-shot smoke test: send an RGB frame through CliRelay gpt-image-2.

This is the same paid request the harness queues when image regeneration is
ticked on and the inspector sets regenerate=yes (~4¢). Uses /v1/images/edits.

Usage (from the repo root, venv active, CliRelay running):
    export CLIRELAY_API_KEY=sk-...
    python scripts/test_regenerate.py --image path.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splat_explorer.agent.cli_relay import CliRelayPolicy
from splat_explorer.agent.regenerate import (
    IMAGE_MODEL,
    REGENERATE_PROMPT,
    ask_regenerate,
    extract_images,
    save_png,
    summarize_payload,
)
from splat_explorer.config import load_config


def _render_one_view(cfg) -> Path:
    from PIL import Image

    from splat_explorer.agent.camera_rig import CameraRig
    from splat_explorer.cli import _resolve_start
    from splat_explorer.rendering import make_renderer
    from splat_explorer.scene import load_scene

    print("== Rendering one test view ==")
    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    renderer = make_renderer(scene, cfg.renderer)
    rig = CameraRig(
        _resolve_start(cfg, scene),
        up_axis=cfg.camera.up_axis,
        yaw_deg=cfg.camera.start_yaw_deg,
    )
    camera = rig.camera(cfg.renderer.width, cfg.renderer.height, cfg.renderer.fov_deg)
    observation = renderer.render(camera)
    out = Path(cfg.output.dir) / "cli_relay_regen_source.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observation).save(out)
    print(f"Source view saved to {out}")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cli_relay.yaml")
    parser.add_argument("--image", type=Path, default=None,
                        help="Existing RGB PNG to send (skips rendering).")
    args = parser.parse_args()
    cfg = load_config(args.config)

    image_path = args.image
    if image_path is None:
        image_path = _render_one_view(cfg)
    elif not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 1

    print(f"\n== Prompt ==\n{REGENERATE_PROMPT}")
    print(f"\n== Sending {image_path} through CliRelay ({IMAGE_MODEL} / images.edit) ==")
    policy = CliRelayPolicy(
        model=cfg.agent.model,
        base_url=cfg.agent.get("relay_base_url", ""),
        api_key=cfg.agent.get("relay_api_key", ""),
        prompt=cfg.agent.get("prompt", ""),
    )
    payload, error = ask_regenerate(policy.client, IMAGE_MODEL, image_path)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cli_relay_regen.json"
    summary_path.write_text(json.dumps({
        "error": error,
        "model": IMAGE_MODEL,
        "prompt": REGENERATE_PROMPT,
        "source": str(image_path),
        "n_images": 0 if error else None,
        "response": None if error else summarize_payload(payload),
    }, indent=2))

    if error:
        print(f"Request error: {error}")
        print(f"Wrote {summary_path}")
        return 1

    images = extract_images(payload)
    summary = json.loads(summary_path.read_text())
    summary["n_images"] = len(images)
    summary["response"] = summarize_payload(payload)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n== CliRelay payload summary written to {summary_path} ==")
    print(json.dumps(summarize_payload(payload), indent=2)[:4000])

    if not images:
        print(
            "\nNo image found in the response. The sidecar JSON is the next "
            "place to look — we may need a dedicated image-edit API next."
        )
        return 2

    for i, data in enumerate(images):
        name = "cli_relay_regen.png" if i == 0 else f"cli_relay_regen_{i + 1}.png"
        path = out_dir / name
        save_png(data, path)
        print(f"Saved image {i + 1}/{len(images)} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
