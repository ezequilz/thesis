#!/usr/bin/env python3
"""One-shot smoke test for the gemini_web backend.

Renders a single view from the scene, sends it to Gemini through the
browser-session client, and prints the raw reply plus the parsed action.
Run this before a full episode to verify cookies work.

Usage (from the repo root, venv active):
    python scripts/test_gemini_web.py [--config configs/gemini_web.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.gemini_web import GeminiWebPolicy, parse_action, render_tool_catalog
from splat_explorer.cli import _resolve_start
from splat_explorer.config import load_config
from splat_explorer.rendering import make_renderer
from splat_explorer.scene import load_scene


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemini_web.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print("== Rendering one test view ==")
    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    renderer = make_renderer(scene, cfg.renderer)
    rig = CameraRig(_resolve_start(cfg, scene), up_axis=cfg.camera.up_axis,
                    yaw_deg=cfg.camera.start_yaw_deg)
    observation = renderer.render(
        rig.camera(cfg.renderer.width, cfg.renderer.height, cfg.renderer.fov_deg)
    )

    out_path = Path(cfg.output.dir) / "gemini_web_smoke.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(observation).save(out_path)
    print(f"View saved to {out_path}")

    print("\n== Tool catalog sent to the model ==")
    print(render_tool_catalog())

    print("\n== Connecting to Gemini (browser session) ==")
    policy = GeminiWebPolicy(
        cookie_file=cfg.agent.get("cookie_file", ""),
        chrome_profile=cfg.agent.get("chrome_profile", ""),
        auto_cookies=bool(cfg.agent.get("auto_cookies", False)),
    )

    prompt = policy._build_prompt(rig.state_description(), step=0)
    from splat_explorer.agent.gemini_web import _png_bytes
    text = policy._ask(prompt, _png_bytes(observation))

    print("\n== Raw Gemini reply ==")
    print(text or "(empty reply — retry once without reinitializing, or refresh cookies)")

    action = parse_action(text) if text else None
    print("\n== Parsed action ==")
    if action:
        print(f"{action.name} {action.args}")
        return 0
    print("FAILED to parse an action from the reply.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
