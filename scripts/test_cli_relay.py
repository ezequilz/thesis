#!/usr/bin/env python3
"""One-shot smoke test for the cli_relay backend.

Renders a single view from the scene, sends it through the CliRelay endpoint,
and prints the raw reply plus the parsed action. Run this before a full
episode to verify the relay, API key, and model routing work.

Usage (from the repo root, venv active, CliRelay running):
    export CLIRELAY_API_KEY=sk-...
    python scripts/test_cli_relay.py [--config configs/cli_relay.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.cli_relay import (
    CliRelayPolicy,
    _png_data_url,
    parse_action,
    render_tool_catalog,
)
from splat_explorer.cli import _resolve_start
from splat_explorer.config import load_config
from splat_explorer.rendering import make_renderer
from splat_explorer.scene import load_scene


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cli_relay.yaml")
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

    out_path = Path(cfg.output.dir) / "cli_relay_smoke.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(observation).save(out_path)
    print(f"View saved to {out_path}")

    print("\n== Tool catalog sent to the model ==")
    print(render_tool_catalog())

    print("\n== Sending one request through CliRelay ==")
    policy = CliRelayPolicy(
        model=cfg.agent.model,
        base_url=cfg.agent.get("relay_base_url", ""),
        api_key=cfg.agent.get("relay_api_key", ""),
    )

    prompt = policy._build_prompt(rig.state_description(), step=0)
    text = policy._ask(prompt, _png_data_url(observation))

    print("\n== Raw model reply ==")
    print(text or "(empty reply — check the relay is up, the key is valid, "
                  "and the model ID is routed; see the relay request logs)")

    action = parse_action(text) if text else None
    print("\n== Parsed action ==")
    if action:
        print(f"{action.name} {action.args}")
        return 0
    print("FAILED to parse an action from the reply.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
