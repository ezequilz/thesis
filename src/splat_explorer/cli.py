"""Command-line entrypoints.

  splat-explorer render-test   render a panorama of test views from the scene center
  splat-explorer explore       run an agent episode (scripted policy by default)
  splat-explorer viewer        serve the interactive viser debug viewer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_config
from .scene import load_scene

logger = logging.getLogger("splat_explorer")


def _resolve_start(cfg, scene):
    start = cfg.camera.start_position
    if start == "auto":
        position = scene.robust_centroid().astype(np.float64)
        position += _up(cfg) * float(cfg.camera.eye_height)
    else:
        position = np.asarray(start, dtype=np.float64)
    return position


def _up(cfg):
    from .rendering.base import up_vector

    return up_vector(cfg.camera.up_axis)


def cmd_render_test(cfg, args) -> None:
    """Render N yaw views + straight up/down from the scene center, plus a contact sheet."""
    from .agent.camera_rig import CameraRig
    from .rendering import make_renderer

    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    logger.info("Scene loaded: %d gaussians (after opacity filter)", scene.num_gaussians)
    mins, maxs = scene.robust_bounds()
    logger.info("Robust bounds: mins=%s maxs=%s", np.round(mins, 2), np.round(maxs, 2))

    renderer = make_renderer(scene, cfg.renderer)
    rig = CameraRig(_resolve_start(cfg, scene), up_axis=cfg.camera.up_axis,
                    yaw_deg=cfg.camera.start_yaw_deg)

    out_dir = Path(cfg.output.dir) / "test_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    num_yaw = args.num_views
    for i in range(num_yaw):
        rig.yaw_deg = 360.0 * i / num_yaw
        rig.pitch_deg = 0.0
        img = renderer.render(rig.camera(cfg.renderer.width, cfg.renderer.height, cfg.renderer.fov_deg))
        path = out_dir / f"yaw_{int(rig.yaw_deg):03d}.png"
        Image.fromarray(img).save(path)
        frames.append(img)
        logger.info("Rendered %s", path)
    for label, pitch in [("up", 89.0), ("down", -89.0)]:
        rig.pitch_deg = pitch
        img = renderer.render(rig.camera(cfg.renderer.width, cfg.renderer.height, cfg.renderer.fov_deg))
        Image.fromarray(img).save(out_dir / f"pitch_{label}.png")
        frames.append(img)

    cols = 4
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
    sheet_path = out_dir / "contact_sheet.png"
    Image.fromarray(sheet).save(sheet_path)
    logger.info("Contact sheet -> %s", sheet_path)


def cmd_explore(cfg, args) -> None:
    from .agent.camera_rig import CameraRig
    from .agent.loop import run_episode
    from .agent.vlm import make_policy
    from .rendering import make_renderer

    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    renderer = make_renderer(scene, cfg.renderer)
    rig = CameraRig(_resolve_start(cfg, scene), up_axis=cfg.camera.up_axis,
                    yaw_deg=cfg.camera.start_yaw_deg)
    policy = make_policy(cfg.agent)

    run_episode(
        renderer=renderer,
        rig=rig,
        policy=policy,
        output_dir=Path(cfg.output.dir),
        width=cfg.renderer.width,
        height=cfg.renderer.height,
        fov_deg=cfg.renderer.fov_deg,
        max_steps=cfg.agent.max_steps,
        max_move_distance=cfg.agent.max_move_distance,
        max_rotate_degrees=cfg.agent.max_rotate_degrees,
    )


def cmd_viewer(cfg, args) -> None:
    from .rendering.viser_viewer import serve_viewer

    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    serve_viewer(scene, host=cfg.viewer.host, port=cfg.viewer.port, max_splats=cfg.viewer.max_splats)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="splat-explorer")
    parser.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("render-test", help="Render test views from the scene center")
    p_test.add_argument("--num-views", type=int, default=8, help="Number of yaw views")
    p_test.set_defaults(func=cmd_render_test)

    p_explore = sub.add_parser("explore", help="Run an agent exploration episode")
    p_explore.set_defaults(func=cmd_explore)

    p_viewer = sub.add_parser("viewer", help="Serve the viser debug viewer")
    p_viewer.set_defaults(func=cmd_viewer)

    args = parser.parse_args()
    cfg = load_config(args.config)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
