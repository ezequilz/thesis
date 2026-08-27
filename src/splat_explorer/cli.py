"""Command-line entrypoints.

  splat-explorer render-test   render a panorama of test views from the scene center
  splat-explorer explore       run an agent episode (scripted policy by default)
  splat-explorer viewer        serve the interactive viser debug viewer
  splat-explorer dashboard     serve the episode control/debug dashboard
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

    from .config import Config
    renderer_cfg = dict(cfg.renderer)
    if renderer_cfg.get("backend") == "viser":
        renderer_cfg["backend"] = "cpu_splats"
        logger.info("render-test is headless; using cpu_splats instead of visor capture")
    renderer = make_renderer(scene, Config(renderer_cfg))
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


def _build_navigation(cfg, scene):
    """Collision world + (for auto start) the bird's-eye spawn selection.

    Returns (world, spawn); spawn is None when start_position is explicit or
    the spawn search finds nothing usable (legacy centroid start applies).
    """
    from .navigation import CollisionWorld, prepare_spawn_selection

    nav_cfg = cfg.navigation
    world = CollisionWorld(
        scene,
        solid_opacity=float(nav_cfg.solid_opacity),
        clearance_radius=float(nav_cfg.clearance_radius),
        collision=nav_cfg.get("collision", "off"),
    )
    spawn = None
    if cfg.camera.start_position == "auto":
        try:
            spawn = prepare_spawn_selection(scene, cfg, world=world)
        except Exception:
            logger.exception("Spawn-point search failed; falling back to centroid start")
    return world, spawn


def cmd_explore(cfg, args) -> None:
    from .agent.camera_rig import CameraRig
    from .agent.loop import run_episode
    from .agent.vlm import make_policy
    from .rendering import make_renderer

    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    renderer = make_renderer(scene, cfg.renderer)
    policy = make_policy(cfg.agent)

    world, spawn = _build_navigation(cfg, scene)
    start = spawn.points[0].position if spawn else _resolve_start(cfg, scene)
    rig = CameraRig(start, up_axis=cfg.camera.up_axis,
                    yaw_deg=cfg.camera.start_yaw_deg)

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
        nav=world,
        spawn=spawn,
        send_depth=bool(cfg.agent.get("send_depth", False)),
        send_map=bool(cfg.agent.get("send_map", True)),
        send_coverage=bool(cfg.agent.get("send_coverage", False)),
        compute_depth=bool(cfg.agent.get("compute_depth", False)),
        compute_coverage=bool(cfg.agent.get("compute_coverage", False)),
        run_meta={"params": {
            "backend": cfg.agent.vlm_backend,
            "model": cfg.agent.model,
            "max_steps": cfg.agent.max_steps,
            "width": cfg.renderer.width,
            "height": cfg.renderer.height,
            "send_depth": bool(cfg.agent.get("send_depth", False)),
            "send_map": bool(cfg.agent.get("send_map", False)),
            "send_coverage": bool(cfg.agent.get("send_coverage", False)),
            "compute_depth": bool(cfg.agent.get("compute_depth", False)),
            "compute_coverage": bool(cfg.agent.get("compute_coverage", False)),
            "prompt": cfg.agent.get("prompt", ""),
            "collision": world.collision,
        }},
    )


def _stop_stale_viewers() -> None:
    """Terminate other local `splat-explorer viewer` processes before starting.

    Each viewer holds the fully decoded scene in RAM (GBs for big splats), so
    starting a new one — e.g. after switching scenes — replaces the old run
    instead of stacking up. Docker viewers are separate containers and are
    managed by compose, not here.
    """
    import os
    import signal
    import subprocess
    import time

    try:
        out = subprocess.run(
            ["pgrep", "-f", "splat-explorer viewer"],
            capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:  # no pgrep (e.g. slim container images)
        return

    def is_viewer(pid: int) -> bool:
        # pgrep -f also matches wrapper shells and sandbox helpers whose
        # command line merely quotes the viewer command; only kill actual
        # python processes running it.
        cmd = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not cmd:
            return False
        exe = Path(cmd.split()[0]).name.lower()
        return exe.startswith("python") or exe == "splat-explorer"

    stale = [int(p) for p in out.split() if int(p) != os.getpid() and is_viewer(int(p))]
    for pid in stale:
        logger.info("Stopping previous viewer (pid %d) to free RAM and its port", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if stale:
        time.sleep(1.0)  # let ports and memory be released before binding


def cmd_viewer(cfg, args) -> None:
    from .rendering.viser_viewer import serve_viewer

    _stop_stale_viewers()
    scene = load_scene(cfg.scene.path, min_opacity=cfg.scene.min_opacity)
    serve_viewer(
        scene,
        host=cfg.viewer.host,
        port=cfg.viewer.port,
        max_splats=cfg.viewer.max_splats,
        up_axis=cfg.camera.up_axis,
        render_port=int(cfg.viewer.get("render_port", 8081)),
        fov_deg=float(cfg.renderer.fov_deg),
    )


def cmd_dashboard(cfg, args) -> None:
    from .web.server import serve_dashboard

    serve_dashboard(cfg, host=cfg.dashboard.host, port=args.port or cfg.dashboard.port)


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

    p_dash = sub.add_parser("dashboard", help="Serve the episode control/debug dashboard")
    p_dash.add_argument("--port", type=int, default=None, help="Override dashboard.port")
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    cfg = load_config(args.config)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
