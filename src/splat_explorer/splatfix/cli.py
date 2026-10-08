"""Manual entry points for the independent splatfix stages."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def select_views(cfg, args):
    from ..agent.camera_rig import CameraRig
    from ..cli import _build_navigation, _resolve_start
    from ..config import Config
    from .viser_capture import CaptureSession
    from ..rendering.birdseye import ExplorationMap
    from ..scene import load_scene
    from .checkpoint import Checkpoint
    from .view_loop import make_view_policy, run_view_finding

    if args.renderer != 'viser':
        raise ValueError('Splatfix view selection requires Viser captures')
    scene_path = Path(args.scene or cfg.scene.path).expanduser().resolve()
    renderer_cfg = Config(dict(cfg.renderer))
    renderer_cfg["backend"] = args.renderer
    from .resolution import profile_size
    renderer_cfg["width"], renderer_cfg["height"] = profile_size(
        cfg.get("splatfix", {}).get("resolution_profile", "training"))
    scene = load_scene(scene_path, min_opacity=cfg.scene.min_opacity,
                       lod_level=int(cfg.scene.get("lod_level", 0)))
    renderer = CaptureSession(scene_path, up_axis=cfg.camera.up_axis,
                              lod_level=int(cfg.scene.get('lod_level', 0)),
                              url=renderer_cfg.get('viser_url') or None)
    world, spawn = _build_navigation(cfg, scene)
    start = spawn.points[0].position if spawn else _resolve_start(cfg, scene)
    rig = CameraRig(start, up_axis=cfg.camera.up_axis,
                    yaw_deg=cfg.camera.start_yaw_deg)
    exploration_map = None
    if spawn is not None and spawn.base_image is not None:
        exploration_map = ExplorationMap(spawn.base_image, spawn.camera,
                                         renderer_cfg.fov_deg, rig.up)
    policy = make_view_policy(cfg.agent)
    checkpoint = Checkpoint.create(
        args.output, scene_path, target_views=args.views,
        metadata={"scene_load": {"min_opacity": cfg.scene.min_opacity,
                                  "lod_level": int(cfg.scene.get("lod_level", 0))},
                  "renderer": dict(renderer_cfg), "up_axis": cfg.camera.up_axis,
                  "model": cfg.agent.model, "vlm_backend": cfg.agent.vlm_backend},
    )
    print(f"Checkpoint: {checkpoint.root}", flush=True)
    run_view_finding(
        renderer, rig, policy, checkpoint,
        width=renderer_cfg.width, height=renderer_cfg.height,
        fov_deg=renderer_cfg.fov_deg, views=args.views,
        max_steps_per_view=args.max_steps, world=world, scene=scene,
        exploration_map=exploration_map,
        max_move=float(cfg.agent.max_move_distance),
        max_rotate=float(cfg.agent.max_rotate_degrees),
    )
    if not args.select_only:
        from .image_repair import repair_images
        repair_images(checkpoint, cfg=cfg)
    return checkpoint.root


def edit_views(cfg, args):
    from .checkpoint import Checkpoint
    from .image_repair import repair_images
    checkpoint = Checkpoint.load(args.checkpoint)
    repair_images(checkpoint, cfg=cfg)
    return checkpoint.root


def inspect_checkpoint(cfg, args):
    from .checkpoint import Checkpoint
    checkpoint = Checkpoint.load(args.checkpoint)
    print(json.dumps(checkpoint.manifest, indent=2))


def repair_splat(cfg, args):
    from .methods import get_method
    run_repair = get_method(getattr(args, 'reconstruction_method', None)).run_repair
    method = getattr(args, "reconstruction_method", "artifixer")
    runtime = dict(cfg.get("splatfix", {}).get("g4splat_runtime" if method == "g4splat" else "runtime", {}))
    runtime["resolution_profile"] = cfg.get("splatfix", {}).get("resolution_profile", "training")
    runtime['split_mode'] = getattr(args, 'split_mode', 'double-split')
    if getattr(args, 'image_cache_insertion', None) is not None:
        runtime['image_cache_insertion'] = args.image_cache_insertion
    for key in ("repo", "python", "checkpoint", "model_id", "hf_home", "source_points3d"):
        value = getattr(args, "artifixer_" + key, None)
        if value is not None:
            runtime[key] = str(value)
    if method == "g4splat":
        for key in ("repo", "python"):
            value = getattr(args, "g4splat_" + key, None)
            if value is not None:
                runtime[key] = str(value)
    if args.scene is not None:
        runtime["scene_path"] = str(args.scene.resolve())
    result = run_repair(
        args.checkpoint, args.output, mode=args.mode, runtime=runtime,
        frames=args.frames, span_fraction=args.span_fraction, seed=args.seed,
        camera_scale=args.camera_scale,
        on_progress=lambda event: print(json.dumps(event), flush=True),
    )
    print(json.dumps(result, indent=2, default=str))
    return result


def add_parser(subparsers):
    parser = subparsers.add_parser("splatfix", help="Select views, checkpoint GPT edits, and rerun reconstruction")
    return _add_stages(parser)


def _add_stages(parser):
    stages = parser.add_subparsers(dest="splatfix_stage", required=True)
    select = stages.add_parser("select", help="Find views and save GPT-image repairs")
    select.add_argument("--scene", type=Path, help="Override scene.path")
    select.add_argument("--output", type=Path, default=Path("outputs/splatfix"))
    select.add_argument("--views", type=positive_int, default=6)
    select.add_argument("--max-steps", type=positive_int, default=40, help="Maximum VLM actions per selected view")
    select.add_argument("--renderer", choices=("viser",), default="viser",
                        help="Viser capture is required; no CPU or CUDA RGB fallback")
    select.add_argument("--select-only", action="store_true", help="Save original views without GPT-image calls")
    select.set_defaults(func=select_views)
    edit = stages.add_parser("edit", help="Create missing GPT-image repairs in an existing checkpoint")
    edit.add_argument("checkpoint", type=Path)
    edit.set_defaults(func=edit_views)
    repair = stages.add_parser("repair", help="Run fresh ArtiFixer3D/3D+ locally on a GPU from saved views")
    from .methods import METHODS, DEFAULT_METHOD
    repair.add_argument("--g4splat-repo", type=Path)
    repair.add_argument("--g4splat-python", type=Path)
    repair.add_argument("--reconstruction-method", choices=tuple(METHODS), default=DEFAULT_METHOD)
    repair.add_argument("checkpoint", type=Path)
    repair.add_argument("--scene", type=Path, help="Relocated source asset on the GPU host (same content)")
    repair.add_argument("--output", required=True, type=Path, help="Parent directory for isolated reconstruction runs")
    repair.add_argument("--mode", choices=("edited", "baseline"), default="edited")
    repair.add_argument('--split-mode', choices=('single-split', 'double-split'), default='double-split',
                        help='Generate whole legs or inward from both reference endpoints (default)')
    repair.add_argument('--image-cache-insertion', action=argparse.BooleanOptionalAction, default=None,
                        help='Insert the clean endpoint latent into KV memory before generation (default: off)')
    repair.add_argument("--frames", type=positive_int, default=25,
                        help="Legacy compatibility only; authored orbit spacing determines frame count")
    repair.add_argument("--span-fraction", type=float, default=.04,
                        help="Legacy compatibility only; new paths interpolate saved cameras")
    repair.add_argument("--seed", type=int, default=42)
    repair.add_argument("--camera-scale", type=float, default=None,
                        help="Explicit manual camera multiplier; default measures scale with cached official MoGe alignment")
    for key in ("repo", "python", "checkpoint", "model_id", "hf_home", "source_points3d"):
        repair.add_argument("--artifixer-" + key.replace("_", "-"), dest="artifixer_" + key)
    repair.set_defaults(func=repair_splat)
    inspect = stages.add_parser("inspect", help="Print checkpoint metadata without network calls")
    inspect.add_argument("checkpoint", type=Path)
    inspect.set_defaults(func=inspect_checkpoint)
    return stages


def main():
    from ..config import load_config, load_dotenv
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="splatfix")
    parser.add_argument("--config", help="YAML overriding configs/default.yaml")
    _add_stages(parser)
    args = parser.parse_args()
    load_dotenv()
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
