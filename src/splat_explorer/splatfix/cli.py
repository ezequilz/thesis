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
    if getattr(args, "horizontal_fov", None) is not None:
        renderer_cfg["fov_deg"] = args.horizontal_fov
    from .resolution import profile_size
    renderer_cfg["width"], renderer_cfg["height"] = profile_size(
        cfg.get("splatfix", {}).get("resolution_profile", "720p"))
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
        metadata={"repair_policy": getattr(args, "repair_policy", "all"), "scene_load": {"min_opacity": cfg.scene.min_opacity,
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
    repair_images(checkpoint, cfg=cfg, image_set=args.image_set)
    return checkpoint.root


def capture_seva_views(cfg, args):
    from .oraclegs.captures import capture_checkpoint
    return capture_checkpoint(args.checkpoint, view_id=args.view_id,
        url=cfg.get('renderer', {}).get('viser_url'),
        on_progress=lambda **event: print(event['message'], flush=True)).root


def inspect_checkpoint(cfg, args):
    from .checkpoint import Checkpoint
    checkpoint = Checkpoint.load(args.checkpoint)
    print(json.dumps(checkpoint.manifest, indent=2))


def repair_splat(cfg, args):
    from .methods import get_method
    run_repair = get_method(getattr(args, 'reconstruction_method', None)).run_repair
    method = getattr(args, "reconstruction_method", "artifixer")
    runtime = dict(cfg.get("splatfix", {}).get(method + "_runtime" if method in ("g4splat", "oraclegs") else "runtime", {}))
    runtime["resolution_profile"] = cfg.get("splatfix", {}).get("resolution_profile", "720p")
    if getattr(args, 'endpoint_reference_mode', None) is not None:
        runtime['endpoint_reference_mode'] = args.endpoint_reference_mode
    if getattr(args, 'image_trajectory', None) is not None:
        runtime['image_trajectory'] = args.image_trajectory
    runtime['split_mode'] = getattr(args, 'split_mode', 'double-split')
    if getattr(args, 'image_cache_insertion', None) is not None:
        runtime['image_cache_insertion'] = args.image_cache_insertion
    if method == 'artifixer':
        for key in ('reconstruction_images', 'reconstruction_initialization'):
            if getattr(args, key, None) is not None:
                runtime[key] = getattr(args, key)
    for key in ("repo", "python", "checkpoint", "model_id", "hf_home", "source_points3d"):
        value = getattr(args, "artifixer_" + key, None)
        if value is not None:
            runtime[key] = str(value)
    if method == "g4splat":
        for key in ("repo", "python"):
            value = getattr(args, "g4splat_" + key, None)
            if value is not None:
                runtime[key] = str(value)
    if method == 'oraclegs':
        for key in ('oracle_source_reference', 'oracle_capture_source', 'oracle_reference_mode', 'oracle_split_mode', 'seva_geometry_guidance', 'oracle_depth_source', 'oracle_initialization', 'oracle_attention_batching', 'oracle_worldexplorer_attention', 'seva_resolution', 'seva_fit', 'oracle_adapter', 'oracle_preset', 'oracle_preset_scale', 'oracle_target_frames'):
            value = getattr(args, key, None)
            if value is not None:
                runtime[key] = value
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
    select.add_argument("--repair-policy", choices=("all", "endpoints"), default="all")
    select.add_argument("--horizontal-fov", type=float)
    edit = stages.add_parser("edit", help="Create missing GPT-image repairs in an existing checkpoint")
    edit.add_argument("checkpoint", type=Path)
    edit.add_argument("--image-set", choices=("saved", "seva"), default="saved")
    edit.set_defaults(func=edit_views)
    capture = stages.add_parser("seva-capture", help="Capture reusable SEVA RGB and depth at saved poses")
    capture.add_argument("checkpoint", type=Path)
    capture.add_argument("--view-id")
    capture.set_defaults(func=capture_seva_views)
    repair = stages.add_parser("repair", help="Run the selected reconstruction method locally on a GPU from saved views")
    from .methods import METHODS, DEFAULT_METHOD
    repair.add_argument("--oracle-capture-source", choices=("seva", "saved"), default="seva")
    repair.add_argument("--seva-fit", choices=("crop", "pad"))
    repair.add_argument("--seva-resolution", choices=("576p", "720p"))
    repair.add_argument("--seva-geometry-guidance", choices=("off", "source_splat"))
    repair.add_argument("--oracle-depth-source", choices=("predicted", "source_splat"))
    repair.add_argument("--oracle-initialization", choices=("sfm", "source_splat", "point_cloud"))
    repair.add_argument("--oracle-preset", choices=("orbit", "spiral", "spiral_expanding", "lemniscate", "shuffle_orbit"))
    repair.add_argument("--oracle-source-reference", choices=("nearest", "fixed_initial"))
    repair.add_argument("--oracle-reference-mode", choices=("scene_memory", "all", "single"))
    repair.add_argument("--oracle-split-mode", choices=("single-split", "double-split"))
    repair.add_argument("--oracle-preset-scale", type=float)
    repair.add_argument("--oracle-adapter", choices=("trajectory", "saved_poses", "per_view_preset"))
    repair.add_argument("--oracle-worldexplorer-attention", choices=["weighted", "unweighted"], help="WorldExplorer VGGT attention (default: weighted)")
    repair.add_argument("--oracle-attention-batching", choices=["within_trajectory", "global"], help="VGGT target shuffle scope (default: within_trajectory)")
    repair.add_argument("--oracle-target-frames", type=int, help="Generated interior images per consecutive viewpoint pair (default: 240)")
    repair.add_argument("--g4splat-repo", type=Path)
    repair.add_argument("--g4splat-python", type=Path)
    repair.add_argument("--reconstruction-method", choices=tuple(METHODS), default=DEFAULT_METHOD)
    repair.add_argument("checkpoint", type=Path)
    repair.add_argument("--scene", type=Path, help="Relocated source asset on the GPU host (same content)")
    repair.add_argument("--output", required=True, type=Path, help="Parent directory for isolated reconstruction runs")
    repair.add_argument("--image-trajectory", choices=("checkpoint", "endpoints"),
                        help="Use checkpoint policy or endpoint-only forward/reverse inference on two or more saved poses")
    repair.add_argument("--endpoint-reference-mode", choices=("both", "initial"))
    repair.add_argument("--mode", choices=("edited", "baseline"), default="edited")
    repair.add_argument('--split-mode', choices=('single-split', 'double-split', 'no-split'), default='double-split',
                        help='Generate one continuous path (no-split), whole legs, or inward halves (default); normal runs retain all references')
    repair.add_argument('--reconstruction-images', choices=('all', 'distilled'), default=None,
                        help='Saved references and ArtiFixer targets only (default: distilled), or all images including WorldExplorer originals')
    repair.add_argument('--reconstruction-initialization', choices=('full_splat', 'point_cloud'), default=None,
                        help='Initialize fresh Gaussians from point positions/colors (default: point_cloud), or import the original PLY')
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
