"""Disposable process using the official ArtiFixer inference API.

Run inside an ArtiFixer environment. No pretrained code/weights are downloaded
by this bridge. The parent provides RGB, actual alpha and calibrated cameras.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch


def load_eval_pipe(opts, device):
    """Build the transformer without a full randomly initialized CPU copy.

    The release checkpoint replaces every parameter strictly. Keep real buffers
    (including rotary frequencies), but materialize empty BF16 parameters on the
    GPU before upstream adds its camera and neighbor-conditioning layers.
    """
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import WanTransformer3DModel
    from model_eval.run_inference import get_eval_pipe
    from model_eval.checkpoint_loading import load_transformer_checkpoint

    original = WanTransformer3DModel.from_config

    def empty_transformer(*args, **kwargs):
        with init_empty_weights(include_buffers=False):
            model = original(*args, **kwargs)
        for name, parameter in list(model.named_parameters()):
            value = torch.empty(parameter.shape, dtype=torch.bfloat16, device=device)
            set_module_tensor_to_device(model, name, device, value=value, dtype=torch.bfloat16)
        return model.to(device=device, dtype=torch.bfloat16)

    print("ARTIFIXER_LOADING constructing transformer directly on GPU (BF16)", flush=True)
    with patch.object(WanTransformer3DModel, "from_config", side_effect=empty_transformer):
        pipe = get_eval_pipe(opts, device)
    print("ARTIFIXER_LOADING copying memory-mapped checkpoint", flush=True)
    load_transformer_checkpoint(pipe.transformer, opts)
    pipe.transformer.eval().requires_grad_(False)
    print("ARTIFIXER_LOADED", flush=True)
    return pipe

try:
    from .starter_inference import install_starter_inference, starter_reference, starter_inputs, generation_policy
    from .periodic_refresh import PeriodicRefresh
except ImportError:  # Executed directly by the GPU worker.
    from starter_inference import install_starter_inference, starter_reference, starter_inputs, generation_policy
    from periodic_refresh import PeriodicRefresh


def crop_camera_conditioning(compute, cameras, indices, neighbors, *, box, source_size, scale):
    """Compute original full-frame rays, then crop without recentering the lens.

    Upstream's camera class rejects negative principal points. Off-center crops
    can validly have these, so retain upstream distortion/pose conventions by
    cropping its original rays and transforming its normalized K matrices.
    """
    import copy
    original = copy.deepcopy(cameras)
    left, top, right, bottom = box
    width, height = right-left, bottom-top
    source_w, source_h = source_size
    for camera in [original, *original['frames']]:
        camera['cx'] = camera.get('cx', cameras['cx']) + left
        camera['cy'] = camera.get('cy', cameras['cy']) + top
        camera['w'], camera['h'] = source_w, source_h
    result = compute(original, indices, neighbors, scale=scale,
                     image_shape=(source_h, source_w), skip_vae_check=True)
    result['camera_rays'] = result['camera_rays'][:, top:bottom, left:right].contiguous()
    for name in ('Ks', 'neighbor_Ks'):
        k = result[name].clone()
        k[:, 0, 0] *= source_w/width
        k[:, 1, 1] *= source_h/height
        k[:, 0, 2] = ((k[:, 0, 2]+.5)*source_w-left)/width-.5
        k[:, 1, 2] = ((k[:, 1, 2]+.5)*source_h-top)/height-.5
        result[name] = k
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from model_eval.run_inference import build_parser, process_item
    from model_training.data.utils import compute_camera_rays, load_encoded_prompt

    if args.preflight:
        opts = build_parser().parse_args([
            "--checkpoint_pt", str(args.checkpoint), "--model_id", args.model_id,
            "--save_dir", "/tmp/artifixer-probe", "--attention_backend", "native",
        ])
        with torch.inference_mode():
            pipe = load_eval_pipe(opts, torch.device("cuda:0"))
        print("ARTIFIXER_READY", flush=True)
        return
    if args.request is None:
        parser.error("--request is required unless --preflight is used")
    root = args.request.resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    options = manifest["options"]
    policy = generation_policy(options)
    exact_starter = policy["exact_starter_preserved"]
    torch.manual_seed(options["seed"])
    opts = build_parser().parse_args([
        "--checkpoint_pt", str(args.checkpoint), "--model_id", args.model_id,
        "--save_dir", str(root / "artifixer-output"),
        "--num_inference_steps", str(options["inference_steps"]),
        "--save_frame_outputs_only", "--max_neighbors_per_encode", "1",
        "--attention_backend", "native", "--sink_size", "1",
        "--local_attn_size", "21", "--replace_if_exists",
    ])
    cameras = manifest["transforms"]
    count = len(cameras["frames"])
    def rgb(path):
        return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2,0,1).float()/255
    # Each sequence starts from its own calibrated GPT edit.
    references = manifest.get("references", [{"path": "anchor.png", "frame_index": 0}])
    segments = manifest.get("segments", [{"start": 0, "count": count}])
    starters = [starter_reference(manifest, segment) for segment in segments]
    refreshers = []
    generated_references = []
    for segment in segments:
        refresh = None
        if options.get('block_schedule') == 'periodic_starter':
            if options.get('source_conditioning') != 'rendered':
                raise ValueError('Periodic starter requires rendered source conditioning')
            indices = list(range(segment['start'], segment['start'] + segment['count']))
            refresh = PeriodicRefresh(root, indices)
            generated_references.extend(refresh.prepare())
        refreshers.append(refresh)
    # All planned repairs finish before model loading or any generation. Use
    # independent upstream neighbor encoding and exact calibrated RGB cameras.
    all_references = references + generated_references
    reference = torch.stack([rgb(root / r['path']) for r in all_references])
    neighbors = [r['frame_index'] for r in all_references]
    (root / 'prepared-references.json').write_text(json.dumps(all_references, indent=2))
    device = torch.device("cuda:0")
    with torch.inference_mode():
        pipe = load_eval_pipe(opts, device)
        source_conditioning = options.get('source_conditioning', 'none')
        if source_conditioning not in ('none', 'rendered'):
            raise ValueError('Unknown source conditioning mode')
        # Leave the upstream method untouched for the comparison. process_item
        # handles padding to complete blocks and trims padded output frames.
        if exact_starter:
            install_starter_inference(pipe, generated_cache=policy['generated_cache'])
        starter_diagnostics = []
        refresh_records = []
        for segment, starter, refresh in zip(segments, starters, refreshers):
            indices = list(range(segment["start"], segment["start"] + segment["count"]))
            if refresh:
                pipe.starter_refresh = refresh
                pipe.starter_rgb_count = len(indices)
            seed = rgb(root / starter["path"])
            renders, opacity = starter_inputs(seed, len(indices))
            if source_conditioning == 'rendered':
                renders = torch.stack([rgb(root / 'inputs' / f'{i:05d}.png') for i in indices])
                alpha = np.load(root / 'opacity.npy', mmap_mode='r', allow_pickle=False)
                opacity = torch.from_numpy(np.array(alpha[indices], dtype=np.float32))
                if opacity.shape != renders.shape[:1] + renders.shape[-2:]:
                    raise ValueError('Rendered opacity dimensions do not match RGB')
                if not torch.isfinite(opacity).all() or (opacity < 0).any() or (opacity > 1).any():
                    raise ValueError('Rendered opacity must be finite and in [0, 1]')
                renders[0], opacity[0] = seed, 1
            item = {"scene_id": "bundle", "rgb_rendered": renders,
                    "rgb_neighbors": reference,
                    "opacity": opacity,
                    "encoded_prompt": load_encoded_prompt([])[0],
                    "frame_indices": torch.tensor(indices),
                    "valid_frames_mask": torch.ones(len(indices), dtype=torch.bool)}
            region = manifest.get("local_region")
            # Upstream keeps neighbor cameras exact, independently of its
            # temporal averaging of target cameras.
            if region:
                item.update(crop_camera_conditioning(compute_camera_rays, cameras, indices, neighbors,
                            box=region["crop"], source_size=region["source_size"],
                            scale=options["camera_scale"]))
            else:
                item.update(compute_camera_rays(cameras, indices, neighbors,
                            scale=options["camera_scale"], image_shape=renders.shape[-2:], skip_vae_check=True))
            # Separate observed viewpoints need separate temporal sequences:
            # concatenating them makes a video cut look like physical motion.
            if hasattr(pipe, "clear_inference_caches"):
                pipe.clear_inference_caches()
            process_item(pipe, item, opts, root / "artifixer-output", 0, device,
                         pipe.vae.config.scale_factor_temporal)
            # Record frame zero before any export replacement. Only the exact
            # adapter removes VAE roundtrip loss; upstream output stays intact.
            import shutil
            first_output = root / "artifixer-output/bundle/frames/batch_0000/pred" / f"{indices[0]:05d}.png"
            prefix = 'starter-vae-roundtrip' if exact_starter else 'upstream-first-frame'
            roundtrip = root / f'{prefix}-{indices[0]:05d}.png'
            shutil.copy2(first_output, roundtrip)
            starter_diagnostics.append({'frame_index': indices[0],
                ('vae_roundtrip' if exact_starter else 'generated_first_frame'): roundtrip.name,
                ('vae_roundtrip_mae_0_1' if exact_starter else 'reference_mae_0_1'):
                    float((rgb(roundtrip) - seed).abs().mean())})
            if exact_starter:
                shutil.copy2(root / starter["path"], first_output)
            if refresh:
                for record in refresh.records:
                    destination = first_output.parent / f"{record['frame_index']:05d}.png"
                    shutil.copy2(destination, root / 'refresh' / f"{record['frame_index']:05d}" / 'vae-roundtrip.png')
                    shutil.copy2(root / record['path'], destination)
                refresh_records.extend(refresh.records)
            del item, renders
    (root / "inference.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "model_id": args.model_id,
        "frames": count, "text_conditioning": "disabled (official zero embedding)",
        "segments": len(segments), "reference_views": len(references) + len(generated_references),
        "initial_reference_views": len(references),
        "generated_references": generated_references,
        "reference": "image-edited anchor; synthetic, not a captured photograph",
        "reference_conditioning": ("all initial and planned repaired references available from generation start; independently encoded with exact RGB cameras"
                                   if options.get('block_schedule') == 'periodic_starter' else
                                   "fixed edited references supplied to every transformer call; separate from temporal KV cache"),
        "anchor_prefit": options.get('anchor_prefit', 'none'),
        **policy,
        "starter_frames": [s["start"] for s in segments],
        "scene_rgb_conditioning": source_conditioning == 'rendered',
        "source_conditioning": source_conditioning,
        "starter_diagnostics": starter_diagnostics,
        "periodic_refreshes": refresh_records,
        "starter_context": ("clean first latent cached at timestep zero before generation" if exact_starter
                            else "upstream blocks from frame zero; edited reference throughout; no exact latent preservation"),
    }, indent=2))


if __name__ == "__main__":
    main()
