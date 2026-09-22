"""Disposable process using the official ArtiFixer inference API.

Run inside an ArtiFixer environment. No pretrained code/weights are downloaded
by this bridge. The parent provides RGB, actual alpha and calibrated cameras.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


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
    from model_eval.run_inference import build_parser, get_eval_pipe, process_item
    from model_eval.checkpoint_loading import load_transformer_checkpoint
    from model_training.data.utils import compute_camera_rays, load_encoded_prompt

    if args.preflight:
        opts = build_parser().parse_args([
            "--checkpoint_pt", str(args.checkpoint), "--model_id", args.model_id,
            "--save_dir", "/tmp/artifixer-probe", "--attention_backend", "native",
        ])
        with torch.inference_mode():
            pipe = get_eval_pipe(opts, torch.device("cuda:0"))
            load_transformer_checkpoint(pipe.transformer, opts)
        print("ARTIFIXER_READY", flush=True)
        return
    if args.request is None:
        parser.error("--request is required unless --preflight is used")
    root = args.request.resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    options = manifest["options"]
    torch.manual_seed(options["seed"])
    opts = build_parser().parse_args([
        "--checkpoint_pt", str(args.checkpoint), "--model_id", args.model_id,
        "--save_dir", str(root / "artifixer-output"),
        "--num_inference_steps", str(options["inference_steps"]),
        "--save_frame_outputs_only", "--max_neighbors_per_encode", "1",
        "--attention_backend", "native",
    ])
    cameras = manifest["transforms"]
    count = len(cameras["frames"])
    def rgb(path):
        return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2,0,1).float()/255
    # Edited reference stays distinct from the corrupted RGB/alpha conditions.
    references = manifest.get("references", [{"path": "anchor.png", "frame_index": 0}])
    reference = torch.stack([rgb(root / r["path"]) for r in references])
    neighbors = [r["frame_index"] for r in references]
    segments = manifest.get("segments", [{"start": 0, "count": count}])
    opacity = np.load(root / "opacity.npy", allow_pickle=False, mmap_mode="r")
    device = torch.device("cuda:0")
    with torch.inference_mode():
        pipe = get_eval_pipe(opts, device)
        load_transformer_checkpoint(pipe.transformer, opts)
        pipe.transformer.eval()
        for segment in segments:
            indices = list(range(segment["start"], segment["start"] + segment["count"]))
            renders = torch.stack([rgb(root / "inputs" / f"{i:05d}.png") for i in indices])
            item = {"scene_id": "bundle", "rgb_rendered": renders,
                    "rgb_neighbors": reference,
                    "opacity": torch.from_numpy(np.array(opacity[indices])),
                    "encoded_prompt": load_encoded_prompt([])[0],
                    "frame_indices": torch.tensor(indices),
                    "valid_frames_mask": torch.ones(len(indices), dtype=torch.bool)}
            item.update(compute_camera_rays(cameras, indices, neighbors,
                        scale=options["camera_scale"], image_shape=renders.shape[-2:], skip_vae_check=True))
            # Separate observed viewpoints need separate temporal sequences:
            # concatenating them makes a video cut look like physical motion.
            if hasattr(pipe, "clear_inference_caches"):
                pipe.clear_inference_caches()
            process_item(pipe, item, opts, root / "artifixer-output", 0, device,
                         pipe.vae.config.scale_factor_temporal)
            del item, renders
    (root / "inference.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "model_id": args.model_id,
        "frames": count, "text_conditioning": "disabled (official zero embedding)",
        "segments": len(segments), "reference_views": len(references),
        "reference": "image-edited anchor; synthetic, not a captured photograph",
    }, indent=2))


if __name__ == "__main__":
    main()
