"""Extended GPU repair transaction, isolated from the baseline GSFix3D path."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from dataclasses import replace
import numpy as np
from PIL import Image
from .config import RUNTIME_DEFAULTS, UPSTREAM_REVISION, validate_options
from .bundle import camera_bundle, continuous_camera_bundle, transforms, repair_camera
from .fitting import BACKGROUND, fit_views, initial_scale_ceiling
from .starter_inference import starter_reference


def validate_runtime(runtime):
    cfg = {**RUNTIME_DEFAULTS, **(runtime or {})}
    repo = Path(cfg["repo"])
    if not (repo / "model_eval/run_inference.py").is_file():
        raise RuntimeError(f"ArtiFixer is not installed at {repo}. See docs/scene-runs-ext.md; baseline remains available.")
    if not Path(cfg["checkpoint"]).is_file():
        raise RuntimeError(f"ArtiFixer checkpoint missing: {cfg['checkpoint']}")
    if shutil.which(cfg["python"]) is None:
        raise RuntimeError(f"ArtiFixer Python executable not found: {cfg['python']}")
    return cfg


class BundleRenderer:
    def __init__(self, scene):
        from ..rendering.gsplat_renderer import GsplatRenderer
        self.renderer = GsplatRenderer(scene, background=BACKGROUND)

    def render(self, camera):
        import gsplat
        r = self.renderer
        torch = r._torch
        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=r.means.device)
        with torch.no_grad():
            image, alpha, _ = gsplat.rasterization(
                means=r.means, quats=r.quats, scales=r.scales, opacities=r.opacities,
                colors=r.colors, viewmats=t(camera.w2c)[None], Ks=t(camera.intrinsics)[None],
                width=camera.width, height=camera.height, backgrounds=r.background[None],
                render_mode="RGB+ED", packed=False)
        rgb = image[0,:,:,:3].clamp(0,1).mul(255).byte().cpu().numpy()
        a = alpha[0,:,:,0].cpu().numpy().astype(np.float32)
        depth = image[0,:,:,3].cpu().numpy().astype(np.float32)
        depth[a < .15] = np.inf
        return rgb, a, depth


def _release_cuda():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def propagate(root, runtime, should_stop):
    cfg = validate_runtime(runtime)
    command = [cfg["python"], str(Path(__file__).with_name("artifixer_bridge.py")),
               "--request", str(root), "--repo", cfg["repo"],
               "--checkpoint", cfg["checkpoint"], "--model-id", cfg["model_id"]]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=cfg["hf_home"])
    # Keep the separately provisioned ArtiFixer environment ahead of baseline deps.
    env.pop("PYTHONPATH", None)
    with (root / "artifixer.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, cwd=cfg["repo"], start_new_session=True)
        try:
            while process.poll() is None:
                if should_stop():
                    raise InterruptedError("ArtiFixer stopped; candidate discarded")
                time.sleep(.25)
            if process.returncode:
                tail = (root / "artifixer.log").read_text(errors="replace")[-3000:]
                raise RuntimeError(f"ArtiFixer exited {process.returncode}: {tail}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    source = root / "artifixer-output/bundle/frames/batch_0000/pred"
    return source


def repair(scene, camera, anchor_path, request_dir, *, options, runtime, proposal,
           should_stop, on_progress, renderer_factory=BundleRenderer,
           propagator=propagate, fitter=fit_views, selected_views=None, scale_ceiling=None):
    """Return a fully fitted clone; never mutate the incumbent on failure/stop."""
    options = validate_options(options)
    scale_ceiling = initial_scale_ceiling(scene) if scale_ceiling is None else scale_ceiling
    exploration_resolution = [camera.width, camera.height]
    with Image.open(anchor_path) as image:
        native_resolution = list(image.size)
        camera = repair_camera(camera, image.size, max_pixels=options["max_repair_pixels"])
    root = Path(request_dir) / "extended"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def check():
        if should_stop():
            raise InterruptedError("Extended repair stopped; candidate discarded")
    check()
    on_progress({"phase": "bundle_render"})
    renderer = renderer_factory(scene)
    validation_cameras, views = [], []
    anchor_depth = None
    references = [{"path": "anchor.png", "frame_index": 0, "kind": "edited_render"}]
    seeds = [{"camera": camera, "step": None}] + list(selected_views or [])
    reference_dir = root / "references"
    reference_dir.mkdir(exist_ok=True)
    validation_dir = root / "validation"
    validation_dir.mkdir(exist_ok=True)
    for seed in seeds:
        view = seed["camera"]
        if view.width * camera.height != view.height * camera.width:
            raise ValueError("Selected views must share the repaired image aspect ratio")
        view = replace(view, width=camera.width, height=camera.height)
        if not np.allclose(view.intrinsics, camera.intrinsics):
            raise ValueError("Selected views must share the anchor camera intrinsics")
        check()
        _, _, depth = renderer.render(view)
        views.append(view)
        if anchor_depth is None:
            anchor_depth = depth
        held_out = camera_bundle(view, depth, frames=9,
                                 span_fraction=2 * options["span_fraction"])[2]
        for diagnostic_view in (view, held_out):
            check()
            rgb, _, _ = renderer.render(diagnostic_view)
            Image.fromarray(rgb).save(validation_dir / f"before-{len(validation_cameras):03d}.png")
            validation_cameras.append(diagnostic_view)
    cameras, waypoint_indices = continuous_camera_bundle(
        views, anchor_depth, frames=options["frames"], span_fraction=options["span_fraction"])
    segments = [{"start": 0, "count": len(cameras)}]
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    alphas = []
    for i, view in enumerate(cameras):
        check()
        rgb, alpha, _ = renderer.render(view)
        if float(np.mean(alpha)) < .01:
            raise ValueError("Local trajectory leaves visible scene support; choose another view or reduce span")
        Image.fromarray(rgb).save(inputs / f"{i:05d}.png")
        alphas.append(alpha)
    del renderer
    _release_cuda()
    np.save(root / "opacity.npy", np.stack(alphas).astype(np.float32))
    with Image.open(anchor_path) as image:
        if image.width * camera.height != image.height * camera.width:
            raise ValueError("Edited anchor aspect ratio changed; cannot assign the original calibrated camera")
        image.convert("RGB").resize((camera.width, camera.height), Image.Resampling.LANCZOS).save(root / "anchor.png")
    digest = hashlib.sha256()
    for value in (scene.means, scene.scales, scene.quats, scene.opacities, scene.colors):
        digest.update(np.ascontiguousarray(value).tobytes())
    manifest = {"protocol": 2, "transforms": transforms(cameras), "options": options,
                "segments": segments, "references": references,
                "selected_camera_indices": waypoint_indices,
                "conditioning_policy": "one edited starter; camera-only waypoints",
                "validation_transforms": transforms(validation_cameras),
                "native_reference_resolution": native_resolution,
                "exploration_resolution": exploration_resolution,
                "proposal": proposal, "parent_scene_sha256": digest.hexdigest(),
                "reference_kind": "edited_render", "camera_convention": "OpenCV c2w",
                "upstream_adapter_revision": UPSTREAM_REVISION}
    for segment in segments:
        starter_reference(manifest, segment)
    manifest["conditioning_mode"] = "gpt-starter-kv-v1"
    manifest["scene_rgb_conditioning"] = False
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2))
    render_seconds = time.monotonic()-started
    # GPT images seed inference directly. Do not deform sparse geometry first:
    # generate the complete supervision set, then reconstruct exactly once.
    candidate = scene.copy()
    anchor_targets = []
    for reference in references:
        with Image.open(root / reference["path"]) as image:
            anchor_targets.append(np.array(image.convert("RGB")))
    manifest["conditioning_scene"] = "diagnostic only; RGB and opacity not supplied to generation"
    manifest["anchor_role"] = "clean_temporal_starter_and_direct_reconstruction_target"
    manifest["fitting_stages"] = 1
    manifest["scale_ceiling"] = scale_ceiling
    manifest["target_sampling"] = "balanced-edited-generated"
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2))
    on_progress({"phase": "artifixer_propagate"})
    predicted = propagator(root, runtime, should_stop)
    check()
    targets = []
    preview = root / "targets"
    preview.mkdir(exist_ok=True)
    for i in range(len(cameras)):
        path = predicted / f"{i:05d}.png"
        if not path.is_file():
            raise RuntimeError(f"ArtiFixer output frame missing: {path.name}")
        with Image.open(path) as image:
            if image.size != (camera.width, camera.height):
                raise RuntimeError("ArtiFixer output dimensions do not match calibrated cameras")
            targets.append(np.asarray(image.convert("RGB")))
        shutil.copy2(path, preview / path.name)
    propagate_seconds = time.monotonic()-started-render_seconds
    # Only frame zero is an edited observation. Do not insert edits at later
    # waypoints or at loop closure; all later images are model predictions.
    edited_indices = []
    for reference, target in zip(references, anchor_targets):
        start = reference["frame_index"]
        targets[start] = target
        Image.fromarray(target).save(preview / f"{start:05d}.png")
        edited_indices.append(start)
    on_progress({"phase": "multiview_fit"})
    closure_l1 = float(np.mean(np.abs(targets[-1].astype(np.float32) - targets[0].astype(np.float32))) / 255)
    # The generated closing frame is a consistency diagnostic, not a second
    # conflicting training target at the exact edited camera.
    metrics = fitter(candidate, cameras[:-1], targets[:-1],
                     iterations=options["fit_iterations"] * len(seeds),
                     should_stop=should_stop, on_progress=on_progress,
                     scale_ceiling=scale_ceiling, edited_indices=edited_indices)
    check()
    on_progress({"phase": "native_validation"})
    renderer = renderer_factory(candidate)
    for i, view in enumerate(validation_cameras):
        check()
        rgb, _, _ = renderer.render(view)
        Image.fromarray(rgb).save(validation_dir / f"after-{i:03d}.png")
    del renderer
    _release_cuda()
    metrics.update(pipeline="extended", backend="artifixer-gsplat", proposal=proposal,
                   render_seconds=render_seconds, propagation_seconds=propagate_seconds,
                   total_seconds=time.monotonic()-started, generated_frames=len(cameras),
                   baseline="single-starter-single-fit-artifixer-v5",
                   conditioning_mode="gpt-starter-kv-v1", scene_rgb_conditioning=False,
                   anchor_role="clean_temporal_starter_and_direct_reconstruction_target",
                   fitting_stages=1, scale_ceiling=scale_ceiling,
                   loop_closure_excluded_from_fit=True,
                   generated_loop_closure_l1=closure_l1,
                   edited_target_indices=edited_indices,
                   selected_steps=[v["step"] for v in seeds[1:]],
                   repair_scope=proposal.get("repair_scope", "local"),
                   generation_segments=len(segments), reference_views=len(references),
                   validation_views=len(validation_cameras),
                   validation_kind="matched native anchor and held-out cameras; visual review, no score gate",
                   native_reference_resolution=native_resolution,
                   exploration_resolution=exploration_resolution,
                   fitting_resolution=[camera.width, camera.height])
    return candidate, metrics
