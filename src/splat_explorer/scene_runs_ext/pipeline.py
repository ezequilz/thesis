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
from .bundle import camera_bundle, transforms, repair_camera
from .fitting import BACKGROUND, fit_views


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
           propagator=propagate, fitter=fit_views, selected_views=None):
    """Return a fully fitted clone; never mutate the incumbent on failure/stop."""
    options = validate_options(options)
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
    cameras, segments, validation_cameras = [], [], []
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
        check()
        _, _, depth = renderer.render(view)
        start = len(cameras)
        cameras.extend(camera_bundle(view, depth, frames=options["frames"],
                                     span_fraction=options["span_fraction"]))
        segments.append({"start": start, "count": options["frames"], "step": seed.get("step")})
        held_out = camera_bundle(view, depth, frames=9,
                                 span_fraction=2 * options["span_fraction"])[2]
        for diagnostic_view in (view, held_out):
            check()
            rgb, _, _ = renderer.render(diagnostic_view)
            Image.fromarray(rgb).save(validation_dir / f"before-{len(validation_cameras):03d}.png")
            validation_cameras.append(diagnostic_view)
        if seed.get("reference_path"):
            with Image.open(seed["reference_path"]) as image:
                if image.width * camera.height != image.height * camera.width:
                    raise ValueError("Selected edited reference aspect ratio changed")
                path = f"references/{len(references):05d}.png"
                image.convert("RGB").resize((camera.width, camera.height), Image.Resampling.LANCZOS).save(root / path)
            references.append({"path": path, "frame_index": start,
                               "kind": "edited_render", "step": seed.get("step")})
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
                "validation_transforms": transforms(validation_cameras),
                "native_reference_resolution": native_resolution,
                "exploration_resolution": exploration_resolution,
                "proposal": proposal, "parent_scene_sha256": digest.hexdigest(),
                "reference_kind": "edited_render", "camera_convention": "OpenCV c2w",
                "upstream_adapter_revision": UPSTREAM_REVISION}
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2))
    render_seconds = time.monotonic()-started
    # Establish the edited observations in 3D before propagating nearby views.
    # Otherwise the opaque corrupted splat dominates ArtiFixer's RGB condition.
    candidate = scene.copy()
    anchor_cameras, anchor_targets = [], []
    for reference in references:
        anchor_cameras.append(cameras[reference["frame_index"]])
        with Image.open(root / reference["path"]) as image:
            anchor_targets.append(np.array(image.convert("RGB")))
    on_progress({"phase": "edited_view_fit"})
    anchor_metrics = fitter(candidate, anchor_cameras, anchor_targets,
        iterations=options["fit_iterations"] * len(anchor_cameras),
        should_stop=should_stop,
        on_progress=lambda p: on_progress({**p, "phase": "edited_view_fit"}))
    check()
    _release_cuda()
    # Preserve the original renders as diagnostics; propagation sees the scene
    # initialized from the intended corrected views, with its actual opacity.
    inputs.rename(root / "inputs-original")
    inputs.mkdir()
    renderer = renderer_factory(candidate)
    alphas = []
    for i, view in enumerate(cameras):
        check()
        rgb, alpha, _ = renderer.render(view)
        Image.fromarray(rgb).save(inputs / f"{i:05d}.png")
        alphas.append(alpha)
    del renderer
    _release_cuda()
    (root / "opacity.npy").rename(root / "opacity-original.npy")
    np.save(root / "opacity.npy", np.stack(alphas).astype(np.float32))
    manifest["conditioning_scene"] = "jointly fitted to edited reference views"
    manifest["anchor_role"] = "initialization_and_direct_reconstruction_target"
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2))
    initialization_seconds = time.monotonic()-started-render_seconds
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
    propagate_seconds = time.monotonic()-started-render_seconds-initialization_seconds
    # At known edited cameras the edited observation is authoritative. Replace
    # the generated target (including the loop's identical closing pose), rather
    # than training on two contradictory images at the same camera.
    edited_indices = []
    for reference, target in zip(references, anchor_targets):
        start = reference["frame_index"]
        segment = next(segment for segment in segments if segment["start"] == start)
        for i in (start, start + segment["count"] - 1):
            targets[i] = target
            Image.fromarray(target).save(preview / f"{i:05d}.png")
            edited_indices.append(i)
    on_progress({"phase": "multiview_fit"})
    metrics = fitter(candidate, cameras, targets,
                     iterations=options["fit_iterations"] * len(segments),
                     should_stop=should_stop, on_progress=on_progress)
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
                   baseline="edited-view-initialized-artifixer-v3",
                   anchor_role="initialization_and_direct_reconstruction_target",
                   edited_view_fit=anchor_metrics, initialization_seconds=initialization_seconds,
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
