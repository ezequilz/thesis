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
import numpy as np
from PIL import Image
from .config import RUNTIME_DEFAULTS, UPSTREAM_REVISION, validate_options
from .bundle import camera_bundle, transforms
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
           propagator=propagate, fitter=fit_views):
    """Return a fully fitted clone; never mutate the incumbent on failure/stop."""
    options = validate_options(options)
    if camera.width % 16 or camera.height % 16:
        raise ValueError("Extended rendering dimensions must be multiples of 16")
    root = Path(request_dir) / "extended"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def check():
        if should_stop():
            raise InterruptedError("Extended repair stopped; candidate discarded")
    check()
    on_progress({"phase": "bundle_render"})
    renderer = renderer_factory(scene)
    _, _, depth = renderer.render(camera)
    cameras = camera_bundle(camera, depth, frames=options["frames"], span_fraction=options["span_fraction"])
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
    manifest = {"protocol": 1, "transforms": transforms(cameras), "options": options,
                "proposal": proposal, "parent_scene_sha256": digest.hexdigest(),
                "reference_kind": "edited_render", "camera_convention": "OpenCV c2w",
                "upstream_adapter_revision": UPSTREAM_REVISION}
    (root / "bundle.json").write_text(json.dumps(manifest, indent=2))
    render_seconds = time.monotonic()-started
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
    # Anchor remains an explicit training view instead of relying on exact propagation.
    with Image.open(root / "anchor.png") as image:
        targets.append(np.asarray(image.convert("RGB")))
    candidate = scene.copy()
    on_progress({"phase": "multiview_fit"})
    metrics = fitter(candidate, cameras+[camera], targets,
                     iterations=options["fit_iterations"],
                     intervention=proposal.get("intervention", "structure"),
                     should_stop=should_stop, on_progress=on_progress)
    check()
    metrics.update(pipeline="extended", backend="artifixer-gsplat", proposal=proposal,
                   render_seconds=render_seconds, propagation_seconds=propagate_seconds,
                   total_seconds=time.monotonic()-started, generated_frames=len(cameras))
    return candidate, metrics
