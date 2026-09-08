"""GSFix3D §3.3 photometric lift (CUDA / gsplat) — research default.

Ports ``scripts/gsfix3d/refine_gs.py`` onto ``GaussianScene``. No color
stamp. Subclass this (or add a sibling backend id) to try a new idea;
leave ``repair_gsfix.GsplatPhotometricRepair`` frozen as the A/B baseline.

Per repaired view (``iters=20``, paper default)::

    I_gs = rasterize(gaussians, camera)
    L = 0.8 ||I_fixed - I_gs||_1 + 0.2 (1 - SSIM)
    backward
    Adam step

Kerbl clone/split densify is not in this loop. The upstream
``scripts/gsfix3d/refine_gs.py`` does call ``gaussians.densify`` every 5
iters; that path lives only on ``GsplatGsfix3dVisPruneRepair``. PLY
opacities are 1-D, and ``.repeat(2, 1)`` then crashes the paper split.

Colors are optimized as SH DC coefficients (RGB2SH / SH2RGB), matching
the INRIA ``GaussianModel``, not as raw RGB. That is what stopped the
neon posterization: unconstrained RGB Adam + a final clip.

``apply_until`` repeats paper 20-iter chunks so the Stop button still
works for research. Each chunk is the paper inner loop. ``Replay all
views`` then runs ``kf_iters=50`` shuffled passes over the repaired
images (our stand-in for the paper's augmented captured dataset — we
do not have the original RGB-D capture).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("LRZ_CUDA_ARCH") or "8.0"

import numpy as np

from .rendering.base import Camera
from .repair_gsfix import (
    _image_to_tensor,
    _inv_sigmoid,
    _rasterize,
    _require_torch,
    _to_uint8,
    photometric_loss,
)
from .scene import GaussianScene
from .scene.ply_loader import SH_C0

logger = logging.getLogger(__name__)

BACKEND_ID = "gsfix-gsplat"
_LAMBDA_DSSIM = 0.2
_SPLIT_N = 2
_SPLIT_SCALE_DIV = 0.8 * _SPLIT_N  # Kerbl / GSFix3D densify_and_split
_BACKGROUND_BLACK = (0.0, 0.0, 0.0)
_BACKGROUND_WHITE = (1.0, 1.0, 1.0)


def rgb_to_sh(rgb):
    """INRIA RGB2SH: f_dc = (rgb - 0.5) / Y_00."""
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh):
    """INRIA SH2RGB DC term."""
    return 0.5 + sh * SH_C0


def _quat_to_rotmat(quats, torch):
    """(N, 4) wxyz → (N, 3, 3)."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rot = torch.empty(quats.shape[0], 3, 3, dtype=quats.dtype, device=quats.device)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _viewspace_grad_norm(means2d, n_gaussians, torch):
    if means2d is None:
        return None
    grad = getattr(means2d, "absgrad", None)
    if grad is None:
        grad = means2d.grad
    if grad is None:
        return None
    g = grad.detach()
    if g.ndim == 3:
        g = g[0]
    if g.shape[0] != n_gaussians:
        return None
    return g[..., :2].norm(dim=-1)


@dataclass
class GsplatGsfix3dRepair:
    """Paper §3.3 CUDA refine. Default ``gsfix-gsplat`` backend.

    Add fields / override ``apply`` in a subclass when testing a new
    method, and register it in ``repair.CUDA_REPAIR_METHODS``.
    Experimental extras (including Kerbl densify) live only on
    ``GsplatGsfix3dVisPruneRepair`` (``gsfix-gsplat-visprune``) — do not
    add hooks here. ``densify`` defaults off so an LRZ params dump cannot
    re-enable clone/split on this class.
    """

    iters: int = 20
    kf_iters: int = 50
    lambda_dssim: float = _LAMBDA_DSSIM
    densify: bool = False
    densify_every: int = 5
    densify_grad_thresh: float = 0.0002
    prune_opacity: float = 0.005
    split_scale: float = 0.1
    max_clone: int = 2048
    max_gaussians: int = 2_500_000
    lr_means: float = 1.6e-4
    lr_colors: float = 0.0025
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    near: float = 0.05
    packed: bool = False
    white_background: bool = False
    max_chunks: int = 1
    on_progress: Callable[[dict], None] | None = None

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        try:
            return self._apply(scene, camera, rendered_rgb, repaired_rgb)
        except RuntimeError as exc:
            if self.packed or "out of memory" not in str(exc).lower():
                raise
            logger.warning("CUDA OOM during GSFix3D refine; retrying with packed=True")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return replace(self, packed=True)._apply(scene, camera, rendered_rgb, repaired_rgb)

    def apply_until(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
        *,
        should_stop=None,
        deadline: float | None = None,
        on_checkpoint=None,
    ) -> dict[str, Any]:
        """Run ``max_chunks`` paper 20-iter passes (default 1 = paper §3.3).

        Set ``max_chunks=0`` to keep going until Stop / deadline (research).
        """
        import time

        last: dict[str, Any] | None = None
        total_iters = 0
        l1_before = None
        chunk = 0
        limit = int(self.max_chunks)
        while True:
            if should_stop is not None and should_stop():
                break
            if deadline is not None and time.time() >= deadline:
                break
            if limit > 0 and chunk >= limit:
                break
            last = self.apply(scene, camera, rendered_rgb, repaired_rgb)
            if l1_before is None:
                l1_before = last.get("l1_before")
            total_iters += int(last.get("n_iters") or 0)
            chunk += 1
            last = dict(last)
            last["n_iters"] = total_iters
            last["n_chunks"] = chunk
            last["n_stamped"] = 0
            last["l1_before"] = l1_before
            last["phase"] = "refine"
            if on_checkpoint is not None:
                on_checkpoint(last)
        if last is None:
            return {
                "backend": BACKEND_ID,
                "n_visible": scene.num_gaussians,
                "n_updated": 0,
                "n_stamped": 0,
                "n_spawned": 0,
                "n_gaussians": scene.num_gaussians,
                "n_iters": 0,
                "n_chunks": 0,
                "l1_before": 0.0,
                "l1_after": None,
            }
        return last

    def refine_extended_dataset(
        self,
        scene: GaussianScene,
        views: Sequence[tuple[Camera, np.ndarray]],
        *,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        """Paper second stage: ``kf_iters`` shuffled passes over repaired views."""
        if not views or int(self.kf_iters) <= 0:
            return {
                "backend": BACKEND_ID,
                "phase": "keyframes",
                "n_iters": 0,
                "n_stamped": 0,
                "n_gaussians": scene.num_gaussians,
            }
        inner = replace(self, iters=1, densify=False, on_progress=None)
        total = 0
        last: dict[str, Any] = {}
        rng = np.random.default_rng(0)
        for pass_i in range(int(self.kf_iters)):
            order = rng.permutation(len(views))
            for idx in order:
                camera, repaired = views[int(idx)]
                last = inner.apply(scene, camera, repaired, repaired)
                total += int(last.get("n_iters") or 0)
            if on_progress is not None:
                on_progress({
                    **last,
                    "phase": "keyframes",
                    "n_iters": total,
                    "n_stamped": 0,
                    "kf_pass": pass_i + 1,
                    "kf_iters": int(self.kf_iters),
                })
        last = dict(last)
        last["backend"] = BACKEND_ID
        last["phase"] = "keyframes"
        last["n_iters"] = total
        last["n_stamped"] = 0
        logger.info(
            "GSFix3D keyframe refine: %d passes × %d views, %d iters, %d gaussians",
            int(self.kf_iters), len(views), total, scene.num_gaussians,
        )
        return last

    def _apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        torch = _require_torch()
        import gsplat

        device = torch.device("cuda")
        h, w = int(camera.height), int(camera.width)
        if self.on_progress is not None:
            props = torch.cuda.get_device_properties(0)
            self.on_progress({
                "phase": "cuda_ready",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_mib": int(props.total_memory / (1024 * 1024)),
                "n_gaussians": int(scene.num_gaussians),
                "n_iters": 0,
                "n_stamped": 0,
            })
        target = _image_to_tensor(repaired_rgb, w, h, torch, device)
        rendered = _image_to_tensor(rendered_rgb, w, h, torch, device)
        l1_before = float(torch.abs(target - rendered).mean().item())
        n0 = scene.num_gaussians

        means = torch.from_numpy(np.asarray(scene.means, dtype=np.float32)).to(device)
        quats = torch.from_numpy(np.asarray(scene.quats, dtype=np.float32)).to(device)
        f_dc = rgb_to_sh(torch.from_numpy(np.asarray(scene.colors, dtype=np.float32)).to(device))
        log_scales = torch.log(torch.clamp(
            torch.from_numpy(np.asarray(scene.scales, dtype=np.float32)).to(device), 1e-8,
        ))
        logit_opacities = _inv_sigmoid(
            torch.from_numpy(np.asarray(scene.opacities, dtype=np.float32)).to(device), torch,
        )
        for t in (means, quats, f_dc, log_scales, logit_opacities):
            t.requires_grad_(True)

        viewmat = torch.from_numpy(np.asarray(camera.w2c, dtype=np.float32)).to(device).unsqueeze(0)
        K = torch.from_numpy(np.asarray(camera.intrinsics, dtype=np.float32)).to(device).unsqueeze(0)
        bg = _BACKGROUND_WHITE if self.white_background else _BACKGROUND_BLACK
        background = torch.tensor(bg, device=device)

        def make_opt():
            return torch.optim.Adam(
                [
                    {"params": [means], "lr": self.lr_means},
                    {"params": [f_dc], "lr": self.lr_colors},
                    {"params": [logit_opacities], "lr": self.lr_opacities},
                    {"params": [log_scales], "lr": self.lr_scales},
                    {"params": [quats], "lr": self.lr_quats},
                ],
                lr=0.0,
                eps=1e-15,
            )

        opt = make_opt()
        last_l1 = l1_before
        rgb = None
        n_visible = int(means.shape[0])

        for it in range(int(self.iters)):
            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
            colors = sh_to_rgb(f_dc)
            rgb, info = _rasterize(
                gsplat, means, quats_n, scales, opacities, colors,
                viewmat, K, w, h, background, packed=self.packed,
            )
            means2d = info.get("means2d") if isinstance(info, dict) else None
            if means2d is not None and means2d.requires_grad:
                means2d.retain_grad()
            loss, l1 = photometric_loss(rgb, target, torch, self.lambda_dssim)
            last_l1 = float(l1.item())
            loss.backward()

            vis_norm = _viewspace_grad_norm(means2d, means.shape[0], torch)
            if vis_norm is not None:
                radii = info.get("radii") if isinstance(info, dict) else None
                vis = vis_norm > 0
                if radii is not None:
                    r = radii.detach()
                    if r.ndim == 2:
                        r = r[0]
                    if r.shape[0] == vis.shape[0]:
                        vis = vis & (r > 0)
                n_visible = int(vis.sum()) if vis.any() else int(means.shape[0])

            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                quats.copy_(torch.nn.functional.normalize(quats, dim=-1))

            if self.on_progress is not None:
                self.on_progress({
                    "phase": "refine",
                    "iter": it + 1,
                    "n_iters": it + 1,
                    "n_updated": int(means.shape[0]),
                    "n_gaussians": int(means.shape[0]),
                    "n_spawned": 0,
                    "n_stamped": 0,
                    "n_visible": int(n_visible),
                    "l1_before": round(l1_before, 6),
                    "l1": round(last_l1, 6),
                })

        with torch.no_grad():
            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
            colors = sh_to_rgb(f_dc)
            rgb, _ = _rasterize(
                gsplat, means, quats_n, scales, opacities, colors,
                viewmat, K, w, h, background, packed=self.packed,
            )
            l1_after = float(torch.abs(rgb - target).mean().item())
            render_rgb = _to_uint8(rgb)

        scene.means = means.detach().float().cpu().numpy().astype(np.float32)
        scene.quats = quats_n.detach().float().cpu().numpy().astype(np.float32)
        scene.scales = scales.detach().float().cpu().numpy().astype(np.float32)
        scene.opacities = opacities.detach().float().cpu().numpy().astype(np.float32)
        scene.colors = torch.clamp(colors, 0.0, 1.0).detach().float().cpu().numpy().astype(np.float32)

        n1 = scene.num_gaussians
        logger.info(
            "GSFix3D refine: %d iters, L1 %.4f -> %.4f, %d -> %d gaussians",
            self.iters, l1_before, l1_after, n0, n1,
        )
        return {
            "backend": BACKEND_ID,
            "n_visible": int(n_visible),
            "n_updated": n1,
            "n_stamped": 0,
            "n_spawned": 0,
            "n_gaussians": n1,
            "n_iters": int(self.iters),
            "l1_before": round(l1_before, 6),
            "l1_after": round(l1_after, 6),
            "render_rgb": render_rgb,
        }


def instantiate_cuda_repair(params: dict | None = None, **overrides):
    """Build the CUDA lift named in ``params['method']`` (LRZ + local)."""
    from dataclasses import fields

    body = dict(params or {})
    body.update(overrides)
    method = str(body.get("method") or body.get("backend") or BACKEND_ID)
    if method in ("gsfix-gsplat-baseline", "baseline"):
        from .repair_gsfix import GsplatPhotometricRepair
        cls = GsplatPhotometricRepair
    elif method in ("gsfix-gsplat-visprune", "visprune"):
        from .repair_gsfix3d_visprune import GsplatGsfix3dVisPruneRepair
        cls = GsplatGsfix3dVisPruneRepair
    else:
        cls = GsplatGsfix3dRepair
    skip = {"on_progress"}
    allowed = {f.name for f in fields(cls) if f.name not in skip}
    kwargs = {k: body[k] for k in allowed if k in body}
    if cls is GsplatGsfix3dRepair:
        kwargs.pop("densify", None)
    return cls(**kwargs)
