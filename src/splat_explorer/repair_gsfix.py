"""Frozen CUDA photometric lift (pre-paper baseline).

This is the gsplat refine that produced neon / needle artifacts when a
single view ran for hundreds of iterations. It is kept as backend
``gsfix-gsplat-baseline`` for A/B against the paper method in
``repair_gsfix3d.py`` (the default ``gsfix-gsplat``).

Known deviations from GSFix3D ``scripts/gsfix3d/refine_gs.py``:

- RGB is Adam-optimized with no SH / sigmoid activation (can leave [0, 1]
  mid-loop; commit clamps — that clamp is the "neon" posterization).
- Densify only clones (no split of large Gaussians) and skips the Adam
  step on densify iterations.
- ``apply_until`` repeats 20-iter chunks until Stop (paper is 20 / view).
- No color stamp — that path was never on CUDA; the repair UI used to
  label ``n_updated`` as "stamped".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Callable

# NGC PyTorch sets TORCH_CUDA_ARCH_LIST to sm_52–sm_90. gsplat 1.5 uses
# cooperative_groups::labeled_partition, which fails on those old archs.
os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("LRZ_CUDA_ARCH") or "8.0"

import numpy as np

from .rendering.base import Camera
from .scene import GaussianScene

logger = logging.getLogger(__name__)

# Same mix as GSFix3D refine_gs.py / Kerbl et al.
_LAMBDA_DSSIM = 0.2
_SSIM_WINDOW = 11
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def gsplat_refine_available() -> bool:
    try:
        import torch
        import gsplat  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _require_torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GSFix refine needs an NVIDIA GPU with CUDA (torch.cuda.is_available() "
            "is false). Use docker/Dockerfile.gpu or pip install '.[gpu]' on a CUDA host."
        )
    return torch


def _inv_sigmoid(x, torch):
    x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    return torch.log(x / (1.0 - x))


def _create_window(window_size: int, channel: int, torch, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / 2.0)
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, torch, window_size: int = _SSIM_WINDOW) -> Any:
    """img1/img2: (3, H, W) in [0, 1]. Matches the 3DGS / GSFix3D SSIM term."""
    pad = window_size // 2
    channel = img1.shape[0]
    window = _create_window(window_size, channel, torch, img1.device)
    mu1 = torch.nn.functional.conv2d(img1.unsqueeze(0), window, padding=pad, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2.unsqueeze(0), window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=pad, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=pad, groups=channel) - mu12
    ssim_map = ((2 * mu12 + _C1) * (2 * sigma12 + _C2)) / (
        (mu1_sq + mu2_sq + _C1) * (sigma1_sq + sigma2_sq + _C2)
    )
    return ssim_map.mean()


def photometric_loss(pred, gt, torch, lambda_dssim: float = _LAMBDA_DSSIM):
    """pred/gt: (H, W, 3) in [0, 1]. Returns (loss, l1) as in refine_gs.py."""
    pred_n = pred.permute(2, 0, 1)
    gt_n = gt.permute(2, 0, 1)
    l1 = torch.abs(pred_n - gt_n).mean()
    loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim(pred_n, gt_n, torch))
    return loss, l1


def _to_uint8(rgb) -> np.ndarray:
    return np.clip(rgb.detach().float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)


@dataclass
class GsplatPhotometricRepair:
    """Frozen CUDA lift. Prefer ``GsplatGsfix3dRepair`` (``gsfix-gsplat``)."""

    iters: int = 20
    lambda_dssim: float = _LAMBDA_DSSIM
    densify: bool = True
    densify_every: int = 5
    densify_grad_thresh: float = 0.0002
    prune_opacity: float = 0.005
    max_clone: int = 2048
    max_gaussians: int = 2_500_000
    lr_means: float = 1.6e-4
    lr_colors: float = 0.0025
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    near: float = 0.05
    packed: bool = False
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
            if (
                self.packed
                or "out of memory" not in str(exc).lower()
            ):
                raise
            logger.warning("CUDA OOM during GSFix refine; retrying with packed=True")
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
        """Keep running 20-iter paper chunks until Stop or deadline."""
        import time

        last: dict[str, Any] | None = None
        total_iters = 0
        l1_before = None
        while True:
            if should_stop is not None and should_stop():
                break
            if deadline is not None and time.time() >= deadline:
                break
            last = self.apply(scene, camera, rendered_rgb, repaired_rgb)
            if l1_before is None:
                l1_before = last.get("l1_before")
            total_iters += int(last.get("n_iters") or 0)
            last = dict(last)
            last["n_iters"] = total_iters
            last["n_stamped"] = 0
            last["l1_before"] = l1_before
            if on_checkpoint is not None:
                on_checkpoint(last)
        if last is None:
            return {
                "backend": "gsfix-gsplat-baseline",
                "n_visible": scene.num_gaussians,
                "n_updated": 0,
                "n_stamped": 0,
                "n_spawned": 0,
                "n_gaussians": scene.num_gaussians,
                "n_iters": 0,
                "l1_before": 0.0,
                "l1_after": None,
            }
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
            })
        target = _image_to_tensor(repaired_rgb, w, h, torch, device)
        rendered = _image_to_tensor(rendered_rgb, w, h, torch, device)
        l1_before = float(torch.abs(target - rendered).mean().item())
        n0 = scene.num_gaussians

        means = torch.from_numpy(np.asarray(scene.means, dtype=np.float32)).to(device)
        quats = torch.from_numpy(np.asarray(scene.quats, dtype=np.float32)).to(device)
        colors = torch.from_numpy(np.asarray(scene.colors, dtype=np.float32)).to(device)
        log_scales = torch.log(torch.clamp(
            torch.from_numpy(np.asarray(scene.scales, dtype=np.float32)).to(device), 1e-8,
        ))
        logit_opacities = _inv_sigmoid(
            torch.from_numpy(np.asarray(scene.opacities, dtype=np.float32)).to(device), torch,
        )
        for t in (means, quats, colors, log_scales, logit_opacities):
            t.requires_grad_(True)

        viewmat = torch.from_numpy(np.asarray(camera.w2c, dtype=np.float32)).to(device).unsqueeze(0)
        K = torch.from_numpy(np.asarray(camera.intrinsics, dtype=np.float32)).to(device).unsqueeze(0)
        background = torch.tensor([0.12, 0.12, 0.13], device=device)
        n_spawned = 0

        def make_opt():
            return torch.optim.Adam(
                [
                    {"params": [means], "lr": self.lr_means},
                    {"params": [colors], "lr": self.lr_colors},
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

        for it in range(int(self.iters)):
            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
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

            if (
                self.densify
                and it > 0
                and it % int(self.densify_every) == 0
                and means.shape[0] < int(self.max_gaussians)
            ):
                packed = _maybe_densify(
                    torch, means, quats, colors, log_scales, logit_opacities,
                    means2d, thresh=self.densify_grad_thresh, max_clone=self.max_clone,
                )
                if packed is not None:
                    means, quats, colors, log_scales, logit_opacities, n_new = packed
                    n_spawned += n_new
                    opt = make_opt()
                    continue

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
                    "n_spawned": int(n_spawned),
                    "l1_before": round(l1_before, 6),
                    "l1": round(last_l1, 6),
                })

        if self.densify and means.shape[0] > 32:
            with torch.no_grad():
                keep = torch.sigmoid(logit_opacities) > float(self.prune_opacity)
                if int(keep.sum()) >= 32 and int((~keep).sum()) > 0:
                    means, quats, colors, log_scales, logit_opacities = (
                        means[keep], quats[keep], colors[keep],
                        log_scales[keep], logit_opacities[keep],
                    )

        with torch.no_grad():
            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
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
            "GSFix baseline refine: %d iters, L1 %.4f -> %.4f, %d -> %d gaussians (+%d)",
            self.iters, l1_before, l1_after, n0, n1, n_spawned,
        )
        return {
            "backend": "gsfix-gsplat-baseline",
            "n_visible": n1,
            "n_updated": n1,
            "n_stamped": 0,
            "n_spawned": int(max(0, n1 - n0) if n_spawned == 0 else n_spawned),
            "n_gaussians": n1,
            "n_iters": int(self.iters),
            "l1_before": round(l1_before, 6),
            "l1_after": round(l1_after, 6),
            "render_rgb": render_rgb,
        }


def _image_to_tensor(image: np.ndarray, width: int, height: int, torch, device):
    from PIL import Image

    arr = np.asarray(image)
    if arr.shape[0] != height or arr.shape[1] != width:
        arr = np.asarray(Image.fromarray(arr).resize((width, height), Image.Resampling.LANCZOS))
    return torch.from_numpy(arr.astype(np.float32) / 255.0).to(device)


def _rasterize(gsplat, means, quats, scales, opacities, colors, viewmat, K, width, height, background, packed=False):
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=int(width),
        height=int(height),
        packed=bool(packed),
    )
    extra = dict(backgrounds=background.unsqueeze(0), render_mode="RGB", absgrad=True)
    try:
        out = gsplat.rasterization(**kwargs, **extra)
    except TypeError:
        extra.pop("absgrad", None)
        try:
            out = gsplat.rasterization(**kwargs, **extra)
        except TypeError:
            out = gsplat.rasterization(**kwargs)
    colors_out, _alphas, info = out[0], out[1], out[2] if len(out) > 2 else {}
    rgb = colors_out[0]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if isinstance(info, dict):
        means2d = info.get("means2d")
        if means2d is not None:
            try:
                means2d.retain_grad()
            except Exception:
                pass
    return rgb.clamp(0.0, 1.0), info if isinstance(info, dict) else {}


def _maybe_densify(
    torch, means, quats, colors, log_scales, logit_opacities, means2d, *, thresh, max_clone,
):
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
    if g.shape[0] != means.shape[0]:
        return None
    mag = g.norm(dim=-1)
    sel = mag > float(thresh)
    n_sel = int(sel.sum())
    if n_sel == 0:
        return None
    if n_sel > int(max_clone):
        idx = torch.topk(mag, k=int(max_clone)).indices
        sel = torch.zeros_like(sel)
        sel[idx] = True
        n_sel = int(max_clone)
    noise = torch.randn_like(means[sel]) * 0.01
    new_means = (means[sel] + noise).detach().requires_grad_(True)
    new_quats = quats[sel].detach().clone().requires_grad_(True)
    new_colors = colors[sel].detach().clone().requires_grad_(True)
    new_log_scales = (log_scales[sel] - 0.6931).detach().requires_grad_(True)  # /2 scale
    new_logit = logit_opacities[sel].detach().clone().requires_grad_(True)
    means = torch.cat([means.detach(), new_means], dim=0).requires_grad_(True)
    quats = torch.cat([quats.detach(), new_quats], dim=0).requires_grad_(True)
    colors = torch.cat([colors.detach(), new_colors], dim=0).requires_grad_(True)
    log_scales = torch.cat([log_scales.detach(), new_log_scales], dim=0).requires_grad_(True)
    logit_opacities = torch.cat([logit_opacities.detach(), new_logit], dim=0).requires_grad_(True)
    return means, quats, colors, log_scales, logit_opacities, n_sel
