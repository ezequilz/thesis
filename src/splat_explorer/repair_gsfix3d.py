"""GSFix3D §3.3 photometric lift (CUDA / gsplat) — research default.

Ports ``scripts/gsfix3d/refine_gs.py`` onto ``GaussianScene``. No color
stamp. Subclass this (or add a sibling backend id) to try a new idea;
leave ``repair_gsfix.GsplatPhotometricRepair`` frozen as the A/B baseline.

Per repaired view (``iters=20``, paper default)::

    I_gs = rasterize(gaussians, camera)
    L = 0.8 ||I_fixed - I_gs||_1 + 0.2 (1 - SSIM)
    backward
    accumulate view-space positional gradients
    every 5 steps: clone small + split large Gaussians (Kerbl ADC)
    last iter: prune opacity < 0.005
    Adam step  (never skipped on densify steps)

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


def densify_clone_split(
    torch,
    means,
    quats,
    f_dc,
    log_scales,
    logit_opacities,
    grad_norm,
    *,
    thresh: float,
    split_scale: float,
    max_clone: int,
    max_gaussians: int,
):
    """Kerbl clone (small) + split (large). Returns new params and n_spawned.

    ``grad_norm`` is the accumulated view-space mean |xy-grad| / count,
    same as ``GaussianModel.densify``. Clone and split masks are disjoint.
    """
    n = int(means.shape[0])
    if n == 0 or n >= int(max_gaussians):
        return None
    mag = grad_norm.detach().reshape(n)
    mag = torch.nan_to_num(mag, nan=0.0)
    scales = torch.exp(log_scales)
    max_s = scales.max(dim=-1).values
    high = mag >= float(thresh)
    clone_sel = high & (max_s <= float(split_scale))
    split_sel = high & (max_s > float(split_scale))
    n_clone = int(clone_sel.sum())
    n_split = int(split_sel.sum())
    budget = max(0, int(max_gaussians) - n)
    if n_clone + n_split == 0 or budget <= 0:
        return None
    # Prefer highest-grad when we would exceed max_clone / remaining slots.
    cap = min(int(max_clone), budget)
    if n_clone + n_split > cap:
        idx = torch.topk(mag, k=min(cap, n)).indices
        pick = torch.zeros_like(high)
        pick[idx] = True
        clone_sel = clone_sel & pick
        split_sel = split_sel & pick
        n_clone = int(clone_sel.sum())
        n_split = int(split_sel.sum())
        if n_clone + n_split == 0:
            return None

    parts_means = [means[~split_sel].detach()]
    parts_quats = [quats[~split_sel].detach()]
    parts_dc = [f_dc[~split_sel].detach()]
    parts_log = [log_scales[~split_sel].detach()]
    parts_op = [logit_opacities[~split_sel].detach()]
    n_spawned = 0

    if n_clone:
        parts_means.append(means[clone_sel].detach())
        parts_quats.append(quats[clone_sel].detach())
        parts_dc.append(f_dc[clone_sel].detach())
        parts_log.append(log_scales[clone_sel].detach())
        parts_op.append(logit_opacities[clone_sel].detach())
        n_spawned += n_clone

    if n_split:
        stds = scales[split_sel].repeat(_SPLIT_N, 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = _quat_to_rotmat(torch.nn.functional.normalize(quats[split_sel], dim=-1), torch)
        rots = rots.repeat(_SPLIT_N, 1, 1)
        new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + means[split_sel].repeat(_SPLIT_N, 1)
        new_scales = scales[split_sel].repeat(_SPLIT_N, 1) / _SPLIT_SCALE_DIV
        parts_means.append(new_means.detach())
        parts_quats.append(quats[split_sel].detach().repeat(_SPLIT_N, 1))
        parts_dc.append(f_dc[split_sel].detach().repeat(_SPLIT_N, 1))
        parts_log.append(torch.log(torch.clamp(new_scales, min=1e-8)).detach())
        parts_op.append(logit_opacities[split_sel].detach().repeat(_SPLIT_N, 1))
        n_spawned += n_split * _SPLIT_N - n_split  # net: each split becomes 2

    def _cat(chunks):
        return torch.cat(chunks, dim=0).detach().requires_grad_(True)

    return (
        _cat(parts_means),
        _cat(parts_quats),
        _cat(parts_dc),
        _cat(parts_log),
        _cat(parts_op),
        n_spawned,
    )


def _viewspace_grad_norm(means2d, n_gaussians, torch, info=None, width=None, height=None):
    """Per-gaussian |xy| screen-space grad, or None if gsplat did not retain it.

    gsplat returns ``means2d`` as ``[C, N, 2]`` (unpacked) or ``[nnz, 2]``
    (packed). ``.grad`` is dropped unless ``retain_grad()`` ran on the exact
    tensor from ``rasterization``; ``absgrad=True`` fills ``.absgrad``.
    Screen-space scaling matches gsplat ``DefaultStrategy`` so Kerbl's
    0.0002 threshold is in the same units.
    """
    if means2d is None:
        return None
    grad = getattr(means2d, "absgrad", None)
    if grad is None:
        grad = means2d.grad
    if grad is None:
        return None
    g = grad.detach()
    info = info if isinstance(info, dict) else {}
    ids = info.get("gaussian_ids")
    if g.ndim == 3:
        g = g.reshape(-1, g.shape[-1])
        if ids is None and g.shape[0] == n_gaussians:
            pass
        elif ids is None and g.shape[0] != n_gaussians:
            return None
    if width and height and g.shape[-1] >= 2:
        g = g.clone()
        g[..., 0] = g[..., 0] * (float(width) / 2.0)
        g[..., 1] = g[..., 1] * (float(height) / 2.0)
    mag = g[..., :2].norm(dim=-1)
    if mag.shape[0] == n_gaussians:
        return mag
    if ids is None or mag.shape[0] != int(ids.reshape(-1).shape[0]):
        return None
    out = torch.zeros(n_gaussians, device=mag.device, dtype=mag.dtype)
    out.index_add_(0, ids.reshape(-1).long(), mag.reshape(-1))
    return out


def _n_visible_from_info(info, n_gaussians, torch):
    if not isinstance(info, dict):
        return n_gaussians
    radii = info.get("radii")
    if radii is None:
        return n_gaussians
    r = radii.detach()
    if r.ndim == 2:
        r = r[0]
    if r.shape[0] != n_gaussians:
        return n_gaussians
    n = int((r > 0).sum())
    return n if n else n_gaussians


@dataclass
class GsplatGsfix3dRepair:
    """Paper §3.3 CUDA refine. Default ``gsfix-gsplat`` backend.

    Add fields / override ``apply`` in a subclass when testing a new
    method, and register it in ``repair.CUDA_REPAIR_METHODS``.
    """

    iters: int = 20
    kf_iters: int = 50
    lambda_dssim: float = _LAMBDA_DSSIM
    densify: bool = True
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
        n_spawned = 0
        xyz_grad_accum = torch.zeros(means.shape[0], device=device)
        xyz_grad_denom = torch.zeros(means.shape[0], device=device)

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
            if means2d is not None:
                try:
                    means2d.retain_grad()
                except Exception:
                    pass
            loss, l1 = photometric_loss(rgb, target, torch, self.lambda_dssim)
            last_l1 = float(l1.item())
            loss.backward()

            n_visible = _n_visible_from_info(info, means.shape[0], torch)
            vis_norm = _viewspace_grad_norm(
                means2d, means.shape[0], torch, info=info, width=w, height=h,
            )
            if vis_norm is not None:
                radii = info.get("radii") if isinstance(info, dict) else None
                vis = vis_norm > 0
                if radii is not None:
                    r = radii.detach()
                    if r.ndim == 2:
                        r = r[0]
                    if r.shape[0] == vis.shape[0]:
                        vis = vis & (r > 0)
                xyz_grad_accum = xyz_grad_accum + vis_norm
                xyz_grad_denom = xyz_grad_denom + vis.to(xyz_grad_accum.dtype)
            elif it == 0:
                logger.warning(
                    "GSFix3D densify: no means2d screen grads (absgrad/retain_grad). "
                    "Clone+split will not spawn until gsplat exposes them."
                )

            # Adam on this iteration's graph, then densify. Recreating the
            # optimizer before step() would drop .grad (unlike INRIA's cat).
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                quats.copy_(torch.nn.functional.normalize(quats, dim=-1))

            last_iter = it == int(self.iters) - 1
            if (
                self.densify
                and (it + 1) % int(self.densify_every) == 0
                and not last_iter
                and means.shape[0] < int(self.max_gaussians)
            ):
                avg = xyz_grad_accum / xyz_grad_denom.clamp(min=1.0)
                n_high = int((avg >= float(self.densify_grad_thresh)).sum())
                packed = densify_clone_split(
                    torch, means, quats, f_dc, log_scales, logit_opacities, avg,
                    thresh=self.densify_grad_thresh,
                    split_scale=self.split_scale,
                    max_clone=self.max_clone,
                    max_gaussians=self.max_gaussians,
                )
                if packed is not None:
                    means, quats, f_dc, log_scales, logit_opacities, n_new = packed
                    n_spawned += n_new
                    xyz_grad_accum = torch.zeros(means.shape[0], device=device)
                    xyz_grad_denom = torch.zeros(means.shape[0], device=device)
                    opt = make_opt()
                elif it == int(self.densify_every) - 1:
                    logger.info(
                        "GSFix3D densify: 0 spawned (max |grad2d|=%.6f, %d above %.6f)",
                        float(avg.max().item()) if avg.numel() else 0.0,
                        n_high,
                        float(self.densify_grad_thresh),
                    )

            if last_iter and self.densify and means.shape[0] > 32:
                with torch.no_grad():
                    keep = torch.sigmoid(logit_opacities) > float(self.prune_opacity)
                    if int(keep.sum()) >= 32 and int((~keep).sum()) > 0:
                        means, quats, f_dc, log_scales, logit_opacities = (
                            means[keep], quats[keep], f_dc[keep],
                            log_scales[keep], logit_opacities[keep],
                        )

            if self.on_progress is not None:
                self.on_progress({
                    "phase": "refine",
                    "iter": it + 1,
                    "n_iters": it + 1,
                    "n_updated": int(means.shape[0]),
                    "n_gaussians": int(means.shape[0]),
                    "n_spawned": int(n_spawned),
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
            "GSFix3D refine: %d iters, L1 %.4f -> %.4f, %d -> %d gaussians (+%d)",
            self.iters, l1_before, l1_after, n0, n1, n_spawned,
        )
        return {
            "backend": BACKEND_ID,
            "n_visible": int(n_visible),
            "n_updated": n1,
            "n_stamped": 0,
            "n_spawned": int(max(0, n1 - n0) if n_spawned == 0 else n_spawned),
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
    else:
        cls = GsplatGsfix3dRepair
    skip = {"on_progress"}
    allowed = {f.name for f in fields(cls) if f.name not in skip}
    kwargs = {k: body[k] for k in allowed if k in body}
    return cls(**kwargs)
