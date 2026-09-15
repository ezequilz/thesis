"""src/splat_explorer/repair_gsfix3d_working_backup.py
GSFix3D §3.3 photometric lift (CUDA / gsplat) — scene-run default.

This is the paper ADC path used by scene-runs. Scene loading stays
outside this module. Later research in ``repair_gsfix3d.py`` showed the
neon artifacts come from two mistakes relative to INRIA / GSFix3D:

1. Optimizing raw RGB, then hard-clipping to ``[0, 1]`` every chunk.
   Saturated channels get zero gradient through the clip, Adam keeps
   pushing, and the next reload starts at the clip wall — posterized
   magenta / cyan / yellow.
2. Reloading those clipped RGB values from the CPU scene between
   20-iter chunks. Scene-runs set ``max_chunks=0`` for a time budget,
   so that cycle ran for minutes on one Qwen view.

This copy therefore keeps the working-backup ADC loop, but uses the
later GPU-resident SH DC tensors: RGB2SH once, Adam on ``f_dc``,
SH2RGB only for rasterization. Checkpoints still write clipped RGB
for the PLY / spectator; the optimizer state is not reset by that clip.

Per repaired view (``iters=20``, paper default)::

    I_gs = rasterize(gaussians, camera)          # unclamped training RGB
    L = 0.8 ||W ⊙ (I_fixed - I_gs)||_1
        + 0.2 (1 - SSIM)
        + ||(1-W) ⊙ (I_orig - I_gs)||_1
        + λ_bound · overshoot(SH2RGB(f_dc))
    backward
    accumulate view-space positional gradients
    every 5 steps: clone small + split large Gaussians (Kerbl ADC)
    last iter of the first chunk: prune opacity < 0.005
    Adam step  (never skipped on densify steps)

``W`` is the Qwen residual, so artifact pixels follow the repaired
image and unchanged pixels stay anchored to the original render.
``apply_until`` repeats paper 20-iter chunks with Gaussians remaining
on the GPU. ``Replay all views`` then runs ``kf_iters=50`` shuffled
passes over the repaired images (our stand-in for the paper's
augmented captured dataset — we do not have the original RGB-D capture).
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
    camera_for_train,
    ssim,
    upsample_uint8,
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


def repaired_view_loss(
    pred,
    target,
    rendered,
    torch,
    *,
    lambda_dssim: float = _LAMBDA_DSSIM,
    lambda_preserve: float = 1.0,
    lambda_color_bound: float = 0.05,
    colors_rgb=None,
    min_fix_weight: float = 0.05,
):
    """Photometric lift of a Qwen view without neon-saturating unchanged pixels.

    ``pred`` is the current differentiable render and must not be
    display-clamped: once a channel exceeds 1, ``clamp`` has zero
    gradient and Adam drives SH DC to the clip wall. ``target`` is the
    repaired image; ``rendered`` is the original 3DGS view. Residual
    weights ``W`` send artifact pixels toward Qwen and keep the rest
    anchored. ``colors_rgb`` (SH2RGB of ``f_dc``) gets a quadratic
    overshoot penalty so per-Gaussian colors cannot run to ±inf.
    """
    change = (target - rendered).abs().mean(dim=-1, keepdim=True)
    scale = change.mean().clamp(min=1e-4)
    w_fix = (change / (change + scale)).clamp(min=float(min_fix_weight), max=1.0)
    l1_fix = (w_fix * (pred - target).abs()).mean()
    l1_keep = ((1.0 - w_fix) * (pred - rendered).abs()).mean()
    pred_disp = pred.clamp(0.0, 1.0)
    ssim_val = ssim(pred_disp.permute(2, 0, 1), target.permute(2, 0, 1), torch)
    loss = (
        (1.0 - float(lambda_dssim)) * l1_fix
        + float(lambda_dssim) * (1.0 - ssim_val)
        + float(lambda_preserve) * l1_keep
    )
    if colors_rgb is not None and float(lambda_color_bound) > 0:
        over = torch.relu(colors_rgb - 1.0).square().mean()
        under = torch.relu(-colors_rgb).square().mean()
        loss = loss + float(lambda_color_bound) * (over + under)
    return loss, (pred_disp - target).abs().mean()


def _as_leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


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


def _repeat_gaussians(tensor, count: int):
    """Repeat dimension 0 while preserving all remaining tensor ranks."""
    return tensor.repeat(*((int(count),) + (1,) * (tensor.ndim - 1)))


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
        stds = _repeat_gaussians(scales[split_sel], _SPLIT_N)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = _quat_to_rotmat(torch.nn.functional.normalize(quats[split_sel], dim=-1), torch)
        rots = _repeat_gaussians(rots, _SPLIT_N)
        new_means = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + _repeat_gaussians(means[split_sel], _SPLIT_N)
        )
        new_scales = (
            _repeat_gaussians(scales[split_sel], _SPLIT_N)
            / _SPLIT_SCALE_DIV
        )
        parts_means.append(new_means.detach())
        parts_quats.append(
            _repeat_gaussians(quats[split_sel].detach(), _SPLIT_N)
        )
        parts_dc.append(
            _repeat_gaussians(f_dc[split_sel].detach(), _SPLIT_N)
        )
        parts_log.append(torch.log(torch.clamp(new_scales, min=1e-8)).detach())
        parts_op.append(
            _repeat_gaussians(logit_opacities[split_sel].detach(), _SPLIT_N)
        )
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


def _viewspace_grad_norm(
    means2d, n_gaussians, torch, *, info=None, width=None, height=None,
):
    if means2d is None:
        return None
    grad = getattr(means2d, "absgrad", None)
    if grad is None:
        grad = means2d.grad
    if grad is None:
        return None
    g = grad.detach()
    info = info if isinstance(info, dict) else {}
    gaussian_ids = info.get("gaussian_ids")
    if g.ndim == 3:
        g = g.reshape(-1, g.shape[-1])
    if width and height and g.shape[-1] >= 2:
        g = g.clone()
        g[..., 0] *= float(width) / 2.0
        g[..., 1] *= float(height) / 2.0
    magnitude = g[..., :2].norm(dim=-1)
    if magnitude.shape[0] == n_gaussians:
        return magnitude
    if (
        gaussian_ids is None
        or magnitude.shape[0] != int(gaussian_ids.reshape(-1).shape[0])
    ):
        return None
    result = torch.zeros(
        n_gaussians, device=magnitude.device, dtype=magnitude.dtype,
    )
    result.index_add_(
        0, gaussian_ids.reshape(-1).long(), magnitude.reshape(-1),
    )
    return result


@dataclass
class GsplatGsfix3dRepair:
    """Paper §3.3 CUDA refine. Default ``gsfix-gsplat`` backend.

    Add fields / override ``apply`` in a subclass when testing a new
    method, and register it in ``repair.CUDA_REPAIR_METHODS``.
    Experimental extras live only on ``GsplatGsfix3dVisPruneRepair``
    (``gsfix-gsplat-visprune``) — do not add hooks here.
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
    train_max_edge: int = 0
    sparse_grad: bool = False
    white_background: bool = False
    max_chunks: int = 1
    lambda_preserve: float = 1.0
    lambda_color_bound: float = 0.05
    min_fix_weight: float = 0.05
    sh_clip: float = 2.3
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
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            edge = int(self.train_max_edge or 0)
            if not self.packed:
                logger.warning("CUDA OOM during GSFix3D refine; retrying packed @ %spx", edge or 512)
                return replace(
                    self, packed=True, sparse_grad=True,
                    train_max_edge=edge or 512,
                )._apply(scene, camera, rendered_rgb, repaired_rgb)
            nxt = 384 if edge <= 0 or edge > 384 else (320 if edge > 320 else 0)
            if nxt:
                logger.warning("CUDA OOM during packed GSFix3D refine; retrying @ %spx", nxt)
                return replace(
                    self, packed=True, sparse_grad=True, train_max_edge=nxt,
                )._apply(scene, camera, rendered_rgb, repaired_rgb)
            raise

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

        Gaussians stay on the GPU across chunks so SH DC is not clipped
        back to RGB between passes. Set ``max_chunks=0`` to keep going
        until Stop / deadline (scene-runs). ADC runs only on chunk 0.
        """
        import time

        last: dict[str, Any] | None = None
        total_iters = 0
        l1_before = None
        chunk = 0
        limit = int(self.max_chunks)
        state: dict[str, Any] = {
            "should_stop": should_stop,
            "deadline": deadline,
        }
        saved_densify = self.densify
        try:
            while True:
                if should_stop is not None and should_stop():
                    break
                if deadline is not None and time.time() >= deadline:
                    break
                if limit > 0 and chunk >= limit:
                    break
                # GSFix3D applies ADC during its 20 iterations for a repaired
                # image. Extra time-budget chunks continue photometric fitting
                # without repeatedly multiplying the topology for the same view.
                if chunk > 0:
                    self.densify = False
                last = self._step_chunk(
                    scene, camera, rendered_rgb, repaired_rgb, state,
                )
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
                last["checkpoint_iters"] = total_iters
                if on_checkpoint is not None:
                    on_checkpoint(last)
        finally:
            self.densify = saved_densify
            ctx = state.get("ctx")
            if isinstance(ctx, dict):
                ctx.pop("opt", None)
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

    def _step_chunk(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One paper 20-iter pass. Reuses ``state['ctx']`` so SH DC stays on GPU."""
        state = state if state is not None else {}
        if "ctx" not in state:
            state["ctx"] = self._gpu_bind(scene, camera, rendered_rgb, repaired_rgb)
        ctx = state["ctx"]
        if "should_stop" in state:
            ctx["should_stop"] = state.get("should_stop")
        if "deadline" in state:
            ctx["deadline"] = state.get("deadline")
        return self._gpu_advance(ctx, scene)

    def _make_opt(self, torch, means, quats, f_dc, log_scales, logit_opacities):
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

    def _gpu_bind(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        torch = _require_torch()
        import gsplat

        device = torch.device("cuda")
        orig_w, orig_h = int(camera.width), int(camera.height)
        camera = camera_for_train(camera, int(self.train_max_edge or 0))
        h, w = int(camera.height), int(camera.width)
        use_packed = bool(self.packed)
        use_sparse = bool(self.sparse_grad)
        if self.on_progress is not None:
            props = torch.cuda.get_device_properties(0)
            self.on_progress({
                "phase": "cuda_ready",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_mib": int(props.total_memory / (1024 * 1024)),
                "n_gaussians": int(scene.num_gaussians),
                "n_iters": 0,
                "n_stamped": 0,
                "packed": use_packed,
                "train_width": w,
                "train_height": h,
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
        opt = self._make_opt(torch, means, quats, f_dc, log_scales, logit_opacities)
        return {
            "torch": torch,
            "gsplat": gsplat,
            "means": means,
            "quats": quats,
            "f_dc": f_dc,
            "log_scales": log_scales,
            "logit_opacities": logit_opacities,
            "opt": opt,
            "viewmat": viewmat,
            "K": K,
            "background": background,
            "target": target,
            "rendered": rendered,
            "l1_before": l1_before,
            "n0": n0,
            "w": w,
            "h": h,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "use_packed": use_packed,
            "use_sparse": use_sparse,
            "n_visible": int(means.shape[0]),
            "n_spawned": 0,
            "total_iters": 0,
            "last_l1": l1_before,
            "xyz_grad_accum": torch.zeros(means.shape[0], device=device),
            "xyz_grad_denom": torch.zeros(means.shape[0], device=device),
        }

    def _commit_scene(self, ctx: dict[str, Any], scene: GaussianScene) -> tuple[np.ndarray, float]:
        """CPU snapshot for PLY / dashboard. Does not clip the GPU SH DC."""
        torch = ctx["torch"]
        gsplat = ctx["gsplat"]
        means = ctx["means"]
        quats = ctx["quats"]
        f_dc = ctx["f_dc"]
        log_scales = ctx["log_scales"]
        logit_opacities = ctx["logit_opacities"]
        with torch.no_grad():
            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
            colors = sh_to_rgb(f_dc)
            rgb, _ = _rasterize(
                gsplat, means, quats_n, scales, opacities, colors,
                ctx["viewmat"], ctx["K"], ctx["w"], ctx["h"], ctx["background"],
                packed=ctx["use_packed"], sparse_grad=ctx["use_sparse"],
                absgrad=False, clamp_rgb=True,
            )
            l1_after = float(torch.abs(rgb - ctx["target"]).mean().item())
            render_rgb = upsample_uint8(_to_uint8(rgb), ctx["orig_w"], ctx["orig_h"])
        scene.means = means.detach().float().cpu().numpy().astype(np.float32)
        scene.quats = quats_n.detach().float().cpu().numpy().astype(np.float32)
        scene.scales = scales.detach().float().cpu().numpy().astype(np.float32)
        scene.opacities = opacities.detach().float().cpu().numpy().astype(np.float32)
        scene.colors = torch.clamp(colors, 0.0, 1.0).detach().float().cpu().numpy().astype(np.float32)
        return render_rgb, l1_after

    def _gpu_advance(self, ctx: dict[str, Any], scene: GaussianScene) -> dict[str, Any]:
        import time

        torch = ctx["torch"]
        gsplat = ctx["gsplat"]
        means = ctx["means"]
        quats = ctx["quats"]
        f_dc = ctx["f_dc"]
        log_scales = ctx["log_scales"]
        logit_opacities = ctx["logit_opacities"]
        opt = ctx["opt"]
        viewmat = ctx["viewmat"]
        K = ctx["K"]
        background = ctx["background"]
        target = ctx["target"]
        rendered = ctx["rendered"]
        l1_before = ctx["l1_before"]
        n0 = ctx["n0"]
        w, h = ctx["w"], ctx["h"]
        use_packed = ctx["use_packed"]
        use_sparse = ctx["use_sparse"]
        n_visible = int(ctx["n_visible"])
        n_spawned = int(ctx["n_spawned"])
        last_l1 = ctx["last_l1"]
        xyz_grad_accum = ctx["xyz_grad_accum"]
        xyz_grad_denom = ctx["xyz_grad_denom"]
        should_stop = ctx.get("should_stop")
        deadline = ctx.get("deadline")
        sh_clip = float(self.sh_clip)
        ran = 0

        for it in range(int(self.iters)):
            if should_stop is not None and should_stop():
                break
            if deadline is not None and time.time() >= deadline:
                break
            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            quats_n = torch.nn.functional.normalize(quats, dim=-1)
            colors = sh_to_rgb(f_dc)
            rgb, info = _rasterize(
                gsplat, means, quats_n, scales, opacities, colors,
                viewmat, K, w, h, background, packed=use_packed,
                sparse_grad=use_sparse, absgrad=not use_packed,
                clamp_rgb=False,
            )
            means2d = info.get("means2d") if isinstance(info, dict) else None
            if means2d is not None and means2d.requires_grad:
                means2d.retain_grad()
            loss, l1 = repaired_view_loss(
                rgb, target, rendered, torch,
                lambda_dssim=self.lambda_dssim,
                lambda_preserve=self.lambda_preserve,
                lambda_color_bound=self.lambda_color_bound,
                colors_rgb=colors,
                min_fix_weight=self.min_fix_weight,
            )
            last_l1 = float(l1.item())
            if ctx["total_iters"] == 0 and it == 0 and self.on_progress is not None:
                self.on_progress({
                    "phase": "refine",
                    "iter": 0,
                    "n_iters": 0,
                    "n_updated": 0,
                    "n_gaussians": int(means.shape[0]),
                    "n_spawned": int(n_spawned),
                    "n_stamped": 0,
                    "n_visible": int(n_visible),
                    "l1_before": round(l1_before, 6),
                    "l1": round(last_l1, 6),
                    "train_width": w,
                    "train_height": h,
                    "message": (
                        f"First rasterize L1 {last_l1:.4f} @ {w}x{h}"
                        + (" · packed" if use_packed else "")
                    ),
                })
            loss.backward()

            vis_norm = _viewspace_grad_norm(
                means2d,
                means.shape[0],
                torch,
                info=info,
                width=w,
                height=h,
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
                n_visible = int(vis.sum()) if vis.any() else int(means.shape[0])
                xyz_grad_accum = xyz_grad_accum + vis_norm
                xyz_grad_denom = xyz_grad_denom + vis.to(xyz_grad_accum.dtype)

            # Adam on this iteration's graph, then densify. Recreating the
            # optimizer before step() would drop .grad (unlike INRIA's cat).
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                quats.copy_(torch.nn.functional.normalize(quats, dim=-1))
                if sh_clip > 0:
                    f_dc.clamp_(-sh_clip, sh_clip)

            last_iter = it == int(self.iters) - 1
            if (
                self.densify
                and (it + 1) % int(self.densify_every) == 0
                and not last_iter
                and means.shape[0] < int(self.max_gaussians)
            ):
                avg = xyz_grad_accum / xyz_grad_denom.clamp(min=1.0)
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
                    xyz_grad_accum = torch.zeros(means.shape[0], device=means.device)
                    xyz_grad_denom = torch.zeros(means.shape[0], device=means.device)
                    opt = self._make_opt(
                        torch, means, quats, f_dc, log_scales, logit_opacities,
                    )

            if last_iter and self.densify and means.shape[0] > 32:
                keep = torch.sigmoid(logit_opacities).reshape(-1) > float(self.prune_opacity)
                if int(keep.sum()) >= 32 and int((~keep).sum()) > 0:
                    means, quats, f_dc, log_scales, logit_opacities = (
                        _as_leaf(means[keep]), _as_leaf(quats[keep]),
                        _as_leaf(f_dc[keep]), _as_leaf(log_scales[keep]),
                        _as_leaf(logit_opacities[keep]),
                    )
                    xyz_grad_accum = torch.zeros(means.shape[0], device=means.device)
                    xyz_grad_denom = torch.zeros(means.shape[0], device=means.device)
                    opt = self._make_opt(
                        torch, means, quats, f_dc, log_scales, logit_opacities,
                    )

            ran += 1
            done = ctx["total_iters"] + ran
            if self.on_progress is not None:
                self.on_progress({
                    "phase": "refine",
                    "iter": done,
                    "n_iters": done,
                    "n_updated": int(means.shape[0]),
                    "n_gaussians": int(means.shape[0]),
                    "n_spawned": int(n_spawned),
                    "n_stamped": 0,
                    "n_visible": int(n_visible),
                    "l1_before": round(l1_before, 6),
                    "l1": round(last_l1, 6),
                    "train_width": w,
                    "train_height": h,
                })

        ctx["means"] = means
        ctx["quats"] = quats
        ctx["f_dc"] = f_dc
        ctx["log_scales"] = log_scales
        ctx["logit_opacities"] = logit_opacities
        ctx["opt"] = opt
        ctx["xyz_grad_accum"] = xyz_grad_accum
        ctx["xyz_grad_denom"] = xyz_grad_denom
        ctx["n_visible"] = n_visible
        ctx["n_spawned"] = n_spawned
        ctx["last_l1"] = last_l1
        ctx["total_iters"] += ran

        render_rgb, l1_after = self._commit_scene(ctx, scene)
        n1 = scene.num_gaussians
        logger.info(
            "GSFix3D refine: %d iters, L1 %.4f -> %.4f, %d -> %d gaussians (+%d)",
            ran, l1_before, l1_after, n0, n1, n_spawned,
        )
        return {
            "backend": BACKEND_ID,
            "n_visible": int(n_visible),
            "n_updated": n1,
            "n_stamped": 0,
            "n_spawned": int(max(0, n1 - n0) if n_spawned == 0 else n_spawned),
            "n_gaussians": n1,
            "n_iters": int(ran),
            "l1_before": round(l1_before, 6),
            "l1_after": round(l1_after, 6),
            "train_width": w,
            "train_height": h,
            "render_rgb": render_rgb,
        }

    def _apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        return self._step_chunk(scene, camera, rendered_rgb, repaired_rgb, {})


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
    return cls(**kwargs)