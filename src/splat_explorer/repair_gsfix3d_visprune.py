"""Experimental GSFix3D lift: prune floaters, anchor nearby original views.

Sibling of ``GsplatGsfix3dRepair`` (backend ``gsfix-gsplat-visprune``).
Owns its own ``_apply`` so the paper path in ``repair_gsfix3d.py`` stays
frozen. Do not add hooks to the paper class.

Per 20-iter chunk, on top of §3.3 photometric refine::

    last iter of a full chunk: opacity prune, then error-mask floater prune
    extra L1 vs original-scene renders at four yaw/pitch micro-rotations

Occlusion-freeze (nearest-center visual mask) is off: it selected in-front
floaters as the only updatable set, so they ballooned into huge opaque discs.

``apply_until`` densifies only the first chunk so a focused 1h run cannot
keep spawning ray-streaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .rendering.base import Camera
from .repair_gsfix import (
    _image_to_tensor,
    _inv_sigmoid,
    _rasterize,
    _require_torch,
    _to_uint8,
    photometric_loss,
)
from .repair_gsfix3d import (
    _BACKGROUND_BLACK,
    _BACKGROUND_WHITE,
    _SPLIT_N,
    _SPLIT_SCALE_DIV,
    _quat_to_rotmat,
    GsplatGsfix3dRepair,
    rgb_to_sh,
    sh_to_rgb,
)
from .scene import GaussianScene

logger = logging.getLogger(__name__)

BACKEND_ID = "gsfix-gsplat-visprune"
_MIN_KEEP = 32


def _repeat_along_n(t, n_rep: int):
    """Repeat along Gaussian dim 0 without promoting a 1-D tensor to 2-D.

    ``t.repeat(2, 1)`` on shape ``(K,)`` prepends a dimension → ``(2, K)``.
    Opacity logits from the PLY are ``(N,)``, so that form cannot be
    concatenated with the kept 1-D slice (the /repair crash on LRZ).
    """
    n_rep = int(n_rep)
    if n_rep == 1:
        return t
    return t.repeat(*((n_rep,) + (1,) * (t.ndim - 1)))


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
    """Kerbl clone (small) + split (large). Same as paper, 1-D opacity-safe."""
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
        stds = _repeat_along_n(scales[split_sel], _SPLIT_N)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = _quat_to_rotmat(torch.nn.functional.normalize(quats[split_sel], dim=-1), torch)
        rots = _repeat_along_n(rots, _SPLIT_N)
        new_means = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + _repeat_along_n(means[split_sel], _SPLIT_N)
        )
        new_scales = _repeat_along_n(scales[split_sel], _SPLIT_N) / _SPLIT_SCALE_DIV
        parts_means.append(new_means.detach())
        parts_quats.append(_repeat_along_n(quats[split_sel].detach(), _SPLIT_N))
        parts_dc.append(_repeat_along_n(f_dc[split_sel].detach(), _SPLIT_N))
        parts_log.append(torch.log(torch.clamp(new_scales, min=1e-8)).detach())
        parts_op.append(_repeat_along_n(logit_opacities[split_sel].detach(), _SPLIT_N))
        n_spawned += n_split * _SPLIT_N - n_split

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

    Packed ``means2d`` and gsplat ``DefaultStrategy`` screen-space scaling
    live here so the paper lift stays on the original unpacked |xy-grad|.
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


def project_centers(
    means: np.ndarray,
    camera: Camera,
    near: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixel (u, v), camera-space z, and in-frustum mask for Gaussian centers."""
    means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    w2c = np.asarray(camera.w2c, dtype=np.float64)
    pcam = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pcam[:, 2]
    n = int(means.shape[0])
    u = np.full(n, np.nan, dtype=np.float64)
    v = np.full(n, np.nan, dtype=np.float64)
    valid = z > float(near)
    if np.any(valid):
        u[valid] = camera.fx * pcam[valid, 0] / z[valid] + camera.width / 2.0
        v[valid] = camera.fy * pcam[valid, 1] / z[valid] + camera.height / 2.0
    on = (
        valid
        & np.isfinite(u)
        & (u >= 0.0)
        & (u < float(camera.width))
        & (v >= 0.0)
        & (v < float(camera.height))
    )
    return u, v, z, on


def front_contributing_mask(
    means: np.ndarray,
    camera: Camera,
    opacities: np.ndarray,
    *,
    near: float = 0.05,
    contrib_thresh: float = 0.05,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    """True for in-frustum Gaussians that win nearest-opaque-center at their pixel.

    Farther Gaussians at the same pixel are treated as occluded. Centers that
    miss the image, fail ``radii > 0``, or sit behind a nearer opaque splat
    return False (do not update).
    """
    u, v, z, on = project_centers(means, camera, near=near)
    n = int(np.asarray(means).reshape(-1, 3).shape[0])
    opacities = np.asarray(opacities, dtype=np.float64).reshape(-1)
    if opacities.shape[0] != n:
        raise ValueError("opacities must be length-N")
    if radii is not None:
        r = np.asarray(radii, dtype=np.float64).reshape(-1)
        if r.shape[0] == n:
            on = on & (r > 0.0)
    out = np.zeros(n, dtype=bool)
    opaque = opacities >= float(contrib_thresh)
    idx = np.flatnonzero(on & opaque)
    h, w = int(camera.height), int(camera.width)
    if idx.size == 0:
        out[on] = True
        return out
    ui = np.clip(np.floor(u[idx]).astype(np.int64), 0, w - 1)
    vi = np.clip(np.floor(v[idx]).astype(np.int64), 0, h - 1)
    nearest = np.full((h, w), np.inf, dtype=np.float64)
    np.minimum.at(nearest, (vi, ui), z[idx])
    on_idx = np.flatnonzero(on)
    ui_on = np.clip(np.floor(u[on]).astype(np.int64), 0, w - 1)
    vi_on = np.clip(np.floor(v[on]).astype(np.int64), 0, h - 1)
    front = z[on] <= nearest[vi_on, ui_on] + 1e-4
    out[on_idx[front]] = True
    return out


def apply_updatable_grads(tensors: Sequence[Any], updatable) -> None:
    """In-place: zero ``.grad`` on rows where ``updatable`` is False."""
    if updatable is None:
        return
    mask = updatable
    for t in tensors:
        grad = getattr(t, "grad", None)
        if grad is None:
            continue
        m = mask.to(device=grad.device, dtype=grad.dtype)
        if grad.ndim == 1:
            grad.mul_(m)
        else:
            grad.mul_(m.reshape([-1] + [1] * (grad.ndim - 1)))


def rgb_l1_residual(original: np.ndarray, repaired: np.ndarray) -> np.ndarray:
    """Per-pixel mean-|RGB| residual in [0, 1]."""
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(repaired, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
        b = b[..., None]
    if a.max() > 1.5 or b.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
    return np.abs(a - b).mean(axis=-1)


def _resize_hw(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    if arr.shape[0] == height and arr.shape[1] == width:
        return np.asarray(arr, dtype=np.float64)
    img = Image.fromarray(
        np.clip(np.asarray(arr, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )
    return np.asarray(
        img.resize((int(width), int(height)), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0


def error_mask_keep(
    means: np.ndarray,
    camera: Camera,
    residual_hw: np.ndarray,
    *,
    error_thresh: float = 0.12,
    max_frac: float = 0.02,
    min_keep: int = _MIN_KEEP,
    near: float = 0.05,
    depth_margin: float = 0.05,
    surface_depth: np.ndarray | None = None,
) -> np.ndarray:
    """Keep mask: drop Gaussians on high-residual pixels in front of a surface.

    Surface depth is the nearest Gaussian on *low*-error pixels, filled into
    high-error pixels by nearest-neighbor. Deletion is capped at ``max_frac``
    of N and never leaves fewer than ``min_keep`` Gaussians.
    """
    means = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    n = int(means.shape[0])
    keep = np.ones(n, dtype=bool)
    if n <= int(min_keep):
        return keep
    residual = _resize_hw(np.asarray(residual_hw, dtype=np.float64), camera.width, camera.height)
    high = residual >= float(error_thresh)
    u, v, z, on = project_centers(means, camera, near=near)
    h, w = int(camera.height), int(camera.width)
    if surface_depth is None:
        surface = np.full((h, w), np.inf, dtype=np.float64)
        on_idx = np.flatnonzero(on)
        if on_idx.size:
            ui = np.clip(np.floor(u[on]).astype(np.int64), 0, w - 1)
            vi = np.clip(np.floor(v[on]).astype(np.int64), 0, h - 1)
            low = ~high[vi, ui]
            if np.any(low):
                np.minimum.at(surface, (vi[low], ui[low]), z[on][low])
        missing = ~np.isfinite(surface) | (surface == np.inf)
        if missing.all() or (~missing).sum() == 0:
            return keep
        if missing.any():
            _, inds = distance_transform_edt(missing, return_indices=True)
            filled = surface.copy()
            filled[missing] = surface[inds[0][missing], inds[1][missing]]
            surface = filled
    else:
        surface = _resize_hw(np.asarray(surface_depth, dtype=np.float64), w, h)
        surface = np.where(np.isfinite(surface), surface, np.inf)

    ui = np.clip(np.floor(np.nan_to_num(u, nan=0.0)).astype(np.int64), 0, w - 1)
    vi = np.clip(np.floor(np.nan_to_num(v, nan=0.0)).astype(np.int64), 0, h - 1)
    hit_high = on & high[vi, ui] & np.isfinite(surface[vi, ui])
    in_front = z < (surface[vi, ui] - float(depth_margin))
    drop = hit_high & in_front
    n_drop = int(drop.sum())
    cap = max(0, min(n_drop, int(n * float(max_frac)), n - int(min_keep)))
    if cap <= 0:
        return keep
    if n_drop > cap:
        drop_idx = np.flatnonzero(drop)
        order = np.argsort(z[drop_idx])
        drop[:] = False
        drop[drop_idx[order[:cap]]] = True
    keep[drop] = False
    if int(keep.sum()) < int(min_keep):
        return np.ones(n, dtype=bool)
    return keep


def _rotate_vec(vec: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.asarray(vec, dtype=np.float64)
    a = a / n
    v = np.asarray(vec, dtype=np.float64)
    th = np.radians(float(deg))
    c, s = np.cos(th), np.sin(th)
    return v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1.0 - c)


def micro_rotation_cameras(
    camera: Camera,
    *,
    yaw_deg: float = 12.0,
    pitch_deg: float = 8.0,
) -> list[Camera]:
    """Four nearby cameras: yaw ±``yaw_deg``, pitch ±``pitch_deg``."""
    right = np.asarray(camera.rotation[:, 0], dtype=np.float64)
    down = np.asarray(camera.rotation[:, 1], dtype=np.float64)
    forward = np.asarray(camera.rotation[:, 2], dtype=np.float64)
    up = -down
    pos = np.asarray(camera.position, dtype=np.float64)
    out: list[Camera] = []
    for delta, axis in (
        (float(yaw_deg), up),
        (-float(yaw_deg), up),
        (float(pitch_deg), right),
        (-float(pitch_deg), right),
    ):
        new_fwd = _rotate_vec(forward, axis, delta)
        n = float(np.linalg.norm(new_fwd))
        if n < 1e-8:
            continue
        out.append(
            Camera.look_at(
                pos, pos + new_fwd, up,
                width=int(camera.width),
                height=int(camera.height),
                fov_deg=float(camera.fov_deg),
            )
        )
    return out


def _radii_np(info, n: int) -> np.ndarray | None:
    if not isinstance(info, dict):
        return None
    radii = info.get("radii")
    if radii is None:
        return None
    r = radii.detach().float().cpu().numpy()
    if r.ndim == 2:
        r = r[0]
    r = np.asarray(r).reshape(-1)
    if r.shape[0] != n:
        return None
    return r


@dataclass
class GsplatGsfix3dVisPruneRepair(GsplatGsfix3dRepair):
    """Paper §3.3 plus error-mask prune and original-scene micro-rotation anchors.

    ``freeze_occluded`` is off by default. The nearest-center visual mask
    treats in-front floaters as the only updatable Gaussians, so all
    photometric error lands on them and they balloon into huge opaque
    discs. Keep the helper for opt-in wild tests; do not send it to LRZ.

    ``_apply`` is a full copy of the refine loop (not getattr hooks on the
    paper class) so GSFix3D stays the original working implementation.
    """

    freeze_occluded: bool = False
    error_prune: bool = True
    error_thresh: float = 0.12
    prune_max_frac: float = 0.02
    prune_min_keep: int = _MIN_KEEP
    depth_margin: float = 0.05
    contrib_thresh: float = 0.05
    anchor_weight: float = 0.3
    yaw_offset_deg: float = 12.0
    pitch_offset_deg: float = 8.0

    def _result_backend(self) -> str:
        return BACKEND_ID

    def apply_until(
        self,
        scene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
        *,
        should_stop=None,
        deadline: float | None = None,
        on_checkpoint=None,
    ) -> dict[str, Any]:
        """Paper chunks, but densify only on the first 20-iter pass."""
        import time

        saved = self.densify
        last: dict[str, Any] | None = None
        total_iters = 0
        l1_before = None
        chunk = 0
        limit = int(self.max_chunks)
        try:
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
                self.densify = False
                last = dict(last)
                last["n_iters"] = total_iters
                last["n_chunks"] = chunk
                last["n_stamped"] = 0
                last["l1_before"] = l1_before
                last["phase"] = "refine"
                last["backend"] = BACKEND_ID
                if on_checkpoint is not None:
                    on_checkpoint(last)
        finally:
            self.densify = saved
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
        on_progress=None,
    ) -> dict[str, Any]:
        last = super().refine_extended_dataset(scene, views, on_progress=on_progress)
        last = dict(last)
        last["backend"] = BACKEND_ID
        return last

    def _updatable_mask(self, means, camera, logit_opacities, info, torch):
        if not self.freeze_occluded:
            return None
        opacities = torch.sigmoid(logit_opacities).detach().float().cpu().numpy().reshape(-1)
        radii = _radii_np(info, int(means.shape[0]))
        mask_np = front_contributing_mask(
            means.detach().float().cpu().numpy(),
            camera,
            opacities,
            near=self.near,
            contrib_thresh=self.contrib_thresh,
            radii=radii,
        )
        return torch.from_numpy(mask_np).to(device=means.device)

    def _setup_anchors(
        self,
        gsplat,
        means,
        quats,
        f_dc,
        log_scales,
        logit_opacities,
        camera,
        device,
        torch,
        w,
        h,
        background,
        packed,
    ):
        if float(self.anchor_weight) <= 0:
            return None
        cams = micro_rotation_cameras(
            camera, yaw_deg=self.yaw_offset_deg, pitch_deg=self.pitch_offset_deg,
        )
        if not cams:
            return None
        with torch.no_grad():
            f_means = means.detach()
            f_quats = torch.nn.functional.normalize(quats.detach(), dim=-1)
            f_dc = f_dc.detach()
            f_scales = torch.exp(log_scales.detach())
            f_op = torch.sigmoid(logit_opacities.detach())
            f_colors = sh_to_rgb(f_dc)
            targets = []
            for cam in cams:
                vm = torch.from_numpy(np.asarray(cam.w2c, dtype=np.float32)).to(device).unsqueeze(0)
                k = torch.from_numpy(np.asarray(cam.intrinsics, dtype=np.float32)).to(device).unsqueeze(0)
                rgb, _ = _rasterize(
                    gsplat, f_means, f_quats, f_scales, f_op, f_colors,
                    vm, k, w, h, background, packed=packed,
                )
                targets.append((vm, k, rgb.detach()))
        return targets

    def _augment_loss(
        self,
        loss,
        gsplat,
        means,
        quats_n,
        scales,
        opacities,
        colors,
        background,
        w,
        h,
        torch,
        packed,
        anchor_state,
    ):
        if not anchor_state or float(self.anchor_weight) <= 0:
            return loss
        extra = None
        for vm, k, target in anchor_state:
            rgb_a, _ = _rasterize(
                gsplat, means, quats_n, scales, opacities, colors,
                vm, k, w, h, background, packed=packed,
            )
            term = torch.abs(rgb_a - target).mean()
            extra = term if extra is None else extra + term
        if extra is None:
            return loss
        return loss + float(self.anchor_weight) * extra / float(len(anchor_state))

    def _error_mask_prune_tensors(
        self,
        last_iter: bool,
        means,
        quats,
        f_dc,
        log_scales,
        logit_opacities,
        camera,
        rendered_rgb,
        repaired_rgb,
        torch,
    ):
        if (
            not last_iter
            or not self.error_prune
            or int(self.iters) < 5
            or int(means.shape[0]) <= int(self.prune_min_keep)
        ):
            return means, quats, f_dc, log_scales, logit_opacities, 0
        residual = rgb_l1_residual(rendered_rgb, repaired_rgb)
        keep_np = error_mask_keep(
            means.detach().float().cpu().numpy(),
            camera,
            residual,
            error_thresh=self.error_thresh,
            max_frac=self.prune_max_frac,
            min_keep=self.prune_min_keep,
            near=self.near,
            depth_margin=self.depth_margin,
        )
        n_drop = int((~keep_np).sum())
        if n_drop == 0:
            return means, quats, f_dc, log_scales, logit_opacities, 0
        keep = torch.from_numpy(keep_np).to(device=means.device)
        logger.info("GSFix3D vis-prune: dropped %d floater gaussians", n_drop)
        return (
            means[keep], quats[keep], f_dc[keep],
            log_scales[keep], logit_opacities[keep], n_drop,
        )

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
        n_error_pruned = 0
        anchor_state = self._setup_anchors(
            gsplat, means, quats, f_dc, log_scales, logit_opacities,
            camera, device, torch, w, h, background, self.packed,
        )

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
            loss = self._augment_loss(
                loss, gsplat, means, quats_n, scales, opacities, colors,
                background, w, h, torch, self.packed, anchor_state,
            )
            loss.backward()

            updatable = self._updatable_mask(
                means, camera, logit_opacities, info, torch,
            )
            apply_updatable_grads(
                (means, quats, f_dc, log_scales, logit_opacities), updatable,
            )

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
                if updatable is not None and updatable.shape[0] == vis.shape[0]:
                    vis = vis & updatable
                contrib = vis_norm
                if updatable is not None:
                    contrib = vis_norm * vis.to(dtype=vis_norm.dtype)
                xyz_grad_accum = xyz_grad_accum + contrib
                xyz_grad_denom = xyz_grad_denom + vis.to(xyz_grad_accum.dtype)
            elif it == 0:
                logger.warning(
                    "GSFix3D vis-prune densify: no means2d screen grads "
                    "(absgrad/retain_grad). Clone+split will not spawn."
                )

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
                        "GSFix3D vis-prune densify: 0 spawned "
                        "(max |grad2d|=%.6f, %d above %.6f)",
                        float(avg.max().item()) if avg.numel() else 0.0,
                        n_high,
                        float(self.densify_grad_thresh),
                    )

            if last_iter and self.densify and means.shape[0] > 32:
                with torch.no_grad():
                    keep = torch.sigmoid(logit_opacities).reshape(-1) > float(self.prune_opacity)
                    if int(keep.sum()) >= 32 and int((~keep).sum()) > 0:
                        means, quats, f_dc, log_scales, logit_opacities = (
                            means[keep], quats[keep], f_dc[keep],
                            log_scales[keep], logit_opacities[keep],
                        )

            means, quats, f_dc, log_scales, logit_opacities, n_drop = (
                self._error_mask_prune_tensors(
                    last_iter, means, quats, f_dc, log_scales, logit_opacities,
                    camera, rendered_rgb, repaired_rgb, torch,
                )
            )
            n_error_pruned += int(n_drop)

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
        scene.opacities = opacities.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        scene.colors = torch.clamp(colors, 0.0, 1.0).detach().float().cpu().numpy().astype(np.float32)

        n1 = scene.num_gaussians
        logger.info(
            "GSFix3D vis-prune refine: %d iters, L1 %.4f -> %.4f, %d -> %d gaussians (+%d)",
            self.iters, l1_before, l1_after, n0, n1, n_spawned,
        )
        return {
            "backend": self._result_backend(),
            "n_visible": int(n_visible),
            "n_updated": n1,
            "n_stamped": 0,
            "n_spawned": int(max(0, n1 - n0) if n_spawned == 0 else n_spawned),
            "n_gaussians": n1,
            "n_iters": int(self.iters),
            "n_error_pruned": int(n_error_pruned),
            "l1_before": round(l1_before, 6),
            "l1_after": round(l1_after, 6),
            "render_rgb": render_rgb,
        }
