"""Experimental GSFix3D lift: freeze occluded Gaussians, prune floaters.

Sibling of ``GsplatGsfix3dRepair`` (backend ``gsfix-gsplat-visprune``).
The paper path in ``repair_gsfix3d.py`` stays the research default.

Per 20-iter chunk, on top of §3.3 photometric refine::

    freeze Adam on Gaussians that are out of frustum or behind the nearest
        opaque center at their pixel
    densify clone+split only from that updatable set
    last iter of a full chunk: opacity prune, then error-mask floater prune
    extra L1 vs original-scene renders at four yaw/pitch micro-rotations

``apply_until`` densifies only the first chunk so a focused 1h run cannot
keep spawning ray-streaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .rendering.base import Camera
from .repair_gsfix import _rasterize
from .repair_gsfix3d import GsplatGsfix3dRepair, sh_to_rgb

logger = logging.getLogger(__name__)

BACKEND_ID = "gsfix-gsplat-visprune"
_MIN_KEEP = 32


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
    """Paper §3.3 plus occlusion freeze, error-mask prune, and original anchors."""

    freeze_occluded: bool = True
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

    def _updatable_mask(self, means, camera, logit_opacities, info, torch):
        if not self.freeze_occluded:
            return None
        opacities = torch.sigmoid(logit_opacities).detach().float().cpu().numpy()
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
