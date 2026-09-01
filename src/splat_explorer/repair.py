"""View-local 3D Gaussian repair after a repaired RGB image comes back.

The live renderer keeps exploring the original reconstruction. The first
repair copies that scene into the episode directory as `scene_original.ply`
(never written again) and a working `scene_repaired.ply` that accumulates
later views.

`make_repair_backend()` picks the lift: CUDA+gsplat (GSFix3D refine), else
Apple Silicon gsplat-mlx (same photometric loop on Metal), else
`ProjectedViewRepair` (CPU color stamp at existing depths).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, uniform_filter

from .rendering.base import Camera
from .scene import GaussianScene, save_ply

logger = logging.getLogger(__name__)

ORIGINAL_PLY = "scene_original.ply"
REPAIRED_PLY = "scene_repaired.ply"
REPAIR_LOG = "repair_log.json"

# Matches the 3DGS / GSFix3D photometric mix (Kerbl et al. use λ ≈ 0.2).
DEFAULT_LAMBDA_SSIM = 0.2
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def repair_meta_name(step: int) -> str:
    return f"step_{int(step):03d}_repair.json"


def repaired_render_name(step: int) -> str:
    return f"step_{int(step):03d}_repaired_render.png"


_BACKEND_ALIASES = {
    "auto": "auto",
    "detect": "auto",
    "mlx": "gsplat-mlx",
    "gsplat-mlx": "gsplat-mlx",
    "apple": "gsplat-mlx",
    "cuda": "gsfix-gsplat",
    "gsplat": "gsfix-gsplat",
    "gsfix": "gsfix-gsplat",
    "gsfix-gsplat": "gsfix-gsplat",
    "cpu": "cpu-project",
    "cpu-project": "cpu-project",
    "project": "cpu-project",
    "cpu-photometric": "cpu-photometric",
    "photometric": "cpu-photometric",
}


def _in_docker() -> bool:
    return Path("/.dockerenv").is_file()


def _cuda_available() -> bool:
    try:
        from .repair_gsfix import gsplat_refine_available
        return bool(gsplat_refine_available())
    except Exception:
        return False


def _mlx_available() -> bool:
    try:
        from .repair_mlx import mlx_refine_available
        return bool(mlx_refine_available())
    except Exception:
        return False


def detect_repair_backend() -> str:
    """Id that `auto` would pick in this process."""
    if _cuda_available():
        return "gsfix-gsplat"
    if _mlx_available():
        return "gsplat-mlx"
    return "cpu-project"


def list_repair_backends() -> dict:
    """Catalog for the repair-studio dropdown (availability is process-local)."""
    cuda = _cuda_available()
    mlx = _mlx_available()
    docker = _in_docker()
    if mlx:
        mlx_detail = "Metal / MLX differentiable refine"
    elif docker:
        mlx_detail = (
            "Not visible in Linux Docker (no Metal). "
            "scripts/start.sh runs the dashboard on the Mac host for gsplat-mlx."
        )
    else:
        mlx_detail = "Install with: pip install -e '.[apple]' (Apple Silicon + MLX)"
    cuda_detail = (
        "CUDA + gsplat photometric refine"
        if cuda
        else "Needs an NVIDIA GPU and pip install -e '.[gpu]'"
    )
    detected = detect_repair_backend()
    return {
        "detected": detected,
        "docker": docker,
        "backends": [
            {
                "id": "auto",
                "label": f"Auto ({detected})",
                "available": True,
                "detail": f"Picks the best stack in this process: {detected}",
            },
            {
                "id": "gsplat-mlx",
                "label": "gsplat-mlx (Apple Silicon / Metal)",
                "available": mlx,
                "detail": mlx_detail,
            },
            {
                "id": "gsfix-gsplat",
                "label": "gsplat CUDA (GSFix3D)",
                "available": cuda,
                "detail": cuda_detail,
            },
            {
                "id": "cpu-project",
                "label": "CPU color stamp",
                "available": True,
                "detail": "Fast stand-in: paint the regen PNG onto existing depths. No backprop.",
            },
            {
                "id": "cpu-photometric",
                "label": "CPU photometric (no rasterizer)",
                "available": True,
                "detail": "Residual-weighted color step at gaussian centers. No 3DGS rasterizer.",
            },
        ],
    }


def _normalize_backend_name(name: str | None) -> str:
    key = str(name or "auto").strip().lower()
    if key not in _BACKEND_ALIASES:
        known = ", ".join(sorted(set(_BACKEND_ALIASES.values())))
        raise ValueError(f"Unknown repair backend {name!r}. Choose one of: {known}")
    return _BACKEND_ALIASES[key]


def make_repair_backend(name: str | None = "auto", *, studio: bool = False):
    """Build a lift backend.

    `name='auto'` detects CUDA gsplat, then Apple Silicon gsplat-mlx, then the
    CPU color stamp. An explicit name must be available here — it will not
    silently fall back (the dashboard uses that to show a real error).

    `studio=True` uses a heavier gsplat-mlx preset (higher training resolution
    / more gaussians) for dashboard replays. Live episodes keep the lighter
    defaults so the harness is not blocked for minutes per view.
    """
    key = _normalize_backend_name(name)
    if key == "auto":
        key = detect_repair_backend()
        logger.info("3D repair backend (auto): %s", key)
        return _build_repair_backend(key, studio=studio, required=False)
    return _build_repair_backend(key, studio=studio, required=True)


def _build_repair_backend(key: str, *, studio: bool, required: bool):
    if key == "gsfix-gsplat":
        from .repair_gsfix import GsplatPhotometricRepair, gsplat_refine_available

        if not gsplat_refine_available():
            msg = (
                "gsplat CUDA refine is not available in this process "
                "(needs NVIDIA GPU + pip install -e '.[gpu]')."
            )
            if required:
                raise RuntimeError(msg)
            logger.info(msg)
        else:
            logger.info("3D repair backend: GSFix refine (gsplat / CUDA)")
            return GsplatPhotometricRepair()
    if key == "gsplat-mlx":
        from .repair_mlx import MlxPhotometricRepair, mlx_refine_available

        if not mlx_refine_available():
            extra = (
                " Linux Docker cannot use Metal — run the host dashboard via scripts/start.sh."
                if _in_docker()
                else " Install with: pip install -e '.[apple]'."
            )
            msg = "gsplat-mlx is not available in this process." + extra
            if required:
                raise RuntimeError(msg)
            logger.info(msg)
        else:
            kwargs = {}
            if studio:
                kwargs = dict(iters=20, max_train_side=256, max_opt_gaussians=8192)
            logger.info(
                "3D repair backend: GSFix refine (gsplat-mlx / Apple Silicon)%s",
                " [studio]" if studio else "",
            )
            return MlxPhotometricRepair(**kwargs)
    if key == "cpu-photometric":
        logger.info("3D repair backend: CPU photometric (no rasterizer)")
        return PhotometricViewRepair()
    logger.info(
        "3D repair backend: CPU projection lift. "
        "Stamps the diffusion image onto the splat at existing depths."
    )
    return ProjectedViewRepair()


def copy_camera(camera: Camera) -> Camera:
    return Camera(
        position=np.array(camera.position, dtype=np.float32, copy=True),
        rotation=np.array(camera.rotation, dtype=np.float32, copy=True),
        width=int(camera.width),
        height=int(camera.height),
        fov_deg=float(camera.fov_deg),
    )


def scene_from_renderer(renderer) -> GaussianScene | None:
    """GaussianScene on cpu_splats / cpu_points / gsplat, or viser's CPU twin."""
    scene = getattr(renderer, "scene", None)
    if isinstance(scene, GaussianScene):
        return scene
    cpu = getattr(renderer, "_cpu", None)
    scene = getattr(cpu, "scene", None)
    return scene if isinstance(scene, GaussianScene) else None


def visible_gaussians(
    scene: GaussianScene,
    camera: Camera,
    near: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gaussians whose projected centers fall inside this camera frustum.

    Returns (indices, u, v, z) for the visible subset. u/v are pixel coords
    in the OpenCV convention of `Camera` (x right, y down, z forward).
    """
    w2c = camera.w2c.astype(np.float32)
    pcam = scene.means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pcam[:, 2]
    keep = z > float(near)
    if not np.any(keep):
        empty = np.empty(0, dtype=np.int64)
        z0 = np.empty(0, dtype=np.float32)
        return empty, z0, z0, z0
    z_kept = z[keep]
    uf = camera.fx * pcam[keep, 0] / z_kept + camera.width / 2.0
    vf = camera.fy * pcam[keep, 1] / z_kept + camera.height / 2.0
    on = (uf >= 0.0) & (uf < camera.width) & (vf >= 0.0) & (vf < camera.height)
    idx = np.flatnonzero(keep)[on]
    return idx, uf[on].astype(np.float32), vf[on].astype(np.float32), z_kept[on]


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _to_float(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def _sample_rgb(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, 3) float image at pixel coords (u, v)."""
    h, w = image.shape[:2]
    u = np.clip(u.astype(np.float32), 0.0, w - 1.001)
    v = np.clip(v.astype(np.float32), 0.0, h - 1.001)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    su = (u - u0).astype(np.float32)[:, None]
    sv = (v - v0).astype(np.float32)[:, None]
    c00 = image[v0, u0]
    c01 = image[v0, u1]
    c10 = image[v1, u0]
    c11 = image[v1, u1]
    return (1.0 - sv) * ((1.0 - su) * c00 + su * c01) + sv * ((1.0 - su) * c10 + su * c11)


def _unproject(camera: Camera, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (u - camera.width / 2.0) * z / camera.fx
    y = (v - camera.height / 2.0) * z / camera.fy
    p_cam = np.stack([x, y, np.broadcast_to(z, u.shape)], axis=-1).astype(np.float32)
    return p_cam @ camera.rotation.T + camera.position.astype(np.float32)


class RepairBackend(Protocol):
    """Mutate `scene` for gaussians visible in this view. Return a stats dict."""

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        ...


@dataclass
class PhotometricViewRepair:
    """GSFix3D-inspired photometric lift, CPU, view-local.

    No differentiable rasterizer: each visible gaussian takes a residual-weighted
    step toward the repaired pixel at its projected center (the gradient of Lpho
    at that pixel w.r.t. the gaussian's color). Adaptive density control fills
    previously empty / under-populated pixels with small isotropic gaussians.
    """

    lambda_ssim: float = DEFAULT_LAMBDA_SSIM
    color_lr: float = 0.55
    near: float = 0.05
    ssim_window: int = 7
    densify: bool = True
    max_new: int = 128
    hole_stride: int = 12
    hole_l1: float = 0.12
    new_scale: float = 0.03
    new_opacity: float = 0.55
    seed: int = 0

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        h, w = int(camera.height), int(camera.width)
        rendered = _to_float(_resize_rgb(rendered_rgb, w, h))
        repaired = _to_float(_resize_rgb(repaired_rgb, w, h))
        l1_before = float(np.mean(np.abs(repaired - rendered)))

        idx, uf, vf, z = visible_gaussians(scene, camera, near=self.near)
        n_visible = int(len(idx))
        n_updated = 0
        n_spawned = 0

        if n_visible:
            n_updated = self._update_colors(scene, idx, uf, vf, rendered, repaired)

        if self.densify:
            n_spawned = self._densify(scene, camera, idx, uf, vf, z, rendered, repaired)

        return {
            "backend": "cpu-photometric",
            "n_visible": n_visible,
            "n_updated": n_updated,
            "n_spawned": n_spawned,
            "n_gaussians": scene.num_gaussians,
            "l1_before": round(l1_before, 6),
        }

    def _update_colors(
        self,
        scene: GaussianScene,
        idx: np.ndarray,
        uf: np.ndarray,
        vf: np.ndarray,
        rendered: np.ndarray,
        repaired: np.ndarray,
    ) -> int:
        k = max(3, int(self.ssim_window) | 1)
        mu_r = uniform_filter(rendered, size=(k, k, 1), mode="nearest")
        mu_f = uniform_filter(repaired, size=(k, k, 1), mode="nearest")
        residual = (1.0 - self.lambda_ssim) * (repaired - rendered) + self.lambda_ssim * (
            mu_f - mu_r
        )
        err = np.mean(np.abs(repaired - rendered), axis=2)
        target = _sample_rgb(repaired, uf, vf)
        step = _sample_rgb(residual, uf, vf)
        err_i = np.clip(_sample_rgb(err[:, :, None], uf, vf)[:, 0] / 0.30, 0.15, 1.0)
        lr = (self.color_lr * err_i).astype(np.float32)[:, None]
        # Residual-weighted blend toward I_fixed, plus an explicit Lpho step.
        updated = (1.0 - lr) * scene.colors[idx] + lr * target + 0.25 * lr * step
        scene.colors[idx] = np.clip(updated, 0.0, 1.0)
        return int(len(idx))

    def _densify(
        self,
        scene: GaussianScene,
        camera: Camera,
        idx: np.ndarray,
        uf: np.ndarray,
        vf: np.ndarray,
        z: np.ndarray,
        rendered: np.ndarray,
        repaired: np.ndarray,
    ) -> int:
        h, w = rendered.shape[:2]
        err = np.mean(np.abs(repaired - rendered), axis=2)
        occupied = np.zeros((h, w), dtype=bool)
        if len(idx):
            u_i = np.clip(np.round(uf).astype(np.int32), 0, w - 1)
            v_i = np.clip(np.round(vf).astype(np.int32), 0, h - 1)
            occupied[v_i, u_i] = True
            occupied = binary_dilation(occupied, iterations=2)
        lum = repaired.mean(axis=2)
        holes = (~occupied) & (err > self.hole_l1) & (lum > 0.04)
        if self.hole_stride > 1:
            grid = np.zeros_like(holes)
            grid[:: self.hole_stride, :: self.hole_stride] = True
            holes &= grid
        ys, xs = np.nonzero(holes)
        if len(xs) == 0:
            return 0
        if len(xs) > self.max_new:
            rng = np.random.default_rng(self.seed)
            pick = rng.choice(len(xs), size=self.max_new, replace=False)
            ys, xs = ys[pick], xs[pick]
        default_z = float(np.median(z)) if len(z) else 2.0
        new_z = np.full(len(xs), default_z, dtype=np.float32)
        u_new = xs.astype(np.float32)
        v_new = ys.astype(np.float32)
        means = _unproject(camera, u_new, v_new, new_z)
        colors = repaired[ys, xs].astype(np.float32)
        n = len(xs)
        scales = np.full((n, 3), self.new_scale, dtype=np.float32)
        quats = np.tile(_IDENTITY_QUAT, (n, 1))
        opacities = np.full((n,), self.new_opacity, dtype=np.float32)
        scene.means = np.concatenate([scene.means, means], axis=0)
        scene.scales = np.concatenate([scene.scales, scales], axis=0)
        scene.quats = np.concatenate([scene.quats, quats], axis=0)
        scene.opacities = np.concatenate([scene.opacities, opacities], axis=0)
        scene.colors = np.concatenate([scene.colors, colors], axis=0)
        return n


def _scatter_depth(
    height: int,
    width: int,
    uf: np.ndarray,
    vf: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor depth from projected gaussian centers (closer wins)."""
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if len(z) == 0:
        return depth
    order = np.argsort(-z.astype(np.float64))
    u_i = np.clip(np.round(uf[order]).astype(np.int32), 0, width - 1)
    v_i = np.clip(np.round(vf[order]).astype(np.int32), 0, height - 1)
    depth[v_i, u_i] = z[order].astype(np.float32)
    missing = ~np.isfinite(depth)
    if np.any(~missing) and np.any(missing):
        _, (iy, ix) = distance_transform_edt(missing, return_indices=True)
        depth[missing] = depth[iy[missing], ix[missing]]
    return depth


def _point_preview(scene: GaussianScene, camera: Camera, near: float = 0.05) -> np.ndarray:
    """Cheap z-buffer of gaussian centers — the 3D-after still for the compare UI."""
    h, w = int(camera.height), int(camera.width)
    img = np.zeros((h, w, 3), dtype=np.float32)
    depth = np.full((h, w), np.inf, dtype=np.float32)
    idx, uf, vf, z = visible_gaussians(scene, camera, near=near)
    if len(idx) == 0:
        return (img * 255).astype(np.uint8)
    order = np.argsort(-z.astype(np.float64))
    u_i = np.clip(np.round(uf[order]).astype(np.int32), 0, w - 1)
    v_i = np.clip(np.round(vf[order]).astype(np.int32), 0, h - 1)
    z_o = z[order]
    col = scene.colors[idx[order]]
    closer = z_o < depth[v_i, u_i]
    depth[v_i[closer], u_i[closer]] = z_o[closer]
    img[v_i[closer], u_i[closer]] = col[closer]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


@dataclass
class ProjectedViewRepair:
    """CPU substitute for GSFix refine: paint the diffusion image into the splat.

    No backprop. Uses the current splat as a depth map, overwrites colors of
    gaussians whose centers fall in the view, fades those that disagree with
    the repair, and injects a dense layer of small gaussians at the repaired
    pixels so Original vs Repaired is a real 3D change.
    """

    near: float = 0.05
    color_lr: float = 1.0
    fade: float = 0.75
    stride: int = 3
    max_new: int = 12000
    error_floor: float = 0.04
    new_opacity: float = 0.9
    seed: int = 0

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        h, w = int(camera.height), int(camera.width)
        rendered = _to_float(_resize_rgb(rendered_rgb, w, h))
        repaired = _to_float(_resize_rgb(repaired_rgb, w, h))
        l1_before = float(np.mean(np.abs(repaired - rendered)))
        err = np.mean(np.abs(repaired - rendered), axis=2)

        idx, uf, vf, z = visible_gaussians(scene, camera, near=self.near)
        n_visible = int(len(idx))
        n_updated = 0
        if n_visible:
            n_updated = self._paint_centers(scene, idx, uf, vf, repaired, err)

        n_spawned = self._inject_layer(scene, camera, idx, uf, vf, z, repaired, err)
        preview = _point_preview(scene, camera, near=self.near)
        l1_after = float(np.mean(np.abs(_to_float(preview) - repaired)))

        return {
            "backend": "cpu-project",
            "n_visible": n_visible,
            "n_updated": n_updated,
            "n_spawned": n_spawned,
            "n_gaussians": scene.num_gaussians,
            "n_iters": 1,
            "l1_before": round(l1_before, 6),
            "l1_after": round(l1_after, 6),
            "render_rgb": preview,
        }

    def _paint_centers(self, scene, idx, uf, vf, repaired, err) -> int:
        target = _sample_rgb(repaired, uf, vf)
        scene.colors[idx] = (1.0 - self.color_lr) * scene.colors[idx] + self.color_lr * target
        scene.colors[idx] = np.clip(scene.colors[idx], 0.0, 1.0)
        err_i = np.clip(_sample_rgb(err[:, :, None], uf, vf)[:, 0], 0.0, 1.0)
        scene.opacities[idx] = np.clip(
            scene.opacities[idx] * (1.0 - self.fade * err_i), 0.02, 0.995,
        )
        return int(len(idx))

    def _inject_layer(
        self, scene, camera, idx, uf, vf, z, repaired, err,
    ) -> int:
        h, w = repaired.shape[:2]
        depth = _scatter_depth(h, w, uf, vf, z)
        if not np.any(np.isfinite(depth)):
            depth[:] = 2.0
        stride = max(1, int(self.stride))
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        ys, xs = ys.ravel(), xs.ravel()
        keep = err[ys, xs] >= float(self.error_floor)
        ys, xs = ys[keep], xs[keep]
        if len(xs) == 0:
            return 0
        scores = err[ys, xs]
        if len(xs) > int(self.max_new):
            pick = np.argpartition(scores, -int(self.max_new))[-int(self.max_new):]
            ys, xs = ys[pick], xs[pick]
        z_pix = depth[ys, xs]
        valid = np.isfinite(z_pix) & (z_pix > float(self.near))
        ys, xs, z_pix = ys[valid], xs[valid], z_pix[valid]
        if len(xs) == 0:
            return 0
        u_new = xs.astype(np.float32)
        v_new = ys.astype(np.float32)
        means = _unproject(camera, u_new, v_new, z_pix)
        colors = repaired[ys, xs].astype(np.float32)
        pixel_m = (float(stride) * z_pix / max(camera.fx, 1e-6)).astype(np.float32)
        scales = np.clip(pixel_m, 0.008, 0.12)[:, None] * np.ones((len(xs), 3), np.float32)
        n = len(xs)
        quats = np.tile(_IDENTITY_QUAT, (n, 1))
        opacities = np.full((n,), self.new_opacity, dtype=np.float32)
        scene.means = np.concatenate([scene.means, means], axis=0)
        scene.scales = np.concatenate([scene.scales, scales], axis=0)
        scene.quats = np.concatenate([scene.quats, quats], axis=0)
        scene.opacities = np.concatenate([scene.opacities, opacities], axis=0)
        scene.colors = np.concatenate([scene.colors, colors], axis=0)
        return n


@dataclass
class RepairResult:
    """Outcome of lifting one repaired RGB view back into 3DGS."""

    step: int
    status: str  # ok | skipped | error
    seconds: float = 0.0
    n_visible: int = 0
    n_updated: int = 0
    n_spawned: int = 0
    n_gaussians: int = 0
    l1_before: float = 0.0
    l1_after: float | None = None
    n_iters: int = 0
    backend: str | None = None
    train_width: int | None = None
    train_height: int | None = None
    render_name: str | None = None
    original_ply: str | None = None
    repaired_ply: str | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "step": self.step,
            "status": self.status,
            "seconds": round(self.seconds, 3),
            "n_visible": self.n_visible,
            "n_updated": self.n_updated,
            "n_updated_gaussians": self.n_updated,
            "n_spawned": self.n_spawned,
            "n_gaussians": self.n_gaussians,
            "n_iters": self.n_iters,
            "l1_before": self.l1_before,
            "l1_after": self.l1_after,
            "backend": self.backend,
            "train_width": self.train_width,
            "train_height": self.train_height,
            "render_name": self.render_name,
            "original_ply": self.original_ply,
            "repaired_ply": self.repaired_ply,
            "error": self.error,
        }


def _write_json(path: Path, body: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(path)


@dataclass
class SceneRepairer:
    """Copy-on-write owner of the repaired 3DGS working set.

    `source` is never mutated (the live renderer / original file stay as-is).
    `backend` defaults to `PhotometricViewRepair` and is the swap point for
    later CUDA/gsplat or paper-faithful variants.
    """

    source: GaussianScene
    backend: RepairBackend | None = None
    _working: GaussianScene | None = field(default=None, init=False, repr=False)
    _pending: dict[int, tuple[Camera, Path, Path]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    results: list[RepairResult] = field(default_factory=list)
    original_ply: Path | None = None
    repaired_ply: Path | None = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = make_repair_backend()

    def make_callback(self, step: int, camera: Camera, source_rgb: Path, episode_dir: Path):
        """Snapshot the view now; return an `on_done` for Regenerator.submit."""
        with self._lock:
            self._pending[int(step)] = (
                copy_camera(camera), Path(source_rgb), Path(episode_dir),
            )

        def on_done(result) -> None:
            self.apply_from_regenerate(result)

        return on_done

    def apply_from_regenerate(self, result) -> RepairResult:
        """Called on the regenerate worker thread as soon as pixels are saved."""
        step = int(getattr(result, "step", -1))
        status = str(getattr(result, "status", "") or "")
        image_name = getattr(result, "image_name", None)
        with self._lock:
            pending = self._pending.get(step)
        if pending is None:
            out = RepairResult(step=step, status="skipped", error="no camera snapshot for this step")
            self._record(None, out)
            return out
        camera, source_rgb, episode_dir = pending
        if status != "ok" or not image_name:
            out = RepairResult(step=step, status="skipped", error=f"regenerate status={status!r}")
            self._record(episode_dir, out)
            return out
        return self.apply_view(
            step=step,
            camera=camera,
            rendered_path=source_rgb,
            repaired_path=episode_dir / str(image_name),
            episode_dir=episode_dir,
        )

    def apply_view(
        self,
        *,
        step: int,
        camera: Camera,
        rendered_path: Path,
        repaired_path: Path,
        episode_dir: Path,
    ) -> RepairResult:
        t0 = time.perf_counter()
        out = RepairResult(step=step, status="error")
        try:
            rendered = _load_rgb(Path(rendered_path))
            repaired = _load_rgb(Path(repaired_path))
            with self._lock:
                working = self._ensure_working(Path(episode_dir))
                backend = self.backend or PhotometricViewRepair()
                stats = backend.apply(working, camera, rendered, repaired)
                repaired_path_out = Path(episode_dir) / REPAIRED_PLY
                save_ply(working, repaired_path_out)
                self.repaired_ply = repaired_path_out
                render_rgb = stats.get("render_rgb")
                render_name = None
                if render_rgb is not None:
                    render_name = repaired_render_name(step)
                    Image.fromarray(np.asarray(render_rgb, dtype=np.uint8)).save(
                        Path(episode_dir) / render_name,
                    )
                out.status = "ok"
                out.n_visible = int(stats.get("n_visible", 0))
                out.n_updated = int(stats.get("n_updated", 0))
                out.n_spawned = int(stats.get("n_spawned", 0))
                out.n_gaussians = int(stats.get("n_gaussians", working.num_gaussians))
                out.n_iters = int(stats.get("n_iters") or 0)
                out.l1_before = float(stats.get("l1_before", 0.0))
                after = stats.get("l1_after")
                out.l1_after = None if after is None else float(after)
                out.backend = stats.get("backend")
                tw, th = stats.get("train_width"), stats.get("train_height")
                out.train_width = None if tw is None else int(tw)
                out.train_height = None if th is None else int(th)
                out.render_name = render_name
                out.original_ply = ORIGINAL_PLY
                out.repaired_ply = REPAIRED_PLY
            out.seconds = time.perf_counter() - t0
            logger.info(
                "Repair step %d [%s]: %d visible, %d spawned, L1=%.4f%s -> %s (%.1fs)",
                step, out.backend or "?", out.n_visible, out.n_spawned, out.l1_before,
                f"/{out.l1_after:.4f}" if out.l1_after is not None else "",
                REPAIRED_PLY, out.seconds,
            )
        except Exception as exc:
            out.seconds = time.perf_counter() - t0
            out.status = "error"
            out.error = f"{type(exc).__name__}: {exc}"
            logger.exception("3DGS repair step %d failed", step)
        self._record(Path(episode_dir), out)
        return out

    def _ensure_working(self, episode_dir: Path) -> GaussianScene:
        if self._working is None:
            self._working = self.source.copy()
            original = episode_dir / ORIGINAL_PLY
            save_ply(self._working, original)
            self.original_ply = original
            logger.info(
                "Copied %d gaussians to %s (original asset left untouched)",
                self._working.num_gaussians, original,
            )
        return self._working

    def _record(self, episode_dir: Path | None, result: RepairResult) -> None:
        with self._lock:
            self.results.append(result)
            snapshot = [r.to_json() for r in self.results]
            original = ORIGINAL_PLY if self.original_ply else None
            repaired = REPAIRED_PLY if self.repaired_ply else None
        if episode_dir is None:
            return
        _write_json(episode_dir / repair_meta_name(result.step), result.to_json())
        _write_json(
            episode_dir / REPAIR_LOG,
            {
                "original_ply": original,
                "repaired_ply": repaired,
                "repairs": snapshot,
            },
        )

    def reset(self) -> None:
        """Drop the working copy so the next apply starts from `source` again."""
        with self._lock:
            self._working = None
            self.results.clear()
            self.original_ply = None
            self.repaired_ply = None
            self._pending.clear()


def regen_png_name(step: int) -> str:
    return f"step_{int(step):03d}_regen.png"


def camera_from_record(
    record: dict,
    *,
    up_axis: str,
    width: int,
    height: int,
    fov_deg: float,
) -> Camera:
    """Rebuild the OpenCV camera that produced this episode step."""
    from .agent.camera_rig import CameraRig

    rig = CameraRig(
        np.asarray(record["position"], dtype=np.float64),
        up_axis=up_axis,
        yaw_deg=float(record.get("yaw_deg") or 0.0),
        pitch_deg=float(record.get("pitch_deg") or 0.0),
    )
    return rig.camera(int(width), int(height), float(fov_deg))


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return int(img.size[0]), int(img.size[1])
    except OSError:
        return None


def discover_repair_views(episode_dir: Path, meta: dict | None = None) -> list[dict]:
    """Episode steps that have a regenerated RGB PNG plus a camera pose.

    Does not require a live episode or an existing 3D repair — only the
    diffusion-fixed image next to the original render.
    """
    episode_dir = Path(episode_dir)
    meta = meta if meta is not None else {}
    try:
        meta = meta or json.loads((episode_dir / "meta.json").read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        meta = meta or {}
    params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
    default_w = int(params.get("width") or 960)
    default_h = int(params.get("height") or 720)
    fov_deg = float(params.get("fov_deg") or 75.0)
    views: list[dict] = []
    actions_path = episode_dir / "actions.jsonl"
    if not actions_path.is_file():
        return views
    try:
        lines = actions_path.read_text().splitlines()
    except OSError:
        return views
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            step = int(rec.get("step", -1))
        except (TypeError, ValueError):
            continue
        if step < 0 or "position" not in rec:
            continue
        rendered_name = rec.get("frame") or f"step_{step:03d}.png"
        rendered_path = episode_dir / rendered_name
        repaired_name = rec.get("regenerate_frame") or regen_png_name(step)
        repaired_path = episode_dir / repaired_name
        if not repaired_path.is_file():
            continue
        size = _image_size(rendered_path) or (default_w, default_h)
        repair_meta = {}
        rp = episode_dir / repair_meta_name(step)
        if rp.is_file():
            try:
                repair_meta = json.loads(rp.read_text())
            except (OSError, json.JSONDecodeError):
                repair_meta = {}
        views.append({
            "step": step,
            "rendered_name": rendered_path.name,
            "repaired_name": repaired_path.name,
            "rendered_path": str(rendered_path),
            "repaired_path": str(repaired_path),
            "position": rec["position"],
            "yaw_deg": float(rec.get("yaw_deg") or 0.0),
            "pitch_deg": float(rec.get("pitch_deg") or 0.0),
            "pose": rec.get("pose"),
            "width": int(size[0]),
            "height": int(size[1]),
            "fov_deg": fov_deg,
            "repair_status": repair_meta.get("status"),
            "repair": repair_meta or None,
            "lift_name": repaired_render_name(step) if (episode_dir / repaired_render_name(step)).is_file() else None,
        })
    return views


def reload_repair_module():
    """Re-import this module so replay picks up in-place code edits."""
    import importlib
    import sys

    for extra in ("splat_explorer.repair_gsfix", "splat_explorer.repair_mlx"):
        if extra in sys.modules:
            importlib.reload(sys.modules[extra])
    name = __name__
    mod = sys.modules[name]
    return importlib.reload(mod)


def replay_episode_repairs(
    source: GaussianScene,
    episode_dir: Path,
    views: list[dict] | None = None,
    *,
    up_axis: str = "+y",
    fov_deg: float | None = None,
    backend: RepairBackend | None = None,
    on_view=None,
    should_stop=None,
) -> list[RepairResult]:
    """Run 3D repair on each regenerated view, in order, from a fresh copy.

    `source` is not mutated. Each successful view updates `scene_repaired.ply`.
    `on_view(index, view, result)` fires after every view (including failures).
    """
    episode_dir = Path(episode_dir)
    if views is None:
        views = discover_repair_views(episode_dir)
    if backend is None:
        backend = make_repair_backend()
    repairer = SceneRepairer(source, backend=backend)
    results: list[RepairResult] = []
    for i, view in enumerate(views):
        if should_stop is not None and should_stop():
            break
        width = int(view.get("width") or 960)
        height = int(view.get("height") or 720)
        fov = float(fov_deg if fov_deg is not None else view.get("fov_deg") or 75.0)
        camera = camera_from_record(
            view, up_axis=up_axis, width=width, height=height, fov_deg=fov,
        )
        result = repairer.apply_view(
            step=int(view["step"]),
            camera=camera,
            rendered_path=Path(view["rendered_path"]),
            repaired_path=Path(view["repaired_path"]),
            episode_dir=episode_dir,
        )
        results.append(result)
        if on_view is not None:
            on_view(i, view, result)
    return results
