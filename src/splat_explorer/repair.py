"""View-local 3D Gaussian repair after a repaired RGB image comes back.

The live renderer keeps exploring the original reconstruction. The first
repair copies that scene into the episode directory as `scene_original.ply`
(never written again) and a working `scene_repaired.ply` that accumulates
later views.

Only gaussians visible in the repaired camera are updated. The default
backend is a CPU stand-in for GSFix3D §3.3 (photometric lift of the
diffusion-fixed image back into 3DGS) that does not need CUDA / gsplat:

  Lpho = (1 − λ) ||I_fixed − I_gs||_1 + λ L_local-mean(I_fixed, I_gs)

plus a sparse adaptive-density pass that spawns small gaussians in holes.
Swap the backend by passing any `RepairBackend` to `SceneRepairer`.
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
from scipy.ndimage import binary_dilation, uniform_filter

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
            "l1_before": self.l1_before,
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
            self.backend = PhotometricViewRepair()

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
                out.status = "ok"
                out.n_visible = int(stats.get("n_visible", 0))
                out.n_updated = int(stats.get("n_updated", 0))
                out.n_spawned = int(stats.get("n_spawned", 0))
                out.n_gaussians = int(stats.get("n_gaussians", working.num_gaussians))
                out.l1_before = float(stats.get("l1_before", 0.0))
                out.original_ply = ORIGINAL_PLY
                out.repaired_ply = REPAIRED_PLY
            out.seconds = time.perf_counter() - t0
            logger.info(
                "Repair step %d: %d visible, %d spawned, L1=%.4f -> %s (%.1fs)",
                step, out.n_visible, out.n_spawned, out.l1_before, REPAIRED_PLY, out.seconds,
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
