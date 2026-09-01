"""GSFix3D-style photometric lift via gsplat-mlx (Apple Silicon / Metal).

Same refine loop as ``repair_gsfix.GsplatPhotometricRepair``, but the
differentiable rasterizer is RobotFlow Labs' MLX port instead of CUDA gsplat:

  for each repaired view, for ``iters`` steps:
      I_gs = rasterize(gaussians, camera)          # gsplat_mlx, Metal
      L = (1-λ) ||I_fixed - I_gs||_1 + λ (1-SSIM)
      backward; Adam step on means / color / opacity / scale / quat

No CUDA, no PyTorch at runtime. ``make_repair_backend()`` picks this when
CUDA+gsplat is missing and ``gsplat_mlx`` + MLX import successfully (M1–M4).

The Tier-2 MLX rasterizer is a Python loop over tiles, so this backend
downsamples the training view and caps the optimized set (visible gaussians
first). That is a local-testing stand-in for the CUDA refine, not a full-res
replay of a million-splat scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .rendering.base import Camera
from .scene import GaussianScene

logger = logging.getLogger(__name__)

_LAMBDA_DSSIM = 0.2
_BACKGROUND = (0.12, 0.12, 0.13)


def mlx_refine_available() -> bool:
    """True when gsplat-mlx can rasterize on this machine (Apple Silicon + MLX)."""
    try:
        import mlx.core as mx
        from gsplat_mlx import rasterization  # noqa: F401
    except ImportError:
        return False
    try:
        mx.eval(mx.zeros((1,), dtype=mx.float32))
    except Exception:
        return False
    return True


def _require_mlx():
    try:
        import mlx.core as mx
        from gsplat_mlx import rasterization
    except ImportError as exc:
        raise RuntimeError(
            "gsplat-mlx refine needs Apple Silicon + MLX. "
            "Install with: pip install -e '.[apple]'"
        ) from exc
    return mx, rasterization


def _to_mx(mx, array: np.ndarray):
    return mx.array(np.ascontiguousarray(array, dtype=np.float32))


def _to_numpy(mx, array) -> np.ndarray:
    mx.eval(array)
    return np.asarray(array, dtype=np.float32)


def _normalize_quats(mx, quats):
    n = mx.sqrt(mx.sum(quats * quats, axis=-1, keepdims=True))
    return quats / mx.maximum(n, mx.array(1e-8, dtype=mx.float32))


def _inv_sigmoid(mx, x):
    x = mx.clip(x, 1e-6, 1.0 - 1e-6)
    return mx.log(x / (1.0 - x))


def _train_hw(width: int, height: int, max_side: int) -> tuple[int, int]:
    width, height = int(width), int(height)
    max_side = max(16, int(max_side))
    if max(width, height) <= max_side:
        return width, height
    scale = max_side / float(max(width, height))
    return max(16, int(round(width * scale))), max(16, int(round(height * scale)))


def _scaled_K(camera: Camera, width: int, height: int) -> np.ndarray:
    sx = width / float(camera.width)
    sy = height / float(camera.height)
    return np.array(
        [
            [camera.fx * sx, 0.0, (camera.width / 2.0) * sx],
            [0.0, camera.fy * sy, (camera.height / 2.0) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _image_to_mx(mx, image: np.ndarray, width: int, height: int):
    arr = np.asarray(image)
    if arr.shape[0] != height or arr.shape[1] != width:
        arr = np.asarray(
            Image.fromarray(arr).resize((width, height), Image.Resampling.LANCZOS),
        )
    return _to_mx(mx, arr.astype(np.float32) / 255.0)


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)


class _Adam:
    """Per-parameter Adam with the 3DGS epsilon (no PyTorch)."""

    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-15):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = None
        self.v = None
        self.t = 0

    def step(self, mx, param, grad):
        self.t += 1
        if self.m is None:
            self.m = mx.zeros_like(param)
            self.v = mx.zeros_like(param)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        updated = param - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)
        mx.eval(updated, self.m, self.v)
        return updated


@dataclass
class MlxPhotometricRepair:
    """Differentiable photometric lift on Apple Silicon via gsplat-mlx."""

    iters: int = 12
    lambda_dssim: float = _LAMBDA_DSSIM
    densify: bool = False
    densify_every: int = 5
    densify_grad_thresh: float = 0.0002
    max_clone: int = 256
    max_gaussians: int = 50_000
    max_opt_gaussians: int = 2048
    max_train_side: int = 128
    lr_means: float = 1.6e-4
    lr_colors: float = 0.05
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    near: float = 0.05
    tile_size: int = 16

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        mx, rasterization = _require_mlx()
        from gsplat_mlx.losses import combined_loss, l1_loss

        from .repair import visible_gaussians

        cam_h, cam_w = int(camera.height), int(camera.width)
        train_w, train_h = _train_hw(cam_w, cam_h, self.max_train_side)
        target = _image_to_mx(mx, repaired_rgb, train_w, train_h)
        rendered = _image_to_mx(mx, rendered_rgb, train_w, train_h)
        mx.eval(target, rendered)
        l1_before = float(_to_numpy(mx, mx.mean(mx.abs(target - rendered))))
        n0 = scene.num_gaussians

        idx, _uf, _vf, z = visible_gaussians(scene, camera, near=self.near)
        idx = np.asarray(idx, dtype=np.int64)
        n_visible = int(len(idx))
        if n_visible > int(self.max_opt_gaussians):
            cap = int(self.max_opt_gaussians)
            if len(z) == len(idx):
                keep = np.argpartition(z, cap - 1)[:cap]
            else:
                keep = np.argpartition(-scene.opacities[idx], cap - 1)[:cap]
            idx = idx[keep]
            n_visible = int(len(idx))
        if len(idx) == 0:
            preview = np.asarray(rendered_rgb, dtype=np.uint8)
            if preview.shape[1] != cam_w or preview.shape[0] != cam_h:
                preview = np.asarray(
                    Image.fromarray(preview).resize((cam_w, cam_h), Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                )
            return {
                "backend": "gsplat-mlx",
                "n_visible": 0,
                "n_updated": 0,
                "n_spawned": 0,
                "n_gaussians": n0,
                "n_iters": 0,
                "l1_before": round(l1_before, 6),
                "l1_after": round(l1_before, 6),
                "render_rgb": preview,
            }

        means = _to_mx(mx, scene.means[idx])
        quats = _to_mx(mx, scene.quats[idx])
        colors = _to_mx(mx, scene.colors[idx])
        log_scales = mx.log(mx.maximum(_to_mx(mx, scene.scales[idx]), mx.array(1e-8)))
        logit_opacities = _inv_sigmoid(mx, _to_mx(mx, scene.opacities[idx]))

        viewmat = _to_mx(mx, np.asarray(camera.w2c, dtype=np.float32))[None]
        K = _to_mx(mx, _scaled_K(camera, train_w, train_h))[None]
        background = _to_mx(mx, np.array(_BACKGROUND, dtype=np.float32))[None]
        n_opt0 = n_visible
        n_spawned = 0

        def make_opts():
            return {
                "means": _Adam(self.lr_means),
                "quats": _Adam(self.lr_quats),
                "log_scales": _Adam(self.lr_scales),
                "logit_opacities": _Adam(self.lr_opacities),
                "colors": _Adam(self.lr_colors),
            }

        opts = make_opts()

        def rasterize(means_, quats_, log_scales_, logit_opacities_, colors_):
            scales = mx.exp(log_scales_)
            opacities = mx.sigmoid(logit_opacities_)
            quats_n = _normalize_quats(mx, quats_)
            rgb, _alphas, info = rasterization(
                means_,
                quats_n,
                scales,
                opacities,
                colors_,
                viewmat,
                K,
                width=int(train_w),
                height=int(train_h),
                near_plane=float(self.near),
                sh_degree=None,
                tile_size=int(self.tile_size),
                backgrounds=background,
                render_mode="RGB",
                differentiable=True,
            )
            rgb = rgb[0]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            return mx.clip(rgb, 0.0, 1.0), info

        def loss_fn(means_, quats_, log_scales_, logit_opacities_, colors_):
            rgb, _info = rasterize(
                means_, quats_, log_scales_, logit_opacities_, colors_,
            )
            if float(self.lambda_dssim) <= 0.0:
                return l1_loss(rgb, target)
            return combined_loss(rgb, target, lambda_ssim=float(self.lambda_dssim))

        value_and_grad = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))

        for it in range(int(self.iters)):
            loss, grads = value_and_grad(
                means, quats, log_scales, logit_opacities, colors,
            )
            mx.eval(loss, *grads)
            g_means, g_quats, g_log_scales, g_logit, g_colors = grads

            if (
                self.densify
                and it > 0
                and it % int(self.densify_every) == 0
                and int(means.shape[0]) < int(self.max_gaussians)
            ):
                packed = _maybe_densify(
                    mx, means, quats, colors, log_scales, logit_opacities, g_means,
                    thresh=self.densify_grad_thresh, max_clone=self.max_clone,
                )
                if packed is not None:
                    means, quats, colors, log_scales, logit_opacities, n_new = packed
                    n_spawned += n_new
                    opts = make_opts()
                    continue

            means = opts["means"].step(mx, means, g_means)
            quats = opts["quats"].step(mx, quats, g_quats)
            log_scales = opts["log_scales"].step(mx, log_scales, g_log_scales)
            logit_opacities = opts["logit_opacities"].step(mx, logit_opacities, g_logit)
            colors = opts["colors"].step(mx, colors, g_colors)
            quats = _normalize_quats(mx, quats)
            mx.eval(means, quats, log_scales, logit_opacities, colors)

        rgb, _info = rasterize(means, quats, log_scales, logit_opacities, colors)
        l1_after = float(_to_numpy(mx, l1_loss(rgb, target)))
        preview = _to_uint8(_to_numpy(mx, rgb))
        if preview.shape[1] != cam_w or preview.shape[0] != cam_h:
            preview = np.asarray(
                Image.fromarray(preview).resize((cam_w, cam_h), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )

        scales = mx.exp(log_scales)
        opacities = mx.sigmoid(logit_opacities)
        quats_n = _normalize_quats(mx, quats)
        colors_c = mx.clip(colors, 0.0, 1.0)
        _commit_subset(
            scene,
            idx,
            n_opt0,
            means=_to_numpy(mx, means),
            quats=_to_numpy(mx, quats_n),
            scales=_to_numpy(mx, scales),
            opacities=_to_numpy(mx, opacities),
            colors=_to_numpy(mx, colors_c),
        )

        n1 = scene.num_gaussians
        logger.info(
            "gsplat-mlx refine: %d iters @ %dx%d, L1 %.4f -> %.4f, %d visible / %d gaussians (+%d)",
            self.iters, train_w, train_h, l1_before, l1_after, n_visible, n1, n_spawned,
        )
        return {
            "backend": "gsplat-mlx",
            "n_visible": n_visible,
            "n_updated": n_visible,
            "n_spawned": int(max(0, n1 - n0) if n_spawned == 0 else n_spawned),
            "n_gaussians": n1,
            "n_iters": int(self.iters),
            "l1_before": round(l1_before, 6),
            "l1_after": round(l1_after, 6),
            "render_rgb": preview,
            "train_width": train_w,
            "train_height": train_h,
        }


def _commit_subset(
    scene: GaussianScene,
    idx: np.ndarray,
    n_opt0: int,
    *,
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Write optimized gaussians back; append any cloned extras."""
    n_now = int(means.shape[0])
    n_orig = int(min(n_opt0, n_now, len(idx)))
    if n_orig:
        dest = idx[:n_orig]
        scene.means[dest] = means[:n_orig]
        scene.quats[dest] = quats[:n_orig]
        scene.scales[dest] = scales[:n_orig]
        scene.opacities[dest] = opacities[:n_orig]
        scene.colors[dest] = colors[:n_orig]
    if n_now > n_opt0:
        extra = slice(n_opt0, n_now)
        scene.means = np.concatenate([scene.means, means[extra]], axis=0)
        scene.quats = np.concatenate([scene.quats, quats[extra]], axis=0)
        scene.scales = np.concatenate([scene.scales, scales[extra]], axis=0)
        scene.opacities = np.concatenate([scene.opacities, opacities[extra]], axis=0)
        scene.colors = np.concatenate([scene.colors, colors[extra]], axis=0)


def _maybe_densify(
    mx, means, quats, colors, log_scales, logit_opacities, grad_means, *, thresh, max_clone,
):
    if grad_means is None:
        return None
    mag = np.asarray(mx.sqrt(mx.sum(grad_means * grad_means, axis=-1)))
    mx.eval(means, quats, colors, log_scales, logit_opacities)
    sel = mag > float(thresh)
    n_sel = int(sel.sum())
    if n_sel == 0:
        return None
    if n_sel > int(max_clone):
        idx = np.argpartition(mag, -int(max_clone))[-int(max_clone):]
        sel = np.zeros_like(sel)
        sel[idx] = True
        n_sel = int(max_clone)
    means_np = np.asarray(means)
    noise = np.random.randn(*means_np[sel].shape).astype(np.float32) * 0.01
    new_means = means_np[sel] + noise
    new_quats = np.asarray(quats)[sel]
    new_colors = np.asarray(colors)[sel]
    new_log_scales = np.asarray(log_scales)[sel] - 0.6931
    new_logit = np.asarray(logit_opacities)[sel]
    means = mx.array(np.concatenate([means_np, new_means], axis=0))
    quats = mx.array(np.concatenate([np.asarray(quats), new_quats], axis=0))
    colors = mx.array(np.concatenate([np.asarray(colors), new_colors], axis=0))
    log_scales = mx.array(np.concatenate([np.asarray(log_scales), new_log_scales], axis=0))
    logit_opacities = mx.array(np.concatenate([np.asarray(logit_opacities), new_logit], axis=0))
    return means, quats, colors, log_scales, logit_opacities, n_sel
