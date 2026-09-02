"""GSFix3D-style photometric lift via gsplat-mlx (Apple Silicon / Metal).

Same refine loop as ``repair_gsfix.GsplatPhotometricRepair``, but the
differentiable rasterizer is RobotFlow Labs' MLX port instead of CUDA gsplat:

  for each repaired view, for ``iters`` steps:
      I_gs = rasterize(gaussians, camera)          # gsplat_mlx, Metal
      L = (1-λ) ||I_fixed - I_gs||_1 + λ (1-SSIM)
      backward; Adam step on means / color / opacity / scale / quat

Default is that paper loop — no color stamp, no frozen RGB, no extra loss
mask. ``stamp_first=True`` is the optional ``gsplat-mlx-stamp`` path that
paints regen RGB onto visible gaussians first (visible updates, worse repair).

No CUDA, no PyTorch at runtime. ``make_repair_backend()`` picks this when
CUDA+gsplat is missing and ``gsplat_mlx`` + MLX import successfully (M1–M4).

The Tier-2 MLX rasterizer is a Python loop over tiles, so this backend
downsamples the training view and caps the optimized set (visible gaussians
first). That is a local-testing stand-in for the CUDA refine, not a full-res
replay of a million-splat scene.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
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


def _clear_metal(mx) -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
        return
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "clear_cache"):
        metal.clear_cache()


def _is_metal_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "Resource limit" in text or "metal::malloc" in text


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


def _select_opt_gaussians(
    idx: np.ndarray,
    uf: np.ndarray,
    vf: np.ndarray,
    z: np.ndarray,
    opacities: np.ndarray,
    rendered_rgb: np.ndarray,
    repaired_rgb: np.ndarray,
    cap: int,
    rank_offset: int = 0,
    rank_mode: str = "residual",
) -> np.ndarray:
    """Keep Gaussians that cover the photometric residual.

    `rank_offset` rotates the ranking so a long single-view run can walk
    across the frustum in chunks instead of re-optimizing the same 256.
    `rank_mode='depth'` walks near-to-far instead of the original-render
    residual (which keeps re-hitting the same splats after a stamp).
    """
    idx = np.asarray(idx, dtype=np.int64)
    cap = int(cap)
    if cap <= 0 or len(idx) == 0:
        return idx
    src = np.asarray(rendered_rgb, dtype=np.float32)
    dst = np.asarray(repaired_rgb, dtype=np.float32)
    if rank_mode == "depth" or src.ndim != 3 or dst.ndim != 3:
        order = np.argsort(np.asarray(z, dtype=np.float32), kind="mergesort")
        ranked = idx[order]
    else:
        h, w = src.shape[:2]
        if dst.shape[0] != h or dst.shape[1] != w:
            dst = np.asarray(
                Image.fromarray(np.clip(dst, 0, 255).astype(np.uint8)).resize(
                    (w, h), Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
        u = np.clip(np.rint(np.asarray(uf, dtype=np.float32)).astype(np.int32), 0, w - 1)
        v = np.clip(np.rint(np.asarray(vf, dtype=np.float32)).astype(np.int32), 0, h - 1)
        err = np.mean(np.abs(dst[v, u] - src[v, u]), axis=-1)
        score = err * np.clip(np.asarray(opacities, dtype=np.float32), 1e-3, 1.0)
        ranked = idx[np.argsort(-score, kind="mergesort")]
    if len(ranked) <= cap:
        return ranked
    start = int(rank_offset) % len(ranked)
    return np.concatenate([ranked[start:], ranked[:start]])[:cap]


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
    """Differentiable photometric lift on Apple Silicon via gsplat-mlx.

    Defaults match ``GsplatPhotometricRepair`` (GSFix3D refine): L1+SSIM on
    the regen view, Adam on means/color/opacity/scale/quat, no color stamp.
    Set ``stamp_first=True`` for the optional visible-stamp variant.
    """

    iters: int = 20
    densify: bool = False
    densify_every: int = 5
    densify_grad_thresh: float = 0.0002
    max_clone: int = 256
    max_gaussians: int = 50_000
    # Tier-2 MLX rasterizer builds one Metal graph node per tile-gaussian.
    # 8k gaussians at 256px blows past Metal's ~5e5 resource cap on M1/M2.
    max_opt_gaussians: int = 512
    max_train_side: int = 64
    lr_means: float = 1.6e-4
    lr_colors: float = 0.0025
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    near: float = 0.05
    tile_size: int = 32
    lambda_dssim: float = _LAMBDA_DSSIM
    stamp_first: bool = False
    freeze_colors: bool = False
    mask_loss: bool = False
    max_scale_ratio: float = 0.0

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
        rank_offset: int = 0,
        rank_mode: str = "residual",
    ) -> dict[str, Any]:
        mx, _rasterization = _require_mlx()
        try:
            return self._apply_once(
                scene, camera, rendered_rgb, repaired_rgb,
                rank_offset=rank_offset, rank_mode=rank_mode,
            )
        except RuntimeError as exc:
            if not _is_metal_limit(exc):
                raise
            next_n = max(64, int(self.max_opt_gaussians) // 2)
            next_side = max(32, int(self.max_train_side) // 2)
            if next_n >= int(self.max_opt_gaussians) and next_side >= int(self.max_train_side):
                raise RuntimeError(
                    "gsplat-mlx hit Metal's resource limit even after shrinking the "
                    "optimized set. Use the CPU color stamp for this view."
                ) from exc
            logger.warning(
                "gsplat-mlx Metal resource limit at %d gaussians / %dpx; retrying at %d / %dpx",
                self.max_opt_gaussians,
                self.max_train_side,
                next_n,
                next_side,
            )
            _clear_metal(mx)
            smaller = replace(
                self,
                max_opt_gaussians=next_n,
                max_train_side=next_side,
                iters=min(int(self.iters), 6),
            )
            return smaller.apply(
                scene, camera, rendered_rgb, repaired_rgb,
                rank_offset=rank_offset, rank_mode=rank_mode,
            )

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
        """Keep refining until stop or deadline.

        Paper path: photometric chunks only. Stamp path: paint regen RGB
        first, then geometry-only chunks (colors stay frozen).
        """
        import time as time_mod

        n_stamped = 0
        if self.stamp_first:
            from .repair import stamp_view_colors

            n_stamped = stamp_view_colors(
                scene, camera, repaired_rgb, near=self.near, color_lr=1.0,
            )
            stats: dict[str, Any] = {
                "backend": "gsplat-mlx",
                "n_visible": n_stamped,
                "n_updated": n_stamped,
                "n_stamped": n_stamped,
                "n_spawned": 0,
                "n_gaussians": scene.num_gaussians,
                "n_iters": 0,
                "n_chunks": 0,
                "l1_before": None,
                "l1_after": None,
                "phase": "stamp",
            }
            if on_checkpoint is not None:
                on_checkpoint(dict(stats))
            # Colors already match the regen. Further subset Adam on RGB is
            # what clipped to neon magenta/green/blue after ~1 minute.
            inner = replace(
                self,
                stamp_first=False,
                freeze_colors=True,
                lr_colors=0.0,
                mask_loss=True,
            )
            rank_mode = "depth"
        else:
            stats = {
                "backend": "gsplat-mlx",
                "n_visible": 0,
                "n_updated": 0,
                "n_stamped": 0,
                "n_spawned": 0,
                "n_gaussians": scene.num_gaussians,
                "n_iters": 0,
                "n_chunks": 0,
                "l1_before": None,
                "l1_after": None,
                "phase": "refine",
            }
            inner = replace(self, stamp_first=False)
            rank_mode = "residual"

        chunk = 0
        total_iters = 0
        l1_before = None
        last = stats
        while True:
            if should_stop is not None and should_stop():
                break
            if deadline is not None and time_mod.time() >= deadline:
                break
            last = inner.apply(
                scene, camera, rendered_rgb, repaired_rgb,
                rank_offset=chunk * int(self.max_opt_gaussians),
                rank_mode=rank_mode,
            )
            if l1_before is None:
                l1_before = last.get("l1_before")
            total_iters += int(last.get("n_iters") or 0)
            chunk += 1
            last = {
                **last,
                "n_stamped": n_stamped,
                "n_updated": max(n_stamped, int(last.get("n_updated") or 0)),
                "n_iters": total_iters,
                "n_chunks": chunk,
                "l1_before": l1_before,
                "phase": "refine",
            }
            if on_checkpoint is not None:
                on_checkpoint(dict(last))
        last["n_stamped"] = n_stamped
        last["n_iters"] = total_iters
        last["n_chunks"] = chunk
        return last

    def _apply_once(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
        rank_offset: int = 0,
        rank_mode: str = "residual",
    ) -> dict[str, Any]:
        mx, rasterization = _require_mlx()
        from gsplat_mlx.losses import combined_loss, l1_loss

        from .repair import stamp_view_colors, visible_gaussians

        cam_h, cam_w = int(camera.height), int(camera.width)
        n_stamped = 0
        if self.stamp_first:
            n_stamped = stamp_view_colors(
                scene, camera, repaired_rgb, near=self.near, color_lr=1.0,
            )
        freeze_colors = bool(self.freeze_colors or self.stamp_first)
        use_logit = not freeze_colors
        train_w, train_h = _train_hw(cam_w, cam_h, self.max_train_side)
        target = _image_to_mx(mx, repaired_rgb, train_w, train_h)
        rendered = _image_to_mx(mx, rendered_rgb, train_w, train_h)
        mx.eval(target, rendered)
        l1_before = float(_to_numpy(mx, mx.mean(mx.abs(target - rendered))))
        n0 = scene.num_gaussians

        idx, uf, vf, z = visible_gaussians(scene, camera, near=self.near)
        idx = np.asarray(idx, dtype=np.int64)
        n_visible = int(len(idx))
        if n_visible > int(self.max_opt_gaussians):
            idx = _select_opt_gaussians(
                idx, uf, vf, z, scene.opacities[idx],
                rendered_rgb, repaired_rgb, int(self.max_opt_gaussians),
                rank_offset=int(rank_offset),
                rank_mode=str(rank_mode or "residual"),
            )
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
        colors_rgb = mx.clip(_to_mx(mx, scene.colors[idx]), 0.0, 1.0)
        color_param = _inv_sigmoid(mx, colors_rgb) if use_logit else colors_rgb
        log_scales = mx.log(mx.maximum(_to_mx(mx, scene.scales[idx]), mx.array(1e-8)))
        log_scales0 = log_scales
        logit_opacities = _inv_sigmoid(mx, _to_mx(mx, scene.opacities[idx]))
        ratio = float(self.max_scale_ratio or 0.0)
        scale_span = math.log(max(1.01, ratio)) if ratio > 1.0 else None

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
                "colors": _Adam(0.0 if freeze_colors else self.lr_colors),
            }

        opts = make_opts()

        def colors_from(param):
            return mx.sigmoid(param) if use_logit else mx.clip(param, 0.0, 1.0)

        def rasterize(means_, quats_, log_scales_, logit_opacities_, color_param_):
            scales = mx.exp(log_scales_)
            opacities = mx.sigmoid(logit_opacities_)
            quats_n = _normalize_quats(mx, quats_)
            rgb, alphas, info = rasterization(
                means_,
                quats_n,
                scales,
                opacities,
                colors_from(color_param_),
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
            alphas = alphas[0]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            rgb = mx.clip(rgb, 0.0, 1.0)
            if alphas.ndim == 3:
                alphas = alphas[..., 0]
            return rgb, mx.clip(alphas, 0.0, 1.0), info

        def loss_fn(means_, quats_, log_scales_, logit_opacities_, color_param_):
            rgb, alphas, _info = rasterize(
                means_, quats_, log_scales_, logit_opacities_, color_param_,
            )
            if self.mask_loss:
                err = mx.abs(rgb - target)
                weight = alphas
                denom = mx.sum(weight) * rgb.shape[-1] + mx.array(1e-6)
                return mx.sum(err * weight[..., None]) / denom
            if float(self.lambda_dssim) <= 0.0:
                return l1_loss(rgb, target)
            return combined_loss(rgb, target, lambda_ssim=float(self.lambda_dssim))

        value_and_grad = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))

        for it in range(int(self.iters)):
            loss, grads = value_and_grad(
                means, quats, log_scales, logit_opacities, color_param,
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
                    mx, means, quats, colors_from(color_param),
                    log_scales, logit_opacities, g_means,
                    thresh=self.densify_grad_thresh, max_clone=self.max_clone,
                )
                if packed is not None:
                    means, quats, colors_rgb, log_scales, logit_opacities, n_new = packed
                    color_param = (
                        _inv_sigmoid(mx, mx.clip(colors_rgb, 0.0, 1.0))
                        if use_logit else mx.clip(colors_rgb, 0.0, 1.0)
                    )
                    log_scales0 = log_scales
                    n_spawned += n_new
                    opts = make_opts()
                    continue

            means = opts["means"].step(mx, means, g_means)
            quats = opts["quats"].step(mx, quats, g_quats)
            log_scales = opts["log_scales"].step(mx, log_scales, g_log_scales)
            if scale_span is not None:
                log_scales = mx.clip(
                    log_scales, log_scales0 - scale_span, log_scales0 + scale_span,
                )
            logit_opacities = opts["logit_opacities"].step(mx, logit_opacities, g_logit)
            if not freeze_colors:
                color_param = opts["colors"].step(mx, color_param, g_colors)
            quats = _normalize_quats(mx, quats)
            mx.eval(means, quats, log_scales, logit_opacities, color_param)
            _clear_metal(mx)

        rgb, _alphas, _info = rasterize(
            means, quats, log_scales, logit_opacities, color_param,
        )
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
        colors_c = colors_from(color_param)
        commit_colors = (
            np.asarray(scene.colors[idx], dtype=np.float32)
            if freeze_colors else _to_numpy(mx, colors_c)
        )
        _commit_subset(
            scene,
            idx,
            n_opt0,
            means=_to_numpy(mx, means),
            quats=_to_numpy(mx, quats_n),
            scales=_to_numpy(mx, scales),
            opacities=_to_numpy(mx, opacities),
            colors=commit_colors,
        )

        n1 = scene.num_gaussians
        logger.info(
            "gsplat-mlx refine: %d iters @ %dx%d, L1 %.4f -> %.4f, %d visible / %d gaussians (+%d)%s",
            self.iters, train_w, train_h, l1_before, l1_after, n_visible, n1, n_spawned,
            " [colors frozen]" if freeze_colors else "",
        )
        return {
            "backend": "gsplat-mlx",
            "n_visible": n_visible,
            "n_updated": int(n_stamped or n_visible),
            "n_stamped": int(n_stamped),
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
