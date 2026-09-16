"""CPU-safe tests for the scene-run GSFix3D working-backup ADC path."""

from __future__ import annotations

import numpy as np
import pytest

from splat_explorer.repair_gsfix3d_working_backup import (
    GsplatGsfix3dRepair,
    _SH_CLIP_UNIT_RGB,
    clamp_log_scales,
    densify_clone_split,
    repaired_view_loss,
    rgb_to_sh,
    sh_to_rgb,
)
from splat_explorer.scene.ply_loader import SH_C0


def test_working_backup_clone_split_keeps_1d_and_column_opacities():
    torch = pytest.importorskip("torch")
    n = 4
    means = torch.zeros(n, 3, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, requires_grad=True)
    f_dc = torch.zeros(n, 3, requires_grad=True)
    scales = torch.tensor([
        [0.01, 0.01, 0.01],
        [0.01, 0.01, 0.01],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    log_scales = torch.log(scales).requires_grad_(True)
    mag = torch.tensor([1.0, 0.0, 1.0, 0.0])

    packed = densify_clone_split(
        torch, means, quats, f_dc, log_scales, torch.zeros(n, requires_grad=True), mag,
        thresh=0.5, split_scale=0.1, max_clone=16, max_gaussians=100,
    )
    assert packed is not None
    means2, _, _, _, logit2, n_spawned = packed
    assert int(means2.shape[0]) == 6
    assert int(n_spawned) == 2
    assert logit2.ndim == 1
    assert int(logit2.shape[0]) == 6

    packed_col = densify_clone_split(
        torch, means, quats, f_dc, log_scales,
        torch.zeros(n, 1, requires_grad=True), mag,
        thresh=0.5, split_scale=0.1, max_clone=16, max_gaussians=100,
    )
    assert packed_col is not None
    assert packed_col[4].shape == (6, 1)


def test_working_backup_apply_until_densifies_only_first_chunk():
    backend = GsplatGsfix3dRepair(max_chunks=0, densify=True)
    flags: list[bool] = []

    def fake_chunk(*_a, **_k):
        flags.append(bool(backend.densify))
        return {"n_iters": 20, "l1_before": 0.5, "l1_after": 0.4}

    backend._step_chunk = fake_chunk
    out = backend.apply_until(
        object(), object(), object(), object(),
        should_stop=lambda: len(flags) >= 3,
    )
    assert flags == [True, False, False]
    assert out["n_chunks"] == 3
    assert backend.densify is True


def test_working_backup_apply_until_reuses_gpu_state():
    backend = GsplatGsfix3dRepair(max_chunks=0)
    seen: list[int] = []

    def fake_chunk(scene, camera, rendered_rgb, repaired_rgb, state):
        seen.append(id(state))
        state["ctx"] = state.get("ctx") or {"resident": True}
        return {"n_iters": 20, "l1_before": 0.5, "l1_after": 0.4}

    backend._step_chunk = fake_chunk
    backend.apply_until(
        object(), object(), object(), object(),
        should_stop=lambda: len(seen) >= 3,
    )
    assert len(seen) == 3
    assert len(set(seen)) == 1


def test_rgb_sh_roundtrip_numpy():
    rgb = np.array([[0.2, 0.5, 0.9], [0.0, 1.0, 0.33]], dtype=np.float64)
    out = sh_to_rgb(rgb_to_sh(rgb))
    np.testing.assert_allclose(out, rgb, atol=1e-5)


def test_unclamped_oversaturated_pred_keeps_color_gradient():
    torch = pytest.importorskip("torch")
    pred = torch.full((16, 16, 3), 2.0, requires_grad=True)
    target = torch.full((16, 16, 3), 0.4)
    rendered = torch.full((16, 16, 3), 0.4)
    loss, _ = repaired_view_loss(
        pred, target, rendered, torch,
        lambda_dssim=0.0, lambda_preserve=0.0, lambda_color_bound=0.0,
    )
    loss.backward()
    assert pred.grad is not None
    assert float(pred.grad.abs().sum()) > 0.0

    stuck = torch.full((16, 16, 3), 2.0, requires_grad=True)
    clipped = (stuck.clamp(0.0, 1.0) - target).abs().mean()
    clipped.backward()
    assert stuck.grad is None or float(stuck.grad.abs().sum()) == 0.0


def test_repaired_view_loss_preserves_unchanged_pixels():
    torch = pytest.importorskip("torch")
    rendered = torch.zeros(16, 16, 3)
    target = rendered.clone()
    target[0, 0] = 1.0
    pred = torch.zeros(16, 16, 3, requires_grad=True)
    loss, _ = repaired_view_loss(
        pred, target, rendered, torch,
        lambda_dssim=0.0, lambda_preserve=1.0, lambda_color_bound=0.0,
        min_fix_weight=0.05,
    )
    loss.backward()
    assert pred.grad is not None
    changed = float(pred.grad[0, 0].abs().sum())
    unchanged = float(pred.grad[8, 8].abs().sum())
    assert changed > unchanged


def test_color_bound_pulls_exploded_sh_rgb_back():
    torch = pytest.importorskip("torch")
    colors = torch.tensor([[1.8, -0.4, 0.5]], requires_grad=True)
    pred = torch.zeros(16, 16, 3)
    target = torch.zeros(16, 16, 3)
    rendered = torch.zeros(16, 16, 3)
    loss, _ = repaired_view_loss(
        pred, target, rendered, torch,
        lambda_dssim=0.0, lambda_preserve=0.0, lambda_color_bound=1.0,
        colors_rgb=colors,
    )
    loss.backward()
    assert colors.grad is not None
    assert float(colors.grad[0, 0]) > 0.0
    assert float(colors.grad[0, 1]) < 0.0
    assert abs(float(colors.grad[0, 2])) < 1e-6


def test_working_backup_keeps_sh_clip_and_preserve_defaults():
    backend = GsplatGsfix3dRepair()
    assert backend.lambda_preserve == 1.0
    assert backend.sh_clip == pytest.approx(_SH_CLIP_UNIT_RGB)
    assert backend.lambda_color_bound == 0.05
    assert backend.lambda_color_reg == 0.02
    assert backend.freeze_geometry_after_first_chunk is True
    rgb_hi = 0.5 + backend.sh_clip * SH_C0
    rgb_lo = 0.5 - backend.sh_clip * SH_C0
    assert rgb_hi == pytest.approx(1.0)
    assert rgb_lo == pytest.approx(0.0)


def test_working_backup_apply_until_freezes_geometry_after_first_chunk():
    backend = GsplatGsfix3dRepair(max_chunks=0)
    seen: list[bool] = []
    frozen_calls: list[int] = []

    def fake_chunk(scene, camera, rendered_rgb, repaired_rgb, state):
        state["ctx"] = state.get("ctx") or {"geometry_frozen": False}
        seen.append(bool(state["ctx"].get("geometry_frozen")))
        return {"n_iters": 20, "l1_before": 0.5, "l1_after": 0.4}

    def fake_freeze(ctx):
        ctx["geometry_frozen"] = True
        frozen_calls.append(1)

    backend._step_chunk = fake_chunk
    backend._freeze_geometry = fake_freeze
    backend.apply_until(
        object(), object(), object(), object(),
        should_stop=lambda: len(seen) >= 3,
    )
    assert seen == [False, True, True]
    assert frozen_calls == [1]


def test_github_original_apply_until_runs_one_chunk_without_freeze():
    backend = GsplatGsfix3dRepair(
        max_chunks=1,
        upstream_gsfix3d=True,
        freeze_geometry_after_first_chunk=False,
    )
    chunks: list[int] = []
    frozen_calls: list[int] = []

    def fake_chunk(scene, camera, rendered_rgb, repaired_rgb, state):
        chunks.append(1)
        return {"n_iters": 20, "l1_before": 0.5, "l1_after": 0.4}

    backend._step_chunk = fake_chunk
    backend._freeze_geometry = lambda ctx: frozen_calls.append(1)
    stats = backend.apply_until(object(), object(), object(), object())
    assert chunks == [1]
    assert frozen_calls == []
    assert stats["n_chunks"] == 1


def test_clamp_log_scales_caps_needles():
    torch = pytest.importorskip("torch")
    log_scales = torch.log(torch.tensor([[1e-6, 1e-6, 80.0], [0.02, 0.02, 0.02]]))
    clamp_log_scales(log_scales, min_scale=1e-6, max_scale=1.0, max_aniso=32.0)
    scales = log_scales.exp()
    assert float(scales[0].max()) <= 32.0 * 1e-6 + 1e-9
    assert float(scales[0].max() / scales[0].min()) <= 32.0 + 1e-5
    assert float(scales.max()) <= 1.0 + 1e-6
    np.testing.assert_allclose(scales[1].detach().cpu().numpy(), [0.02, 0.02, 0.02], atol=1e-6)
