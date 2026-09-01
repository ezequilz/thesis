"""gsplat-mlx photometric lift helpers (skipped when MLX is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from splat_explorer.repair import SceneRepairer, make_repair_backend, repaired_render_name
from splat_explorer.repair_mlx import mlx_refine_available


def test_mlx_availability_is_boolean():
    assert mlx_refine_available() in (True, False)


def test_apply_until_stops_after_stamp():
    from splat_explorer.repair_mlx import MlxPhotometricRepair
    from splat_explorer.rendering.base import Camera
    from splat_explorer.scene import GaussianScene

    scene = GaussianScene(
        means=np.zeros((4, 3), np.float32),
        scales=np.full((4, 3), 0.1, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)),
        opacities=np.full((4,), 0.9, np.float32),
        colors=np.full((4, 3), 0.1, np.float32),
    )
    camera = Camera.look_at(
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        width=16, height=12, fov_deg=75.0,
    )
    src = np.full((12, 16, 3), 30, np.uint8)
    dst = np.full((12, 16, 3), 220, np.uint8)
    backend = MlxPhotometricRepair(stamp_first=True)
    stats = backend.apply_until(
        scene, camera, src, dst, should_stop=lambda: True,
    )
    assert stats["n_stamped"] >= 1
    assert stats["n_chunks"] == 0
    assert stats["phase"] == "stamp"
    assert scene.colors.mean() > 0.5


def test_select_opt_gaussians_prefers_residual():
    from splat_explorer.repair_mlx import _select_opt_gaussians

    n = 8
    idx = np.arange(n)
    uf = np.array([0, 1, 2, 3, 0, 1, 2, 3], np.float32)
    vf = np.zeros(n, np.float32)
    z = np.linspace(1.0, 8.0, n).astype(np.float32)
    opacities = np.ones(n, np.float32)
    rendered = np.zeros((1, 4, 3), np.uint8)
    repaired = np.zeros((1, 4, 3), np.uint8)
    repaired[0, 3] = 255
    keep = _select_opt_gaussians(idx, uf, vf, z, opacities, rendered, repaired, cap=3)
    assert len(keep) == 3
    assert 3 in set(keep.tolist())


def test_select_opt_gaussians_depth_walks_near_to_far():
    from splat_explorer.repair_mlx import _select_opt_gaussians

    idx = np.arange(6)
    uf = np.zeros(6, np.float32)
    vf = np.zeros(6, np.float32)
    z = np.array([6, 5, 4, 3, 2, 1], np.float32)
    opacities = np.ones(6, np.float32)
    rendered = np.zeros((1, 1, 3), np.uint8)
    repaired = np.zeros((1, 1, 3), np.uint8)
    keep = _select_opt_gaussians(
        idx, uf, vf, z, opacities, rendered, repaired, cap=2, rank_mode="depth",
    )
    assert list(keep) == [5, 4]
    keep2 = _select_opt_gaussians(
        idx, uf, vf, z, opacities, rendered, repaired,
        cap=2, rank_offset=2, rank_mode="depth",
    )
    assert list(keep2) == [3, 2]


def test_metal_limit_detects_malloc_error():
    from splat_explorer.repair_mlx import _is_metal_limit

    assert _is_metal_limit(RuntimeError("[metal::malloc] Resource limit (499000) exceeded."))
    assert not _is_metal_limit(RuntimeError("shape mismatch"))


def test_mlx_apply_retries_after_metal_limit(monkeypatch):
    from splat_explorer import repair_mlx as mod

    monkeypatch.setattr(mod, "_require_mlx", lambda: (object(), None))
    monkeypatch.setattr(mod, "_clear_metal", lambda mx: None)
    seen = []

    def fake_once(self, scene, camera, rendered_rgb, repaired_rgb, rank_offset=0, rank_mode="residual"):
        seen.append((self.max_opt_gaussians, self.max_train_side, self.iters))
        if self.max_opt_gaussians > 128:
            raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
        return {"backend": "gsplat-mlx", "n_visible": 4, "n_iters": self.iters}

    monkeypatch.setattr(mod.MlxPhotometricRepair, "_apply_once", fake_once)
    backend = mod.MlxPhotometricRepair(iters=8, max_opt_gaussians=256, max_train_side=64)
    out = backend.apply(None, None, None, None)
    assert out["n_visible"] == 4
    assert seen[0] == (256, 64, 8)
    assert seen[-1] == (128, 32, 6)


@pytest.mark.skipif(not mlx_refine_available(), reason="gsplat-mlx / MLX not installed")
def test_studio_mlx_caps_stay_under_metal_limit():
    from splat_explorer.repair_mlx import MlxPhotometricRepair

    backend = make_repair_backend("gsplat-mlx", studio=True)
    assert isinstance(backend, MlxPhotometricRepair)
    assert backend.max_opt_gaussians == 256
    assert backend.max_train_side == 128
    assert backend.iters == 20
    assert backend.lambda_dssim == 0.0
    assert backend.stamp_first is True
    assert backend.freeze_colors is True
    assert backend.lr_colors == 0.0
    assert backend.mask_loss is True
    focused = make_repair_backend("gsplat-mlx", studio=True, focused=True)
    assert focused.freeze_colors is True
    assert focused.lr_colors == 0.0


def test_make_repair_backend_prefers_mlx_over_cpu_stamp():
    backend = make_repair_backend()
    from splat_explorer.repair import ProjectedViewRepair
    from splat_explorer.repair_gsfix import gsplat_refine_available

    if gsplat_refine_available():
        pytest.skip("CUDA gsplat takes priority over MLX")
    if mlx_refine_available():
        from splat_explorer.repair_mlx import MlxPhotometricRepair
        assert isinstance(backend, MlxPhotometricRepair)
    else:
        assert isinstance(backend, ProjectedViewRepair)


@pytest.mark.skipif(not mlx_refine_available(), reason="gsplat-mlx / MLX not installed")
def test_mlx_refine_reduces_l1(tmp_path):
    from PIL import Image

    from splat_explorer.repair_mlx import MlxPhotometricRepair
    from splat_explorer.rendering.base import Camera
    from splat_explorer.scene import GaussianScene

    n = 64
    rng = np.random.default_rng(0)
    means = np.zeros((n, 3), np.float32)
    means[:, 0] = rng.uniform(-0.4, 0.4, n)
    means[:, 1] = rng.uniform(-0.3, 0.3, n)
    scene = GaussianScene(
        means=means,
        scales=np.full((n, 3), 0.15, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 0.85, np.float32),
        colors=np.full((n, 3), 0.2, np.float32),
    )
    camera = Camera.look_at(
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        width=64, height=48, fov_deg=75.0,
    )
    Image.fromarray(np.full((48, 64, 3), 40, np.uint8)).save(tmp_path / "src.png")
    Image.fromarray(np.full((48, 64, 3), 220, np.uint8)).save(tmp_path / "fix.png")
    repairer = SceneRepairer(
        scene,
        backend=MlxPhotometricRepair(iters=6, densify=False, lambda_dssim=0.0),
    )
    result = repairer.apply_view(
        step=0, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok", result.error
    assert result.backend == "gsplat-mlx"
    assert result.l1_after is not None
    assert result.l1_after < result.l1_before
    assert result.n_iters == 6
    assert result.n_visible > 0
    assert (tmp_path / repaired_render_name(0)).is_file()


@pytest.mark.skipif(not mlx_refine_available(), reason="gsplat-mlx / MLX not installed")
def test_mlx_refine_writes_back_without_aliasing_source(tmp_path):
    from PIL import Image

    from splat_explorer.repair_mlx import MlxPhotometricRepair
    from splat_explorer.rendering.base import Camera
    from splat_explorer.scene import GaussianScene

    n = 32
    rng = np.random.default_rng(1)
    means = np.zeros((n, 3), np.float32)
    means[:, 0] = rng.uniform(-0.3, 0.3, n)
    means[:, 1] = rng.uniform(-0.3, 0.3, n)
    scene = GaussianScene(
        means=means,
        scales=np.full((n, 3), 0.2, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 0.9, np.float32),
        colors=np.full((n, 3), 0.15, np.float32),
    )
    before = scene.colors.copy()
    camera = Camera.look_at(
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        width=32, height=32, fov_deg=75.0,
    )
    Image.fromarray(np.full((32, 32, 3), 30, np.uint8)).save(tmp_path / "src.png")
    Image.fromarray(np.full((32, 32, 3), 200, np.uint8)).save(tmp_path / "fix.png")
    repairer = SceneRepairer(
        scene,
        backend=MlxPhotometricRepair(iters=4, densify=False, lambda_dssim=0.0, max_train_side=32),
    )
    result = repairer.apply_view(
        step=1, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok", result.error
    np.testing.assert_allclose(scene.colors, before)
    working = repairer._working
    assert working is not None
    assert working.colors.mean() > before.mean() + 0.02
