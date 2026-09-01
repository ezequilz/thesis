"""gsplat-mlx photometric lift helpers (skipped when MLX is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from splat_explorer.repair import SceneRepairer, make_repair_backend, repaired_render_name
from splat_explorer.repair_mlx import mlx_refine_available


def test_mlx_availability_is_boolean():
    assert mlx_refine_available() in (True, False)


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
