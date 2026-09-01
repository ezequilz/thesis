"""GSFix-style photometric lift helpers (CPU-safe; CUDA tests skipped)."""

from __future__ import annotations

import numpy as np
import pytest

from splat_explorer.repair import (
    PhotometricViewRepair,
    ProjectedViewRepair,
    SceneRepairer,
    make_repair_backend,
    repaired_render_name,
)
from splat_explorer.repair_gsfix import photometric_loss, ssim


def test_repaired_render_name():
    assert repaired_render_name(2) == "step_002_repaired_render.png"


def test_make_repair_backend_falls_back_without_cuda():
    backend = make_repair_backend()
    from splat_explorer.repair_gsfix import gsplat_refine_available

    if gsplat_refine_available():
        from splat_explorer.repair_gsfix import GsplatPhotometricRepair
        assert isinstance(backend, GsplatPhotometricRepair)
    else:
        assert isinstance(backend, ProjectedViewRepair)


def test_ssim_and_l1_on_cpu():
    torch = pytest.importorskip("torch")
    a = torch.zeros(32, 32, 3)
    b = torch.ones(32, 32, 3)
    loss, l1 = photometric_loss(a, a, torch)
    assert float(l1) == 0.0
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    loss2, l1b = photometric_loss(a, b, torch)
    assert float(l1b) == pytest.approx(1.0, abs=1e-5)
    assert float(loss2) > 0.5
    same = ssim(a.permute(2, 0, 1), a.permute(2, 0, 1), torch)
    assert float(same) == pytest.approx(1.0, abs=1e-4)


def test_apply_view_records_backend(tmp_path):
    from PIL import Image

    from splat_explorer.agent.camera_rig import CameraRig
    from splat_explorer.repair import ORIGINAL_PLY, REPAIRED_PLY
    from splat_explorer.scene import GaussianScene

    n = 4
    scene = GaussianScene(
        means=np.zeros((n, 3), np.float32),
        scales=np.full((n, 3), 0.05, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 0.8, np.float32),
        colors=np.full((n, 3), 0.4, np.float32),
    )
    scene.means[0] = [0.0, 0.0, 0.0]
    camera = CameraRig(np.array([0.0, 0.0, -2.0]), up_axis="+y").camera(32, 24, 75.0)
    Image.fromarray(np.full((24, 32, 3), 80, np.uint8)).save(tmp_path / "src.png")
    Image.fromarray(np.full((24, 32, 3), 200, np.uint8)).save(tmp_path / "fix.png")
    repairer = SceneRepairer(scene, backend=PhotometricViewRepair(densify=False))
    result = repairer.apply_view(
        step=7, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok"
    assert result.backend == "cpu-photometric"
    body = result.to_json()
    assert body["backend"] == "cpu-photometric"
    assert "l1_after" in body
    assert (tmp_path / ORIGINAL_PLY).is_file()
    assert (tmp_path / REPAIRED_PLY).is_file()


@pytest.mark.skipif(
    __import__("splat_explorer.repair_gsfix", fromlist=["gsplat_refine_available"]).gsplat_refine_available() is False,
    reason="CUDA+gsplat not available",
)
def test_gsplat_refine_reduces_l1(tmp_path):
    from PIL import Image

    from splat_explorer.agent.camera_rig import CameraRig
    from splat_explorer.repair_gsfix import GsplatPhotometricRepair
    from splat_explorer.scene import GaussianScene

    n = 64
    rng = np.random.default_rng(0)
    means = np.zeros((n, 3), np.float32)
    means[:, 2] = 2.0
    means[:, 0] = rng.uniform(-0.3, 0.3, n)
    means[:, 1] = rng.uniform(-0.3, 0.3, n)
    scene = GaussianScene(
        means=means,
        scales=np.full((n, 3), 0.08, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 0.7, np.float32),
        colors=np.full((n, 3), 0.2, np.float32),
    )
    camera = CameraRig(np.array([0.0, 0.0, 0.0]), up_axis="+y").camera(64, 48, 75.0)
    Image.fromarray(np.full((48, 64, 3), 40, np.uint8)).save(tmp_path / "src.png")
    Image.fromarray(np.full((48, 64, 3), 220, np.uint8)).save(tmp_path / "fix.png")
    repairer = SceneRepairer(
        scene,
        backend=GsplatPhotometricRepair(iters=8, densify=False),
    )
    result = repairer.apply_view(
        step=0, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok"
    assert result.backend == "gsfix-gsplat"
    assert result.l1_after is not None
    assert result.l1_after < result.l1_before
    assert (tmp_path / repaired_render_name(0)).is_file()
