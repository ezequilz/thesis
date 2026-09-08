"""CPU-safe tests for the paper GSFix3D CUDA lift (no GPU required)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from splat_explorer.repair_gsfix3d import (
    GsplatGsfix3dRepair,
    densify_clone_split,
    instantiate_cuda_repair,
    rgb_to_sh,
    sh_to_rgb,
)


def test_gsfix3d_module_does_not_stamp():
    from splat_explorer import repair_gsfix3d

    src = inspect.getsource(repair_gsfix3d)
    assert "stamp_view_colors" not in src
    assert "stamp_first" not in src


def test_instantiate_cuda_repair_dispatches_baseline():
    from splat_explorer.repair_gsfix import GsplatPhotometricRepair

    paper = instantiate_cuda_repair(method="gsfix-gsplat", iters=7)
    assert isinstance(paper, GsplatGsfix3dRepair)
    assert paper.iters == 7
    assert paper.kf_iters == 50
    base = instantiate_cuda_repair(method="gsfix-gsplat-baseline", iters=3)
    assert isinstance(base, GsplatPhotometricRepair)
    assert base.iters == 3


def test_rgb_sh_roundtrip_numpy():
    rgb = np.array([[0.2, 0.5, 0.9], [0.0, 1.0, 0.33]], dtype=np.float64)
    out = sh_to_rgb(rgb_to_sh(rgb))
    np.testing.assert_allclose(out, rgb, atol=1e-5)


def test_rgb_sh_roundtrip():
    torch = pytest.importorskip("torch")
    rgb = torch.tensor([[0.2, 0.5, 0.9], [0.0, 1.0, 0.33]])
    out = sh_to_rgb(rgb_to_sh(rgb))
    assert torch.allclose(out, rgb, atol=1e-5)


def test_densify_clone_split_small_vs_large():
    torch = pytest.importorskip("torch")
    n = 4
    means = torch.zeros(n, 3, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, requires_grad=True)
    f_dc = torch.zeros(n, 3, requires_grad=True)
    scales = torch.tensor([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    log_scales = torch.log(scales).requires_grad_(True)
    logit = torch.zeros(n, requires_grad=True)
    mag = torch.tensor([1.0, 0.0, 1.0, 0.0])
    packed = densify_clone_split(
        torch, means, quats, f_dc, log_scales, logit, mag,
        thresh=0.5, split_scale=0.1, max_clone=16, max_gaussians=100,
    )
    assert packed is not None
    means2, _, _, _, _, n_spawned = packed
    # clone gaussian 0 (+1) and split gaussian 2 into 2 (+1 net) → 6
    assert int(means2.shape[0]) == 6
    assert int(n_spawned) == 2


@pytest.mark.skipif(
    __import__("splat_explorer.repair_gsfix", fromlist=["gsplat_refine_available"]).gsplat_refine_available() is False,
    reason="CUDA+gsplat not available",
)
def test_gsfix3d_refine_reduces_l1(tmp_path):
    from PIL import Image

    from splat_explorer.agent.camera_rig import CameraRig
    from splat_explorer.repair import SceneRepairer, repaired_render_name
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
        backend=GsplatGsfix3dRepair(iters=8, densify=False),
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
