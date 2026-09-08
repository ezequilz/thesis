"""CPU-safe tests for the experimental GSFix3D vis-prune lift."""

from __future__ import annotations

import numpy as np
import pytest

from splat_explorer.rendering.base import Camera
from splat_explorer.repair_gsfix3d import GsplatGsfix3dRepair, instantiate_cuda_repair
from splat_explorer.repair_gsfix3d_visprune import (
    BACKEND_ID,
    GsplatGsfix3dVisPruneRepair,
    _repeat_along_n,
    _viewspace_grad_norm,
    apply_updatable_grads,
    densify_clone_split,
    error_mask_keep,
    front_contributing_mask,
    micro_rotation_cameras,
    project_centers,
)


def _cam(width=64, height=48) -> Camera:
    return Camera.look_at(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        width=width,
        height=height,
        fov_deg=75.0,
    )


def test_instantiate_visprune_alias():
    vis = instantiate_cuda_repair(method="visprune")
    assert isinstance(vis, GsplatGsfix3dVisPruneRepair)
    assert vis._result_backend() == BACKEND_ID
    assert vis.freeze_occluded is False
    assert vis.error_prune is True
    assert vis._updatable_mask(None, None, None, None, None) is None


def test_paper_instantiate_is_not_visprune():
    paper = instantiate_cuda_repair(method="gsfix-gsplat")
    assert type(paper) is GsplatGsfix3dRepair
    assert GsplatGsfix3dVisPruneRepair._apply is not GsplatGsfix3dRepair._apply


def test_front_contributing_mask_freezes_far_gaussian():
    cam = _cam()
    means = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    opacities = np.array([0.9, 0.9])
    u, v, z, on = project_centers(means, cam)
    assert on.all()
    assert z[0] < z[1]
    mask = front_contributing_mask(means, cam, opacities)
    assert mask[0]
    assert not mask[1]


def test_occlusion_freeze_zeros_far_grad():
    torch = pytest.importorskip("torch")
    cam = _cam()
    means_np = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    means = torch.tensor(means_np, requires_grad=True)
    (means.sum()).backward()
    assert means.grad is not None
    assert float(means.grad.abs().sum()) > 0
    mask_np = front_contributing_mask(means_np, cam, np.array([0.9, 0.9]))
    apply_updatable_grads([means], torch.from_numpy(mask_np))
    assert float(means.grad[0].abs().sum()) > 0
    assert float(means.grad[1].abs().sum()) == 0


def test_error_mask_prune_drops_floater_keeps_surface():
    cam = _cam(width=64, height=48)
    rng = np.random.default_rng(0)
    n_surf = 40
    xy = rng.uniform(-0.6, 0.6, size=(n_surf, 2))
    surface = np.column_stack([xy[:, 0], xy[:, 1], np.full(n_surf, 2.0)])
    floater = np.array([[0.0, 0.0, 0.4]])
    means = np.vstack([floater, surface])
    residual = np.zeros((48, 64), dtype=np.float64)
    u, v, _, on = project_centers(floater, cam)
    assert on[0]
    ui = int(np.clip(np.floor(u[0]), 0, 63))
    vi = int(np.clip(np.floor(v[0]), 0, 47))
    residual[vi, ui] = 1.0
    keep = error_mask_keep(
        means, cam, residual,
        error_thresh=0.12,
        max_frac=0.5,
        min_keep=8,
        depth_margin=0.05,
    )
    assert not keep[0]
    assert keep[1:].all()


def test_error_mask_prune_keeps_low_residual_surface():
    cam = _cam()
    means = np.array([[0.0, 0.0, 2.0], [0.3, 0.0, 2.0]], dtype=np.float64)
    residual = np.zeros((48, 64), dtype=np.float64)
    keep = error_mask_keep(
        means, cam, residual,
        error_thresh=0.12, max_frac=0.5, min_keep=1,
    )
    assert keep.all()


def test_micro_rotation_cameras_are_four_distinct_views():
    cam = _cam()
    extra = micro_rotation_cameras(cam, yaw_deg=12.0, pitch_deg=8.0)
    assert len(extra) == 4
    fwds = [c.rotation[:, 2] for c in extra]
    dots = [float(np.dot(fwds[i], cam.rotation[:, 2])) for i in range(4)]
    assert all(d < 0.999 for d in dots)
    pos = [tuple(np.round(c.position, 6)) for c in extra]
    assert len(set(pos)) == 1


def test_repeat_along_n_does_not_promote_1d():
    """``.repeat(2, 1)`` on (K,) prepends a dim; densify cat then dies 1 vs 2."""

    class _Fake:
        def __init__(self, ndim):
            self.ndim = ndim
            self.sizes = None

        def repeat(self, *sizes):
            self.sizes = sizes
            return self

    t1 = _Fake(1)
    assert _repeat_along_n(t1, 2) is t1
    assert t1.sizes == (2,)
    t2 = _Fake(2)
    _repeat_along_n(t2, 2)
    assert t2.sizes == (2, 1)
    t3 = _Fake(3)
    _repeat_along_n(t3, 2)
    assert t3.sizes == (2, 1, 1)


def test_viewspace_grad_norm_unpacked_and_packed():
    torch = pytest.importorskip("torch")
    g = torch.zeros(1, 4, 2)
    g[0, 1, 0] = 0.4
    means2d = type("M", (), {"grad": g, "absgrad": None})()
    mag = _viewspace_grad_norm(means2d, 4, torch, width=100, height=80)
    assert mag is not None
    assert mag.shape == (4,)
    assert float(mag[1]) == pytest.approx(0.4 * 50.0)
    assert float(mag[0]) == 0.0

    nnz = torch.tensor([[0.1, 0.0], [0.2, 0.0]])
    packed = type("M", (), {"grad": nnz, "absgrad": None})()
    ids = torch.tensor([2, 0])
    mag2 = _viewspace_grad_norm(
        packed, 4, torch, info={"gaussian_ids": ids}, width=10, height=10,
    )
    assert mag2 is not None
    assert float(mag2[2]) == pytest.approx(0.1 * 5.0)
    assert float(mag2[0]) == pytest.approx(0.2 * 5.0)
    assert float(mag2[1]) == 0.0


def test_densify_clone_split_keeps_1d_opacities():
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
    means2, _, _, _, logit2, n_spawned = packed
    assert int(means2.shape[0]) == 6
    assert int(n_spawned) == 2
    assert logit2.ndim == 1
    assert int(logit2.shape[0]) == 6

    logit_col = torch.zeros(n, 1, requires_grad=True)
    packed_col = densify_clone_split(
        torch, means, quats, f_dc, log_scales, logit_col, mag,
        thresh=0.5, split_scale=0.1, max_clone=16, max_gaussians=100,
    )
    assert packed_col is not None
    assert packed_col[4].shape == (6, 1)


def test_visprune_apply_until_densify_only_first_chunk():
    backend = GsplatGsfix3dVisPruneRepair(max_chunks=0, densify=True)
    flags: list[bool] = []

    def fake_apply(*_a, **_k):
        flags.append(bool(backend.densify))
        return {"n_iters": 20, "l1_before": 0.5, "l1_after": 0.4, "backend": BACKEND_ID}

    backend.apply = fake_apply
    out = backend.apply_until(
        object(), object(), object(), object(),
        should_stop=lambda: len(flags) >= 3,
    )
    assert flags == [True, False, False]
    assert out["n_chunks"] == 3
    assert out["backend"] == BACKEND_ID
    assert backend.densify is True
