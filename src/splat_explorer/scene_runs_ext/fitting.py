"""Fixed-topology 3DGS continuation on ArtiFixer-generated views.

Asset-only adaptation, not the official fresh 3DGRUT reconstruction recipe.
All represented parameters train jointly; no VLM-controlled gradient routing.
"""
from __future__ import annotations
import numpy as np

BACKGROUND = (.12, .12, .13)


def initial_scale_ceiling(scene):
    """Compute once from the incoming run asset, never from a fitted checkpoint."""
    scales = np.asarray(scene.scales)
    if not scales.size or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Initial Gaussian scales must be finite and positive")
    return max(float(np.max(scales)) * 2, 1e-4)


def view_schedule(count, iterations, edited_indices=()):
    """Balance supervision sources rather than overweighting correlated frames.

    Both pools are shuffled without replacement. With both sources present,
    half the updates use direct edits and half use generated views, independent
    of trajectory length. No source weighting is inferred from image quality.
    """
    edited = sorted(set(edited_indices))
    if count < 1 or iterations < 1 or any(type(i) is not int or not 0 <= i < count for i in edited):
        raise ValueError("Invalid fitting view schedule")
    edited_set = set(edited)
    generated = [i for i in range(count) if i not in edited_set]
    pools = [p for p in (edited, generated) if p]
    rng = np.random.default_rng(0)
    pending = [[] for _ in pools]
    for step in range(iterations):
        source = step % len(pools)
        if not pending[source]:
            pending[source] = list(rng.permutation(pools[source]))
        yield int(pending[source].pop())


def shape_metrics(scales):
    values = np.asarray(scales)
    ratio = values.max(axis=1) / values.min(axis=1)
    return {"max_scale": float(values.max()), "max_axis_ratio": float(ratio.max()),
            "axis_ratio_p99": float(np.quantile(ratio, .99)),
            "axis_ratio_above_100": int(np.count_nonzero(ratio > 100))}


def fit_views(scene, cameras, targets, *, iterations, should_stop, on_progress, device="cuda",
              scale_ceiling=None, edited_indices=()):
    import torch
    import gsplat
    from ..repair_gsfix import photometric_loss
    if not cameras or len(cameras) != len(targets):
        raise ValueError("Fitting requires one target per camera")
    if iterations < 1:
        raise ValueError("Fitting requires at least one iteration")
    for c, target in zip(cameras, targets):
        if np.asarray(target).shape != (c.height, c.width, 3):
            raise ValueError("Fitting target dimensions must match its camera")
    def tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).clone()
    means = tensor(scene.means).requires_grad_(True)
    scales = tensor(scene.scales).clamp_min(1e-8).log().requires_grad_(True)
    quats = tensor(scene.quats).requires_grad_(True)
    opacity = torch.logit(tensor(scene.opacities).clamp(1e-5, 1-1e-5)).requires_grad_(True)
    colors = tensor(scene.colors[:, :3]).requires_grad_(True)
    groups = [{"params": [colors], "lr": .0025},
              {"params": [means], "lr": .00016}, {"params": [scales], "lr": .002},
              {"params": [quats], "lr": .001}, {"params": [opacity], "lr": .01}]
    optimizer = torch.optim.Adam(groups)
    # Targets stay on CPU: video models and fitting do not coexist in GPU memory.
    targets = [np.asarray(x, dtype=np.float32)/255 for x in targets]
    bg = tensor(BACKGROUND)[None]
    max_scale = initial_scale_ceiling(scene) if scale_ceiling is None else float(scale_ceiling)
    if not np.isfinite(max_scale) or max_scale <= 0:
        raise ValueError("Scale ceiling must be finite and positive")
    edited_indices = list(edited_indices)
    schedule = view_schedule(len(cameras), iterations, edited_indices)
    edited_set = set(edited_indices)
    shape_before = shape_metrics(scene.scales)
    def render(i):
        c = cameras[i]
        out, _, _ = gsplat.rasterization(
            means=means, quats=torch.nn.functional.normalize(quats, dim=-1),
            scales=scales.exp(), opacities=opacity.sigmoid(), colors=colors,
            viewmats=tensor(c.w2c)[None], Ks=tensor(c.intrinsics)[None],
            width=c.width, height=c.height, backgrounds=bg, packed=False,
        )
        return out[0]
    def measure():
        mse, l1 = [], []
        with torch.no_grad():
            for i in range(len(cameras)):
                if should_stop():
                    raise InterruptedError("Extended repair stopped during fitting metrics")
                delta = render(i)-tensor(targets[i])
                l1.append(float(delta.abs().mean()))
                mse.append(float(delta.square().mean()))
        error = max(float(np.mean(mse)), 1e-12)
        result = {"target_l1": float(np.mean(l1)), "target_psnr_db": float(-10*np.log10(error)),
                  "per_view_l1": l1}
        for name, indices in (("edited", sorted(edited_set)),
                              ("generated", [i for i in range(len(cameras)) if i not in edited_set])):
            if indices:
                result[f"{name}_target_l1"] = float(np.mean([l1[i] for i in indices]))
        if "edited_target_l1" in result and "generated_target_l1" in result:
            result["source_balanced_target_l1"] = .5 * (result["edited_target_l1"] + result["generated_target_l1"])
        return result
    before = measure()
    visits = np.zeros(len(cameras), dtype=int)
    for step, i in enumerate(schedule):
        if should_stop():
            raise InterruptedError("Extended repair stopped; candidate discarded")
        visits[i] += 1
        optimizer.zero_grad(set_to_none=True)
        loss, l1 = photometric_loss(render(i), tensor(targets[i]), torch, .2)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite multiview loss; candidate discarded")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            colors.clamp_(0, 1)
            scales.clamp_(np.log(1e-8), np.log(max_scale))
            opacity.clamp_(-12, 12)
        if step % 20 == 0:
            on_progress({"phase": "multiview_fit", "iteration": step+1,
                         "iterations": iterations, "loss": float(loss.detach()), "target_l1": float(l1.detach())})
    after = measure()
    for value in (means, scales, quats, opacity, colors):
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite candidate parameters")
    def cpu(x):
        return x.detach().cpu().numpy().astype(np.float32)
    scene.means = cpu(means)
    scene.scales = cpu(scales.exp())
    scene.quats = cpu(torch.nn.functional.normalize(quats, dim=-1))
    scene.opacities = cpu(opacity.sigmoid())
    scene.colors = cpu(colors)
    return {"before": before, "after": after, "n_iters": iterations,
            "n_views": len(cameras), "n_gaussians": scene.num_gaussians,
            "metric_reference": "synthetic fitting targets, not ground truth",
            "quality_gate": "disabled", "densification": False,
            "fit_recipe": "fixed-topology-gsplat-l1-ssim",
            "scale_ceiling": max_scale,
            "shape_before": shape_before, "shape_after": shape_metrics(scene.scales),
            "sampling": "balanced-edited-generated" if edited_set and len(edited_set) < len(cameras) else "uniform",
            "edited_updates": int(visits[list(edited_set)].sum()) if edited_set else 0,
            "trainable_parameters": ["means", "scales", "quats", "opacities", "colors"],
            "loss": "0.8*L1 + 0.2*(1-SSIM)", "view_updates": visits.tolist()}
