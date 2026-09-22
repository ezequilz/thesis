"""Minimal shared-optimizer multiview fit; no selective weighting or densification."""
from __future__ import annotations
import numpy as np

BACKGROUND = (.12, .12, .13)


def fit_views(scene, cameras, targets, *, iterations, intervention, should_stop, on_progress):
    import torch
    import gsplat
    device = "cuda"
    def tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).clone()
    geometry = intervention == "structure"
    means = tensor(scene.means).requires_grad_(geometry)
    scales = tensor(scene.scales).clamp_min(1e-8).log().requires_grad_(geometry)
    quats = tensor(scene.quats).requires_grad_(geometry)
    opacity = torch.logit(tensor(scene.opacities).clamp(1e-5, 1-1e-5)).requires_grad_(geometry)
    colors = tensor(scene.colors[:, :3]).requires_grad_(True)
    groups = [{"params": [colors], "lr": .0025}]
    if geometry:
        groups += [{"params": [means], "lr": .00016}, {"params": [scales], "lr": .002},
                   {"params": [quats], "lr": .001}, {"params": [opacity], "lr": .01}]
    optimizer = torch.optim.Adam(groups)
    # Targets stay on CPU: video models and fitting do not coexist in GPU memory.
    targets = [np.asarray(x, dtype=np.float32)/255 for x in targets]
    bg = tensor(BACKGROUND)[None]
    max_scale = max(float(np.max(scene.scales))*2, 1e-4)
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
        return {"target_l1": float(np.mean(l1)), "target_psnr_db": float(-10*np.log10(error))}
    before = measure()
    rng = np.random.default_rng(0)
    order = []
    for step in range(iterations):
        if should_stop():
            raise InterruptedError("Extended repair stopped; candidate discarded")
        if not order:
            order = list(rng.permutation(len(cameras)))
        i = int(order.pop())
        optimizer.zero_grad(set_to_none=True)
        loss = (render(i)-tensor(targets[i])).abs().mean()
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
                         "iterations": iterations, "target_l1": float(loss.detach())})
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
            "quality_gate": "disabled", "densification": False}
