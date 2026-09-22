"""Fixed-topology 3DGS continuation on ArtiFixer-generated views.

Asset-only adaptation, not the official fresh 3DGRUT reconstruction recipe.
All represented parameters train jointly; no VLM-controlled gradient routing.
"""
from __future__ import annotations
import numpy as np

BACKGROUND = (.12, .12, .13)


def fit_views(scene, cameras, targets, *, iterations, should_stop, on_progress, device="cuda"):
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
        return {"target_l1": float(np.mean(l1)), "target_psnr_db": float(-10*np.log10(error)),
                "per_view_l1": l1}
    before = measure()
    rng = np.random.default_rng(0)
    order = []
    visits = np.zeros(len(cameras), dtype=int)
    for step in range(iterations):
        if should_stop():
            raise InterruptedError("Extended repair stopped; candidate discarded")
        if not order:
            order = list(rng.permutation(len(cameras)))
        i = int(order.pop())
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
            "trainable_parameters": ["means", "scales", "quats", "opacities", "colors"],
            "loss": "0.8*L1 + 0.2*(1-SSIM)", "view_updates": visits.tolist()}
