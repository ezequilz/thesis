"""Original-scene RGB, opacity and depth for cached diffusion inputs."""
from __future__ import annotations

import gc
import numpy as np


class BundleRenderer:
    def __init__(self, scene):
        from ..rendering.gsplat_renderer import GsplatRenderer
        self.renderer = GsplatRenderer(scene)

    def render(self, camera):
        import gsplat
        renderer = self.renderer
        torch = renderer._torch
        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=renderer.means.device)
        with torch.no_grad():
            image, alpha, _ = gsplat.rasterization(
                means=renderer.means, quats=renderer.quats, scales=renderer.scales,
                opacities=renderer.opacities, colors=renderer.colors,
                viewmats=tensor(camera.w2c)[None], Ks=tensor(camera.intrinsics)[None],
                width=camera.width, height=camera.height,
                backgrounds=renderer.background[None], render_mode="RGB+ED", packed=False,
            )
        rgb = image[0, :, :, :3].clamp(0, 1).mul(255).byte().cpu().numpy()
        opacity = alpha[0, :, :, 0].cpu().numpy().astype(np.float32)
        depth = image[0, :, :, 3].cpu().numpy().astype(np.float32)
        depth[opacity < .15] = np.inf
        return rgb, opacity, depth


def release_cuda():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
