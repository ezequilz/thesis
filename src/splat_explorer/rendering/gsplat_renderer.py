"""Real 3DGS rasterization via gsplat (https://github.com/nerfstudio-project/gsplat).

STUB STATUS: written against the gsplat 1.x rasterization API but not yet
exercised — it requires an NVIDIA GPU with CUDA (install the [gpu] extra and
use docker/Dockerfile.gpu). Validate outputs against the viser viewer before
trusting renders. Higher-order SH is not wired up yet (the SOG loader
currently only decodes the DC color term).
"""

from __future__ import annotations

import numpy as np

from ..scene import GaussianScene
from .base import Camera


class GsplatRenderer:
    def __init__(self, scene: GaussianScene, background: tuple[float, float, float] = (0.12, 0.12, 0.13)):
        try:
            import torch
            import gsplat  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The gsplat backend requires torch and gsplat (pip install '.[gpu]'), "
                "plus an NVIDIA GPU. Use renderer.backend=cpu_points otherwise."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat rasterization requires CUDA, but no GPU is available.")

        self._torch = torch
        device = torch.device("cuda")
        self.means = torch.from_numpy(scene.means).to(device)
        self.quats = torch.from_numpy(scene.quats).to(device)      # (w, x, y, z)
        self.scales = torch.from_numpy(scene.scales).to(device)
        self.opacities = torch.from_numpy(scene.opacities).to(device)
        self.colors = torch.from_numpy(scene.colors).to(device)
        self.background = torch.tensor(background, device=device)

    def render(self, camera: Camera) -> np.ndarray:
        import gsplat

        torch = self._torch
        device = self.means.device
        viewmat = torch.from_numpy(camera.w2c).unsqueeze(0).to(device)
        K = torch.from_numpy(camera.intrinsics).unsqueeze(0).to(device)

        with torch.no_grad():
            renders, _, _ = gsplat.rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.colors,
                viewmats=viewmat,
                Ks=K,
                width=camera.width,
                height=camera.height,
                backgrounds=self.background.unsqueeze(0),
            )
        img = renders[0].clamp(0, 1).mul(255).byte().cpu().numpy()
        return img
