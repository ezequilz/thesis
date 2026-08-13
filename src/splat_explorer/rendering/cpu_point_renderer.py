"""Dependency-free CPU renderer: projects gaussian centers as depth-sorted points.

This is deliberately crude — no covariance rasterization, no alpha blending —
but it runs anywhere (Docker on a Mac included), which makes it the default
backend for smoke tests and for exercising the agent loop end to end.
Swap in the gsplat backend on a CUDA machine for faithful renders.
"""

from __future__ import annotations

import numpy as np

from ..scene import GaussianScene
from .base import Camera


class CpuPointRenderer:
    def __init__(
        self,
        scene: GaussianScene,
        point_radius_px: int = 1,
        near: float = 0.05,
        background: tuple[int, int, int] = (30, 30, 34),
    ):
        self.scene = scene
        self.point_radius_px = max(0, int(point_radius_px))
        self.near = near
        self.background = background

    def render(self, camera: Camera) -> np.ndarray:
        w2c = camera.w2c
        pts_cam = self.scene.means @ w2c[:3, :3].T + w2c[:3, 3]

        z = pts_cam[:, 2]
        in_front = z > self.near
        pts_cam = pts_cam[in_front]
        colors = self.scene.colors[in_front]
        z = z[in_front]

        u = (camera.fx * pts_cam[:, 0] / z + camera.width / 2.0).astype(np.int32)
        v = (camera.fy * pts_cam[:, 1] / z + camera.height / 2.0).astype(np.int32)
        r = self.point_radius_px
        on_screen = (u >= r) & (u < camera.width - r) & (v >= r) & (v < camera.height - r)
        u, v, z, colors = u[on_screen], v[on_screen], z[on_screen], colors[on_screen]

        # Painter's algorithm: draw far-to-near so near points win.
        order = np.argsort(-z)
        u, v, colors = u[order], v[order], colors[order]

        img = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
        img[:] = self.background
        rgb = (colors * 255).astype(np.uint8)
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                img[v + dv, u + du] = rgb
        return img
