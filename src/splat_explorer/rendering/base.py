"""Camera model and renderer interface.

Convention: the camera frame is OpenCV-style — x right, y down, z forward.
Poses are stored camera-to-world; world coordinates follow the loaded scene
(y-up for SOG assets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..scene import GaussianScene

_AXES = {
    "+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
    "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
    "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0]),
}


def up_vector(axis: str) -> np.ndarray:
    try:
        return _AXES[axis].copy()
    except KeyError:
        raise ValueError(f"up_axis must be one of {sorted(_AXES)}, got {axis!r}")


def quats_to_covariances(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Covariance = R diag(s^2) R^T for (w,x,y,z) quats. Returns (N, 3, 3)."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = np.empty((len(quats), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    S2 = scales[:, None, :] ** 2  # broadcast diag(s^2)
    return np.einsum("nij,njk->nik", R * S2, R.transpose(0, 2, 1))


@dataclass
class Camera:
    """Pinhole camera with a camera-to-world pose."""

    position: np.ndarray                 # (3,) world-space eye position
    rotation: np.ndarray                 # (3, 3) camera-to-world rotation
    width: int = 960
    height: int = 720
    fov_deg: float = 75.0                # horizontal field of view

    @property
    def fx(self) -> float:
        return self.width / (2.0 * np.tan(np.radians(self.fov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.width / 2.0],
             [0, self.fy, self.height / 2.0],
             [0, 0, 1.0]],
            dtype=np.float32,
        )

    @property
    def c2w(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.position
        return m

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)

    @staticmethod
    def look_at(
        position: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        **kwargs,
    ) -> "Camera":
        """Build a camera at `position` looking toward `target`."""
        forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward], axis=1).astype(np.float32)
        return Camera(position=np.asarray(position, dtype=np.float32), rotation=rotation, **kwargs)


class Renderer(Protocol):
    """Renders a GaussianScene from a Camera into an RGB uint8 image."""

    def render(self, camera: Camera) -> np.ndarray:
        """Return an (H, W, 3) uint8 image."""
        ...


def make_renderer(scene: GaussianScene, renderer_cfg) -> Renderer:
    """Factory dispatching on config renderer.backend."""
    backend = renderer_cfg.backend
    if backend == "cpu_splats":
        from .cpu_splat_renderer import CpuSplatRenderer

        return CpuSplatRenderer(
            scene, max_splat_radius_px=renderer_cfg.get("max_splat_radius_px", 40)
        )
    if backend == "cpu_points":
        from .cpu_point_renderer import CpuPointRenderer

        return CpuPointRenderer(scene, point_radius_px=renderer_cfg.get("point_radius_px", 1))
    if backend == "gsplat":
        from .gsplat_renderer import GsplatRenderer

        return GsplatRenderer(scene)
    raise ValueError(f"Unknown renderer backend: {backend!r}")
