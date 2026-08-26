"""Top-down overview render of a (ceiling-stripped) scene.

Places a pinhole camera above the scene looking straight down along the up
axis, at an altitude chosen so the robust ground-plane bounds fit in the
frame. The caller strips the ceiling first (navigation.strip_ceiling) so the
render shows the room interior instead of the roof.
"""

from __future__ import annotations

import numpy as np

from .base import Camera, up_vector
from .cpu_splat_renderer import CpuSplatRenderer


def render_birdseye(
    scene,
    up_axis: str,
    width: int,
    height: int,
    fov_deg: float = 55.0,
    margin: float = 1.15,
    max_splat_radius_px: int = 120,
) -> tuple[np.ndarray, Camera]:
    """Render the scene from above. Returns (RGB uint8 image, camera used)."""
    from ..navigation import ground_basis  # local import to avoid a cycle

    up = up_vector(up_axis).astype(np.float64)
    e0, e1 = ground_basis(up)

    means = scene.means.astype(np.float64)
    x, y, h = means @ e0, means @ e1, means @ up
    x0, x1 = np.percentile(x, [1.0, 99.0])
    y0, y1 = np.percentile(y, [1.0, 99.0])
    h_top = float(np.percentile(h, 99.0))
    h_mid = float(np.percentile(h, 50.0))

    cx, cy = float(x0 + x1) / 2.0, float(y0 + y1) / 2.0
    extent_x, extent_y = float(x1 - x0), float(y1 - y0)

    # Altitude above the highest remaining splats so the horizontal FOV covers
    # extent_x and the (aspect-scaled) vertical FOV covers extent_y.
    tan_half = np.tan(np.radians(fov_deg) / 2.0)
    altitude = margin * max(
        extent_x / (2.0 * tan_half),
        extent_y / (2.0 * tan_half * height / width),
    )

    center = cx * e0 + cy * e1 + h_mid * up
    position = cx * e0 + cy * e1 + (h_top + altitude) * up
    camera = Camera.look_at(position, center, up=e1,
                            width=width, height=height, fov_deg=fov_deg)

    renderer = CpuSplatRenderer(scene, max_splat_radius_px=max_splat_radius_px)
    return renderer.render(camera), camera


class ExplorationMap:
    """Cached ceiling-stripped bird's-eye plus the agent's path overlay.

    The splat render is done once (at spawn / episode start). add_pose() only
    re-paints the walked path and camera frustums so the dashboard and the
    optional VLM map stay cheap to refresh after every step.
    """

    def __init__(
        self,
        base_image: np.ndarray,
        camera: Camera,
        fov_deg: float,
        up: np.ndarray,
    ):
        self.base_image = np.asarray(base_image)
        self.camera = camera
        self.fov_deg = float(fov_deg)
        self.up = np.asarray(up, dtype=np.float64)
        self.poses: list[dict] = []

    def add_pose(self, position: np.ndarray, heading: np.ndarray, step: int) -> None:
        h = np.asarray(heading, dtype=np.float64)
        n = np.linalg.norm(h)
        if n > 1e-8:
            h = h / n
        self.poses.append({
            "position": np.asarray(position, dtype=np.float64).copy(),
            "heading": h,
            "step": int(step),
        })

    def render(self) -> np.ndarray:
        from .annotate import draw_path_map

        return draw_path_map(
            self.base_image, self.camera, self.poses,
            fov_deg=self.fov_deg, up=self.up,
        )
