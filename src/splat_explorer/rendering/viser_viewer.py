"""Interactive debug viewer using viser (browser-based, no GPU required).

Serves the splat as real gaussians via viser's WebGL splat renderer, so you
can fly around the scene, sanity-check the SOG decoding, and pick sensible
start poses / up axes for the agent. Big scenes are subsampled since the
browser can't handle millions of splats.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..scene import GaussianScene
from .base import quats_to_covariances

logger = logging.getLogger(__name__)


def serve_viewer(
    scene: GaussianScene,
    host: str = "0.0.0.0",
    port: int = 8080,
    max_splats: int = 0,  # 0 = serve everything
    up_axis: str = "-y",
):
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is not installed — pip install '.[viewer]'") from exc

    from .base import up_vector

    n = scene.num_gaussians
    if max_splats and n > max_splats:
        idx = np.random.default_rng(0).choice(n, size=max_splats, replace=False)
        logger.info("Subsampling %d -> %d splats for the browser", n, max_splats)
    else:
        idx = np.arange(n)
        logger.info("Serving all %d splats to the browser", n)

    server = viser.ViserServer(host=host, port=port)
    server.scene.add_gaussian_splats(
        "/splat",
        centers=scene.means[idx],
        rgbs=scene.colors[idx],
        opacities=scene.opacities[idx, None],
        covariances=quats_to_covariances(scene.quats[idx], scene.scales[idx]),
    )
    server.scene.add_frame("/origin", axes_length=0.5, axes_radius=0.01)

    # Start clients inside the room, gravity-aligned, instead of viser's
    # default exterior orbit pose.
    up = up_vector(up_axis).astype(np.float64)
    center = scene.robust_centroid().astype(np.float64)
    seed = np.array([0.0, 0.0, -1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    forward = seed - up * np.dot(seed, up)
    forward /= np.linalg.norm(forward)

    @server.on_client_connect
    def _(client: "viser.ClientHandle") -> None:
        client.camera.up_direction = up
        client.camera.position = center
        client.camera.look_at = center + forward

    logger.info("Viser viewer running at http://%s:%d", host, port)
    # TODO: overlay the agent's camera frustum + trajectory during episodes.
    while True:
        time.sleep(1.0)
