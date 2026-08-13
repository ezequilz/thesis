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

logger = logging.getLogger(__name__)


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


def serve_viewer(scene: GaussianScene, host: str = "0.0.0.0", port: int = 8080, max_splats: int = 1_000_000):
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is not installed — pip install '.[viewer]'") from exc

    n = scene.num_gaussians
    if n > max_splats:
        idx = np.random.default_rng(0).choice(n, size=max_splats, replace=False)
        logger.info("Subsampling %d -> %d splats for the browser", n, max_splats)
    else:
        idx = np.arange(n)

    server = viser.ViserServer(host=host, port=port)
    server.scene.add_gaussian_splats(
        "/splat",
        centers=scene.means[idx],
        rgbs=scene.colors[idx],
        opacities=scene.opacities[idx, None],
        covariances=quats_to_covariances(scene.quats[idx], scene.scales[idx]),
    )
    server.scene.add_frame("/origin", axes_length=0.5, axes_radius=0.01)
    logger.info("Viser viewer running at http://%s:%d", host, port)
    # TODO: overlay the agent's camera frustum + trajectory during episodes.
    while True:
        time.sleep(1.0)
