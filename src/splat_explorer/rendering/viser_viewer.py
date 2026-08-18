"""Interactive debug viewer using viser (browser-based, no GPU required).

Serves the splat as real gaussians via viser's WebGL splat renderer, so you
can fly around the scene, sanity-check the SOG decoding, and pick sensible
start poses / up axes for the agent. Big scenes are subsampled since the
browser can't handle millions of splats.

Live agent overlay: the viewer polls outputs/live/agent_state.json (written by
the episode dashboard after every step) and draws the agent's camera frustum —
with the current screenshot inside — plus its trajectory. A "Follow agent"
checkbox snaps connected browser cameras to the agent pose after each step.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from ..scene import GaussianScene
from .base import quats_to_covariances

logger = logging.getLogger(__name__)

LIVE_STATE_PATH = Path("outputs/live/agent_state.json")


def _load_live_state(path: Path) -> dict | None:
    """Read the agent state file, tolerating missing/partially-written files."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _frustum_image(state: dict, max_width: int = 200) -> np.ndarray | None:
    """Downscaled copy of the agent's current frame to embed in the frustum.

    Tries the absolute path first, then the repo-relative one — the writer
    (dashboard) and this viewer may run in different roots (Docker vs local).
    """
    from PIL import Image

    for key in ("frame", "frame_rel"):
        path = state.get(key)
        if not path:
            continue
        try:
            img = Image.open(path)
        except OSError:
            continue
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
        return np.asarray(img.convert("RGB"))
    return None


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

    # --- live agent overlay (fed by the episode dashboard) -------------------
    follow_agent = server.gui.add_checkbox(
        "Follow agent", initial_value=False,
        hint="Snap the browser camera to the agent pose after every step.")
    show_frame = server.gui.add_checkbox(
        "Frame in frustum", initial_value=True,
        hint="Show the agent's current screenshot inside its camera frustum.")
    agent_status = server.gui.add_markdown("**Agent**: no live episode data yet.")

    logger.info("Viser viewer running at http://%s:%d", host, port)
    last_key: tuple | None = None
    while True:
        time.sleep(0.5)
        state = _load_live_state(LIVE_STATE_PATH)
        if state is None:
            continue
        # updated_at changes on every publish (new live step OR a step pinned
        # from the dashboard), so clicked steps re-pose the frustum too.
        key = (state.get("episode"), state.get("step"), state.get("updated_at"), show_frame.value)
        if key == last_key:
            continue
        last_key = key

        position = np.asarray(state["position"], dtype=np.float64)
        image = _frustum_image(state) if show_frame.value else None
        server.scene.add_camera_frustum(
            "/agent/camera",
            fov=np.radians(state.get("fov_deg", 75.0)),
            aspect=state.get("aspect", 4 / 3),
            scale=0.4,
            color=(255, 80, 80),
            wxyz=np.asarray(state["wxyz"], dtype=np.float64),
            position=position,
            image=image,
        )
        trajectory = np.asarray(state.get("trajectory", []), dtype=np.float64)
        if len(trajectory) >= 2:
            server.scene.add_spline_catmull_rom(
                "/agent/trajectory", points=trajectory,
                color=(255, 80, 80), line_width=3.0)
        agent_status.content = (
            f"**Agent**: episode `{state.get('episode', '?')}` step {state.get('step', '?')}  \n"
            f"{state.get('pose', '')}"
        )
        if follow_agent.value:
            look_at = position + np.asarray(state.get("view_dir", forward), dtype=np.float64)
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.look_at = look_at
