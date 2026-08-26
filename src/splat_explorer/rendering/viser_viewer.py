"""Interactive debug viewer using viser (browser-based, no GPU required).

Serves the splat as real gaussians via viser's WebGL splat renderer, so you
can fly around the scene, sanity-check the SOG decoding, and pick sensible
start poses / up axes for the agent. Big scenes are subsampled since the
browser can't handle millions of splats.

Live agent overlay: the viewer polls outputs/live/agent_state.json (written by
the episode dashboard after every step) and draws the agent's camera frustum —
with the current screenshot inside — plus its trajectory. A "Follow agent"
checkbox snaps connected browser cameras to the agent pose after each step.

Harness capture: a small HTTP API on render_port (default 8081) lets the
episode loop request a WebGL frame from a connected visor tab, so the VLM
sees the same image the user does.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

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


def _start_render_api(host: str, port: int, capture, viewer_url: str) -> ThreadingHTTPServer:
    """HTTP API the harness uses to grab a visor frame.

    GET  /health  -> {clients, capture_ready, viewports, viewer_url}
    POST /render  -> JPEG of the WebGL view at the requested camera
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            logger.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj: dict, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self._send_json({"error": "not found"}, 404)
                return
            n, err, viewports = capture.client_info()
            ready = sum(1 for v in viewports if v.get("usable"))
            self._send_json({
                "clients": n,
                "capture_ready": ready,
                "viewports": viewports,
                "viewer_url": viewer_url,
                "error": err,
            })

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/render":
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                params = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, 400)
                return
            try:
                jpeg = capture.render(params)
            except CaptureError as exc:
                self._send_json({"error": str(exc)}, 503)
                return
            except Exception as exc:
                logger.exception("Visor capture failed")
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self._send(200, jpeg, "image/jpeg")

    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot bind visor capture API on {host}:{port} ({exc}). "
            "Stop whatever is using that port or set viewer.render_port."
        ) from exc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    logger.info("Visor capture API at http://%s:%d/render", host if host != "0.0.0.0" else "localhost", port)
    return httpd


class CaptureError(RuntimeError):
    pass


class _VisorCapture:
    """Hides overlay gizmos, captures the full-page visor tab, restores gizmos.

    The dashboard preview iframe is a wide, short canvas. viser's get_render
    samples that canvas, so using it produces the squashed/mushy agent frames.
    We only capture from a client whose viewport is tall enough and not
    ultra-wide — i.e. the dedicated http://localhost:8080 tab.
    """

    GET_RENDER_TIMEOUT_S = 20.0
    # Skip the dashboard preview strip (typically ~3–6∶1) and tiny windows.
    MAX_ASPECT = 2.2
    MIN_VIEWPORT_FRAC = 0.8

    def __init__(self, server, lock: threading.Lock, overlay_handles: list):
        self._server = server
        self._lock = lock
        self._overlay = overlay_handles
        self._good_client_id: int | None = None

    def client_info(self) -> tuple[int, str | None, list[dict]]:
        try:
            clients = self._server.get_clients()
        except Exception as exc:
            return 0, str(exc), []
        viewports = []
        for cid, client in clients.items():
            w, h = self._viewport(client)
            viewports.append({
                "id": cid, "width": w, "height": h,
                "usable": self._usable(w, h, 960, 720),
            })
        return len(clients), None, viewports

    def client_count(self) -> tuple[int, str | None]:
        n, err, _ = self.client_info()
        return n, err

    @staticmethod
    def _viewport(client) -> tuple[int, int]:
        try:
            return int(client.camera.image_width), int(client.camera.image_height)
        except Exception:
            return 0, 0

    @classmethod
    def _usable(cls, width: int, height: int, need_w: int, need_h: int) -> bool:
        if width < 16 or height < 16:
            return False
        if height < need_h * cls.MIN_VIEWPORT_FRAC or width < need_w * cls.MIN_VIEWPORT_FRAC:
            return False
        if width / height > cls.MAX_ASPECT:
            return False
        return True

    def _pick_clients(self, clients: dict, need_w: int, need_h: int) -> list:
        ranked = []
        for cid, client in clients.items():
            w, h = self._viewport(client)
            if not self._usable(w, h, need_w, need_h):
                continue
            preferred = cid == self._good_client_id
            ranked.append((not preferred, -h, -w, cid, client))
        ranked.sort()
        return [(cid, client) for _, _, _, cid, client in ranked]

    def render(self, params: dict) -> bytes:
        width = int(params["width"])
        height = int(params["height"])
        if width < 16 or height < 16 or width > 1920 or height > 1440:
            raise CaptureError(f"Unsupported capture size {width}x{height}")
        with self._lock:
            clients = self._server.get_clients()
            if not clients:
                raise CaptureError(
                    "No visor tab connected. Open http://localhost:8080 in its own "
                    "window (not the dashboard preview) and keep it visible."
                )
            order = self._pick_clients(clients, width, height)
            if not order:
                sizes = [
                    f"{cid}:{self._viewport(c)[0]}x{self._viewport(c)[1]}"
                    for cid, c in clients.items()
                ]
                raise CaptureError(
                    "No full-page visor tab to capture from "
                    f"(connected viewports: {', '.join(sizes) or 'none'}). "
                    "Open http://localhost:8080 as its own tab, make the window "
                    f"at least ~{width}x{height}, and do not use the dashboard preview."
                )
            hidden = []
            for handle in self._overlay:
                if handle is not None and getattr(handle, "visible", False):
                    handle.visible = False
                    hidden.append(handle)
            img, errors = None, []
            try:
                self._server.flush()
                for cid, client in order:
                    try:
                        img = client.get_render(
                            height=height,
                            width=width,
                            wxyz=np.asarray(params["wxyz"], dtype=np.float64),
                            position=np.asarray(params["position"], dtype=np.float64),
                            fov=float(params["fov"]),
                            transport_format="jpeg",
                            timeout=self.GET_RENDER_TIMEOUT_S,
                        )
                        self._good_client_id = cid
                        logger.info("Visor capture via client %s (%dx%d viewport)",
                                    cid, *self._viewport(client))
                        break
                    except Exception as exc:
                        errors.append(f"client {cid}: {type(exc).__name__}: {exc}")
                        logger.warning("Visor capture via client %s failed: %s", cid, exc)
            finally:
                for handle in hidden:
                    handle.visible = True
                self._server.flush()
        if img is None or getattr(img, "size", 0) == 0:
            raise CaptureError(
                "; ".join(errors) or "Visor returned an empty frame"
            )
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = arr[..., :3]
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(
            buf, format="JPEG", quality=90
        )
        return buf.getvalue()


def serve_viewer(
    scene: GaussianScene,
    host: str = "0.0.0.0",
    port: int = 8080,
    max_splats: int = 0,  # 0 = serve everything
    up_axis: str = "-y",
    render_port: int = 8081,
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
    origin = server.scene.add_frame("/origin", axes_length=0.5, axes_radius=0.01)
    try:
        server.scene.world_axes.visible = False
    except Exception:
        pass

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

    overlay_lock = threading.Lock()
    overlay_handles: list = [origin]
    viewer_url = f"http://localhost:{port}"
    _start_render_api(host, render_port, _VisorCapture(server, overlay_lock, overlay_handles), viewer_url)

    logger.info("Viser viewer running at %s — keep a tab open for agent captures", viewer_url)
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
        with overlay_lock:
            cam = server.scene.add_camera_frustum(
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
            traj = None
            if len(trajectory) >= 2:
                traj = server.scene.add_spline_catmull_rom(
                    "/agent/trajectory", points=trajectory,
                    color=(255, 80, 80), line_width=3.0)
            overlay_handles[:] = [origin, cam, traj]
        agent_status.content = (
            f"**Agent**: episode `{state.get('episode', '?')}` step {state.get('step', '?')}  \n"
            f"{state.get('pose', '')}"
        )
        if follow_agent.value:
            look_at = position + np.asarray(state.get("view_dir", forward), dtype=np.float64)
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.look_at = look_at
