"""Interactive debug viewer using viser (browser-based, no GPU required).

Serves the splat as real gaussians via viser's WebGL splat renderer, so you
can fly around the scene, sanity-check the SOG decoding, and pick sensible
start poses / up axes for the agent. Big scenes are subsampled since the
browser can't handle millions of splats.

Live agent overlay: the viewer polls outputs/live/agent_state.json (written by
the episode dashboard after every step) and draws the agent's camera frustum —
with the current screenshot inside — plus its trajectory. A "Follow agent"
checkbox snaps connected browser cameras to the agent pose after each step.

Harness capture: a small HTTP API on render_port (default 8081) grabs a
WebGL frame from a connected visor tab (prefer the full-page spectator).
The VLM receives a center-cropped, downscaled copy (default 960x720); the
tab itself stays HD.
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

# Viser's splat shader applies last frame's projection. Requesting 4:3 from a
# 16:9 (or retina) tab therefore squeezes the image. Capture at the tab's
# native aspect, then crop + downscale to the VLM size.
_VLM_MAX_W, _VLM_MAX_H = 1920, 1440
_CAPTURE_MAX_W, _CAPTURE_MAX_H = 1920, 1080


def _center_crop_and_resize(arr: np.ndarray, need_w: int, need_h: int) -> np.ndarray:
    """Crop to `need_w`/`need_h` aspect (never stretch), then resize.

    A wider visor frame is cropped on the sides; a taller one on top/bottom.
    That matches a pinhole camera that keeps the same vertical FOV.
    """
    img = Image.fromarray(arr)
    src_w, src_h = img.size
    if src_w < 2 or src_h < 2:
        return np.asarray(img.resize((need_w, need_h), Image.Resampling.NEAREST).convert("RGB"))
    need_aspect = need_w / need_h
    src_aspect = src_w / src_h
    if src_aspect > need_aspect + 1e-3:
        new_w = max(1, int(round(src_h * need_aspect)))
        left = max(0, (src_w - new_w) // 2)
        img = img.crop((left, 0, min(src_w, left + new_w), src_h))
    elif src_aspect < need_aspect - 1e-3:
        new_h = max(1, int(round(src_w / need_aspect)))
        top = max(0, (src_h - new_h) // 2)
        img = img.crop((0, top, src_w, min(src_h, top + new_h)))
    if img.size != (need_w, need_h):
        img = img.resize((need_w, need_h), Image.Resampling.LANCZOS)
    return np.asarray(img.convert("RGB"))


def _native_capture_size(view_w: int, view_h: int) -> tuple[int, int]:
    """Same aspect as the visor canvas, capped so get_render stays responsive."""
    view_w, view_h = max(int(view_w), 16), max(int(view_h), 16)
    scale = min(1.0, _CAPTURE_MAX_W / view_w, _CAPTURE_MAX_H / view_h)
    cap_w = max(16, int(round(view_w * scale)))
    cap_h = max(16, int(round(view_h * scale)))
    return cap_w, cap_h


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
    """Hides overlay gizmos, captures a visor tab, restores gizmos.

    viser's Gaussian shader uses the live canvas aspect (one frame delayed).
    We therefore capture at that native aspect, then crop and downscale to
    the VLM size so pixels are never stretched. Prefers the largest visible
    tab (full-page spectator / :8080); the dashboard pip is a fallback.
    """

    GET_RENDER_TIMEOUT_S = 45.0
    ATTEMPTS = 3
    MIN_VIEW_W, MIN_VIEW_H = 160, 90

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
                "usable": self._usable(w, h),
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
    def _usable(cls, width: int, height: int) -> bool:
        return width >= cls.MIN_VIEW_W and height >= cls.MIN_VIEW_H

    def _pick_clients(self, clients: dict) -> list:
        """Largest usable canvas first; last successful client wins ties."""
        ranked = []
        for cid, client in clients.items():
            w, h = self._viewport(client)
            if not self._usable(w, h):
                continue
            ranked.append((cid != self._good_client_id, -(w * h), -h, -w, cid, client))
        ranked.sort()
        return [(cid, client) for _, _, _, _, cid, client in ranked]

    def _sync_fov(self, client, fov: float):
        """Match the live canvas FOV to the agent so the delayed splat
        projection is not a zoomed-out spectator view."""
        try:
            old = float(client.camera.fov)
        except Exception:
            return None
        if abs(old - fov) < 1e-4:
            return None
        try:
            client.camera.fov = fov
            self._server.flush()
            time.sleep(0.05)
            return old
        except Exception:
            return None

    def render(self, params: dict) -> bytes:
        width = int(params["width"])
        height = int(params["height"])
        if width < 16 or height < 16 or width > _VLM_MAX_W or height > _VLM_MAX_H:
            raise CaptureError(f"Unsupported VLM size {width}x{height}")
        fov = float(params["fov"])
        wxyz = np.asarray(params["wxyz"], dtype=np.float64)
        position = np.asarray(params["position"], dtype=np.float64)
        with self._lock:
            clients = self._server.get_clients()
            if not clients:
                raise CaptureError(
                    "No visor tab connected. Open the spectator page or "
                    "http://localhost:8080 and keep that window visible."
                )
            order = self._pick_clients(clients)
            if not order:
                sizes = [
                    f"{cid}:{self._viewport(c)[0]}x{self._viewport(c)[1]}"
                    for cid, c in clients.items()
                ]
                raise CaptureError(
                    "No visor tab large enough to capture from "
                    f"(connected viewports: {', '.join(sizes) or 'none'}). "
                    "Open /spectator or http://localhost:8080 and keep it visible."
                )
            hidden = []
            for handle in self._overlay:
                if handle is not None and getattr(handle, "visible", False):
                    handle.visible = False
                    hidden.append(handle)
            img, errors, used = None, [], None
            try:
                self._server.flush()
                for attempt in range(1, self.ATTEMPTS + 1):
                    for cid, client in order:
                        view_w, view_h = self._viewport(client)
                        cap_w, cap_h = _native_capture_size(view_w, view_h)
                        old_fov = self._sync_fov(client, fov)
                        try:
                            img = client.get_render(
                                height=cap_h,
                                width=cap_w,
                                wxyz=wxyz,
                                position=position,
                                fov=fov,
                                transport_format="jpeg",
                                timeout=self.GET_RENDER_TIMEOUT_S,
                            )
                            used = (cid, view_w, view_h, cap_w, cap_h)
                            self._good_client_id = cid
                            break
                        except Exception as exc:
                            errors.append(
                                f"client {cid} attempt {attempt}: {type(exc).__name__}: {exc}"
                            )
                            logger.warning(
                                "Visor capture via client %s failed (attempt %d): %s",
                                cid, attempt, exc,
                            )
                        finally:
                            if old_fov is not None:
                                try:
                                    client.camera.fov = old_fov
                                    self._server.flush()
                                except Exception:
                                    pass
                    if img is not None:
                        break
            finally:
                for handle in hidden:
                    handle.visible = True
                self._server.flush()
        if img is None or getattr(img, "size", 0) == 0:
            raise CaptureError(
                "; ".join(errors[-3:]) or "Visor returned an empty frame. "
                "Keep the spectator/visor window visible — Chrome pauses hidden "
                "tabs, so get_render never returns."
            )
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = arr[..., :3]
        fitted = _center_crop_and_resize(arr.astype(np.uint8), width, height)
        if used:
            cid, view_w, view_h, cap_w, cap_h = used
            logger.info(
                "Visor capture client %s viewport %dx%d -> %dx%d crop/resize %dx%d",
                cid, view_w, view_h, cap_w, cap_h, width, height,
            )
        buf = io.BytesIO()
        Image.fromarray(fitted, mode="RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def serve_viewer(
    scene: GaussianScene,
    host: str = "0.0.0.0",
    port: int = 8080,
    max_splats: int = 0,  # 0 = serve everything
    up_axis: str = "-y",
    render_port: int = 8081,
    fov_deg: float = 75.0,
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

    # viser's fov is vertical. Match the agent's vertical FOV at 4:3 so a
    # 16:9 spectator crop lines up with the VLM's 960x720 pinhole.
    hfov = np.radians(float(fov_deg))
    vfov = float(2.0 * np.arctan(np.tan(hfov / 2.0) * 3.0 / 4.0))

    @server.on_client_connect
    def _(client: "viser.ClientHandle") -> None:
        client.camera.up_direction = up
        client.camera.position = center
        client.camera.look_at = center + forward
        try:
            client.camera.fov = vfov
        except Exception:
            pass

    # --- live agent overlay (fed by the episode dashboard) -------------------
    server.gui.configure_theme(
        control_layout="floating",
        control_width="small",
        show_logo=False,
        show_share_button=False,
        dark_mode=True,
    )
    server.gui.set_panel_label("Visor")
    # Collapse the floating panel on load and keep it compact. <script> inside
    # dangerouslySetInnerHTML does not run; an iframe srcdoc does.
    server.gui.add_html(
        '<iframe title="" style="width:0;height:0;border:0;position:absolute" srcdoc="'
        "&lt;script&gt;"
        "(function(){function go(){"
        "var d=parent.document;"
        "var h=d.querySelector('[data-testid=floating-panel-handle]');"
        "if(!h){setTimeout(go,50);return;}"
        "if(parent.__splatVisorChrome)return;"
        "parent.__splatVisorChrome=1;"
        "var s=d.createElement('style');"
        "s.textContent='[data-testid=floating-panel]{width:9.5em!important;max-width:9.5em!important;"
        "font-size:12px!important}[data-testid=floating-panel-handle]{height:1.7em!important;"
        "padding:0 .4em!important;font-size:11px!important}';"
        "d.head.appendChild(s);"
        "setTimeout(function(){h.dispatchEvent(new MouseEvent('click',{bubbles:true}));},150);"
        "}go();})();"
        "&lt;/script&gt;"
        '"></iframe>'
    )
    with server.gui.add_folder("Agent", expand_by_default=False):
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

    logger.info(
        "Viser spectator at %s (HD). Agent frames are cropped/downscaled from this view.",
        viewer_url,
    )
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
