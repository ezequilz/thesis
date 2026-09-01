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
WebGL frame from the dashboard's always-on visor iframe, which is sized to
the VLM resolution (default 960x720). Captures are requested at that size
directly — no HD readback or crop.
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

# Viser's splat shader applies last frame's projection. The dashboard visor
# iframe is therefore sized to the VLM's 4:3 so get_render(W,H) matches the
# live canvas and we never read back a larger buffer.
_VLM_MAX_W, _VLM_MAX_H = 1920, 1440
_ASPECT_TOL = 0.08

# Debug gizmos that must never appear in VLM frames. `/agent` is the parent
# frame for the frustum + path, so hiding it covers children even if the
# handle list is stale. Fat Line2 segments that pass through the capture
# camera smear into a thin unnaturally straight red streak — the VLM then
# reports that streak as a scene artifact.
_OVERLAY_NODE_NAMES = (
    "/origin",
    "/WorldAxes",
    "/agent",
    "/agent/camera",
    "/agent/trajectory",
)
# One extra frame after a client-local hide so Three.js applies visibility
# before get_render starts (the client pauses the message queue once a
# render request is in flight).
_OVERLAY_HIDE_SETTLE_S = 0.05


def overlay_node_names(handles) -> tuple[str, ...]:
    """Stable scene-tree names to hide for a VLM capture."""
    names: list[str] = []
    seen: set[str] = set()
    extra = (
        getattr(handle, "name", None)
        for handle in handles
        if handle is not None
    )
    for name in (*_OVERLAY_NODE_NAMES, *extra):
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def queue_client_overlay_visibility(client, names: tuple[str, ...], visible: bool) -> bool:
    """Hide/show overlay nodes on one client's websocket, not the broadcast bus.

    ``handle.visible = False`` is broadcast. ``get_render`` is client-local.
    Those two buffers race: if the render request is processed first, the
    client stops handling messages until the JPEG is sent, so the hide never
    applies and the red frustum/path leaks into the agent view.
    """
    conn = getattr(client, "_websock_connection", None)
    if conn is None:
        return False
    try:
        from viser._messages import SetSceneNodeVisibilityMessage
    except ImportError:
        return False
    for name in names:
        conn.queue_message(SetSceneNodeVisibilityMessage(name, visible))
    try:
        client.flush()
    except Exception:
        pass
    return True


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


def _aspect_match(width: int, height: int, need_w: int, need_h: int) -> bool:
    if width < 2 or height < 2 or need_h < 1:
        return False
    return abs((width / height) - (need_w / need_h)) <= _ASPECT_TOL


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


def _view_from_center(center: np.ndarray, up_axis: str, fov_deg: float) -> dict:
    """Gravity-aligned look-from at `center` for the given up axis."""
    from .base import up_vector

    up = up_vector(up_axis).astype(np.float64)
    center = np.asarray(center, dtype=np.float64)
    seed = np.array([0.0, 0.0, -1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    forward = seed - up * np.dot(seed, up)
    norm = float(np.linalg.norm(forward))
    forward = forward / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    hfov = np.radians(float(fov_deg))
    vfov = float(2.0 * np.arctan(np.tan(hfov / 2.0) * 3.0 / 4.0))
    return {"up": up, "center": center, "forward": forward, "vfov": vfov, "up_axis": up_axis}


def _view_pose(scene: GaussianScene, up_axis: str, fov_deg: float) -> dict:
    """Gravity-aligned start pose inside the reconstructed volume."""
    return _view_from_center(scene.robust_centroid(), up_axis, fov_deg)


def _splat_index(scene: GaussianScene, max_splats: int) -> np.ndarray:
    n = scene.num_gaussians
    if max_splats and n > max_splats:
        logger.info("Subsampling %d -> %d splats for the browser", n, max_splats)
        return np.random.default_rng(0).choice(n, size=max_splats, replace=False)
    logger.info("Serving all %d splats to the browser", n)
    return np.arange(n)


def _install_splats(server, scene: GaussianScene, max_splats: int) -> None:
    try:
        server.scene.remove_by_name("/splat")
    except Exception:
        pass
    idx = _splat_index(scene, max_splats)
    server.scene.add_gaussian_splats(
        "/splat",
        centers=scene.means[idx],
        rgbs=scene.colors[idx],
        opacities=scene.opacities[idx, None],
        covariances=quats_to_covariances(scene.quats[idx], scene.scales[idx]),
    )


def _apply_view(server, view: dict) -> None:
    try:
        clients = server.get_clients().values()
    except Exception:
        return
    for client in clients:
        client.camera.up_direction = view["up"]
        client.camera.position = view["center"]
        client.camera.look_at = view["center"] + view["forward"]
        try:
            client.camera.fov = view["vfov"]
        except Exception:
            pass


def _start_render_api(
    host: str,
    port: int,
    capture,
    viewer_url: str,
    health_extra=None,
) -> ThreadingHTTPServer:
    """HTTP API the harness uses to grab a visor frame.

    GET  /health  -> {clients, capture_ready, viewports, viewer_url, scene?}
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
            payload = {
                "clients": n,
                "capture_ready": ready,
                "viewports": viewports,
                "viewer_url": viewer_url,
                "error": err,
            }
            if health_extra is not None:
                try:
                    payload.update(health_extra())
                except Exception:
                    logger.exception("health_extra failed")
            self._send_json(payload)

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
    """Hides overlay gizmos on the capture tab, grabs a frame, restores gizmos.

    The dashboard iframe is sized to the VLM resolution (4:3), so we request
    that size from get_render and skip HD readback. 16:9 spectator tabs are
    ignored — they are the slow, easy-to-throttle path. Overlay hide is sent
    on the same client websocket as get_render so the red frustum/path cannot
    race into the JPEG the VLM sees; other tabs keep the gizmos.
    """

    GET_RENDER_TIMEOUT_S = 12.0
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
    def _usable(cls, width: int, height: int, need_w: int = 4, need_h: int = 3) -> bool:
        """Dashboard capture visor is 4:3 at the VLM size. Skip 16:9 spectator tabs."""
        if width < cls.MIN_VIEW_W or height < cls.MIN_VIEW_H:
            return False
        return _aspect_match(width, height, need_w, need_h)

    def _pick_clients(self, clients: dict, need_w: int, need_h: int) -> list:
        """Prefer a 4:3 canvas closest to the VLM size (the dashboard iframe)."""
        target_px = need_w * need_h
        ranked = []
        for cid, client in clients.items():
            w, h = self._viewport(client)
            if not self._usable(w, h, need_w, need_h):
                continue
            if w < need_w * 0.9 or h < need_h * 0.9:
                continue
            aspect_err = abs((w / h) - (need_w / need_h))
            px_err = abs(w * h - target_px)
            ranked.append((cid != self._good_client_id, aspect_err, px_err, cid, client))
        ranked.sort()
        return [(cid, client) for _, _, _, cid, client in ranked]

    def _get_render(self, client, *, width: int, height: int, wxyz, position, fov):
        """Call get_render at the VLM size, with a timeout so a frozen tab
        cannot stall the episode for a minute."""
        kwargs = dict(
            height=height,
            width=width,
            wxyz=wxyz,
            position=position,
            fov=fov,
            transport_format="jpeg",
        )
        box: dict = {}

        def run() -> None:
            try:
                box["img"] = client.get_render(**kwargs)
            except Exception as exc:
                box["err"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(self.GET_RENDER_TIMEOUT_S)
        if thread.is_alive():
            raise CaptureError(
                f"get_render timed out after {self.GET_RENDER_TIMEOUT_S:.0f}s "
                "(Chrome likely throttled the visor). Keep the dashboard tab visible."
            )
        if "err" in box:
            raise box["err"]
        img = box.get("img")
        if img is None or getattr(img, "size", 0) == 0:
            raise CaptureError("Visor returned an empty frame")
        return img

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

    def _set_overlays_visible_on_client(
        self, client, names: tuple[str, ...], visible: bool
    ) -> None:
        """Toggle debug gizmos on `client` only. Spectator tabs stay put."""
        if queue_client_overlay_visibility(client, names, visible):
            if not visible:
                time.sleep(_OVERLAY_HIDE_SETTLE_S)
            return
        for handle in self._overlay:
            if handle is None:
                continue
            try:
                handle.visible = visible
            except Exception:
                pass
        try:
            self._server.flush()
        except Exception:
            pass
        if not visible:
            time.sleep(_OVERLAY_HIDE_SETTLE_S * 2)

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
                    "No visor connected. Keep the episode dashboard open — it "
                    "runs a 4:3 capture visor at the selected VLM resolution."
                )
            order = self._pick_clients(clients, width, height)
            if not order:
                sizes = [
                    f"{cid}:{self._viewport(c)[0]}x{self._viewport(c)[1]}"
                    for cid, c in clients.items()
                ]
                raise CaptureError(
                    "No 4:3 capture visor ready "
                    f"(connected viewports: {', '.join(sizes) or 'none'}). "
                    "Keep the episode dashboard tab visible; the visor iframe "
                    f"must stay at {width}x{height}."
                )
            img, errors, used = None, [], None
            names = overlay_node_names(self._overlay)
            try:
                self._server.flush()
                for attempt in range(1, self.ATTEMPTS + 1):
                    for cid, client in order:
                        view_w, view_h = self._viewport(client)
                        old_fov = self._sync_fov(client, fov)
                        try:
                            self._set_overlays_visible_on_client(client, names, False)
                            img = self._get_render(
                                client,
                                width=width,
                                height=height,
                                wxyz=wxyz,
                                position=position,
                                fov=fov,
                            )
                            used = (cid, view_w, view_h, width, height)
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
                            self._set_overlays_visible_on_client(client, names, True)
                            if old_fov is not None:
                                try:
                                    client.camera.fov = old_fov
                                    self._server.flush()
                                except Exception:
                                    pass
                    if img is not None:
                        break
            finally:
                self._server.flush()
        if img is None or getattr(img, "size", 0) == 0:
            raise CaptureError(
                "; ".join(errors[-3:]) or "Visor returned an empty frame. "
                "Keep the episode dashboard tab visible so Chrome does not "
                "throttle the capture visor."
            )
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = arr[..., :3]
        arr = arr.astype(np.uint8)
        if arr.shape[1] != width or arr.shape[0] != height:
            logger.warning(
                "Visor returned %dx%d, expected %dx%d — resizing (should be rare)",
                arr.shape[1], arr.shape[0], width, height,
            )
            arr = _center_crop_and_resize(arr, width, height)
        if used:
            cid, view_w, view_h, cap_w, cap_h = used
            logger.info(
                "Visor capture client %s viewport %dx%d -> %dx%d (direct)",
                cid, view_w, view_h, cap_w, cap_h,
            )
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def serve_viewer(
    scene: GaussianScene,
    host: str = "0.0.0.0",
    port: int = 8080,
    max_splats: int = 0,  # 0 = serve everything
    up_axis: str = "-y",
    render_port: int = 8081,
    fov_deg: float = 75.0,
    min_opacity: float = 0.0,
    initial_spec=None,
    generation: int = 0,
):
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is not installed — pip install '.[viewer]'") from exc

    server = viser.ViserServer(host=host, port=port)
    _install_splats(server, scene, max_splats)
    origin = server.scene.add_frame("/origin", axes_length=0.5, axes_radius=0.01)
    try:
        server.scene.world_axes.visible = False
    except Exception:
        pass

    # Start clients inside the room, gravity-aligned, instead of viser's
    # default exterior orbit pose. Mutated in place when the dashboard
    # switches scenes (outputs/live/scene.json).
    view = _view_pose(scene, up_axis, fov_deg)

    @server.on_client_connect
    def _(client: "viser.ClientHandle") -> None:
        client.camera.up_direction = view["up"]
        client.camera.position = view["center"]
        client.camera.look_at = view["center"] + view["forward"]
        try:
            client.camera.fov = view["vfov"]
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
    scene_state = {
        "id": getattr(initial_spec, "id", None),
        "label": getattr(initial_spec, "label", None),
        "path": str(initial_spec.path) if initial_spec is not None else "",
        "up_axis": up_axis,
        "lod_level": int(getattr(initial_spec, "lod_level", 0) or 0),
        "generation": int(generation or 0),
        "num_gaussians": scene.num_gaussians,
        "status": "ready",
        "error": None,
    }

    def health_extra() -> dict:
        return {"scene": dict(scene_state)}

    _start_render_api(
        host, render_port,
        _VisorCapture(server, overlay_lock, overlay_handles),
        viewer_url,
        health_extra=health_extra,
    )

    logger.info(
        "Viser at %s. Agent frames are captured at VLM size from the dashboard visor.",
        viewer_url,
    )

    def _maybe_reload_scene() -> None:
        from ..scene import load_scene
        from ..scene.catalog import read_live_scene

        req = read_live_scene()
        if not req or not req.get("path"):
            return
        gen = int(req.get("generation") or 0)
        path = str(req["path"])
        if gen < scene_state["generation"]:
            return
        # Same asset: ack the new generation. Re-pose if only the up axis changed.
        # `reload` is set when a repaired PLY is overwritten in place.
        if (
            path == scene_state["path"]
            and scene_state["status"] == "ready"
            and not req.get("reload")
        ):
            scene_state["generation"] = max(scene_state["generation"], gen)
            scene_state["id"] = req.get("id", scene_state["id"])
            scene_state["label"] = req.get("label", scene_state["label"])
            up_ax = str(req.get("up_axis") or view["up_axis"])
            if up_ax != scene_state.get("up_axis"):
                logger.info("Viser flipping up axis %s -> %s", scene_state.get("up_axis"), up_ax)
                view.update(_view_from_center(view["center"], up_ax, fov_deg))
                _apply_view(server, view)
                scene_state["up_axis"] = up_ax
            return
        if scene_state["status"] == "loading":
            return

        scene_state.update(status="loading", error=None, id=req.get("id"), label=req.get("label"))
        lod = int(req.get("lod_level") or 0)
        up_ax = str(req.get("up_axis") or view["up_axis"])
        logger.info("Viser loading scene %s (generation %s)", path, gen)
        try:
            new_scene = load_scene(path, min_opacity=min_opacity, lod_level=lod)
        except Exception as exc:
            logger.exception("Viser scene reload failed")
            scene_state["status"] = "error"
            scene_state["error"] = f"{type(exc).__name__}: {exc}"
            return
        new_view = _view_pose(new_scene, up_ax, fov_deg)
        with overlay_lock:
            _install_splats(server, new_scene, max_splats)
            view.update(new_view)
            try:
                server.scene.remove_by_name("/agent")
            except Exception:
                pass
            overlay_handles[:] = [origin]
        _apply_view(server, view)
        scene_state.update(
            path=path,
            up_axis=up_ax,
            lod_level=lod,
            generation=gen,
            num_gaussians=new_scene.num_gaussians,
            status="ready",
            error=None,
        )
        logger.info("Viser scene ready: %d gaussians", new_scene.num_gaussians)

    last_key: tuple | None = None
    while True:
        time.sleep(0.5)
        _maybe_reload_scene()
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
            agent_root = server.scene.add_frame("/agent", show_axes=False)
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
            else:
                try:
                    server.scene.remove_by_name("/agent/trajectory")
                except Exception:
                    pass
            overlay_handles[:] = [origin, agent_root, cam, traj]
        agent_status.content = (
            f"**Agent**: episode `{state.get('episode', '?')}` step {state.get('step', '?')}  \n"
            f"{state.get('pose', '')}"
        )
        if follow_agent.value:
            look_at = position + np.asarray(state.get("view_dir", view["forward"]), dtype=np.float64)
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.look_at = look_at
