"""Temporary full-screen visor for one scene-run's original vs repaired PLY.

This is a second ViserServer, not the capture visor on :8080. Opening
``/{run_id}/viser`` therefore cannot steal the live VLM/WebGL client.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ..repair import ORIGINAL_PLY, REPAIRED_PLY
from ..scene_runs.store import RUN_ID_PATTERN
from .scene_run_studio import _cfg_get

logger = logging.getLogger(__name__)

WHICH = ("original", "repaired")
_PLY_NAME = {"original": ORIGINAL_PLY, "repaired": REPAIRED_PLY}
_REVIEW_PORT_OFFSET = 2
_IDLE_TIMEOUT_S = 45.0
_WATCHDOG_S = 5.0


def parse_run_viser_page(path: str) -> str | None:
    """Return the run id for ``/{run_id}/viser``, else None."""
    raw = str(path or "").rstrip("/")
    suffix = "/viser"
    if not raw.startswith("/") or not raw.endswith(suffix):
        return None
    run_id = raw[1:-len(suffix)]
    if "/" in run_id or RUN_ID_PATTERN.fullmatch(run_id) is None:
        return None
    return run_id


def parse_run_viser_api(path: str) -> str | None:
    """Return the run id for ``/api/scene-runs/{run_id}/viser``, else None."""
    prefix = "/api/scene-runs/"
    suffix = "/viser"
    raw = str(path or "").rstrip("/")
    if not raw.startswith(prefix) or not raw.endswith(suffix):
        return None
    run_id = raw[len(prefix):-len(suffix)]
    if not run_id or "/" in run_id:
        return None
    return run_id


class SceneRunViser:
    """At most one review ViserServer, started only after Open visor is clicked.

    Opening ``/{run_id}/viser`` POSTs to start it. Extra run tabs rebind this
    same server instead of spawning another process. Closing the last visor
    tab (or an idle timeout) stops it.
    """

    def __init__(
        self,
        studio: Any,
        *,
        background: bool = True,
        server_factory: Callable[..., Any] | None = None,
        load_scene: Callable[..., Any] | None = None,
        idle_timeout: float | None = _IDLE_TIMEOUT_S,
    ):
        self.studio = studio
        self.cfg = studio.cfg
        self._background = background
        self._server_factory = server_factory
        self._load_scene = load_scene
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()
        self._server = None
        self._port: int | None = None
        self._run_id: str | None = None
        self._which: str | None = None
        self._status = "idle"
        self._error: str | None = None
        self._generation = 0
        self._view: dict | None = None
        self._scenes: dict[tuple[str, str], Any] = {}
        self._clients: dict[str, float] = {}
        self._watchdog_started = False

    def ply_paths(self, run_id: str) -> dict[str, Path | None]:
        return {
            name: self.studio.artifact_path(str(run_id), filename)
            for name, filename in _PLY_NAME.items()
        }

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run_id = str(run_id)
        paths = self.ply_paths(run_id)
        with self._lock:
            showing = self._run_id == run_id
            return self._snapshot_locked(run_id, paths, showing=showing)

    def heartbeat(self, run_id: str, client: str | None = None) -> dict[str, Any]:
        """Keep the review visor alive without starting or reloading it."""
        with self._lock:
            self._touch_client_locked(client)
        return self.snapshot(run_id)

    def release(self, run_id: str, client: str | None = None) -> dict[str, Any]:
        """Drop one visor tab. Stop the server when none remain."""
        with self._lock:
            if client:
                self._clients.pop(str(client), None)
            elif self._run_id == str(run_id):
                self._clients.clear()
            if not self._clients:
                self._stop_locked()
        return self.snapshot(run_id)

    def _touch_client_locked(self, client: str | None) -> None:
        token = str(client or "").strip()
        if token:
            self._clients[token] = time.monotonic()

    def _stop_locked(self) -> None:
        server = self._server
        self._generation += 1
        self._server = None
        self._port = None
        self._run_id = None
        self._which = None
        self._status = "idle"
        self._error = None
        self._view = None
        self._scenes.clear()
        self._clients.clear()
        if server is None:
            return
        stopper = getattr(server, "stop", None)
        if callable(stopper):
            try:
                stopper()
            except Exception:
                logger.debug("Scene-run review visor stop failed", exc_info=True)
        logger.info("Stopped scene-run review visor")

    def _schedule_watchdog_locked(self) -> None:
        if self._idle_timeout is None or self._watchdog_started:
            return
        self._watchdog_started = True
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self) -> None:
        timeout = self._idle_timeout
        while timeout is not None:
            time.sleep(_WATCHDOG_S)
            now = time.monotonic()
            with self._lock:
                stale = [
                    token for token, seen in self._clients.items()
                    if now - seen > timeout
                ]
                for token in stale:
                    self._clients.pop(token, None)
                if self._server is None:
                    self._watchdog_started = False
                    return
                if self._clients:
                    continue
                self._stop_locked()
                self._watchdog_started = False
                return

    def show(
        self,
        run_id: str,
        which: str | None = None,
        *,
        toggle: bool = False,
        client: str | None = None,
    ) -> dict[str, Any]:
        run_id = str(run_id)
        if self.studio.detail(run_id) is None:
            return {
                "ok": False,
                "run_id": run_id,
                "which": None,
                "status": "error",
                "viewer_url": None,
                "original": False,
                "repaired": False,
                "viser_path": f"/{run_id}/viser",
                "message": f"Scene-run {run_id} not found.",
                "error": "not found",
            }

        paths = self.ply_paths(run_id)
        requested = str(which or "").strip().lower() or None
        if requested is not None and requested not in WHICH:
            return {
                **self.snapshot(run_id),
                "ok": False,
                "message": "which must be 'original' or 'repaired'.",
                "error": "invalid which",
            }

        with self._lock:
            current = self._which if self._run_id == run_id else None
            if toggle:
                requested = "original" if current == "repaired" else "repaired"
            if requested is None:
                requested = "repaired" if paths["repaired"] is not None else "original"

            missing = _PLY_NAME[requested]
            if paths[requested] is None:
                snap = self._snapshot_locked(run_id, paths, showing=self._run_id == run_id)
                snap.update(
                    ok=False,
                    message=f"No {missing} yet for {run_id}.",
                    error="missing ply",
                )
                return snap

            try:
                self._touch_client_locked(client)
                self._ensure_server_locked()
                self._schedule_watchdog_locked()
            except Exception as exc:
                self._status = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                snap = self._snapshot_locked(run_id, paths, showing=False)
                snap.update(ok=False, message=self._error, error=self._error)
                return snap

            keep_camera = (
                self._run_id == run_id
                and self._view is not None
                and self._status in {"ready", "loading"}
            )
            already = (
                self._run_id == run_id
                and self._which == requested
                and self._status in {"ready", "loading"}
            )
            if already:
                snap = self._snapshot_locked(run_id, paths, showing=True)
                snap["ok"] = True
                snap["message"] = (
                    f"Loading {missing}…"
                    if self._status == "loading"
                    else f"Showing {missing}."
                )
                return snap

            self._generation += 1
            generation = self._generation
            self._run_id = run_id
            self._which = requested
            self._status = "loading"
            self._error = None
            up_axis = self._up_axis(run_id)
            ply = paths[requested]
            snap = self._snapshot_locked(run_id, paths, showing=True)
            snap["ok"] = True
            snap["message"] = f"Loading {missing}…"

        args = (run_id, requested, ply, generation, keep_camera, up_axis)
        if self._background:
            threading.Thread(target=self._load, args=args, daemon=True).start()
        else:
            self._load(*args)
            snap = self.snapshot(run_id)
            snap["ok"] = snap["status"] != "error"
            snap["message"] = (
                snap.get("error") or f"Showing {_PLY_NAME[requested]}."
            )
        return snap

    def _snapshot_locked(
        self,
        run_id: str,
        paths: Mapping[str, Path | None],
        *,
        showing: bool,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "run_id": run_id,
            "which": self._which if showing else None,
            "status": self._status if showing else "idle",
            "viewer_url": (
                f"http://localhost:{self._port}" if self._port is not None else None
            ),
            "original": paths["original"] is not None,
            "repaired": paths["repaired"] is not None,
            "viser_path": f"/{run_id}/viser",
            "message": self._error if showing and self._status == "error" else "",
            "error": self._error if showing else None,
        }

    def _up_axis(self, run_id: str) -> str:
        default = str(_cfg_get(self.cfg, "camera.up_axis", "+y") or "+y")
        try:
            detail = self.studio.detail(run_id) or {}
        except Exception:
            return default
        config = detail.get("config") if isinstance(detail, Mapping) else None
        scene_id = ""
        if isinstance(config, Mapping):
            scene_id = str(config.get("scene_id") or "")
        if not scene_id:
            return default
        try:
            from ..scene.catalog import spec_by_id

            spec = spec_by_id(self.cfg, scene_id)
        except Exception:
            return default
        if spec is None:
            return default
        return str(spec.up_axis or default)

    def _ensure_server_locked(self) -> None:
        if self._server is not None:
            return
        host = str(_cfg_get(self.cfg, "viewer.host", "0.0.0.0") or "0.0.0.0")
        base = int(_cfg_get(self.cfg, "viewer.port", 8080) or 8080)
        port = base + _REVIEW_PORT_OFFSET
        factory = self._server_factory
        if factory is None:
            try:
                import viser
            except ImportError as exc:
                raise RuntimeError(
                    "viser is not installed — pip install '.[viewer]'"
                ) from exc

            def factory(**kwargs):
                return viser.ViserServer(verbose=False, **kwargs)
        server = factory(host=host, port=port)
        self._configure_server(server)
        getter = getattr(server, "get_port", None)
        self._port = int(getter()) if callable(getter) else int(
            getattr(server, "_port", port) or port
        )
        self._server = server
        logger.info("Scene-run review visor at http://localhost:%s", self._port)

    def _configure_server(self, server: Any) -> None:
        gui = getattr(server, "gui", None)
        if gui is not None:
            try:
                gui.configure_theme(
                    control_layout="floating",
                    control_width="small",
                    show_logo=False,
                    show_share_button=False,
                    dark_mode=True,
                )
                gui.set_panel_label("Review")
                gui.add_html(
                    '<iframe title="" style="width:0;height:0;border:0;position:absolute" srcdoc="'
                    "&lt;script&gt;"
                    "(function(){function go(){"
                    "var d=parent.document;"
                    "var h=d.querySelector('[data-testid=floating-panel-handle]');"
                    "if(!h){setTimeout(go,50);return;}"
                    "if(parent.__splatRunVisorChrome)return;"
                    "parent.__splatRunVisorChrome=1;"
                    "var s=d.createElement('style');"
                    "s.textContent='[data-testid=floating-panel]{width:9.5em!important;"
                    "max-width:9.5em!important;font-size:12px!important}"
                    "[data-testid=floating-panel-handle]{height:1.7em!important;"
                    "padding:0 .4em!important;font-size:11px!important}';"
                    "d.head.appendChild(s);"
                    "setTimeout(function(){h.dispatchEvent(new MouseEvent('click',{bubbles:true}));},150);"
                    "}go();})();"
                    "&lt;/script&gt;"
                    '"></iframe>'
                )
            except Exception:
                logger.debug("Could not theme the scene-run review visor", exc_info=True)
        scene = getattr(server, "scene", None)
        if scene is not None:
            try:
                scene.world_axes.visible = False
            except Exception:
                pass

        visor = self

        def _on_connect(client: Any) -> None:
            with visor._lock:
                view = visor._view
            if not view:
                return
            try:
                client.camera.up_direction = view["up"]
                client.camera.position = view["center"]
                client.camera.look_at = view["center"] + view["forward"]
                client.camera.fov = view["vfov"]
            except Exception:
                pass

        on_connect = getattr(server, "on_client_connect", None)
        if callable(on_connect):
            on_connect(_on_connect)

    def _load(
        self,
        run_id: str,
        which: str,
        path: Path,
        generation: int,
        keep_camera: bool,
        up_axis: str,
    ) -> None:
        from ..rendering.viser_viewer import (
            _apply_up,
            _apply_view,
            _install_splats,
            _restore_client_cameras,
            _snapshot_client_cameras,
            _view_pose,
        )
        from ..scene import load_scene

        loader = self._load_scene or load_scene
        try:
            scene = self._scenes.get((run_id, which))
            if scene is None:
                min_opacity = float(_cfg_get(self.cfg, "scene.min_opacity", 0.0) or 0.0)
                lod = int(_cfg_get(self.cfg, "scene.lod_level", 0) or 0)
                scene = loader(path, min_opacity=min_opacity, lod_level=lod)
                self._scenes = {
                    key: value for key, value in self._scenes.items() if key[0] == run_id
                }
                self._scenes[(run_id, which)] = scene
        except Exception as exc:
            logger.exception("Scene-run visor failed to load %s", path)
            with self._lock:
                if generation != self._generation:
                    return
                self._status = "error"
                self._error = f"{type(exc).__name__}: {exc}"
            return

        fov_deg = float(_cfg_get(self.cfg, "camera.fov_deg", 75.0) or 75.0)
        max_splats = int(_cfg_get(self.cfg, "viewer.max_splats", 0) or 0)
        with self._lock:
            if generation != self._generation or self._server is None:
                return
            server = self._server
            saved = _snapshot_client_cameras(server) if keep_camera else []
            _install_splats(server, scene, max_splats)
            view = _view_pose(scene, up_axis, fov_deg)
            self._view = view
            if saved:
                _restore_client_cameras(saved)
                _apply_up(server, view)
            else:
                _apply_view(server, view)
            self._status = "ready"
            self._error = None
            logger.info(
                "Scene-run visor showing %s %s (%d gaussians)",
                run_id, _PLY_NAME[which], scene.num_gaussians,
            )
