"""Local debug dashboard server (stdlib only, no extra dependencies).

Serves a single-page dashboard (static/index.html) to start/stop agent
episodes from the browser and watch them in near real time: per-step frames,
the exact VLM reply and what was parsed from it, render/decide wall times,
and the agent pose. The frontend polls GET /api/state once per second.

It also writes outputs/live/agent_state.json after every step, which the
viser viewer (rendering/viser_viewer.py, :8080) picks up to draw the agent's
camera frustum + trajectory inside the live splat view.

The scene is decoded once at server startup and shared across all runs, so
starting a new episode from the browser costs nothing but the rendering.

Endpoints:
  GET  /                  dashboard page
  GET  /api/state         full dashboard state (scene status + current run)
  POST /api/run           start an episode  {backend, model, max_steps, width, height}
  POST /api/stop          request cooperative stop of the running episode
  POST /api/select        pin the viser overlay to a step {step: int} / back to live {step: null}
  GET  /frames/<ep>/<png> step screenshots from outputs/episodes/
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from ..config import Config

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
LIVE_STATE_PATH = Path("outputs/live/agent_state.json")

RUN_DEFAULTS = {"backend": "scripted", "model": "", "max_steps": 5, "width": 480, "height": 360}
RUN_LIMITS = {"max_steps": 200, "width": 1920, "height": 1440}


def load_dotenv(path: Path = Path(".env")) -> list[str]:
    """Load KEY=VALUE lines from .env into the environment (no overrides).

    Lets browser-started cli_relay runs find CLIRELAY_API_KEY without the
    user exporting it in the shell that launched the dashboard.
    """
    loaded = []
    if not path.is_file():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def rotation_to_wxyz(R: np.ndarray) -> list[float]:
    """Rotation matrix -> quaternion (w, x, y, z), for viser frustum poses."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return [float(w), float(x), float(y), float(z)]


class DashboardApp:
    """Owns the shared scene/renderer and at most one running episode."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.scene = None
        self.renderer = None
        self.nav_world = None
        self.spawn = None
        self.scene_status = "loading"
        self.scene_info: dict = {"path": str(cfg.scene.path)}
        self.run: dict | None = None
        self._stop = threading.Event()
        self._run_thread: threading.Thread | None = None
        # Step index the UI pinned the viser overlay to; None = follow live.
        self._pinned: int | None = None
        threading.Thread(target=self._load_scene, daemon=True).start()

    # --- scene ----------------------------------------------------------------
    def _load_scene(self) -> None:
        from ..cli import _build_navigation
        from ..rendering import make_renderer
        from ..scene import load_scene

        try:
            t0 = time.perf_counter()
            scene = load_scene(self.cfg.scene.path, min_opacity=self.cfg.scene.min_opacity)
            renderer = make_renderer(scene, self.cfg.renderer)
            # Collision world + bird's-eye spawn selection, shared by all runs.
            nav_world, spawn = _build_navigation(self.cfg, scene)
            with self.lock:
                self.scene, self.renderer = scene, renderer
                self.nav_world, self.spawn = nav_world, spawn
                self.scene_status = "ready"
                self.scene_info.update(
                    num_gaussians=scene.num_gaussians,
                    load_seconds=round(time.perf_counter() - t0, 1),
                    renderer=self.cfg.renderer.backend,
                )
            logger.info("Scene ready: %d gaussians in %.1fs",
                        scene.num_gaussians, time.perf_counter() - t0)
        except Exception as exc:
            logger.exception("Scene load failed")
            with self.lock:
                self.scene_status = "error"
                self.scene_info["error"] = f"{type(exc).__name__}: {exc}"

    # --- run control ------------------------------------------------------------
    def start_run(self, params: dict) -> tuple[bool, str]:
        clean = dict(RUN_DEFAULTS)
        for key in clean:
            if key in params and params[key] not in ("", None):
                clean[key] = params[key]
        for key, cap in RUN_LIMITS.items():
            clean[key] = max(1, min(int(clean[key]), cap))

        with self.lock:
            if self.scene_status != "ready":
                return False, f"Scene not ready (status: {self.scene_status})."
            if self.run and self.run["status"] in ("running", "stopping"):
                return False, "An episode is already running."
            self._stop.clear()
            self._pinned = None
            self.run = {
                "id": time.strftime("%Y%m%d_%H%M%S"),
                "episode": None,
                "status": "running",
                "params": clean,
                "steps": [],
                "artifacts": [],
                "summary": None,
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
            }
        self._run_thread = threading.Thread(target=self._run_episode, args=(clean,), daemon=True)
        self._run_thread.start()
        return True, "Episode started."

    def stop_run(self) -> tuple[bool, str]:
        with self.lock:
            if not self.run or self.run["status"] != "running":
                return False, "No running episode."
            self.run["status"] = "stopping"
        self._stop.set()
        return True, "Stop requested; finishing the current step."

    def _make_policy(self, params: dict):
        from ..agent.vlm import make_policy

        agent_cfg = dict(self.cfg["agent"])
        agent_cfg["vlm_backend"] = params["backend"]
        if params["model"]:
            agent_cfg["model"] = params["model"]
        return make_policy(Config(agent_cfg))

    def _run_episode(self, params: dict) -> None:
        from ..agent.camera_rig import CameraRig
        from ..agent.loop import run_episode
        from ..cli import _resolve_start

        try:
            policy = self._make_policy(params)
            start = (self.spawn.points[0].position if self.spawn
                     else _resolve_start(self.cfg, self.scene))
            rig = CameraRig(
                start,
                up_axis=self.cfg.camera.up_axis,
                yaw_deg=self.cfg.camera.start_yaw_deg,
            )
            self._trajectory: list[list[float]] = []
            run_episode(
                renderer=self.renderer,
                rig=rig,
                policy=policy,
                output_dir=Path(self.cfg.output.dir),
                width=params["width"],
                height=params["height"],
                fov_deg=self.cfg.renderer.fov_deg,
                max_steps=params["max_steps"],
                max_move_distance=self.cfg.agent.max_move_distance,
                max_rotate_degrees=self.cfg.agent.max_rotate_degrees,
                nav=self.nav_world,
                spawn=self.spawn,
                on_step=self._on_step,
                should_stop=self._stop.is_set,
            )
            with self.lock:
                self.run["status"] = "stopped" if self._stop.is_set() else "completed"
        except Exception as exc:
            logger.exception("Episode failed")
            with self.lock:
                self.run["status"] = "error"
                self.run["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self.run["finished_at"] = time.time()

    def _on_step(self, record: dict, frame_path: Path, rig) -> None:
        episode = frame_path.parent.name
        step = dict(record)
        step["frame_url"] = f"/frames/{episode}/{frame_path.name}"
        action = record["action"]
        with self.lock:
            self.run["episode"] = episode
            self.run["steps"].append(step)
            if action["name"] == "report_artifact":
                self.run["artifacts"].append({"step": record["step"], **action["args"]})
            if action["name"] == "done":
                self.run["summary"] = action["args"].get("summary", "")
            self._trajectory.append(record["position"])
            params = self.run["params"]
            pinned = self._pinned
            trajectory = list(self._trajectory)
        # While the UI has an older step pinned, don't yank the viser overlay
        # back to the live pose on every new step.
        if pinned is None:
            self._publish_pose(record, frame_path, params, trajectory)

    def select_step(self, step: int | None) -> tuple[bool, str]:
        """Pin the viser overlay to a specific step, or return to live (None)."""
        with self.lock:
            if step is None:
                self._pinned = None
            else:
                if not self.run or not (0 <= int(step) < len(self.run["steps"])):
                    return False, f"No step {step} in the current run."
                self._pinned = int(step)
            run = self.run
            if not run or not run["steps"]:
                return True, "Following live agent."
            index = self._pinned if self._pinned is not None else len(run["steps"]) - 1
            record = run["steps"][index]
            params = run["params"]
            episode = run["episode"]
            trajectory = [s["position"] for s in run["steps"]]
        frame_path = Path(self.cfg.output.dir) / "episodes" / episode / record["frame"]
        self._publish_pose(record, frame_path, params, trajectory)
        if step is None:
            return True, "Viewer back to live agent."
        return True, f"Viewer camera set to step {step}."

    def _publish_pose(self, record: dict, frame_path: Path, params: dict,
                      trajectory: list) -> None:
        """Publish a step's camera pose for the viser viewer overlay (atomic write)."""
        from ..agent.camera_rig import CameraRig

        try:
            rig = CameraRig(
                np.asarray(record["position"], dtype=np.float64),
                up_axis=self.cfg.camera.up_axis,
                yaw_deg=record["yaw_deg"],
                pitch_deg=record["pitch_deg"],
            )
            camera = rig.camera(params["width"], params["height"], self.cfg.renderer.fov_deg)
            state = {
                "episode": frame_path.parent.name,
                "step": record["step"],
                "pose": record["pose"],
                "position": record["position"],
                "wxyz": rotation_to_wxyz(np.asarray(camera.rotation, dtype=np.float64)),
                "view_dir": rig.view_direction().tolist(),
                "fov_deg": self.cfg.renderer.fov_deg,
                "aspect": params["width"] / params["height"],
                "frame": str(frame_path.resolve()),
                "frame_rel": str(frame_path),
                "trajectory": trajectory,
                "updated_at": time.time(),
            }
            LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = LIVE_STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, LIVE_STATE_PATH)
        except Exception:
            logger.exception("Failed to write live agent state")

    # --- state snapshot --------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "scene": {"status": self.scene_status, **self.scene_info},
                "run": self.run,
                "defaults": {
                    **RUN_DEFAULTS,
                    "model": self.cfg.agent.model,
                    "viewer_url": f"http://localhost:{self.cfg.viewer.port}",
                },
                "now": time.time(),
            }


class DashboardHandler(BaseHTTPRequestHandler):
    app: DashboardApp  # injected by serve_dashboard

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send_json(self.app.snapshot())
        elif path.startswith("/frames/"):
            self._serve_frame(path)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"ok": False, "message": "Invalid JSON body."}, 400)
            return
        if path == "/api/run":
            ok, message = self.app.start_run(body)
        elif path == "/api/stop":
            ok, message = self.app.stop_run()
        elif path == "/api/select":
            ok, message = self.app.select_step(body.get("step"))
        else:
            self._send_json({"error": "not found"}, 404)
            return
        self._send_json({"ok": ok, "message": message}, 200 if ok else 409)

    def _serve_frame(self, path: str) -> None:
        episodes_dir = (Path(self.app.cfg.output.dir) / "episodes").resolve()
        target = (episodes_dir / path[len("/frames/"):]).resolve()
        if not target.is_relative_to(episodes_dir) or not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        self._send(200, target.read_bytes(), "image/png")

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("%s %s", self.address_string(), fmt % args)


def serve_dashboard(cfg: Config, host: str, port: int) -> None:
    loaded = load_dotenv()
    if loaded:
        logger.info("Loaded from .env: %s", ", ".join(loaded))
    DashboardHandler.app = DashboardApp(cfg)
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    logger.info("Dashboard running at http://%s:%d (scene loading in background)",
                "localhost" if host in ("0.0.0.0", "") else host, port)
    server.serve_forever()
