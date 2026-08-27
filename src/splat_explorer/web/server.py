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
  GET  /spectator         HD visor for looking around (not used for VLM captures)
  GET  /video/<id>        player tab: loading, then the stitched episode video
  GET  /api/state         full dashboard state (scene status + current run)
  GET  /api/episodes      list all past runs on disk (meta.json summaries)
  GET  /api/episodes/<id>      full trace of one past run (steps + artifacts)
  GET  /api/episodes/<id>/log  that run's episode.log
  GET  /api/episodes/<id>/video  RGB|map video (renders once, then cached on disk)
  POST /api/run           start an episode  {backend, model, max_steps, width, height, send_depth, send_map, send_coverage, compute_depth, compute_coverage}
  POST /api/stop          request cooperative stop of the running episode
  POST /api/select        pin the viser overlay to a step {step: int} / back to live {step: null}
  GET  /frames/<ep>/<png> step screenshots from outputs/episodes/
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from ..config import Config
from ..rendering.base import rotation_to_wxyz

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
LIVE_STATE_PATH = Path("outputs/live/agent_state.json")

RUN_DEFAULTS = {"backend": "cli_relay", "model": "", "max_steps": 10,
                "width": 960, "height": 720, "send_depth": False,
                "send_map": False, "send_coverage": False,
                "compute_depth": False, "compute_coverage": False}
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
        # One lock per episode so two video-tab opens don't double-render.
        self._video_locks: dict[str, threading.Lock] = {}
        self._video_locks_guard = threading.Lock()
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
        clean["send_depth"] = bool(clean["send_depth"])
        clean["send_map"] = bool(clean["send_map"])
        clean["send_coverage"] = bool(clean["send_coverage"])
        clean["compute_depth"] = bool(clean["compute_depth"])
        clean["compute_coverage"] = bool(clean["compute_coverage"])
        # Sending a view implies computing it.
        if clean["send_depth"]:
            clean["compute_depth"] = True
        if clean["send_coverage"]:
            clean["compute_coverage"] = True

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
            meta_params = dict(params)
            if self.nav_world is not None:
                meta_params["collision"] = self.nav_world.collision
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
                send_depth=params["send_depth"],
                send_map=params["send_map"],
                send_coverage=params["send_coverage"],
                compute_depth=params["compute_depth"],
                compute_coverage=params["compute_coverage"],
                run_meta={"params": meta_params},
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
        depth_name = record.get("depth_frame")
        if depth_name:
            step["depth_frame_url"] = f"/frames/{episode}/{depth_name}"
        map_name = record.get("map_frame")
        if map_name:
            step["map_frame_url"] = f"/frames/{episode}/{map_name}"
        coverage_name = record.get("coverage_frame")
        if coverage_name:
            step["coverage_frame_url"] = f"/frames/{episode}/{coverage_name}"
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
                "wxyz": rotation_to_wxyz(np.asarray(camera.rotation, dtype=np.float64)).tolist(),
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

    # --- run history -------------------------------------------------------------
    def _episodes_dir(self) -> Path:
        return Path(self.cfg.output.dir) / "episodes"

    def episode_path(self, ep_id: str) -> Path | None:
        """Resolve an episode id to its directory, rejecting path escapes."""
        root = self._episodes_dir().resolve()
        target = (root / ep_id).resolve()
        return target if target.parent == root and target.is_dir() else None

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def list_episodes(self) -> list[dict]:
        """All episode dirs on disk, newest first, with meta.json summaries."""
        root = self._episodes_dir()
        entries = []
        if not root.is_dir():
            return entries
        for d in sorted(root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta = self._read_json(d / "meta.json") or {}
            steps = meta.get("steps")
            if steps is None:  # runs from before meta.json existed
                try:
                    with open(d / "actions.jsonl") as f:
                        steps = sum(1 for line in f if line.strip())
                except OSError:
                    steps = 0
            entries.append({
                "id": d.name,
                "status": meta.get("status", "unknown"),
                "steps": steps,
                "params": meta.get("params"),
                "error": meta.get("error"),
                "artifact_count": meta.get("artifact_count"),
                "started_at": meta.get("started_at"),
                "finished_at": meta.get("finished_at"),
                "has_log": (d / "episode.log").is_file(),
            })
        return entries

    def episode_detail(self, ep_id: str) -> dict | None:
        """Full record of one past run: meta + per-step trace + artifacts."""
        d = self.episode_path(ep_id)
        if d is None:
            return None
        steps = []
        try:
            with open(d / "actions.jsonl") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    rec["frame_url"] = f"/frames/{d.name}/{rec['frame']}"
                    if rec.get("depth_frame"):
                        rec["depth_frame_url"] = f"/frames/{d.name}/{rec['depth_frame']}"
                    if rec.get("map_frame"):
                        rec["map_frame_url"] = f"/frames/{d.name}/{rec['map_frame']}"
                    if rec.get("coverage_frame"):
                        rec["coverage_frame_url"] = f"/frames/{d.name}/{rec['coverage_frame']}"
                    steps.append(rec)
        except (OSError, json.JSONDecodeError):
            pass
        return {
            "id": d.name,
            "meta": self._read_json(d / "meta.json") or {},
            "steps": steps,
            "artifacts": self._read_json(d / "artifacts.json") or [],
            "has_log": (d / "episode.log").is_file(),
        }

    def episode_video(self, ep_id: str):
        """Render-or-reuse the stitched RGB|map video. Returns (path, mime) or (None, error)."""
        from ..rendering.episode_video import EpisodeVideoError, render_episode_video

        d = self.episode_path(ep_id)
        if d is None:
            return None, "not found"
        with self._video_locks_guard:
            lock = self._video_locks.setdefault(ep_id, threading.Lock())
        with lock:
            try:
                result = render_episode_video(d)
            except EpisodeVideoError as exc:
                return None, str(exc)
            except Exception:
                logger.exception("Episode video failed for %s", ep_id)
                return None, "Video render failed."
        return result, None

    # --- state snapshot --------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "scene": {"status": self.scene_status, **self.scene_info},
                "run": self.run,
                "defaults": {
                    **RUN_DEFAULTS,
                    "model": self.cfg.agent.model,
                    "send_depth": bool(self.cfg.agent.get("send_depth", False)),
                    "send_map": bool(self.cfg.agent.get("send_map", False)),
                    "send_coverage": bool(self.cfg.agent.get("send_coverage", False)),
                    "compute_depth": bool(self.cfg.agent.get("compute_depth", False)),
                    "compute_coverage": bool(self.cfg.agent.get("compute_coverage", False)),
                    "viewer_url": f"http://localhost:{self.cfg.viewer.port}",
                    "spectator_path": "/spectator",
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
        elif path in ("/spectator", "/spectator.html"):
            self._send(200, (STATIC_DIR / "spectator.html").read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/video/"):
            ep = path[len("/video/"):]
            if not ep or "/" in ep:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send(200, (STATIC_DIR / "video.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send_json(self.app.snapshot())
        elif path == "/api/episodes":
            self._send_json({"episodes": self.app.list_episodes()})
        elif path.startswith("/api/episodes/"):
            self._serve_episode(path[len("/api/episodes/"):])
        elif path.startswith("/frames/"):
            self._serve_frame(path)
        else:
            self._send_json({"error": "not found"}, 404)

    def _serve_episode(self, rest: str) -> None:
        if rest.endswith("/log"):
            ep_dir = self.app.episode_path(rest[:-len("/log")])
            log = ep_dir / "episode.log" if ep_dir else None
            if log is not None and log.is_file():
                self._send(200, log.read_bytes(), "text/plain; charset=utf-8")
            else:
                self._send_json({"error": "not found"}, 404)
            return
        if rest.endswith("/video"):
            self._serve_episode_video(rest[:-len("/video")])
            return
        detail = self.app.episode_detail(rest)
        if detail is None:
            self._send_json({"error": "not found"}, 404)
        else:
            self._send_json(detail)

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

    def _serve_episode_video(self, ep_id: str) -> None:
        result, error = self.app.episode_video(ep_id)
        if error or result is None:
            code = 404 if error in ("not found",) else 409
            self._send_json({"error": error or "not found"}, code)
            return
        name = result.path.name
        self.send_response(200)
        self.send_header("Content-Type", result.content_type)
        self.send_header("Content-Length", str(result.path.stat().st_size))
        self.send_header("Content-Disposition", f'inline; filename="{name}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(result.path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

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
