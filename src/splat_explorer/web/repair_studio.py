"""Replay 3DGS repair on a past (or live) episode and preview original vs repaired.

Lives beside the episode dashboard. Regenerated RGB PNGs are enough to rerun
the photometric lift after code edits — no live VLM / image-model run needed.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..repair import (
    ORIGINAL_PLY,
    REPAIRED_PLY,
    discover_repair_views,
    reload_repair_module,
    replay_episode_repairs,
)
from ..scene.catalog import SceneSpec, publish_live_scene

logger = logging.getLogger(__name__)


def _ply_info(path: Path) -> dict | None:
    if not path.is_file():
        return None
    st = path.stat()
    return {"name": path.name, "bytes": st.st_size, "mtime": st.st_mtime}


class RepairStudio:
    def __init__(self, app):
        self.app = app
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.job: dict = self._idle_job()
        self.showing: str | None = None  # original | repaired | None (catalog)
        self.showing_episode: str | None = None

    @staticmethod
    def _idle_job(episode: str | None = None) -> dict:
        return {
            "status": "idle",
            "episode": episode,
            "current_index": None,
            "current_step": None,
            "n_views": 0,
            "n_done": 0,
            "error": None,
            "message": None,
            "started_at": None,
            "finished_at": None,
            "results": [],
            "reload_code": False,
        }

    def snapshot(self, episode_id: str | None = None) -> dict:
        with self.app.lock:
            run = self.app.run
            live_ep = run.get("episode") if run else None
            run_status = run.get("status") if run else None
            scene_status = self.app.scene_status
            spec = self.app._scene_spec
            scene_id = spec.id if spec is not None else None
            up_axis = spec.up_axis if spec is not None else None
        with self._lock:
            job = dict(self.job)
            showing = self.showing
            showing_episode = self.showing_episode
        ep = episode_id or job.get("episode") or live_ep
        detail = self.episode_review(ep) if ep else None
        episode_scene = None
        if detail and isinstance(detail.get("meta"), dict):
            params = detail["meta"].get("params")
            if isinstance(params, dict):
                episode_scene = params.get("scene")
        return {
            "job": job,
            "showing": showing,
            "showing_episode": showing_episode,
            "live_episode": live_ep,
            "run_status": run_status,
            "scene_status": scene_status,
            "scene_id": scene_id,
            "episode_scene": episode_scene,
            "up_axis": up_axis,
            "episode": detail,
            "viewer_url": f"http://localhost:{self.app.cfg.viewer.port}",
        }

    def list_episodes(self) -> list[dict]:
        entries = []
        for entry in self.app.list_episodes():
            d = self.app.episode_path(entry["id"])
            if d is None:
                continue
            n_regen = sum(1 for p in d.glob("step_*_regen.png") if p.stem.endswith("_regen"))
            entry = dict(entry)
            entry["n_regen"] = n_regen
            entry["has_original_ply"] = (d / ORIGINAL_PLY).is_file()
            entry["has_repaired_ply"] = (d / REPAIRED_PLY).is_file()
            entries.append(entry)
        return entries

    def episode_review(self, ep_id: str) -> dict | None:
        d = self.app.episode_path(ep_id)
        if d is None:
            return None
        meta = self.app._read_json(d / "meta.json") or {}
        views = discover_repair_views(d, meta=meta)
        for view in views:
            step = view["step"]
            view["rendered_url"] = f"/frames/{ep_id}/{view['rendered_name']}"
            view["repaired_url"] = f"/frames/{ep_id}/{view['repaired_name']}"
            del view["rendered_path"]
            del view["repaired_path"]
        log = self.app._read_json(d / "repair_log.json")
        return {
            "id": ep_id,
            "meta": meta,
            "n_views": len(views),
            "views": views,
            "original_ply": _ply_info(d / ORIGINAL_PLY),
            "repaired_ply": _ply_info(d / REPAIRED_PLY),
            "repair_log": log,
        }

    def start_replay(self, episode_id: str, *, reload_code: bool = True) -> tuple[bool, str]:
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        meta = self.app._read_json(d / "meta.json") or {}
        views = discover_repair_views(d, meta=meta)
        if not views:
            return False, "No regenerated RGB views in this episode (need step_NNN_regen.png)."
        with self.app.lock:
            if self.app.scene_status != "ready" or self.app.scene is None:
                return False, "Dashboard scene is not ready."
        with self._lock:
            if self.job["status"] == "running":
                return False, "A repair replay is already running."
            self._stop.clear()
            self.job = {
                **self._idle_job(episode_id),
                "status": "running",
                "n_views": len(views),
                "started_at": time.time(),
                "reload_code": bool(reload_code),
                "message": f"Replaying {len(views)} view(s)…",
            }
        self._thread = threading.Thread(
            target=self._run, args=(episode_id, d, meta, views, bool(reload_code)),
            daemon=True,
        )
        self._thread.start()
        return True, f"Replaying 3D repair on {len(views)} view(s)."

    def stop_replay(self) -> tuple[bool, str]:
        with self._lock:
            if self.job["status"] != "running":
                return False, "No repair replay running."
            self.job["status"] = "stopping"
            self.job["message"] = "Stop requested…"
        self._stop.set()
        return True, "Stop requested."

    def show(self, episode_id: str, which: str, *, force: bool = False) -> tuple[bool, str]:
        which = str(which or "").strip().lower()
        if which not in ("original", "repaired"):
            return False, "which must be 'original' or 'repaired'."
        with self.app.lock:
            if self.app.run and self.app.run["status"] in ("running", "stopping"):
                return False, (
                    "An episode is capturing from the visor — leave the catalog "
                    "scene loaded until it finishes."
                )
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        ply = d / (ORIGINAL_PLY if which == "original" else REPAIRED_PLY)
        if not ply.is_file():
            return False, f"{ply.name} is not on disk yet — run a replay first."
        with self._lock:
            if (
                self.showing == which
                and self.showing_episode == episode_id
                and not force
            ):
                return True, f"Viser already showing {which}."
        with self.app.lock:
            spec = self.app._scene_spec
            up_axis = spec.up_axis if spec is not None else "+y"
            self.app._scene_generation += 1
            generation = self.app._scene_generation
        preview = SceneSpec(
            id=f"repair-{which}",
            label=f"{episode_id} ({which})",
            path=ply.resolve(),
            up_axis=up_axis,
        )
        publish_live_scene(preview, generation, reload=True)
        with self._lock:
            self.showing = which
            self.showing_episode = episode_id
        logger.info("Viser preview %s -> %s", which, ply)
        return True, f"Viser showing {which} splat."

    def look_at(self, episode_id: str, step: int) -> tuple[bool, str]:
        """Point the viser frustum overlay at a regenerated view's camera."""
        with self.app.lock:
            if self.app.run and self.app.run["status"] in ("running", "stopping"):
                return False, "Episode is using the visor overlay."
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        meta = self.app._read_json(d / "meta.json") or {}
        views = discover_repair_views(d, meta=meta)
        view = next((v for v in views if int(v["step"]) == int(step)), None)
        if view is None:
            return False, f"No regenerated view at step {step}."
        record = {
            "step": view["step"],
            "pose": view.get("pose") or f"step {step}",
            "position": view["position"],
            "yaw_deg": view["yaw_deg"],
            "pitch_deg": view["pitch_deg"],
        }
        frame = d / view["rendered_name"]
        self.app._publish_pose(
            record, frame,
            {"width": view["width"], "height": view["height"]},
            [view["position"]],
            snap_camera=True,
        )
        return True, f"Viewer camera set to step {step}."

    def _source_for_episode(self, meta: dict):
        params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
        scene_id = params.get("scene")
        with self.app.lock:
            spec = self.app._scene_spec
            scene = self.app.scene
        if scene is None:
            raise RuntimeError("No Gaussian scene loaded in the dashboard.")
        if scene_id and spec is not None and spec.id != scene_id:
            logger.warning(
                "Episode scene %s differs from dashboard %s; repairing the loaded scene",
                scene_id, spec.id,
            )
        return scene

    def _run(self, episode_id: str, episode_dir: Path, meta: dict,
             views: list[dict], reload_code: bool) -> None:
        try:
            module = reload_repair_module() if reload_code else None
            replay = replay_episode_repairs if module is None else module.replay_episode_repairs
            source = self._source_for_episode(meta)
            params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
            with self.app.lock:
                spec = self.app._scene_spec
            up_axis = str(
                params.get("up_axis")
                or (spec.up_axis if spec is not None else None)
                or self.app.cfg.camera.up_axis
            )
            fov = float(params.get("fov_deg") or self.app.cfg.renderer.fov_deg)

            def on_view(index, view, result) -> None:
                body = result.to_json() if hasattr(result, "to_json") else dict(result)
                with self._lock:
                    self.job["current_index"] = index
                    self.job["current_step"] = view.get("step")
                    self.job["n_done"] = index + 1
                    self.job["results"] = list(self.job["results"]) + [body]
                    self.job["message"] = (
                        f"View {index + 1}/{self.job['n_views']} "
                        f"(step {view.get('step')}) {body.get('status')}"
                    )
                if result.status == "ok" and self.showing == "repaired":
                    self.show(episode_id, "repaired", force=True)

            replay(
                source,
                episode_dir,
                views,
                up_axis=up_axis,
                fov_deg=fov,
                on_view=on_view,
                should_stop=self._stop.is_set,
            )
            with self._lock:
                stopped = self._stop.is_set()
                self.job["status"] = "stopped" if stopped else "completed"
                self.job["finished_at"] = time.time()
                n_ok = sum(1 for r in self.job["results"] if r.get("status") == "ok")
                self.job["message"] = (
                    f"{'Stopped after' if stopped else 'Finished'} "
                    f"{n_ok}/{self.job['n_views']} view(s)."
                )
        except Exception as exc:
            logger.exception("Repair replay failed")
            with self._lock:
                self.job["status"] = "error"
                self.job["error"] = f"{type(exc).__name__}: {exc}"
                self.job["message"] = self.job["error"]
                self.job["finished_at"] = time.time()
