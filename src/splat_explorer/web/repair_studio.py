"""Replay 3DGS repair on a past (or live) episode and preview original vs repaired.

Lives beside the episode dashboard. Regenerated RGB PNGs are enough to rerun
the photometric lift after code edits — no live VLM / image-model run needed.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

from ..repair import (
    ORIGINAL_PLY,
    REPAIRED_PLY,
    discover_repair_views,
    list_repair_backends,
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
            "backend": "auto",
            "mode": "episode",
            "max_seconds": None,
            "resume": None,
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
        episode_scene_label = None
        if detail and isinstance(detail.get("meta"), dict):
            params = detail["meta"].get("params")
            if isinstance(params, dict):
                episode_scene = params.get("scene")
                episode_scene_label = params.get("scene_label") or episode_scene
        return {
            "job": job,
            "showing": showing,
            "showing_episode": showing_episode,
            "live_episode": live_ep,
            "run_status": run_status,
            "scene_status": scene_status,
            "scene_id": scene_id,
            "episode_scene": episode_scene,
            "episode_scene_label": episode_scene_label,
            "up_axis": up_axis,
            "backends": list_repair_backends(),
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
            lift = view.get("lift_name")
            view["lifted_url"] = f"/frames/{ep_id}/{lift}" if lift else None
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

    def start_replay(
        self,
        episode_id: str,
        *,
        reload_code: bool = True,
        backend: str = "auto",
        step: int | None = None,
        max_seconds: float = 3600.0,
        resume: bool = True,
    ) -> tuple[bool, str]:
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        meta = self.app._read_json(d / "meta.json") or {}
        views = discover_repair_views(d, meta=meta)
        if not views:
            return False, "No regenerated RGB views in this episode (need step_NNN_regen.png)."
        focused = step is not None
        if focused:
            views = [v for v in views if int(v["step"]) == int(step)]
            if not views:
                return False, f"No regenerated view at step {step}."
        has_ply = (d / REPAIRED_PLY).is_file() or (d / ORIGINAL_PLY).is_file()
        with self.app.lock:
            catalog_ready = self.app.scene_status == "ready" and self.app.scene is not None
        if not has_ply and not catalog_ready:
            return False, (
                "Dashboard scene is not ready. Wait for the catalog to finish "
                "loading, or run scripts/start.sh and try again."
            )
        backend_name = str(backend or "auto").strip() or "auto"
        cap = max(30.0, float(max_seconds or 3600.0))
        resume = bool(resume)
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
                "backend": backend_name,
                "mode": "view" if focused else "episode",
                "max_seconds": cap if focused else None,
                "resume": resume,
                "message": (
                    f"{'Continuing' if resume and (d / REPAIRED_PLY).is_file() else 'Repairing'} "
                    f"step {int(step)} until Stop (max {int(cap)}s)…"
                    if focused
                    else f"Replaying {len(views)} view(s) with {backend_name}…"
                ),
            }
        self._thread = threading.Thread(
            target=self._run,
            args=(
                episode_id, d, meta, views, bool(reload_code), backend_name,
                focused, cap, resume,
            ),
            daemon=True,
        )
        self._thread.start()
        if focused:
            return True, f"Repairing step {int(step)} — press Stop when it looks right (max {int(cap / 60)} min)."
        return True, f"Replaying 3D repair on {len(views)} view(s) ({backend_name})."

    def stop_replay(self) -> tuple[bool, str]:
        with self._lock:
            if self.job["status"] != "running":
                return False, "No repair replay running."
            self.job["status"] = "stopping"
            self.job["message"] = "Stop requested…"
        self._stop.set()
        return True, "Stop requested."

    def reset_repair(self, episode_id: str) -> tuple[bool, str]:
        """Copy scene_original.ply back over scene_repaired.ply."""
        with self._lock:
            if self.job["status"] == "running":
                return False, "Stop the current repair before resetting."
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        original = d / ORIGINAL_PLY
        repaired = d / REPAIRED_PLY
        if not original.is_file():
            return False, "No scene_original.ply yet — run a repair once to snapshot the catalog."
        shutil.copy2(original, repaired)
        logger.info("Reset %s from %s", repaired.name, original.name)
        ok, message = self.show(episode_id, "original", force=True)
        if not ok:
            return True, f"Restored {repaired.name} from original. {message}"
        return True, f"Restored {repaired.name} from original."

    def ensure_catalog_scene(self, episode_id: str) -> tuple[bool, str]:
        """Load the episode's catalog scene into the shared visor (one 3DGS)."""
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        meta = self.app._read_json(d / "meta.json") or {}
        params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
        scene_id = str(params.get("scene") or "").strip()
        if not scene_id:
            return False, "Episode has no scene id in meta.json."
        with self.app.lock:
            if self.app.run and self.app.run["status"] in ("running", "stopping"):
                return False, "Episode is using the visor."
            spec = self.app._scene_spec
            current = spec.id if spec is not None else None
            status = self.app.scene_status
        with self._lock:
            previewing = self.showing is not None
        if current == scene_id and status == "loading":
            return True, f"Loading {scene_id}…"
        if current == scene_id and status == "ready" and not previewing:
            return True, f"Viser already on {scene_id}."
        if current == scene_id and status == "ready" and previewing:
            return self._republish_catalog()
        return self.app.select_scene(scene_id)

    def _republish_catalog(self) -> tuple[bool, str]:
        """Put the catalog splat back in viser after an original/repaired PLY preview."""
        with self.app.lock:
            spec = self.app._scene_spec
            if spec is None:
                return False, "No catalog scene loaded."
            self.app._scene_generation += 1
            generation = self.app._scene_generation
        publish_live_scene(spec, generation, reload=True, catalog_id=spec.id)
        with self._lock:
            self.showing = None
            self.showing_episode = None
        return True, f"Viser restored to {spec.id}."

    def show(self, episode_id: str, which: str, *, force: bool = True) -> tuple[bool, str]:
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
            already = (
                self.showing == which
                and self.showing_episode == episode_id
            )
        if already and not force:
            return True, f"Viser already showing {which}."
        with self.app.lock:
            spec = self.app._scene_spec
            up_axis = spec.up_axis if spec is not None else "+y"
            self.app._scene_generation += 1
            generation = self.app._scene_generation
        meta = self.app._read_json(d / "meta.json") or {}
        params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
        up_axis = str(params.get("up_axis") or up_axis)
        episode_scene = str(params.get("scene") or "").strip()
        catalog_id = episode_scene or (
            spec.id if spec is not None and not str(spec.id).startswith("repair-") else None
        )
        preview = SceneSpec(
            id=f"repair-{which}",
            label=f"{episode_id} ({which})",
            path=ply,
            up_axis=up_axis,
        )
        publish_live_scene(preview, generation, reload=True, catalog_id=catalog_id)
        with self._lock:
            self.showing = which
            self.showing_episode = episode_id
        logger.info("Viser preview %s -> %s (generation %s)", which, ply, generation)
        return True, f"Viser showing {which} splat ({ply.name})."

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

    def _source_for_episode(self, meta: dict, episode_dir: Path, *, resume: bool):
        from ..scene import load_ply

        repaired = Path(episode_dir) / REPAIRED_PLY
        original = Path(episode_dir) / ORIGINAL_PLY
        if resume and repaired.is_file():
            logger.info("Continuing 3D repair from %s", repaired)
            return load_ply(repaired)
        if original.is_file():
            logger.info("Starting 3D repair from %s", original)
            return load_ply(original)
        params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
        scene_id = params.get("scene")
        with self.app.lock:
            spec = self.app._scene_spec
            scene = self.app.scene
            status = self.app.scene_status
        if scene is None or status != "ready":
            raise RuntimeError(
                "No episode PLY and catalog scene is still loading. "
                "Wait until the visor is ready, or run scripts/start.sh."
            )
        if scene_id and spec is not None and spec.id != scene_id:
            raise RuntimeError(
                f"Visor is on {spec.id}, not episode scene {scene_id}. "
                "Wait for the catalog room to load before replaying."
            )
        return scene

    def _run(self, episode_id: str, episode_dir: Path, meta: dict,
             views: list[dict], reload_code: bool, backend_name: str = "auto",
             focused: bool = False, max_seconds: float = 3600.0,
             resume: bool = True) -> None:
        try:
            module = reload_repair_module() if reload_code else None
            replay = replay_episode_repairs if module is None else module.replay_episode_repairs
            make_backend = (
                module.make_repair_backend if module is not None else None
            )
            from ..repair import make_repair_backend as _default_backend
            source = self._source_for_episode(meta, episode_dir, resume=resume)
            factory = make_backend or _default_backend
            try:
                backend = factory(backend_name, studio=True, focused=focused)
            except TypeError:
                backend = factory(backend_name)
            params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
            with self.app.lock:
                spec = self.app._scene_spec
            up_axis = str(
                params.get("up_axis")
                or (spec.up_axis if spec is not None else None)
                or self.app.cfg.camera.up_axis
            )
            fov = float(params.get("fov_deg") or self.app.cfg.renderer.fov_deg)
            deadline = (time.time() + float(max_seconds)) if focused else None
            started = time.time()

            def on_progress(stats) -> None:
                elapsed = time.time() - started
                phase = stats.get("phase") or "refine"
                with self._lock:
                    self.job["message"] = (
                        f"Step {views[0].get('step')} {phase}"
                        + (f" · chunk {stats.get('n_chunks')}" if stats.get("n_chunks") else "")
                        + (f" · {int(stats.get('n_stamped') or stats.get('n_updated') or 0)} stamped")
                        + (f" · {stats.get('n_iters') or 0} iters")
                        + (f" · {elapsed:.0f}s")
                        + (
                            f" · train {stats['train_width']}x{stats['train_height']}"
                            if stats.get("train_width") and stats.get("train_height")
                            else ""
                        )
                    )
                self.show(episode_id, "repaired", force=True)

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
                        + (f" · {body['backend']}" if body.get("backend") else "")
                        + (f" · {body['seconds']:.1f}s" if body.get("seconds") is not None else "")
                        + (
                            f" · train {body['train_width']}x{body['train_height']}"
                            if body.get("train_width") and body.get("train_height")
                            else ""
                        )
                    )
                if result.status == "ok":
                    self.show(episode_id, "repaired", force=True)

            replay(
                source,
                episode_dir,
                views,
                up_axis=up_axis,
                fov_deg=fov,
                backend=backend,
                on_view=on_view,
                should_stop=self._stop.is_set,
                until_stop=focused,
                deadline=deadline,
                on_progress=on_progress if focused else None,
            )
            with self._lock:
                stopped = self._stop.is_set()
                self.job["status"] = "stopped" if stopped else "completed"
                self.job["finished_at"] = time.time()
                n_ok = sum(1 for r in self.job["results"] if r.get("status") == "ok")
                if focused:
                    elapsed = (self.job["finished_at"] or time.time()) - (self.job["started_at"] or time.time())
                    self.job["message"] = (
                        f"{'Stopped' if stopped else 'Reached 1h cap'} on step "
                        f"{views[0].get('step')} after {elapsed:.0f}s. Toggle Repaired to inspect."
                    )
                else:
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
