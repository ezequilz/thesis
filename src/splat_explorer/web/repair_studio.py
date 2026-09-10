"""Replay 3DGS repair on a past (or live) episode and preview original vs repaired.

Lives beside the episode dashboard. Regenerated RGB PNGs are enough to rerun
the photometric lift after code edits — no live VLM / image-model run needed.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from ..repair import (
    HIGHLIGHT_PLY,
    METRICS_JSON,
    ORIGINAL_PLY,
    REPAIRED_PLY,
    REPAIR_LOG,
    append_custom_repair_view,
    discover_repair_views,
    list_repair_backends,
    next_repair_step,
    reload_repair_module,
    repair_meta_name,
    repair_progress_suffix,
    replay_episode_repairs,
)
from ..scene.catalog import SceneSpec, publish_live_scene

logger = logging.getLogger(__name__)

# Focused "Repair this view" safety net. CUDA GSFix3D used to exit after one
# 20-iter paper chunk (~11s) and then always print "Reached 1h cap".
FOCUSED_REPAIR_MAX_SECONDS = 12 * 3600
_SPECTATOR_ASPECT = 16.0 / 9.0
_SPECTATOR_ASPECT_TOL = 0.08
_REPAIRED_SAVE_RE = re.compile(r"^scene_repaired_(\d+)\.ply$")


def repaired_save_name(index: int) -> str:
    return f"scene_repaired_{int(index)}.ply"


def repaired_save_index(name: str) -> int | None:
    match = _REPAIRED_SAVE_RE.fullmatch(Path(name).name)
    return int(match.group(1)) if match else None


def list_repaired_saves(episode_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in Path(episode_dir).glob("scene_repaired_*.ply"):
        index = repaired_save_index(path.name)
        if index is not None and path.is_file():
            found.append((index, path))
    found.sort(key=lambda item: item[0])
    return found


def next_repaired_save_index(episode_dir: Path) -> int:
    saves = list_repaired_saves(episode_dir)
    return (saves[-1][0] + 1) if saves else 1


def parse_repaired_save_id(value) -> int | None:
    if value in (None, "", "current", "repaired", 0, "0"):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("save must be a positive integer.") from exc
    if index < 1:
        raise ValueError("save must be a positive integer.")
    return index


def pick_interactive_camera(cameras: list[dict]) -> dict | None:
    """Choose the visor tab the user is flying, not the HD spectator.

    Prefers the most recently updated non-16:9 client. On a timestamp tie,
    skip the 4:3 episode-capture visor so the repair-page iframe wins.
    """
    if not cameras:
        return None

    def is_spectator(cam: dict) -> bool:
        w, h = int(cam.get("width") or 0), int(cam.get("height") or 0)
        if h < 90 or w < 160:
            return False
        return abs((w / h) - _SPECTATOR_ASPECT) <= _SPECTATOR_ASPECT_TOL

    candidates = [c for c in cameras if not is_spectator(c)] or list(cameras)

    def sort_key(cam: dict):
        w, h = int(cam.get("width") or 0), int(cam.get("height") or 0)
        usable = bool(cam.get("usable"))
        return (
            float(cam.get("updated_at") or 0.0),
            0 if usable else 1,
            w * h,
        )

    return max(candidates, key=sort_key)


def focused_cap_label(seconds: float) -> str:
    sec = max(0.0, float(seconds))
    hours = sec / 3600.0
    if sec >= 3600 and abs(hours - round(hours)) < 1e-6:
        return f"{int(round(hours))}h"
    if sec >= 60:
        return f"{int(round(sec / 60.0))} min"
    return f"{int(round(sec))}s"


def focused_finish_message(
    *,
    stopped: bool,
    hit_deadline: bool,
    step,
    elapsed: float,
    cap: float,
) -> str:
    if stopped:
        reason = "Stopped"
    elif hit_deadline:
        reason = f"Reached {focused_cap_label(cap)} cap"
    else:
        reason = "Finished"
    return (
        f"{reason} on step {step} after {elapsed:.0f}s. "
        "Toggle Repaired to inspect."
    )


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
        self.showing_highlight: bool = False
        self.showing_save: int | None = None  # numbered snapshot, or None = working ply
        self._last_preview_at: float = 0.0
        self._regenerator = None
        self._regenerator_lock = threading.Lock()

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
            "backend": "gsfix-gsplat",
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
            showing_highlight = self.showing_highlight
            showing_save = self.showing_save
        ep = episode_id or job.get("episode") or live_ep
        detail = self.episode_review(ep) if ep else None
        episode_scene = None
        episode_scene_label = None
        capture_width, capture_height = 960, 720
        params = None
        if detail and isinstance(detail.get("meta"), dict):
            params = detail["meta"].get("params")
            if isinstance(params, dict):
                episode_scene = params.get("scene")
                episode_scene_label = params.get("scene_label") or episode_scene
        if not isinstance(params, dict):
            renderer = getattr(self.app.cfg, "renderer", None)
            if isinstance(renderer, dict):
                params = renderer
            elif renderer is not None:
                params = {
                    "width": getattr(renderer, "width", None),
                    "height": getattr(renderer, "height", None),
                }
        if isinstance(params, dict):
            capture_width = int(params.get("width") or capture_width)
            capture_height = int(params.get("height") or capture_height)
        return {
            "job": job,
            "showing": showing,
            "showing_episode": showing_episode,
            "showing_highlight": showing_highlight,
            "showing_save": showing_save,
            "live_episode": live_ep,
            "run_status": run_status,
            "scene_status": scene_status,
            "scene_id": scene_id,
            "episode_scene": episode_scene,
            "episode_scene_label": episode_scene_label,
            "capture_width": capture_width,
            "capture_height": capture_height,
            "up_axis": up_axis,
            "backends": list_repair_backends(),
            "episode": detail,
            "pending_regen": bool(
                detail
                and any(
                    (v.get("custom") and (v.get("regen_status") in ("queued", None))
                     and not v.get("has_repaired"))
                    for v in (detail.get("views") or [])
                )
            ),
            "viewer_url": f"http://localhost:{self.app.cfg.viewer.port}",
            "gpu_url": "/repair/gpu",
        }

    def gpu_snapshot(
        self,
        *,
        probe: bool = False,
        force: bool = False,
        history_start: str | None = None,
        history_end: str | None = None,
    ) -> dict:
        from ..repair_lrz import lrz_dashboard_snapshot

        with self._lock:
            job = dict(self.job)
        return lrz_dashboard_snapshot(
            repair_job=job,
            request_probe=probe,
            force_probe=force,
            history_start=history_start,
            history_end=history_end,
        )

    def gpu_allocate(
        self,
        hours: int = 8,
        after: bool | str | None = False,
        begin: str | None = None,
        partition: str | None = None,
    ) -> dict:
        from ..repair_lrz import allocate_lrz_gpu

        result = allocate_lrz_gpu(hours, after=after, begin=begin, partition=partition)
        snap = self.gpu_snapshot(probe=True, force=True)
        snap["allocate"] = result
        snap["ok"] = True
        snap["message"] = result.get("message")
        return snap

    def gpu_use_job(self, job_id: str) -> dict:
        from ..repair_lrz import use_lrz_job

        result = use_lrz_job(job_id)
        snap = self.gpu_snapshot(probe=True, force=True)
        snap["ok"] = True
        snap["message"] = result.get("message")
        return snap

    def gpu_cancel_job(self, job_id: str, *, confirm: bool = False) -> dict:
        from ..repair_lrz import cancel_lrz_job

        result = cancel_lrz_job(job_id, confirm=confirm)
        snap = self.gpu_snapshot(probe=True, force=True)
        snap["ok"] = True
        snap["message"] = result.get("message")
        snap["cancel"] = result
        return snap

    def gpu_load_setup(self, *, force: bool = True) -> dict:
        from ..repair_lrz import request_lrz_setup

        result = request_lrz_setup(force=force)
        snap = self.gpu_snapshot()
        snap["ok"] = True
        snap["setup"] = result
        snap["message"] = result.get("message") or "Loading GPU setup…"
        return snap

    def gpu_widen(self, job_id: str | None = None, partition: str | None = None) -> dict:
        from ..repair_lrz import widen_lrz_job

        result = widen_lrz_job(job_id, partition=partition)
        snap = self.gpu_snapshot(probe=True, force=True)
        snap["ok"] = True
        snap["message"] = result.get("message")
        snap["widen"] = result
        return snap

    def gpu_review(self, *, force: bool = True) -> dict:
        from ..repair_lrz import review_lrz_partitions

        result = review_lrz_partitions(force=force)
        snap = self.gpu_snapshot()
        if not (result.get("inflight") and not result.get("nodes") and not result.get("partitions")):
            snap["partitions"] = result.get("partitions")
            snap["nodes"] = result.get("nodes")
            snap["summary"] = result.get("summary")
            snap["default_free"] = result.get("default_free")
        snap["ok"] = True
        snap["message"] = result.get("message")
        snap["reviewing"] = bool(result.get("inflight") or snap.get("reviewing"))
        return snap

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
        views = discover_repair_views(d, meta=meta, include_pending_custom=True)
        for view in views:
            step = view["step"]
            view["rendered_url"] = f"/frames/{ep_id}/{view['rendered_name']}"
            if view.get("has_repaired"):
                view["repaired_url"] = f"/frames/{ep_id}/{view['repaired_name']}"
            else:
                view["repaired_url"] = None
            lift = view.get("lift_name")
            view["lifted_url"] = f"/frames/{ep_id}/{lift}" if lift else None
            del view["rendered_path"]
            del view["repaired_path"]
        log = self.app._read_json(d / "repair_log.json")
        has_metrics = (d / METRICS_JSON).is_file() or any(
            isinstance(v.get("repair"), dict) and v["repair"].get("l1_after") is not None
            for v in views
        )
        repaired_saves = []
        for index, path in list_repaired_saves(d):
            info = _ply_info(path)
            if info is None:
                continue
            info["id"] = index
            repaired_saves.append(info)
        return {
            "id": ep_id,
            "meta": meta,
            "n_views": len(views),
            "views": views,
            "original_ply": _ply_info(d / ORIGINAL_PLY),
            "repaired_ply": _ply_info(d / REPAIRED_PLY),
            "repaired_saves": repaired_saves,
            "repair_log": log,
            "has_metrics": has_metrics,
            "metrics_url": f"/api/repair/metrics?episode={ep_id}",
        }

    def metrics_payload(
        self, episode_id: str, step: int | None = None,
    ) -> tuple[dict | None, str]:
        """JSON for the metrics.json download (latest repair, or one step)."""
        d = self.app.episode_path(episode_id)
        filename = "metrics.json"
        if d is None:
            return None, filename
        body = None
        if step is not None:
            filename = f"step_{int(step):03d}_metrics.json"
            meta = self.app._read_json(d / repair_meta_name(int(step)))
            if isinstance(meta, dict):
                body = meta
        if body is None and step is None:
            latest = self.app._read_json(d / METRICS_JSON)
            if isinstance(latest, dict):
                body = latest
                filename = "metrics.json"
        if body is None:
            log = self.app._read_json(d / REPAIR_LOG)
            repairs = log.get("repairs") if isinstance(log, dict) else None
            if isinstance(repairs, list):
                if step is not None:
                    for row in reversed(repairs):
                        if isinstance(row, dict) and int(row.get("step", -1)) == int(step):
                            body = row
                            break
                elif repairs and isinstance(repairs[-1], dict):
                    body = repairs[-1]
        if not isinstance(body, dict):
            return None, filename
        out = dict(body)
        before, after = out.get("l1_before"), out.get("l1_after")
        if out.get("l1_improved") is None and before is not None and after is not None:
            try:
                out["l1_improved"] = float(after) < float(before)
            except (TypeError, ValueError):
                pass
        return out, filename

    def start_replay(
        self,
        episode_id: str,
        *,
        reload_code: bool = True,
        backend: str = "gsfix-gsplat",
        step: int | None = None,
        max_seconds: float = FOCUSED_REPAIR_MAX_SECONDS,
        resume: bool = True,
        ssh_password: str | None = None,
    ) -> tuple[bool, str]:
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        meta = self.app._read_json(d / "meta.json") or {}
        views = discover_repair_views(d, meta=meta)
        focused = step is not None
        if focused:
            views = [v for v in views if int(v["step"]) == int(step)]
            if not views:
                return False, f"No regenerated view at step {step}."
        else:
            views = [v for v in views if not v.get("custom")]
            if not views:
                return False, "No regenerated RGB views in this episode (need step_NNN_regen.png)."
        has_ply = (d / REPAIRED_PLY).is_file() or (d / ORIGINAL_PLY).is_file()
        with self.app.lock:
            catalog_ready = self.app.scene_status == "ready" and self.app.scene is not None
        if not has_ply and not catalog_ready:
            return False, (
                "Dashboard scene is not ready. Wait for the catalog to finish "
                "loading, or run scripts/start.sh and try again."
            )
        backend_name = str(backend or "gsfix-gsplat").strip() or "gsfix-gsplat"
        cap = max(30.0, float(max_seconds or FOCUSED_REPAIR_MAX_SECONDS))
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
                    f"step {int(step)} until Stop (max {focused_cap_label(cap)})…"
                    if focused
                    else f"Replaying {len(views)} view(s) with {backend_name}…"
                )
                + (
                    " Open scripts/lrz/ssh-session.sh once if ControlMaster is down."
                    if str(backend_name) in (
                        "gsfix-gsplat", "gsfix-gsplat-baseline",
                        "gsfix-gsplat-visprune",
                        "cuda", "gsplat", "gsfix", "visprune",
                    )
                    else ""
                ),
            }
        self._thread = threading.Thread(
            target=self._run,
            args=(
                episode_id, d, meta, views, bool(reload_code), backend_name,
                focused, cap, resume, ssh_password or None,
            ),
            daemon=True,
        )
        self._thread.start()
        if focused:
            return True, (
                f"Repairing step {int(step)} — press Stop when it looks right "
                f"(max {focused_cap_label(cap)})."
            )
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
        """Copy scene_original.ply back over scene_repaired.ply.

        Numbered snapshots from Save repair stay on disk for the dropdown.
        """
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
        n_saves = len(list_repaired_saves(d))
        note = f"Restored {repaired.name} from original."
        if n_saves:
            note = f"{note} {n_saves} saved snapshot(s) still available."
        ok, message = self.show(episode_id, "original", force=True)
        if not ok:
            note = f"{note} {message}"
        with self._lock:
            # Drop the last job's traceback so /repair does not keep showing
            # a failed LRZ run after the splat has been restored.
            self.job = {**self._idle_job(episode_id), "message": note}
            self.showing_save = None
        return True, note

    def save_repair(self, episode_id: str) -> tuple[bool, str, dict]:
        """Copy the working repaired ply to the next numbered snapshot."""
        extra: dict = {}
        with self._lock:
            if self.job["status"] == "running":
                return False, "Stop the current repair before saving a snapshot.", extra
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found.", extra
        repaired = d / REPAIRED_PLY
        if not repaired.is_file():
            return False, "No scene_repaired.ply yet — run a repair first.", extra
        index = next_repaired_save_index(d)
        dest = d / repaired_save_name(index)
        shutil.copy2(repaired, dest)
        logger.info("Saved repair snapshot %s -> %s", index, dest.name)
        extra = {
            "save": index,
            "name": dest.name,
            "bytes": dest.stat().st_size,
            "mtime": dest.stat().st_mtime,
        }
        return True, f"Saved snapshot {index} as {dest.name}.", extra

    def add_view(self, episode_id: str) -> tuple[bool, str, dict]:
        """Capture the live visor camera as a dashboard-only repair view.

        Writes `repair_custom_views.jsonl` + `step_NNN.png` and queues gpt-image-2
        on the studio Regenerator thread pool. Does not touch actions.jsonl,
        meta.json, or a running harness episode.
        """
        extra: dict = {}
        with self._lock:
            if self.job["status"] == "running":
                return False, "Wait for the current 3D repair to finish (or Stop) before adding a view.", extra
        with self.app.lock:
            run = self.app.run
            if run and run.get("status") in ("running", "stopping"):
                return False, "Stop the episode before adding a custom repair view.", extra
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found.", extra
        meta = self.app._read_json(d / "meta.json") or {}
        params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
        width = int(params.get("width") or 960)
        height = int(params.get("height") or 720)
        fov_deg = float(params.get("fov_deg") or getattr(self.app.cfg.renderer, "fov_deg", 75.0) or 75.0)
        up_axis = str(
            params.get("up_axis")
            or getattr(self.app.cfg.camera, "up_axis", None)
            or "+y"
        )
        try:
            cameras = self._visor_cameras()
        except Exception as exc:
            return False, f"Could not read visor cameras: {exc}", extra
        cam = pick_interactive_camera(cameras)
        if cam is None:
            return False, (
                "No visor camera yet. Keep the 3D visor on this page visible "
                "and fly to the view you want, then try again. "
                "If this dashboard was already running, restart the visor so it serves /cameras."
            ), extra
        from ..agent.camera_rig import CameraRig

        look_at = cam.get("look_at")
        position = cam.get("position")
        if not look_at or not position:
            return False, "Visor camera is missing position/look_at.", extra
        rig = CameraRig.from_look_at(position, look_at, up_axis=up_axis)
        camera = rig.camera(width, height, fov_deg)
        try:
            rgb = self._capture_view_rgb(camera)
        except Exception as exc:
            return False, f"Could not capture this visor view: {exc}", extra
        step = next_repair_step(d)
        frame_name = f"step_{step:03d}.png"
        regen_name = f"step_{step:03d}_regen.png"
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(d / frame_name)
        record = {
            "step": step,
            "position": np.asarray(rig.position, dtype=np.float64).tolist(),
            "yaw_deg": float(rig.yaw_deg),
            "pitch_deg": float(rig.pitch_deg),
            "pose": rig.state_description(),
            "frame": frame_name,
            "regenerate_frame": regen_name,
            "width": width,
            "height": height,
            "fov_deg": fov_deg,
        }
        append_custom_repair_view(d, record)
        queued = False
        regen_error = None
        try:
            regen = self._ensure_regenerator()
            regen.submit(d / frame_name, d, step)
            queued = True
        except Exception as exc:
            logger.exception("Could not queue image repair for custom step %s", step)
            regen_error = f"{type(exc).__name__}: {exc}"
        extra = {
            "step": step,
            "pose": record["pose"],
            "custom": True,
            "queued": queued,
        }
        if queued:
            return True, (
                f"Added custom step {step} from the visor and queued image repair. "
                "It appears at the end of the list; Repair this view unlocks once the PNG returns."
            ), extra
        return True, (
            f"Added custom step {step} from the visor, but image repair did not queue "
            f"({regen_error})."
        ), extra

    def _visor_render_url(self) -> str:
        viewer = self.app.cfg.viewer
        port = 8081
        if hasattr(viewer, "get"):
            port = int(viewer.get("render_port", 8081) or 8081)
        else:
            port = int(getattr(viewer, "render_port", 8081) or 8081)
        return f"http://localhost:{port}"

    def _visor_cameras(self) -> list[dict]:
        import urllib.error
        import urllib.request

        url = self._visor_render_url() + "/cameras"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    "Visor capture API has no /cameras yet — restart the visor "
                    "(scripts/start.sh) and reload /repair."
                ) from exc
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"GET /cameras -> HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not reach visor cameras at {url}: {exc}") from exc
        cams = payload.get("cameras") if isinstance(payload, dict) else None
        if not isinstance(cams, list):
            raise RuntimeError("Visor /cameras returned no camera list.")
        return cams

    def _capture_view_rgb(self, camera) -> "np.ndarray":
        import io
        import urllib.error
        import urllib.request

        from PIL import Image

        body = json.dumps({
            "position": np.asarray(camera.position, dtype=np.float64).tolist(),
            "wxyz": camera.rotation_wxyz().tolist(),
            "fov": camera.vertical_fov_rad(),
            "width": int(camera.width),
            "height": int(camera.height),
            "any_client": True,
        }).encode()
        url = self._visor_render_url() + "/render"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"POST /render -> HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Visor capture at {url} failed: {exc}") from exc
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        need_w, need_h = int(camera.width), int(camera.height)
        if img.shape[1] != need_w or img.shape[0] != need_h:
            from ..rendering.viser_viewer import _center_crop_and_resize
            img = _center_crop_and_resize(img, need_w, need_h)
        return img

    def _ensure_regenerator(self):
        with self._regenerator_lock:
            if self._regenerator is None:
                from ..agent.regenerate import regenerator_from_config
                self._regenerator = regenerator_from_config(self.app.cfg)
            return self._regenerator

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
            return True, f"Viser showing {self.showing} ply."
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
            self.showing_highlight = False
            self.showing_save = None
        return True, f"Viser restored to {spec.id}."

    def show(
        self, episode_id: str, which: str, *, force: bool = False,
        highlight: bool = False, save: int | str | None = None,
    ) -> tuple[bool, str]:
        which = str(which or "").strip().lower()
        if which not in ("original", "repaired"):
            return False, "which must be 'original' or 'repaired'."
        try:
            save_id = parse_repaired_save_id(save) if which == "repaired" else None
        except ValueError as exc:
            return False, str(exc)
        use_highlight = bool(highlight) and which == "repaired"
        with self.app.lock:
            if self.app.run and self.app.run["status"] in ("running", "stopping"):
                return False, (
                    "An episode is capturing from the visor — leave the catalog "
                    "scene loaded until it finishes."
                )
        d = self.app.episode_path(episode_id)
        if d is None:
            return False, f"Episode {episode_id} not found."
        if which == "original":
            ply = d / ORIGINAL_PLY
        elif save_id is not None:
            ply = d / repaired_save_name(save_id)
        else:
            ply = d / REPAIRED_PLY
        if use_highlight:
            try:
                ply = self._highlight_ply(d, repaired=ply)
            except Exception as exc:
                return False, f"Could not build highlight splat: {exc}"
        elif not ply.is_file():
            if save_id is not None:
                return False, f"{ply.name} is not on disk."
            return False, f"{ply.name} is not on disk yet — run a replay first."
        with self._lock:
            already = (
                self.showing == which
                and self.showing_episode == episode_id
                and bool(self.showing_highlight) == bool(use_highlight)
                and (which != "repaired" or self.showing_save == save_id)
            )
        if already and not force:
            if use_highlight:
                label = "repaired highlight"
            elif save_id is not None:
                label = f"save {save_id}"
            else:
                label = which
            return True, f"Viser already showing {label}."
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
        if use_highlight:
            tag = "highlight"
        elif save_id is not None:
            tag = f"save-{save_id}"
        else:
            tag = which
        preview = SceneSpec(
            id=f"repair-{tag}",
            label=f"{episode_id} ({tag})",
            path=ply,
            up_axis=up_axis,
        )
        publish_live_scene(preview, generation, reload=True, catalog_id=catalog_id)
        with self._lock:
            self.showing = which
            self.showing_episode = episode_id
            self.showing_highlight = use_highlight
            if which == "repaired":
                self.showing_save = save_id
        logger.info("Viser preview %s -> %s (generation %s)", tag, ply, generation)
        if use_highlight:
            return True, f"Viser showing repaired splat with changed gaussians in red ({ply.name})."
        if save_id is not None:
            return True, f"Viser showing save {save_id} ({ply.name})."
        return True, f"Viser showing {which} splat ({ply.name})."

    def _highlight_ply(self, episode_dir: Path, repaired: Path | None = None) -> Path:
        """Build (or reuse) a red overlay of gaussians that differ from original."""
        from ..repair import highlight_repaired_scene
        from ..scene import load_ply, save_ply

        original = Path(episode_dir) / ORIGINAL_PLY
        repaired = Path(repaired) if repaired is not None else Path(episode_dir) / REPAIRED_PLY
        if repaired_save_index(repaired.name) is not None:
            out = Path(episode_dir) / f"{repaired.stem}_highlight.ply"
        else:
            out = Path(episode_dir) / HIGHLIGHT_PLY
        if not original.is_file():
            raise FileNotFoundError(
                "scene_original.ply is missing — need it to mark changed gaussians."
            )
        if not repaired.is_file():
            raise FileNotFoundError(
                f"{repaired.name} is not on disk yet — run a replay first."
            )
        src_mtime = max(original.stat().st_mtime, repaired.stat().st_mtime)
        if out.is_file() and out.stat().st_mtime >= src_mtime:
            return out
        scene = highlight_repaired_scene(load_ply(original), load_ply(repaired))
        save_ply(scene, out)
        return out

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
            views = discover_repair_views(d, meta=meta, include_pending_custom=True)
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
             focused: bool = False, max_seconds: float = FOCUSED_REPAIR_MAX_SECONDS,
             resume: bool = True, ssh_password: str | None = None) -> None:
        try:
            if ssh_password:
                from ..repair_lrz import set_ssh_password
                set_ssh_password(ssh_password)
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
            if hasattr(backend, "should_stop"):
                backend.should_stop = self._stop.is_set
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
                    if str(phase) in ("awaiting_ssh", "packed"):
                        cmd = stats.get("command") or "scripts/lrz/ssh-session.sh"
                        self.job["message"] = (
                            f"Packed for LRZ. If CUDA is greyed out, run `{cmd}` "
                            "and type your password once (ControlMaster ~/.ssh/cm-lrz)."
                        )
                    elif str(phase) in ("rsync_up", "srun", "rsync_down"):
                        self.job["message"] = (
                            f"Step {views[0].get('step')} {phase} via LRZ ControlMaster"
                            f" · {elapsed:.0f}s"
                        )
                    else:
                        self.job["message"] = (
                            f"Step {views[0].get('step')} {phase}"
                            + repair_progress_suffix(stats)
                            + (f" · {elapsed:.0f}s")
                            + (
                                f" · train {stats['train_width']}x{stats['train_height']}"
                                if stats.get("train_width") and stats.get("train_height")
                                else ""
                            )
                        )
                now = time.time()
                if now - self._last_preview_at >= 10.0:
                    self._last_preview_at = now
                    self.show(
                        episode_id, "repaired", force=True,
                        highlight=self.showing_highlight,
                    )

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
                    self.show(
                        episode_id, "repaired", force=True,
                        highlight=self.showing_highlight,
                    )

            if hasattr(backend, "on_progress"):
                backend.on_progress = on_progress

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
                on_progress=on_progress,
            )
            with self._lock:
                stopped = self._stop.is_set()
                self.job["status"] = "stopped" if stopped else "completed"
                self.job["finished_at"] = time.time()
                n_ok = sum(1 for r in self.job["results"] if r.get("status") == "ok")
                if focused:
                    elapsed = (self.job["finished_at"] or time.time()) - (self.job["started_at"] or time.time())
                    hit_deadline = (
                        not stopped
                        and deadline is not None
                        and self.job["finished_at"] >= deadline
                    )
                    self.job["message"] = focused_finish_message(
                        stopped=stopped,
                        hit_deadline=hit_deadline,
                        step=views[0].get("step"),
                        elapsed=elapsed,
                        cap=max_seconds,
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
        finally:
            try:
                from ..repair_lrz import set_ssh_password
                set_ssh_password(None)
            except Exception:
                pass
