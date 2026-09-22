"""Disk-backed scene-run dashboard adapter.

This module deliberately owns no worker thread.  It validates browser input
and delegates queue, detail, stop, and artifact operations to SceneRunStore;
the separate scene-run manager consumes queued runs.
"""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from ..image_edit import SCENE_RUN_GPT_IMAGE_MODEL


SCENE_RUN_DEFAULTS = {
    "scene_id": "venetian-balcony",
    "backend": "cli_relay",
    "model": "gpt-5.6-luna",
    "width": 960,
    "height": 720,
    "duration_seconds": 3600,
    "send_map": True,
    "image_edit_backend": SCENE_RUN_GPT_IMAGE_MODEL,
    "repair_backend": "gsfix-gsplat",
    "repair_trigger": "regenerate_yes",
    "repair_type": "original",
    "repair_seconds": 180,
}

ACTIVE_STATUSES = {"queued", "waiting_gpu", "starting", "running", "stopping"}
REPAIR_TRIGGERS = {"every_step", "every_artifact", "regenerate_yes"}
REPAIR_TYPES = {"original", "looped"}
REPAIR_TYPE_ALIASES = {
    "original": "original",
    "original_gsfix3d": "original",
    "gsfix3d": "original",
    "paper": "original",
    "looped": "looped",
    "loop": "looped",
}


class SceneRunValidationError(ValueError):
    """A scene-run form contains an invalid value."""


def _jsonable(value: Any) -> Any:
    """Convert store dataclasses and common scalar wrappers to JSON values."""
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        return _jsonable(to_json())
    return value


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    value = cfg
    for part in dotted.split("."):
        try:
            if isinstance(value, Mapping):
                value = value[part]
            else:
                value = getattr(value, part)
        except (AttributeError, KeyError, TypeError):
            return default
    return value


def _positive_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise SceneRunValidationError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SceneRunValidationError(f"{name} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise SceneRunValidationError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return parsed


class SceneRunStudio:
    """Thin, lazy bridge between the web server and ``SceneRunStore``."""

    def __init__(self, app: Any, store: Any | None = None, *, pipeline: str = "baseline"):
        self.app = app
        self.pipeline = pipeline
        self.cfg = app.cfg
        self.root = Path(_cfg_get(self.cfg, "output.dir", "outputs")) / "scene-runs"
        self._store = store
        self._visor = None

    @property
    def store(self):
        if self._store is None:
            # Lazy so the regular dashboard can still start while scene-run
            # dependencies are being installed or developed independently.
            from ..scene_runs import SceneRunStore

            self._store = SceneRunStore(root=self.root)
        return self._store

    @property
    def visor(self):
        if self._visor is None:
            from .scene_run_viser import SceneRunViser

            self._visor = SceneRunViser(self)
        return self._visor

    def preferred_visor_scene_id(self) -> str | None:
        """Catalog room for a queued or live scene-run, if any.

        The shared :8080 capture visor otherwise falls back to Starter Scene
        after ``scripts/start.sh`` clears ``outputs/live/scene.json``.
        """
        for run in self.list_runs():
            if self._status(run) not in ACTIVE_STATUSES:
                continue
            config = run.get("config") if isinstance(run, Mapping) else None
            if not isinstance(config, Mapping):
                continue
            scene_id = str(config.get("scene_id") or "").strip()
            if scene_id:
                return scene_id
        return None

    def ply_status(self, run_id: str) -> dict:
        """Whether this run has original/repaired PLYs, plus the visor URL."""
        visor_path = f"/{run_id}/viser"
        return {
            "original": self.artifact_path(str(run_id), "scene_original.ply") is not None,
            "repaired": self.artifact_path(str(run_id), "scene_repaired.ply") is not None,
            "viser_path": visor_path,
        }

    def scenes(self) -> list[dict]:
        from ..scene.catalog import list_scenes

        try:
            return [_jsonable(scene) for scene in list_scenes(self.cfg)]
        except (AttributeError, KeyError, TypeError):
            # Small fake configs used by API clients/tests need not define the
            # complete renderer and camera configuration.
            return []

    def defaults(self) -> dict:
        defaults = dict(SCENE_RUN_DEFAULTS)
        defaults["model"] = str(
            _cfg_get(self.cfg, "agent.model", "") or defaults["model"]
        )
        if self.pipeline == "extended":
            from ..scene_runs_ext.config import DEFAULTS
            defaults.update(pipeline="extended", extended=dict(DEFAULTS),
                            width=640, height=480, repair_backend="artifixer-gsplat")
        return defaults

    def validate_config(self, body: Mapping[str, Any] | None) -> dict:
        if not isinstance(body, Mapping):
            raise SceneRunValidationError("Request body must be a JSON object.")
        clean = self.defaults()
        aliases = {
            "scene": "scene_id",
            "duration": "duration_seconds",
            "duration_s": "duration_seconds",
        }
        for raw_key, value in body.items():
            key = aliases.get(str(raw_key), str(raw_key))
            if key in clean and value not in (None, ""):
                clean[key] = value

        clean["scene_id"] = str(clean["scene_id"]).strip()
        clean["backend"] = str(clean["backend"]).strip()
        clean["model"] = str(clean["model"]).strip()
        clean["image_edit_backend"] = str(clean["image_edit_backend"]).strip()
        clean["repair_backend"] = str(clean["repair_backend"]).strip()
        clean["repair_trigger"] = str(clean["repair_trigger"]).strip()
        raw_type = str(clean.get("repair_type") or "original").strip().lower()
        clean["repair_type"] = REPAIR_TYPE_ALIASES.get(raw_type, raw_type)
        for key in ("scene_id", "backend", "image_edit_backend", "repair_backend"):
            if not clean[key]:
                raise SceneRunValidationError(f"{key} is required.")
        if clean["backend"] != "cli_relay":
            raise SceneRunValidationError(
                "Automated scene-runs require cli_relay so every action comes "
                "from the configured VLM."
            )
        if not clean["model"]:
            raise SceneRunValidationError("model is required for CliRelay.")
        from ..image_edit import canonical_image_edit_choice

        try:
            clean["image_edit_backend"] = canonical_image_edit_choice(
                clean["image_edit_backend"],
            )
        except ValueError as exc:
            raise SceneRunValidationError(str(exc)) from exc
        expected_backend = "artifixer-gsplat" if self.pipeline == "extended" else "gsfix-gsplat"
        if clean["repair_backend"] != expected_backend:
            raise SceneRunValidationError(
                "The first automated scene-run release supports gsfix-gsplat only."
            )

        scene_ids = {str(scene.get("id") or "") for scene in self.scenes()}
        if scene_ids and clean["scene_id"] not in scene_ids:
            raise SceneRunValidationError(f"Unknown scene {clean['scene_id']!r}.")
        if clean["repair_trigger"] not in REPAIR_TRIGGERS:
            allowed = ", ".join(sorted(REPAIR_TRIGGERS))
            raise SceneRunValidationError(f"repair_trigger must be one of: {allowed}.")
        if clean["repair_type"] not in REPAIR_TYPES:
            allowed = ", ".join(sorted(REPAIR_TYPES))
            raise SceneRunValidationError(f"repair_type must be one of: {allowed}.")

        clean["width"] = _positive_int(clean["width"], "width", minimum=64, maximum=3840)
        clean["height"] = _positive_int(clean["height"], "height", minimum=64, maximum=2160)
        clean["duration_seconds"] = _positive_int(
            clean["duration_seconds"], "duration_seconds", minimum=1, maximum=7 * 86400,
        )
        clean["repair_seconds"] = _positive_int(
            clean["repair_seconds"], "repair_seconds", minimum=1, maximum=12 * 3600,
        )
        # Maps are part of the isolated scene-run protocol and cannot be
        # disabled by a crafted browser request.
        clean["send_map"] = True
        if self.pipeline == "extended":
            from ..scene_runs_ext.config import validate_options
            from ..image_edit import is_qwen_backend
            try:
                clean["extended"] = validate_options(clean.get("extended"))
            except ValueError as exc:
                raise SceneRunValidationError(str(exc)) from exc
            if clean["width"] % 16 or clean["height"] % 16:
                raise SceneRunValidationError("Extended width and height must be multiples of 16")
            if is_qwen_backend(clean["image_edit_backend"]):
                raise SceneRunValidationError("Extended runs currently use CliRelay image editing; choose a GPT image model")
            clean["pipeline"] = "extended"
        return clean

    def list_runs(self) -> list[dict]:
        runs = _jsonable(self.store.list_runs())
        return [run for run in runs if (run.get("config") or {}).get("pipeline", "baseline") == self.pipeline] if isinstance(runs, list) else []

    def detail(self, run_id: str) -> dict | None:
        try:
            detail = self.store.detail(str(run_id))
        except (FileNotFoundError, ValueError):
            return None
        if detail is None:
            return None
        result = _jsonable(detail)
        if isinstance(result, dict):
            if "files" not in result:
                result["files"] = self._artifact_listing(str(run_id))
            ply = self.ply_status(str(run_id))
            result["viser_path"] = ply["viser_path"]
            result["ply"] = {"original": ply["original"], "repaired": ply["repaired"]}
        return result

    def create(self, body: Mapping[str, Any] | None) -> dict:
        create = getattr(self.store, "create", None)
        if not callable(create):
            create = getattr(self.store, "create_run", None)
        if not callable(create):
            raise RuntimeError("SceneRunStore has no create method.")
        run = _jsonable(create(self.validate_config(body)))
        if isinstance(run, str):
            detail = self.detail(run)
            return detail or {"id": run, "status": "queued"}
        if not isinstance(run, dict):
            raise RuntimeError("SceneRunStore.create() returned no run detail.")
        scene_id = ""
        config = run.get("config")
        if isinstance(config, Mapping):
            scene_id = str(config.get("scene_id") or "").strip()
        select = getattr(self.app, "select_scene", None)
        if callable(select) and scene_id:
            try:
                select(scene_id)
            except Exception:
                pass
        return run

    def request_stop(self, run_id: str) -> dict:
        result = _jsonable(self.store.request_stop(str(run_id)))
        if isinstance(result, dict):
            return result
        detail = self.detail(run_id)
        if detail is not None:
            return detail
        return {"id": str(run_id), "stop_requested": bool(result)}

    def artifact_path(self, run_id: str, relative: str) -> Path | None:
        """Resolve one existing run artifact without permitting path escapes."""
        relative = str(relative or "")
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            return None
        for method_name in ("artifact_path", "safe_artifact_path", "safe_path"):
            method = getattr(self.store, method_name, None)
            if not callable(method):
                continue
            try:
                candidate = method(str(run_id), relative)
            except (FileNotFoundError, ValueError):
                return None
            if candidate is None:
                return None
            path = Path(candidate)
            return path if path.is_file() else None

        # Compatibility fallback for early stores whose safe primitive is the
        # run directory itself.
        safe_run_path = getattr(self.store, "safe_run_path", None)
        if not callable(safe_run_path):
            safe_run_path = getattr(self.store, "run_path", None)
        try:
            run_dir = Path(safe_run_path(str(run_id))) if callable(safe_run_path) else (
                self.root / str(run_id)
            )
            resolved_run = run_dir.resolve()
            target = (resolved_run / relative).resolve()
        except (OSError, RuntimeError, ValueError):
            return None
        if not target.is_relative_to(resolved_run) or not target.is_file():
            return None
        return target

    def _artifact_listing(self, run_id: str) -> list[dict]:
        safe_run_path = getattr(self.store, "safe_run_path", None)
        if not callable(safe_run_path):
            safe_run_path = getattr(self.store, "run_path", None)
        if not callable(safe_run_path):
            return []
        try:
            run_dir = Path(safe_run_path(run_id)).resolve()
        except (OSError, TypeError, ValueError):
            return []
        if not run_dir.is_dir():
            return []
        extensions = {
            ".png": "image", ".jpg": "image", ".jpeg": "image",
            ".webp": "image", ".gif": "image", ".ply": "ply",
            ".log": "log", ".txt": "log", ".jsonl": "log",
        }
        files = []
        try:
            candidates = sorted(run_dir.rglob("*"))
        except OSError:
            return []
        for path in candidates:
            kind = extensions.get(path.suffix.lower())
            if kind is None or not path.is_file():
                continue
            try:
                relative = path.resolve().relative_to(run_dir).as_posix()
                size = path.stat().st_size
            except (OSError, ValueError):
                continue
            files.append({
                "name": path.name,
                "relative": relative,
                "kind": kind,
                "size": size,
                "url": (
                    f"/scene-run-files/{quote(run_id, safe='')}/"
                    + "/".join(quote(part, safe="") for part in relative.split("/"))
                ),
            })
        return files

    def state(self) -> dict:
        runs = self.list_runs()
        active = [
            run for run in runs
            if self._status(run) in ACTIVE_STATUSES
        ]
        active_run = next(
            (run for run in active if self._status(run)
             in {"starting", "running", "stopping"}),
            active[0] if active else None,
        )
        manager = self._manager_state(active_run)
        return {
            "defaults": self.defaults(),
            "scenes": self.scenes(),
            "runs": runs,
            "active": bool(active),
            "active_run": active_run,
            "manager": manager,
        }

    @staticmethod
    def _status(run: Mapping[str, Any]) -> str:
        status = run.get("status")
        if status is None and isinstance(run.get("state"), Mapping):
            status = run["state"].get("status")
        if isinstance(status, Mapping):
            status = status.get("status")
        return str(status or "").lower()

    @staticmethod
    def _run_id(run: Mapping[str, Any] | None) -> str | None:
        if not run:
            return None
        value = run.get("id") or run.get("run_id")
        return str(value) if value else None

    def _manager_state(self, active_run: dict | None) -> dict:
        for method_name in ("manager_status", "manager_state"):
            method = getattr(self.store, method_name, None)
            if callable(method):
                value = _jsonable(method())
                if isinstance(value, dict):
                    return value

        for name in ("manager.json", "manager_status.json", ".manager.json"):
            path = self.root / name
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                value = _jsonable(value)
                updated = value.get("updated_at")
                fresh = (
                    isinstance(updated, (int, float))
                    and time.time() - float(updated) < 120.0
                )
                value["active"] = value.get("status") == "running" and fresh
                value["stale"] = value.get("status") == "running" and not fresh
                value["run_id"] = self._run_id(active_run)
                return value
        return {
            "status": "not_detected",
            "active": False,
            "run_id": self._run_id(active_run),
        }
