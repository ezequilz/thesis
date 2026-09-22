"""Typed values and pure helpers for scene-run orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping

from splat_explorer.agent.actions import Action, wants_regenerate
from splat_explorer.image_edit import SCENE_RUN_GPT_IMAGE_MODEL, canonical_image_edit_choice


class RepairTrigger(str, Enum):
    """Policy controlling when a scene-run requests a repair."""

    EVERY_STEP = "every_step"
    EVERY_ARTIFACT = "every_artifact"
    REGENERATE_YES = "regenerate_yes"


class RepairType(str, Enum):
    """How each triggered view is lifted back into the 3D Gaussians."""

    ORIGINAL = "original"
    LOOPED = "looped"


REPAIR_TYPE_ALIASES = {
    "original": RepairType.ORIGINAL,
    "original_gsfix3d": RepairType.ORIGINAL,
    "gsfix3d": RepairType.ORIGINAL,
    "paper": RepairType.ORIGINAL,
    "looped": RepairType.LOOPED,
    "loop": RepairType.LOOPED,
}


class RunStatus(str, Enum):
    """Lifecycle states persisted by :class:`SceneRunStore`."""

    QUEUED = "queued"
    WAITING_GPU = "waiting_gpu"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    GPU_EXPIRED = "gpu_expired"


@dataclass(frozen=True)
class SceneRunConfig:
    """Validated, JSON-serializable configuration for one scene run."""

    scene_id: str = "venetian-balcony"
    backend: str = "cli_relay"
    model: str = "gpt-5.6-luna"
    width: int = 960
    height: int = 720
    duration_seconds: int = 3600
    send_map: bool = True
    image_edit_backend: str = SCENE_RUN_GPT_IMAGE_MODEL
    repair_backend: str = "gsfix-gsplat"
    repair_trigger: RepairTrigger = RepairTrigger.REGENERATE_YES
    repair_type: RepairType = RepairType.ORIGINAL
    repair_seconds: int = 180
    repair_width: int = 0
    repair_height: int = 0
    pipeline: str = "baseline"
    extended: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pipeline not in {"baseline", "extended"}:
            raise ValueError("pipeline must be baseline or extended")
        if self.pipeline == "extended":
            from ..scene_runs_ext.config import validate_options, validate_repair_resolution
            object.__setattr__(self, "extended", validate_options(self.extended))
            if self.width % 16 or self.height % 16:
                raise ValueError("Extended width and height must be multiples of 16")
            repair_width, repair_height = validate_repair_resolution(
                self.width, self.height, self.repair_width, self.repair_height,
            )
            object.__setattr__(self, "repair_width", repair_width)
            object.__setattr__(self, "repair_height", repair_height)
        for name in ("scene_id", "backend", "image_edit_backend", "repair_backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        try:
            image_backend = canonical_image_edit_choice(self.image_edit_backend)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        object.__setattr__(self, "image_edit_backend", image_backend)
        if not isinstance(self.model, str):
            raise ValueError("model must be a string")
        for name in ("width", "height", "duration_seconds", "repair_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.send_map is not True:
            raise ValueError("send_map is fixed to true")
        try:
            trigger = RepairTrigger(self.repair_trigger)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in RepairTrigger)
            raise ValueError(f"repair_trigger must be one of: {choices}") from exc
        object.__setattr__(self, "repair_trigger", trigger)
        raw_type = self.repair_type
        if isinstance(raw_type, RepairType):
            kind = raw_type
        else:
            key = str(raw_type or "").strip().lower()
            kind = REPAIR_TYPE_ALIASES.get(key)
        if kind is None:
            choices = ", ".join(item.value for item in RepairType)
            raise ValueError(f"repair_type must be one of: {choices}")
        object.__setattr__(self, "repair_type", kind)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None = None) -> "SceneRunConfig":
        """Build a config from partial JSON-like input, applying defaults."""

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("config must be a mapping")
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in dict(value).items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repair_trigger"] = self.repair_trigger.value
        data["repair_type"] = self.repair_type.value
        if self.pipeline == "baseline":
            data.pop("pipeline")
            data.pop("extended")
            data.pop("repair_width", None)
            data.pop("repair_height", None)
        elif not self.repair_width or not self.repair_height:
            data.pop("repair_width", None)
            data.pop("repair_height", None)
        return data


@dataclass(frozen=True)
class RunState:
    """Current persisted lifecycle state for a scene run."""

    status: RunStatus
    created_at: str
    updated_at: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            status = RunStatus(self.status)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in RunStatus)
            raise ValueError(f"status must be one of: {choices}") from exc
        object.__setattr__(self, "status", status)
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at must be a non-empty ISO timestamp")
        if not isinstance(self.updated_at, str) or not self.updated_at:
            raise ValueError("updated_at must be a non-empty ISO timestamp")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string")
        if not isinstance(self.details, dict):
            raise ValueError("details must be an object")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunState":
        if not isinstance(value, Mapping):
            raise TypeError("state must be a mapping")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class SceneRun:
    """A complete run detail returned by the store."""

    run_id: str
    path: str
    config: SceneRunConfig
    state: RunState
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "stop_requested": self.stop_requested,
        }


def utc_now() -> datetime:
    """Return the current UTC time (separate for simple test injection)."""

    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime | None = None) -> str:
    """Render a datetime as an ISO-8601 UTC timestamp."""

    value = value or utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_slurm_duration(value: str | int | None) -> timedelta | None:
    """Parse Slurm durations (``D-HH:MM:SS``, ``HH:MM:SS``, or minutes).

    ``UNLIMITED``, ``N/A``, empty values, and ``None`` return ``None``.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"UNLIMITED", "INFINITE", "N/A", "NONE", "UNKNOWN"}:
        return None
    if text.isdigit():
        return timedelta(minutes=int(text))

    days = 0
    has_days = "-" in text
    clock = text
    if has_days:
        day_text, clock = text.split("-", 1)
        if not day_text.isdigit():
            raise ValueError(f"invalid Slurm duration: {value!r}")
        days = int(day_text)
    parts = clock.split(":")
    if len(parts) == 1 and has_days:
        hours, minutes, seconds = parts[0], "0", "0"
    elif len(parts) == 2 and has_days:
        hours, minutes, seconds = parts[0], parts[1], "0"
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"invalid Slurm duration: {value!r}")
    try:
        hours_i, minutes_i, seconds_i = int(hours), int(minutes), int(seconds)
    except ValueError as exc:
        raise ValueError(f"invalid Slurm duration: {value!r}") from exc
    if min(hours_i, minutes_i, seconds_i, days) < 0 or minutes_i >= 60 or seconds_i >= 60:
        raise ValueError(f"invalid Slurm duration: {value!r}")
    return timedelta(days=days, hours=hours_i, minutes=minutes_i, seconds=seconds_i)


def parse_slurm_end(value: str | datetime | None) -> datetime | None:
    """Parse an ISO-like Slurm ``EndTime`` value.

    Slurm's non-date sentinel values return ``None``. A trailing ``Z`` is
    accepted; timezone-less Slurm timestamps remain timezone-less.
    """

    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.upper() in {
        "N/A",
        "NONE",
        "UNKNOWN",
        "UNLIMITED",
        "NOT_SET",
        "INVALID",
    }:
        return None
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"invalid Slurm end time: {value!r}") from exc


def effective_deadline(
    user_end: str | datetime | None,
    gpu_expected_end: str | datetime | None,
    *,
    safety_seconds: int | float = 300,
) -> datetime | None:
    """Return ``min(user_end, gpu_expected_end - safety_seconds)``.

    Either bound may be absent. When one timestamp is timezone-less and the
    other is aware, the timezone-less value is interpreted in the other's
    timezone so the bounds remain comparable.
    """

    if isinstance(safety_seconds, bool) or safety_seconds < 0:
        raise ValueError("safety_seconds must be non-negative")
    user = parse_slurm_end(user_end)
    gpu = parse_slurm_end(gpu_expected_end)
    gpu_safe = gpu - timedelta(seconds=float(safety_seconds)) if gpu else None
    if user is None:
        return gpu_safe
    if gpu_safe is None:
        return user
    if (user.tzinfo is None) != (gpu_safe.tzinfo is None):
        if user.tzinfo is None:
            user = user.replace(tzinfo=gpu_safe.tzinfo)
        else:
            gpu_safe = gpu_safe.replace(tzinfo=user.tzinfo)
    return min(user, gpu_safe)


def should_trigger_repair(
    trigger: RepairTrigger | str,
    action: Action | Mapping[str, Any],
) -> bool:
    """Evaluate a repair policy for an ``Action`` or JSON-like action mapping."""

    policy = RepairTrigger(trigger)
    if isinstance(action, Action):
        name, args = action.name, action.args
    elif isinstance(action, Mapping):
        name = action.get("name", action.get("action"))
        args = action.get("args", action.get("arguments", {}))
    else:
        raise TypeError("action must be an Action or mapping")
    if not isinstance(name, str) or not isinstance(args, Mapping):
        return False
    if policy is RepairTrigger.EVERY_STEP:
        return True
    if name != "report_artifact":
        return False
    if policy is RepairTrigger.EVERY_ARTIFACT:
        return True
    return wants_regenerate(dict(args))
