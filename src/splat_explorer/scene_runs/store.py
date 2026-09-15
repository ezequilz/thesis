"""Filesystem-backed control-plane state for scene runs.

The store deliberately uses only files and atomic filesystem operations so a
web process, launcher, and worker can coordinate without sharing Python state.
Each run directory contains ``config.json``, ``status.json``, ``events.jsonl``
and, after a stop request, a ``STOP`` marker.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .models import (
    RunState,
    RunStatus,
    SceneRun,
    SceneRunConfig,
    isoformat_utc,
    utc_now,
)

DEFAULT_ROOT = Path("outputs/scene-runs")
CONFIG_NAME = "config.json"
STATUS_NAME = "status.json"
EVENTS_NAME = "events.jsonl"
STOP_NAME = "STOP"
GPU_LEASE_NAME = ".gpu-lease"
GPU_OWNER_NAME = "owner.json"
RUN_ID_PATTERN = re.compile(r"run_(\d{8})_(\d{6})(?:_(\d+))?")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON via a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class GpuLease:
    """An owned cross-process GPU lease returned by ``acquire_gpu_lease``.

    Call :meth:`release` or use the lease as a context manager. Release checks
    the random ownership token, so one run cannot remove another run's lease.
    """

    run_id: str
    token: str
    owner: dict[str, Any]
    _store: "SceneRunStore" = field(repr=False, compare=False)
    _released: bool = field(default=False, init=False, repr=False, compare=False)
    _release_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def release(self) -> bool:
        """Release this lease once; return whether this call removed it."""

        with self._release_lock:
            if self._released:
                return False
            released = self._store.release_gpu_lease(self)
            if released:
                self._released = True
            return released

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class SceneRunStore:
    """Persistent scene-run state shared safely between local processes.

    Main integration methods are :meth:`create_run`, :meth:`get_run`,
    :meth:`list_runs`, :meth:`set_status`, :meth:`append_event`,
    :meth:`request_stop`, and :meth:`acquire_gpu_lease`.
    """

    def __init__(
        self,
        root: str | os.PathLike[str] = DEFAULT_ROOT,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.root = Path(root)
        self._clock = clock

    def _ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root.resolve()

    def run_path(self, run_id: str) -> Path:
        """Resolve a validated run ID without allowing path traversal."""

        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"invalid run_id: {run_id!r}")
        root = self._ensure_root()
        unresolved = root / run_id
        if unresolved.is_symlink():
            raise ValueError(f"unsafe run path for {run_id!r}")
        candidate = unresolved.resolve()
        if candidate.parent != root:
            raise ValueError(f"unsafe run path for {run_id!r}")
        return candidate

    def create_run(
        self,
        config: SceneRunConfig | Mapping[str, Any] | None = None,
        *,
        now: datetime | None = None,
    ) -> SceneRun:
        """Allocate a collision-safe run directory and persist initial state."""

        if isinstance(config, SceneRunConfig):
            validated = config
        else:
            validated = SceneRunConfig.from_dict(config)
        instant = now or self._clock()
        base_id = f"run_{instant.strftime('%Y%m%d_%H%M%S')}"
        suffix = 1
        while True:
            run_id = base_id if suffix == 1 else f"{base_id}_{suffix}"
            path = self.run_path(run_id)
            try:
                path.mkdir()
                break
            except FileExistsError:
                suffix += 1

        timestamp = isoformat_utc(instant)
        state = RunState(
            status=RunStatus.QUEUED,
            created_at=timestamp,
            updated_at=timestamp,
        )
        try:
            _atomic_write_json(path / CONFIG_NAME, validated.to_dict())
            _atomic_write_json(path / STATUS_NAME, state.to_dict())
            self.append_event(run_id, "created", status=RunStatus.QUEUED.value)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return SceneRun(run_id, str(path), validated, state, False)

    def get_run(self, run_id: str) -> SceneRun:
        """Load one run's config, current state, and stop flag."""

        path = self.run_path(run_id)
        if not path.is_dir():
            raise FileNotFoundError(f"scene run not found: {run_id}")
        config = SceneRunConfig.from_dict(_read_json(path / CONFIG_NAME))
        state = RunState.from_dict(_read_json(path / STATUS_NAME))
        return SceneRun(run_id, str(path), config, state, (path / STOP_NAME).is_file())

    # A short alias is convenient in HTTP detail handlers.
    detail = get_run

    def list_runs(self) -> list[SceneRun]:
        """Return all valid runs, newest IDs first.

        Incomplete or corrupt directories are skipped so one interrupted create
        does not prevent the control plane from listing healthy runs.
        """

        root = self._ensure_root()
        runs: list[SceneRun] = []
        def sort_key(path: Path) -> tuple[str, int]:
            match = RUN_ID_PATTERN.fullmatch(path.name)
            if match is None:
                return "", 0
            return match.group(1) + match.group(2), int(match.group(3) or 1)

        for path in sorted(root.glob("run_*"), key=sort_key, reverse=True):
            if not path.is_dir():
                continue
            try:
                runs.append(self.get_run(path.name))
            except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return runs

    def set_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        message: str = "",
        details: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> RunState:
        """Atomically replace current status and append a status event."""

        previous = self.get_run(run_id).state
        state = RunState(
            status=RunStatus(status),
            created_at=previous.created_at,
            updated_at=isoformat_utc(now or self._clock()),
            message=message,
            details=dict(details or {}),
        )
        _atomic_write_json(self.run_path(run_id) / STATUS_NAME, state.to_dict())
        self.append_event(
            run_id,
            "status",
            status=state.status.value,
            message=state.message,
            details=state.details,
        )
        return state

    def update_status(
        self,
        run_id: str,
        status: RunStatus | str | None = None,
        *,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
        now: datetime | None = None,
        **fields: Any,
    ) -> RunState:
        """Merge progress fields into the current state.

        Long-running workers update phase/step/metrics independently. Keeping
        those values under ``details`` preserves the small stable state schema
        while allowing callers to omit an unchanged lifecycle status.
        """

        previous = self.get_run(run_id).state
        merged = dict(previous.details)
        if details:
            merged.update(dict(details))
        merged.update(fields)
        return self.set_status(
            run_id,
            status or previous.status,
            message=previous.message if message is None else str(message),
            details=merged,
            now=now,
        )

    def append_event(
        self,
        run_id: str,
        event: str | Mapping[str, Any],
        **data: Any,
    ) -> dict[str, Any]:
        """Append one JSON object to ``events.jsonl`` using ``O_APPEND``."""

        path = self.run_path(run_id)
        if not path.is_dir():
            raise FileNotFoundError(f"scene run not found: {run_id}")
        if isinstance(event, str):
            record: dict[str, Any] = {"event": event, **data}
        elif isinstance(event, Mapping):
            if data:
                raise ValueError("keyword data cannot accompany a mapping event")
            record = dict(event)
        else:
            raise TypeError("event must be a string or mapping")
        record.setdefault("timestamp", isoformat_utc(self._clock()))
        encoded = _json_bytes(record)
        fd = os.open(path / EVENTS_NAME, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(fd, encoded)
            if written != len(encoded):
                raise OSError("short write while appending scene-run event")
        finally:
            os.close(fd)
        return record

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        """Read all event records; a missing event log is an empty list."""

        path = self.run_path(run_id) / EVENTS_NAME
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("event lines must be JSON objects")
                    records.append(value)
        return records

    def request_stop(self, run_id: str) -> Path:
        """Atomically create the idempotent ``STOP`` marker for a run."""

        path = self.run_path(run_id)
        if not path.is_dir():
            raise FileNotFoundError(f"scene run not found: {run_id}")
        marker = path / STOP_NAME
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return marker
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes({"requested_at": isoformat_utc(self._clock())}))
            stream.flush()
            os.fsync(stream.fileno())
        self.append_event(run_id, "stop_requested")
        return marker

    def stop_requested(self, run_id: str) -> bool:
        """Return whether the run's ``STOP`` marker exists."""

        return (self.run_path(run_id) / STOP_NAME).is_file()

    @property
    def gpu_lease_path(self) -> Path:
        """Path of the singleton GPU lease directory."""

        return self._ensure_root() / GPU_LEASE_NAME

    def gpu_lease_owner(self) -> dict[str, Any] | None:
        """Return current lease metadata, or ``None`` when unowned."""

        lease_path = self.gpu_lease_path
        if not lease_path.is_dir():
            return None
        try:
            return _read_json(lease_path / GPU_OWNER_NAME)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return None

    def acquire_gpu_lease(
        self,
        run_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> GpuLease | None:
        """Try to own the singleton GPU lease using atomic ``mkdir``.

        If the current owner is on this host and its PID no longer exists, the
        stale lease is recovered and acquisition is retried once. Otherwise
        contention returns ``None`` immediately.
        """

        # Require a real run and keep arbitrary caller metadata from replacing
        # the identity fields used for stale detection and safe release.
        self.get_run(run_id)
        lease_path = self.gpu_lease_path
        token = uuid.uuid4().hex
        owner = dict(metadata or {})
        owner.update(
            {
                "run_id": run_id,
                "token": token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": isoformat_utc(self._clock()),
            }
        )
        for attempt in range(2):
            try:
                lease_path.mkdir()
            except FileExistsError:
                if attempt or not self._recover_stale_local_lease():
                    return None
                continue
            try:
                _atomic_write_json(lease_path / GPU_OWNER_NAME, owner)
            except BaseException:
                shutil.rmtree(lease_path, ignore_errors=True)
                raise
            return GpuLease(run_id=run_id, token=token, owner=owner, _store=self)
        return None

    def _recover_stale_local_lease(self) -> bool:
        lease_path = self.gpu_lease_path
        owner = self.gpu_lease_owner()
        if not owner or owner.get("hostname") != socket.gethostname():
            return False
        pid = owner.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or _pid_alive(pid):
            return False
        quarantine = self._ensure_root() / f".gpu-lease-stale-{uuid.uuid4().hex}"
        try:
            lease_path.rename(quarantine)
        except (FileNotFoundError, FileExistsError, OSError):
            return False
        shutil.rmtree(quarantine, ignore_errors=True)
        return True

    def release_gpu_lease(self, lease: GpuLease) -> bool:
        """Release ``lease`` only if its ownership token still matches."""

        if not isinstance(lease, GpuLease):
            raise TypeError("lease must be a GpuLease")
        owner = self.gpu_lease_owner()
        if owner is None or owner.get("token") != lease.token:
            return False
        lease_path = self.gpu_lease_path
        quarantine = self._ensure_root() / f".gpu-lease-release-{lease.token}"
        try:
            lease_path.rename(quarantine)
        except (FileNotFoundError, FileExistsError, OSError):
            return False
        shutil.rmtree(quarantine, ignore_errors=True)
        return True
