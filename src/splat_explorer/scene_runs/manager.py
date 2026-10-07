"""Persistent local queue manager for automated scene-runs."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from ..config import Config, load_dotenv
from .runner import SceneRunExecutor

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = frozenset({"starting", "running", "stopping"})
QUEUE_STATUSES = frozenset({"queued", "waiting_gpu"})


class SceneRunManager:
    """Consume disk-backed queued runs; execute at most one at a time."""

    def __init__(
        self,
        cfg: Config,
        *,
        store=None,
        executor_factory=SceneRunExecutor,
        poll_seconds: float = 1.0,
        gpu_probe_seconds: float = 600.0,
    ):
        if store is None:
            from .store import SceneRunStore

            store = SceneRunStore(Path(cfg.output.dir) / "scene-runs")
        self.cfg = cfg
        self.store = store
        self.executor_factory = executor_factory
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.gpu_probe_seconds = max(30.0, float(gpu_probe_seconds))
        self.stop_event = threading.Event()
        self._last_probe_at = 0.0
        self._last_probe: dict[str, Any] | None = None
        self.root = Path(cfg.output.dir) / "scene-runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def stop(self, *_args) -> None:
        self.stop_event.set()

    def run_forever(self) -> None:
        import fcntl
        manager_lock = (self.root / ".manager.lock").open("a")
        try:
            fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            manager_lock.close()
            raise RuntimeError("A scene-run manager is already running")
        self._install_signals()
        self._recover_interrupted()
        logger.info("Scene-run manager ready (%s)", self.root)
        while not self.stop_event.is_set():
            self._heartbeat()
            candidate = self._next_queued()
            if candidate is None:
                self.stop_event.wait(self.poll_seconds)
                continue
            run_id = str(candidate.get("id") or candidate.get("run_id") or "")
            if not run_id:
                self.stop_event.wait(self.poll_seconds)
                continue
            if self._stop_requested(run_id) and not (candidate.get("state") or {}).get("details", {}).get("remote_dir"):
                self._update(run_id, status="stopped", phase="finished",
                             message="Stopped while queued", finished_at=time.time())
                continue
            from ..splatfix.jobs import requires_gpu
            resuming = (candidate.get("state") or {}).get("details", {}).get("remote_dir")
            gpu = self._gpu_ready() if requires_gpu(candidate.get("config", {})) and not resuming else {"ready": True, "message": "Local stage"}
            if not gpu.get("ready"):
                self._update(
                    run_id,
                    status="waiting_gpu",
                    phase="waiting_gpu",
                    message=gpu.get("message") or "Waiting for an LRZ job in ST=R",
                    gpu=gpu,
                )
                # Do not turn the manager's one-second disk poll into an
                # expensive scheduler poll.
                self.stop_event.wait(min(self.gpu_probe_seconds, 30.0))
                continue
            self._apply_gpu_deadline(run_id, gpu)
            lease = self._acquire_lease(run_id, gpu)
            if lease is None:
                self._update(
                    run_id,
                    status="waiting_gpu",
                    phase="waiting_gpu",
                    message="GPU is owned by another repair or scene-run",
                )
                self.stop_event.wait(5.0)
                continue
            try:
                # Another manager may have completed this candidate while we waited.
                fresh = getattr(self.store, "get_run", lambda _: None)(run_id)
                if fresh is not None and fresh.state.status.value not in QUEUE_STATUSES:
                    continue
                executor = self.executor_factory(self.cfg, self.store)
                executor.execute(run_id)
            except Exception as exc:
                logger.exception("Uncaught scene-run manager failure for %s", run_id)
                self._update(
                    run_id,
                    status="error",
                    phase="finished",
                    error=f"{type(exc).__name__}: {exc}",
                    message=str(exc),
                    finished_at=time.time(),
                )
            finally:
                self._release_lease(lease)
        self._heartbeat(stopped=True)
        manager_lock.close()

    def run_once(self) -> bool:
        """Run one ready queue item; useful for tests and manual recovery."""
        candidate = self._next_queued()
        if candidate is None:
            return False
        from ..splatfix.jobs import requires_gpu
        resuming = (candidate.get("state") or {}).get("details", {}).get("remote_dir")
        gpu = self._gpu_ready(force=True) if requires_gpu(candidate.get("config", {})) and not resuming else {"ready": True}
        if not gpu.get("ready"):
            return False
        run_id = str(candidate.get("id") or candidate.get("run_id"))
        self._apply_gpu_deadline(run_id, gpu)
        lease = self._acquire_lease(run_id, gpu)
        if lease is None:
            return False
        try:
            fresh = getattr(self.store, "get_run", lambda _: None)(run_id)
            if fresh is not None and fresh.state.status.value not in QUEUE_STATUSES:
                return False
            self.executor_factory(self.cfg, self.store).execute(run_id)
        finally:
            self._release_lease(lease)
        return True

    def _list(self) -> list[dict[str, Any]]:
        for name in ("list_runs", "list"):
            fn = getattr(self.store, name, None)
            if callable(fn):
                value = fn()
                if isinstance(value, dict):
                    value = value.get("runs") or []
                rows = []
                for row in value or []:
                    dump = getattr(row, "to_dict", None)
                    rows.append(dict(dump() if callable(dump) else row))
                return rows
        return []

    def _next_queued(self) -> dict[str, Any] | None:
        queued = []
        for row in self._list():
            state = row.get("state", row.get("status"))
            status = state.get("status") if isinstance(state, dict) else state
            from ..splatfix.jobs import is_due
            if status in QUEUE_STATUSES and (row.get("stop_requested") or is_due(row.get("config", {}))):
                queued.append(row)
        from ..splatfix.jobs import requires_gpu
        queued.sort(key=lambda r: (
            0 if (r.get("state") or {}).get("details", {}).get("remote_dir") else (2 if requires_gpu(r.get("config", {})) else 1),
            str((r.get("state") or {}).get("created_at") or r.get("created_at") or ""),
            str(r.get("id") or r.get("run_id") or ""),
        ))
        return queued[0] if queued else None

    def _update(self, run_id: str, **fields: Any) -> None:
        fn = getattr(self.store, "update_status")
        try:
            fn(run_id, **fields)
        except TypeError:
            fn(run_id, fields)

    def _stop_requested(self, run_id: str) -> bool:
        fn = getattr(self.store, "stop_requested", None)
        if callable(fn):
            return bool(fn(run_id))
        return (self.root / run_id / "STOP").is_file()

    def _recover_interrupted(self) -> None:
        fn = getattr(self.store, "recover_interrupted", None)
        if callable(fn):
            fn()
            return
        for row in self._list():
            state = row.get("state", row.get("status"))
            status = state.get("status") if isinstance(state, dict) else state
            if status not in ACTIVE_STATUSES:
                continue
            run_id = str(row.get("id") or row.get("run_id") or "")
            config = row.get("config") or {}
            details = state.get("details", {}) if isinstance(state, dict) else {}
            if config.get("pipeline") == "splatfix" and config.get("splatfix", {}).get("stage") in ("repair", "benchmark") and details.get("remote_dir"):
                self._update(run_id, status="queued", message="Reattaching to the existing GPU reconstruction after restart")
                continue
            if status == "starting":
                self._update(
                    run_id,
                    status="queued",
                    phase="queued",
                    error=None,
                    message="Requeued after scene-run manager restart",
                )
                continue
            self._update(
                run_id,
                status="error",
                phase="finished",
                error="Scene-run manager restarted while this run was active",
                message="Interrupted by manager restart; latest PLY checkpoint was preserved",
                finished_at=time.time(),
            )

    def _gpu_ready(self, force: bool = False) -> dict[str, Any]:
        # Setup publishes a durable, job-bound deadline. Read it locally on
        # every decision so job switches/expiry never use an old probe's TTL.
        from .. import repair_lrz
        from .allocation_cache import loaded_allocation
        cfg = repair_lrz.load_lrz_config()
        try:
            allocation = loaded_allocation(cfg)
            return {**allocation, 'ready': True,
                    'message': f"LRZ job {allocation['job_id']} loaded; using cached allocation deadline"}
        except ValueError as exc:
            return {'ready': False, 'job_id': str(cfg.get('job_id') or ''), 'message': str(exc)}

    def _apply_gpu_deadline(self, run_id: str, gpu: dict[str, Any]) -> None:
        detail = None
        for name in ("detail", "get", "read"):
            fn = getattr(self.store, name, None)
            if callable(fn):
                detail = fn(run_id)
                if detail:
                    break
        dump = getattr(detail, "to_dict", None)
        detail = dict(dump() if callable(dump) else (detail or {}))
        config = dict(detail.get("config") or detail.get("params") or {})
        state = dict(detail.get("state") or detail.get("status") or {})
        status = dict(state.get("details") or state)
        started = time.time()
        requested = float(
            status.get("requested_deadline")
            or config.get("requested_deadline")
            or (started + float(config.get("duration_seconds") or 3600))
        )
        effective = min(requested, float(status.get('effective_deadline') or requested))
        expected = gpu.get("expected_end")
        stop_buffer = float(self.cfg.get('splatfix', {}).get('stop_before_gpu_end_seconds', 300))
        if not math.isfinite(stop_buffer) or stop_buffer < 60:
            raise ValueError('stop_before_gpu_end_seconds must be at least 60')
        gpu_end = None
        if expected:
            try:
                from .models import parse_slurm_end
                from zoneinfo import ZoneInfo

                parsed = parse_slurm_end(expected)
                if parsed is not None:
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=ZoneInfo('Europe/Berlin'))
                    gpu_end = float(parsed.timestamp())
            except (TypeError, ValueError):
                logger.warning('Could not parse GPU deadline %r', expected)
        if gpu_end is None and gpu.get('time_left'):
            from .models import parse_slurm_duration
            remaining = parse_slurm_duration(gpu['time_left'])
            if remaining is not None:
                gpu_end = float(gpu.get('observed_at_epoch') or started) + remaining.total_seconds()
        if gpu_end is not None:
            effective = min(effective, gpu_end - stop_buffer)
        self._update(
            run_id,
            requested_deadline=requested,
            effective_deadline=effective,
            stop_before_gpu_end_seconds=stop_buffer,
            gpu=gpu,
        )

    def _acquire_lease(
        self, run_id: str, gpu: dict[str, Any],
    ):
        fn = getattr(self.store, "acquire_gpu_lease", None)
        if callable(fn):
            try:
                return fn(run_id, metadata={"job_id": gpu.get("job_id")})
            except (RuntimeError, OSError):
                return None
        # Compatibility with a standalone lease class.
        try:
            from .store import GpuLease

            lease = GpuLease(self.root / ".gpu-lease")
            lease.acquire(run_id, job_id=gpu.get("job_id"))
            return lease
        except (ImportError, RuntimeError, OSError):
            return None

    def _release_lease(self, lease) -> None:
        fn = getattr(self.store, "release_gpu_lease", None)
        if callable(fn):
            try:
                fn(lease)
                return
            except TypeError:
                pass
        release = getattr(lease, "release", None)
        if callable(release):
            release()

    def _heartbeat(self, *, stopped: bool = False) -> None:
        body = {
            "pid": os.getpid(),
            "status": "stopped" if stopped else "running",
            "updated_at": time.time(),
        }
        path = self.root / "manager.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=2))
        tmp.replace(path)

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.stop)


def run_manager(cfg: Config, *, once: bool = False) -> int:
    loaded = load_dotenv()
    if loaded:
        logger.info("Loaded scene-run credentials from .env: %s", ", ".join(loaded))
    manager = SceneRunManager(cfg)
    if once:
        return 0 if manager.run_once() else 1
    manager.run_forever()
    return 0
