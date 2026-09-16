"""Persistent local queue manager for automated scene-runs."""

from __future__ import annotations

import json
import logging
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
            if self._stop_requested(run_id):
                self._update(run_id, status="stopped", phase="finished",
                             message="Stopped while queued", finished_at=time.time())
                continue
            gpu = self._gpu_ready()
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

    def run_once(self) -> bool:
        """Run one ready queue item; useful for tests and manual recovery."""
        candidate = self._next_queued()
        if candidate is None:
            return False
        gpu = self._gpu_ready(force=True)
        if not gpu.get("ready"):
            return False
        run_id = str(candidate.get("id") or candidate.get("run_id"))
        self._apply_gpu_deadline(run_id, gpu)
        lease = self._acquire_lease(run_id, gpu)
        if lease is None:
            return False
        try:
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
            if status in QUEUE_STATUSES:
                queued.append(row)
        queued.sort(key=lambda r: (
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
        now = time.time()
        if (
            not force
            and self._last_probe is not None
            and now - self._last_probe_at < self.gpu_probe_seconds
        ):
            return dict(self._last_probe)
        body: dict[str, Any]
        try:
            from .. import repair_lrz
            from .lrz_transport import active_job_squeue_command

            cfg = repair_lrz.load_lrz_config()
            job_id = str(cfg.get("job_id") or "")
            result = repair_lrz._ssh_run(
                cfg, active_job_squeue_command(job_id), timeout=25,
            )
            slurm = repair_lrz.parse_squeue_line(result.stdout or "") or {}
            ready = str(slurm.get("state") or "").upper() == "R"
            body = {
                "ready": ready,
                "job_id": str(slurm.get("job_id") or cfg.get("job_id") or ""),
                "state": slurm.get("state"),
                "partition": slurm.get("partition"),
                "node": slurm.get("node"),
                "expected_end": slurm.get("expected_end"),
                "time_left": slurm.get("time_left"),
                "timelimit": slurm.get("timelimit"),
                "message": (
                    f"LRZ job {slurm.get('job_id')} is ready"
                    if ready
                    else f"LRZ job is {slurm.get('state') or 'unavailable'}, waiting for ST=R"
                ),
            }
        except Exception as exc:
            body = {
                "ready": False,
                "message": f"{type(exc).__name__}: {exc}",
            }
        self._last_probe = body
        self._last_probe_at = now
        return dict(body)

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
        effective = requested
        expected = gpu.get("expected_end")
        if expected:
            try:
                from .store import parse_slurm_end

                parsed = parse_slurm_end(expected)
                if parsed is not None:
                    effective = min(effective, float(parsed.timestamp()) - 120.0)
            except (ImportError, TypeError, ValueError):
                pass
        self._update(
            run_id,
            requested_deadline=requested,
            effective_deadline=effective,
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
