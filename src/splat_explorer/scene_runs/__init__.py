"""Scene-run control-plane primitives.

Typical use::

    store = SceneRunStore()
    run = store.create_run({"model": "my-model"})
    store.set_status(run.run_id, RunStatus.RUNNING)
    lease = store.acquire_gpu_lease(run.run_id)
    if lease is not None:
        with lease:
            ...  # only one local process can hold this block's lease

``acquire_gpu_lease`` can return ``None`` when another process owns the GPU,
so callers must check it before entering the context manager.
"""

from .models import (
    RepairTrigger,
    RepairType,
    RunState,
    RunStatus,
    SceneRun,
    SceneRunConfig,
    effective_deadline,
    isoformat_utc,
    parse_slurm_duration,
    parse_slurm_end,
    should_trigger_repair,
)
from .store import DEFAULT_ROOT, GpuLease, SceneRunStore

__all__ = [
    "DEFAULT_ROOT",
    "GpuLease",
    "RepairTrigger",
    "RepairType",
    "RunState",
    "RunStatus",
    "SceneRun",
    "SceneRunConfig",
    "SceneRunStore",
    "effective_deadline",
    "isoformat_utc",
    "parse_slurm_duration",
    "parse_slurm_end",
    "should_trigger_repair",
]
