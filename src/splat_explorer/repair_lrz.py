"""Ship a GSFix3D CUDA refine job to an allocated LRZ GPU.

Open ``scripts/lrz/ssh-session.sh`` once (type the LRZ password). Later rsync
and srun reuse ``~/.ssh/cm-lrz``. If that socket is missing, apply() errors
with “open the LRZ SSH session first”.

Optional fallbacks: repair-page password field (SSH_ASKPASS for that run) or
``scripts/lrz/run-repair.sh <job-id>``.

The A100 worker is :func:`apply_packed_job` (``python -m splat_explorer.repair_lrz``).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image

from .rendering.base import Camera
from .scene import GaussianScene, load_ply, save_ply

logger = logging.getLogger(__name__)


class RepairStopped(RuntimeError):
    """Cooperative Stop during an LRZ CUDA srun — not a failed refine."""

_PASSWORD = threading.local()

SCENE_PLY = "scene.ply"
RENDERED_PNG = "rendered.png"
REPAIRED_PNG = "repaired.png"
CAMERA_JSON = "camera.json"
PARAMS_JSON = "params.json"
STATUS_JSON = "status.json"
METRICS_JSON = "metrics.json"
OUT_PLY = "scene_repaired.ply"
OUT_RENDER = "repaired_render.png"
RUN_TXT = "RUN.txt"
STOP_NAME = "STOP"
STOP_GRACE_S = 120.0

_DEFAULTS = {
    "user": "go73kaf2",
    "host": "login.ai.lrz.de",
    "job_id": "",
    "workspace": "/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer",
    "container": "/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer/containers/pytorch.sqsh",
    "cpus": 4,
    "mem": "64G",
    "container_name": "splat-repair",
}

# Both A100 partitions: HGX-only sat in PD (Priority) while a DGX A100 was free.
DEFAULT_PARTITION = "lrz-hgx-a100-80x4,lrz-dgx-a100-80x8"
REVIEW_PARTITIONS = (
    "lrz-v100x2,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4"
)
HOLD_HOURS = (2, 8, 24)
MAX_HOLD_HOURS = 336  # Matches the 14-day partition time limit.
PARTITION_CATALOG = (
    {"id": "lrz-hgx-a100-80x4", "label": "HGX A100 80GB ×4", "family": "A100", "default": True},
    {"id": "lrz-dgx-a100-80x8", "label": "DGX A100 80GB ×8", "family": "A100", "default": True},
    {"id": "lrz-hgx-h100-94x4", "label": "HGX H100 94GB ×4", "family": "H100", "default": False},
    {"id": "lrz-v100x2", "label": "V100 ×2", "family": "V100", "default": False},
)
FAMILY_CUDA_ARCH = {"A100": "8.0", "H100": "9.0", "V100": "7.0"}
SQUEUE_FORMAT = "%i|%t|%P|%N|%M|%l|%r|%j|%S|%m"


def catalog_entry_for_partition(partition: str | None) -> dict[str, Any] | None:
    """Match a Slurm partition string, including truncated squeue names."""
    text = str(partition or "").strip()
    if not text:
        return None
    first = text.split(",")[0].strip()
    for spec in PARTITION_CATALOG:
        if first == spec["id"]:
            return dict(spec)
    hits = [
        spec for spec in PARTITION_CATALOG
        if spec["id"].startswith(first) or first.startswith(spec["id"])
    ]
    if len(hits) == 1:
        return dict(hits[0])
    return dict(hits[0]) if hits else None


def gpu_family_from_name(name: str | None) -> str | None:
    text = str(name or "").upper()
    for family in ("H100", "A100", "V100"):
        if family in text:
            return family
    return None


def expected_cuda_arch(family: str | None) -> str:
    return FAMILY_CUDA_ARCH.get(str(family or ""), "")


def _arch_major(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(".", 1)[0]


def remember_live_allocation(
    *,
    job_id: str = "",
    state: str = "",
    mem: str = "",
    partition: str = "",
    node: str = "",
) -> None:
    with _LIVE_ALLOC["lock"]:
        _LIVE_ALLOC["job_id"] = str(job_id or "")
        _LIVE_ALLOC["state"] = str(state or "")
        _LIVE_ALLOC["mem"] = str(mem or "")
        _LIVE_ALLOC["partition"] = str(partition or "")
        _LIVE_ALLOC["node"] = str(node or "")
        _LIVE_ALLOC["at"] = time.time()


def live_allocation(job_id: str | None = None) -> dict[str, str]:
    with _LIVE_ALLOC["lock"]:
        body = {
            "job_id": str(_LIVE_ALLOC.get("job_id") or ""),
            "state": str(_LIVE_ALLOC.get("state") or ""),
            "mem": str(_LIVE_ALLOC.get("mem") or ""),
            "partition": str(_LIVE_ALLOC.get("partition") or ""),
            "node": str(_LIVE_ALLOC.get("node") or ""),
        }
    wanted = str(job_id or "").strip()
    if wanted and body["job_id"] and body["job_id"] != wanted:
        return {"job_id": wanted, "state": "", "mem": "", "partition": "", "node": ""}
    return body


def reset_live_allocation() -> None:
    remember_live_allocation()


def setup_matches_allocation(
    marker: dict | None,
    *,
    job_id: str = "",
    slurm: dict | None = None,
    connected: dict | None = None,
) -> tuple[bool, str]:
    """True when the DSS setup marker belongs to this job and GPU family."""
    marker = marker if isinstance(marker, dict) else {}
    if not marker.get("ok"):
        return False, "GPU setup has not been loaded on this allocation."
    marker_job = str(marker.get("job_id") or "").strip()
    job = str(job_id or (slurm or {}).get("job_id") or "").strip()
    if job and marker_job and marker_job != job:
        return False, (
            f"DSS setup marker is for job {marker_job}, not the connected job {job}. "
            "Load GPU setup on /repair/gpu for this allocation."
        )
    if job and not marker_job:
        return False, (
            "GPU setup marker is missing a job id. Reload setup so this allocation "
            "gets a job-scoped Pyxis container."
        )
    marker_family = gpu_family_from_name(marker.get("gpu") or marker.get("family"))
    live_family = gpu_family_from_name((connected or {}).get("name"))
    if not live_family:
        spec = catalog_entry_for_partition((slurm or {}).get("partition"))
        live_family = (spec or {}).get("family")
    if marker_family and live_family and marker_family != live_family:
        return False, (
            f"Setup was built for {marker_family} ({marker.get('gpu') or 'CUDA'}), "
            f"but this allocation is {live_family}. Reload GPU setup so gsplat "
            "matches the connected card."
        )
    marker_arch = str(marker.get("cuda_arch") or marker.get("compute_cap") or "").strip()
    live_arch = str((connected or {}).get("compute_cap") or "").strip()
    if not live_arch:
        live_arch = expected_cuda_arch(live_family)
    if (
        marker_arch and live_arch
        and _arch_major(marker_arch) != _arch_major(live_arch)
    ):
        return False, (
            f"gsplat was compiled for sm {marker_arch}, this GPU is sm {live_arch}. "
            "Reload GPU setup on /repair/gpu."
        )
    return True, ""


def gpu_target_snapshot(
    cfg: dict | None = None,
    *,
    slurm: dict | None = None,
    gpu: dict | None = None,
    setup: dict | None = None,
) -> dict[str, Any]:
    """Shared /repair ↔ /gpu identity of the connected LRZ allocation."""
    cfg = cfg or {}
    job = str((slurm or {}).get("job_id") or cfg.get("job_id") or "").strip()
    live = live_allocation(job)
    slurm = dict(slurm or {})
    if not slurm.get("mem"):
        slurm["mem"] = live.get("mem") or ""
    if not slurm.get("partition"):
        slurm["partition"] = live.get("partition") or ""
    if not slurm.get("node"):
        slurm["node"] = live.get("node") or ""
    rows = (gpu or {}).get("gpus") if isinstance(gpu, dict) else gpu
    connected = pick_connected_gpu(rows if isinstance(rows, list) else None)
    spec = catalog_entry_for_partition(slurm.get("partition"))
    family = gpu_family_from_name((connected or {}).get("name")) or (spec or {}).get("family")
    marker = None
    if isinstance(setup, dict):
        marker = setup.get("detail") if isinstance(setup.get("detail"), dict) else setup
    matches, reason = setup_matches_allocation(
        marker, job_id=job, slurm=slurm, connected=connected,
    )
    gpu_name = (connected or {}).get("name") or (marker or {}).get("gpu")
    arch = (
        (connected or {}).get("compute_cap")
        or (marker or {}).get("cuda_arch")
        or expected_cuda_arch(family)
    )
    return {
        "job_id": job,
        "partition": slurm.get("partition") or (spec or {}).get("id"),
        "family": family,
        "label": (spec or {}).get("label") or family,
        "node": slurm.get("node") or ((gpu or {}).get("node") if isinstance(gpu, dict) else None),
        "gpu_name": gpu_name,
        "cuda_arch": arch,
        "compute_cap": (connected or {}).get("compute_cap"),
        "job_mem": slurm.get("mem") or cfg.get("job_mem") or cfg.get("mem"),
        "needs_reload": not matches,
        "reload_reason": reason if not matches else "",
    }


def set_ssh_password(password: str | None) -> None:
    """Hold a one-run password on this thread. Never write it to disk."""
    if password:
        _PASSWORD.value = str(password)
    elif hasattr(_PASSWORD, "value"):
        delattr(_PASSWORD, "value")


def get_ssh_password() -> str | None:
    value = getattr(_PASSWORD, "value", None)
    return str(value) if value else None


def _config_paths() -> list[Path]:
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    paths: list[Path] = []
    for root in roots:
        local = root / "configs" / "lrz.local.yaml"
        example = root / "configs" / "lrz.example.yaml"
        if local not in paths:
            paths.append(local)
        if example not in paths:
            paths.append(example)
    return paths


def load_lrz_config() -> dict[str, Any]:
    """Env vars override ``configs/lrz.local.yaml``, which overrides the example."""
    cfg = dict(_DEFAULTS)
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        for path in _config_paths():
            if not path.is_file():
                continue
            try:
                body = yaml.safe_load(path.read_text()) or {}
            except OSError:
                continue
            if not isinstance(body, dict):
                continue
            for key, value in body.items():
                if value is not None and str(value).strip() != "":
                    cfg[key] = value
            if path.name == "lrz.local.yaml":
                break
    env_map = {
        "user": "LRZ_USER",
        "host": "LRZ_HOST",
        "job_id": "LRZ_JOB_ID",
        "workspace": "LRZ_WORKSPACE",
        "container": "LRZ_CONTAINER",
        "container_name": "LRZ_CONTAINER_NAME",
        "cpus": "LRZ_CPUS",
        "mem": "LRZ_MEM",
    }
    for key, env in env_map.items():
        raw = os.environ.get(env)
        if raw is not None and str(raw).strip() != "":
            cfg[key] = raw.strip() if key != "cpus" else int(raw)
    cfg["job_id"] = str(cfg.get("job_id") or "").strip()
    cfg["user"] = str(cfg.get("user") or "").strip()
    cfg["host"] = str(cfg.get("host") or "").strip()
    cfg["workspace"] = str(cfg.get("workspace") or "").rstrip("/")
    cfg["cpus"] = int(cfg.get("cpus") or 4)
    return cfg


def lrz_configured() -> bool:
    cfg = load_lrz_config()
    return bool(
        cfg["user"]
        and cfg["host"]
        and cfg["workspace"]
        and str(cfg["job_id"]).isdigit()
    )


def control_path() -> Path:
    raw = os.environ.get("LRZ_SSH_CONTROL_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".ssh" / "cm-lrz"


def lrz_session_alive(cfg: dict | None = None, *, force: bool = False) -> bool:
    """True when ``scripts/lrz/ssh-session.sh`` left a working ControlMaster.

    ``ssh -O check`` shares ControlMaster with probes. Cache it and skip the
    check while a mux command is already running so reloads cannot pile up.
    """
    sock = control_path()
    if not sock.exists():
        _remember_session(False)
        return False
    cached = _cached_session(force)
    if cached is not None:
        return cached
    acquired = _MUX["lock"].acquire(blocking=False)
    if not acquired:
        cached = _cached_session(False)
        if cached is not None:
            return cached
        return True
    try:
        _mux_stagger_locked()
        cfg = cfg or load_lrz_config()
        try:
            result = subprocess.run(
                [
                    _ssh_bin(),
                    "-o", f"ControlPath={sock}",
                    "-o", f"ConnectTimeout={int(SSH_CONNECT_TIMEOUT_S)}",
                    "-O", "check",
                    f"{cfg['user']}@{cfg['host']}",
                ],
                capture_output=True,
                text=True,
                timeout=SESSION_CHECK_TIMEOUT_S,
            )
            alive = result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            alive = False
        _mux_mark_locked()
        _remember_session(alive)
        return alive
    finally:
        _MUX["lock"].release()


def session_required_message() -> str:
    return (
        "Open the LRZ SSH session first: scripts/lrz/ssh-session.sh "
        f"(ControlMaster socket {control_path()})."
    )


def lrz_status() -> dict[str, Any]:
    cfg = load_lrz_config()
    configured = lrz_configured()
    can_ssh = bool(cfg["user"] and cfg["host"])
    alive = lrz_session_alive(cfg) if can_ssh else False
    sock = control_path()
    setup = lrz_setup_status(cfg)
    return {
        "configured": configured,
        "session": alive,
        "user": cfg["user"],
        "host": cfg["host"],
        "job_id": cfg["job_id"],
        "workspace": cfg["workspace"],
        "container": cfg.get("container"),
        "cpus": cfg.get("cpus"),
        "mem": cfg.get("mem"),
        "container_name": cfg.get("container_name"),
        "control_path": str(sock),
        "control_socket_exists": sock.exists(),
        "session_script": "scripts/lrz/ssh-session.sh",
        "run_script": "scripts/lrz/run-repair.sh",
        "allocate_script": "scripts/lrz/allocate.sh",
        "gpu_shell_script": "scripts/lrz/gpu-shell.sh",
        "setup_script": "scripts/lrz/load-setup.sh",
        "status_script": "scripts/lrz/status.sh",
        "gpu_url": "/repair/gpu",
        "setup": setup,
        "gpu_target": gpu_target_snapshot(cfg, setup=setup),
        "partition": os.environ.get("LRZ_PARTITION") or DEFAULT_PARTITION,
        "hold_hours": list(HOLD_HOURS),
        "max_hold_hours": MAX_HOLD_HOURS,
        "catalog": [dict(p) for p in PARTITION_CATALOG],
    }


def lrz_scripts(hours: int = 8, *, after: bool = False, begin: str | None = None) -> dict[str, str]:
    hours = normalize_hold_hours(hours)
    alloc = f"scripts/lrz/allocate.sh {hours}h"
    if begin:
        alloc += f" --begin {begin}"
    if after:
        alloc += " --after"
    return {
        "session": "scripts/lrz/ssh-session.sh",
        "allocate": alloc,
        "allocate_8h": "scripts/lrz/allocate.sh 8h",
        "allocate_24h": "scripts/lrz/allocate.sh 24h",
        "allocate_after": f"scripts/lrz/allocate.sh {hours}h --after",
        "widen": "scripts/lrz/allocate.sh --widen",
        "cancel": "scancel <job-id>",
        "status": "scripts/lrz/status.sh",
        "status_sinfo": "scripts/lrz/status.sh --sinfo",
        "gpu_shell": "scripts/lrz/gpu-shell.sh",
        "bootstrap": "scripts/lrz/bootstrap.sh",
        "setup": "scripts/lrz/load-setup.sh",
        "run": "scripts/lrz/run-repair.sh",
    }


def normalize_hold_hours(hours: int | str) -> int:
    raw = str(hours).strip().lower().rstrip("h")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Hold duration must be 1–{MAX_HOLD_HOURS} hours.") from exc
    if value < 1 or value > MAX_HOLD_HOURS:
        raise ValueError(f"Hold duration must be 1–{MAX_HOLD_HOURS} hours.")
    return value


def hold_sleep_seconds(hours: int | str) -> int:
    return normalize_hold_hours(hours) * 3600


def slurm_time_limit(hours: int | str) -> str:
    return f"{normalize_hold_hours(hours):02d}:00:00"


def slurm_begin_spec(begin: str | None) -> str | None:
    """Return a Slurm --begin= value, or None for an immediate start."""
    raw = str(begin or "").strip()
    if not raw or raw.lower() in ("now", "immediate"):
        return None
    lowered = raw.lower()
    if lowered in ("tomorrow", "midnight", "noon"):
        return lowered
    stamp = raw.replace(" ", "T")
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", stamp):
        stamp += ":00"
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", stamp):
        raise ValueError("Start time must be now, tomorrow, or YYYY-MM-DDTHH:MM.")
    return stamp


def normalize_partition(partition: str | None) -> str:
    raw = (partition or os.environ.get("LRZ_PARTITION") or DEFAULT_PARTITION).strip()
    known = {p["id"] for p in PARTITION_CATALOG}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Select at least one GPU partition.")
    bad = [p for p in parts if p not in known]
    if bad:
        raise ValueError(f"Unknown partition(s): {', '.join(bad)}")
    return ",".join(dict.fromkeys(parts))


def sbatch_hold_command(
    hours: int | str = 8,
    *,
    partition: str | None = None,
    job_name: str | None = None,
    after_job: str | None = None,
    begin: str | None = None,
    nodelist: str | None = None,
    cpus: int = 4,
    mem: str = "64G",
    gres: str = "gpu:1",
) -> str:
    """One sleep hold job. Does not wait in the queue."""
    hours = normalize_hold_hours(hours)
    partition = normalize_partition(partition)
    name = job_name or f"gs-{hours}h"
    begin_spec = slurm_begin_spec(begin)
    parts = [
        "sbatch",
        f"--job-name={name}",
        f"--partition={partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres={gres}",
        f"--cpus-per-task={int(cpus)}",
        f"--mem={mem}",
        f"--time={slurm_time_limit(hours)}",
        f"--output={name}-%j.log",
    ]
    nodes = ",".join(n.strip() for n in str(nodelist or "").split(",") if n.strip())
    if nodes:
        parts.append(f"--nodelist={nodes}")
    if begin_spec:
        parts.append(f"--begin={begin_spec}")
    if after_job:
        job = str(after_job).strip()
        if not job.isdigit():
            raise ValueError("after_job must be a numeric Slurm id.")
        parts.append(f"--dependency=afterany:{job}")
    parts.append(f"--wrap='sleep {hold_sleep_seconds(hours)}'")
    return " ".join(parts)


def parse_sbatch_output(text: str) -> str:
    match = re.search(r"Submitted batch job\s+(\d+)", text or "")
    if not match:
        raise RuntimeError((text or "").strip() or "sbatch produced no job id.")
    return match.group(1)


def lrz_local_config_path() -> Path:
    cwd = Path.cwd() / "configs" / "lrz.local.yaml"
    pkg = Path(__file__).resolve().parents[2] / "configs" / "lrz.local.yaml"
    if cwd.is_file():
        return cwd
    if pkg.is_file():
        return pkg
    return cwd


def write_lrz_job_id(job_id: str, path: Path | None = None) -> Path:
    """Rewrite ``job_id`` in configs/lrz.local.yaml. Password is never written."""
    job_id = str(job_id).strip()
    if job_id and not job_id.isdigit():
        raise ValueError("job_id must be a numeric Slurm id.")
    path = Path(path) if path is not None else lrz_local_config_path()
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing. Copy configs/lrz.example.yaml to configs/lrz.local.yaml first."
        )
    text = path.read_text()
    replacement = f'job_id: "{job_id}"'
    new, n = re.subn(r"(?m)^job_id:\s*.*$", replacement, text, count=1)
    if n != 1:
        if not text.endswith("\n"):
            text += "\n"
        new = text + replacement + "\n"
    path.write_text(new)
    return path


def jobs_root() -> Path:
    root = Path.cwd() / "outputs" / "lrz-jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def mem_to_mb(mem: str | None) -> int:
    """Parse Slurm ``--mem`` values like ``32G`` / ``64000M`` into MiB."""
    text = str(mem or _DEFAULTS["mem"]).strip().upper().replace(" ", "")
    if not text:
        text = str(_DEFAULTS["mem"])
    try:
        if text.endswith("G"):
            return int(float(text[:-1]) * 1024)
        if text.endswith("M"):
            return int(float(text[:-1]))
        if text.endswith("K"):
            return max(1, int(float(text[:-1]) / 1024))
        return int(float(text))
    except ValueError:
        return 64 * 1024


def hold_mem_mb(cfg: dict | None = None) -> int:
    """Host RAM of the connected sleep-hold, never larger than the live squeue value.

    configs/lrz.local.yaml may say 64G while an older 48h job was submitted with
    32G. Overlapping srun must fit the *actual* allocation or Slurm rejects the
    CUDA step with "Memory required by task is not available".
    """
    cfg = cfg or {}
    yaml_mb = mem_to_mb(cfg.get("mem"))
    known: list[int] = []
    job_mem = str(cfg.get("job_mem") or "").strip()
    if job_mem:
        known.append(mem_to_mb(job_mem))
    live = live_allocation(str(cfg.get("job_id") or ""))
    if live.get("mem"):
        known.append(mem_to_mb(live["mem"]))
    if known:
        return min(min(known), yaml_mb)
    return yaml_mb


# Pyxis + the sleep-hold + a 1G occupancy probe share the job cgroup.
# Requesting hold-1G (31G of a 32G allocation) lets unpacked gsplat rasterize
# of ~400k Gaussians cgroup-OOM after cuda_ready, with no L1 / metrics updates.
SRUN_HEADROOM_MB = 8 * 1024
SRUN_WIDE_HEADROOM_MB = 2 * 1024
SRUN_MIN_WORKER_MB = 8 * 1024
# 32G holds pack @ 512px. 64G holds run unpacked native GSFix3D once gsplat
# CUDA is prebuilt in GPU setup (nvcc JIT during repair was the 56G OOM).
TIGHT_HOST_RAM_MB = 40 * 1024
TIGHT_TRAIN_MAX_EDGE = 512
TIGHT_TRAIN_RETRY_EDGES = (512, 384, 320)


def host_cgroup_mem_mb() -> int | None:
    """Slurm step cgroup limit in MiB, or None if unbounded / unreadable."""
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if not raw or raw.lower() == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= 1 << 60:
            continue
        return max(1, value // (1024 * 1024))
    return None


def tight_host_ram(
    cfg: dict | None = None,
    *,
    cgroup_mb: int | None = None,
    use_cgroup: bool = False,
) -> bool:
    """True on the current 32G DGX A100 hold (H100 64G holds stay False).

    Local dashboard code uses the live Slurm hold. The CUDA worker passes
    ``use_cgroup=True`` so a 32G step cgroup is detected even without squeue.
    """
    mb = cgroup_mb
    if mb is None and use_cgroup:
        mb = host_cgroup_mem_mb()
    if mb is not None and 0 < mb <= TIGHT_HOST_RAM_MB:
        return True
    return hold_mem_mb(cfg) <= TIGHT_HOST_RAM_MB


def srun_mem_flag(
    cfg: dict | None = None,
    *,
    probe: bool = False,
    mem: str | None = None,
) -> str:
    """Probes stay at 1G so they cannot cgroup-OOM a 32G hold running repair."""
    if probe:
        return "--mem=1G"
    mb = mem_to_mb(mem) if mem else hold_mem_mb(cfg)
    headroom = SRUN_WIDE_HEADROOM_MB if mb >= 48 * 1024 else SRUN_HEADROOM_MB
    worker = max(SRUN_MIN_WORKER_MB, mb - headroom)
    if worker % 1024 == 0:
        return f"--mem={worker // 1024}G"
    return f"--mem={worker}M"


def read_status_file(job_dir: Path) -> dict[str, Any]:
    path = Path(job_dir) / STATUS_JSON
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def progress_from_status(body: dict[str, Any] | None) -> dict[str, Any]:
    body = body or {}
    payload = {
        "phase": body.get("phase") or "srun",
        "n_iters": int(body.get("n_iters") or body.get("iter") or 0),
        "n_updated": int(body.get("n_updated") or 0),
        "n_gaussians": body.get("n_gaussians"),
        "message": body.get("message"),
    }
    for key in (
        "iter", "l1", "l1_before", "l1_after", "gpu_name", "n_visible", "n_stamped",
        "packed", "train_width", "train_height", "checkpoint_iters", "has_ply",
        "n_chunks",
    ):
        if key in body:
            payload[key] = body[key]
    return payload


def overlay_running_job_message(job: dict[str, Any], packed: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the /repair ticker moving while rsync/srun block the worker thread."""
    if job.get("status") not in ("running", "stopping"):
        return job
    job = dict(job)
    started = float(job.get("started_at") or 0) or time.time()
    elapsed = max(0.0, time.time() - started)
    if packed and (packed.get("message") or packed.get("phase")):
        job["packed"] = packed
        msg = packed.get("message") or packed.get("phase")
        job["message"] = f"{msg} · {elapsed:.0f}s"
    elif job.get("message"):
        job["message"] = re.sub(r" · \d+s$", "", str(job["message"])) + f" · {elapsed:.0f}s"
    return job


def format_remote_command_error(
    argv: list[str],
    result: subprocess.CompletedProcess,
    *,
    log_tail: str = "",
) -> str:
    """User-facing SSH/srun failure. Do not dump Python INFO next to Slurm OOM."""
    err = (result.stderr or "").strip()
    out = (result.stdout or "").strip()
    blob = f"{err}\n{out}\n{log_tail}".lower()
    lines = [
        ln.strip() for ln in f"{err}\n{out}\n{log_tail}".splitlines()
        if ln.strip() and (
            "slurmstepd" in ln.lower()
            or ln.lower().startswith("srun:")
            or "oom" in ln.lower()
            or "out of memory" in ln.lower()
        )
    ]
    hint = "; ".join(lines[-4:]) if lines else (err or out or log_tail)[-500:]
    if "memory required by task is not available" in blob:
        live = live_allocation()
        job_mem = live.get("mem") or "the live hold"
        return (
            "Slurm refused the CUDA srun: this allocation does not have enough "
            f"free host RAM (hold is {job_mem}). Overlapping steps share one "
            "cgroup — leftover Load GPU setup or occupancy probes can consume it. "
            "Repair this view now sizes --mem from the live squeue hold, not the "
            "64G yaml default. Retry after setup is idle, or Reload GPU setup on "
            "/repair/gpu if a prior occupant was wiped. "
            + hint
        )
    if "oom_kill" in blob or "out of memory" in blob:
        live = live_allocation()
        job_mem = live.get("mem") or "the live hold"
        return (
            "LRZ CUDA step ran out of host RAM (Slurm cgroup OOM) during the first "
            "gsplat rasterize — that is why L1 / metrics.json never moved past "
            "cuda_ready. This sleep-hold shares one host-RAM cgroup across every "
            f"overlapping srun (repair, Load GPU setup, occupancy nvidia-smi). "
            f"This allocation is {job_mem}; unpacked rasterize of a large splat "
            "does not fit. Repair this view now retries packed gsplat automatically. "
            "Do not click Reload GPU setup during a refine. "
            + hint
        )
    if (
        result.returncode in (137, 143)
        or "force terminated" in blob
        or "killed" in blob
    ):
        return (
            "GPU step was killed. Load GPU setup must not run while a repair is on "
            "the card; retry after setup is idle. "
            + hint
        )
    cmd = " ".join(str(p) for p in argv[:8])
    return f"command failed ({result.returncode}): {cmd}… {hint}"


def _write_json(path: Path, body: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(path)


def job_stop_requested(job_dir: Path) -> bool:
    return (Path(job_dir) / STOP_NAME).is_file()


def write_job_checkpoint(job_dir: Path, scene: GaussianScene, stats: dict[str, Any]) -> None:
    """Atomic ply/render/metrics so the dashboard can download without a re-upload."""
    job_dir = Path(job_dir)
    save_ply(scene, job_dir / OUT_PLY)
    render_rgb = stats.get("render_rgb")
    if render_rgb is not None:
        Image.fromarray(np.asarray(render_rgb, dtype=np.uint8)).save(job_dir / OUT_RENDER)
    metrics = {k: _jsonable(v) for k, v in stats.items() if k != "render_rgb" and v is not None}
    metrics["job_id"] = job_dir.name
    _write_json(job_dir / METRICS_JSON, metrics)
    iters = int(stats.get("n_iters") or stats.get("checkpoint_iters") or 0)
    extra = {k: v for k, v in metrics.items() if k not in ("job_id", "phase")}
    write_status(
        job_dir,
        phase=str(stats.get("phase") or "refine"),
        has_ply=True,
        checkpoint_iters=iters,
        **extra,
    )


def enable_packed_job_params(job_dir: Path) -> bool:
    """Flip params.json to packed gsplat. False if already packed or missing."""
    path = Path(job_dir) / PARAMS_JSON
    if not path.is_file():
        return False
    try:
        params = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(params, dict) or params.get("packed"):
        return False
    params["packed"] = True
    _write_json(path, params)
    return True


def enable_tighter_job_params(job_dir: Path) -> bool:
    """Pack gsplat and shrink the train image so a 32G cgroup can finish an iter."""
    path = Path(job_dir) / PARAMS_JSON
    if not path.is_file():
        return False
    try:
        params = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(params, dict):
        return False
    changed = False
    if not params.get("packed"):
        params["packed"] = True
        changed = True
    edge = int(params.get("train_max_edge") or 0)
    next_edge = edge
    for candidate in TIGHT_TRAIN_RETRY_EDGES:
        if edge <= 0 or edge > candidate:
            next_edge = candidate
            break
    if next_edge and next_edge != edge:
        params["train_max_edge"] = int(next_edge)
        changed = True
    if changed:
        _write_json(path, params)
    return changed


def overlapping_step_ids(text: str, job_id: str) -> list[str]:
    """Parse ``squeue -s`` rows into cancellable job.step ids (not the sleep hold)."""
    job = str(job_id or "").strip()
    found: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if "." not in token:
            continue
        base, _, step = token.partition(".")
        if base != job:
            continue
        if step.lower() in ("batch", "extern", "interactive"):
            continue
        if token not in seen:
            seen.add(token)
            found.append(token)
    return found


def cancel_overlapping_repair_steps(cfg: dict) -> list[str]:
    """SIGTERM leftover CUDA srun steps so they do not sit on the hold cgroup."""
    job = str((cfg or {}).get("job_id") or "").strip()
    if not job.isdigit():
        return []
    listed = _ssh_run(
        cfg, f"squeue -s --job={shlex.quote(job)} -h -o '%i %t' || true", timeout=20,
    )
    steps = overlapping_step_ids(
        (listed.stdout or "") + "\n" + (listed.stderr or ""), job,
    )
    if not steps:
        return []
    ids = " ".join(shlex.quote(s) for s in steps)
    _ssh_run(cfg, f"scancel --signal=TERM {ids} || true", timeout=20)
    logger.info("Cancelled leftover CUDA steps on job %s: %s", job, steps)
    return steps


def write_status(job_dir: Path, **fields) -> None:
    path = Path(job_dir) / STATUS_JSON
    body = {}
    if path.is_file():
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            body = {}
    body.update(fields)
    body["updated_at"] = time.time()
    _write_json(path, body)


def camera_to_dict(camera: Camera) -> dict:
    return {
        "position": np.asarray(camera.position, dtype=np.float32).tolist(),
        "rotation": np.asarray(camera.rotation, dtype=np.float32).tolist(),
        "width": int(camera.width),
        "height": int(camera.height),
        "fov_deg": float(camera.fov_deg),
    }


def camera_from_dict(body: dict) -> Camera:
    return Camera(
        position=np.asarray(body["position"], dtype=np.float32),
        rotation=np.asarray(body["rotation"], dtype=np.float32),
        width=int(body["width"]),
        height=int(body["height"]),
        fov_deg=float(body["fov_deg"]),
    )


def _assign_scene(dst: GaussianScene, src: GaussianScene) -> None:
    dst.means = src.means
    dst.scales = src.scales
    dst.quats = src.quats
    dst.opacities = src.opacities
    dst.colors = src.colors


def pack_refine_job(
    scene: GaussianScene,
    camera: Camera,
    rendered_rgb: np.ndarray,
    repaired_rgb: np.ndarray,
    *,
    params: dict | None = None,
    job_id: str | None = None,
) -> Path:
    job_id = job_id or uuid.uuid4().hex[:12]
    job_dir = jobs_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    save_ply(scene, job_dir / SCENE_PLY)
    Image.fromarray(np.asarray(rendered_rgb, dtype=np.uint8)).save(job_dir / RENDERED_PNG)
    Image.fromarray(np.asarray(repaired_rgb, dtype=np.uint8)).save(job_dir / REPAIRED_PNG)
    _write_json(job_dir / CAMERA_JSON, camera_to_dict(camera))
    _write_json(job_dir / PARAMS_JSON, dict(params or {}))
    script = "scripts/lrz/ssh-session.sh"
    (job_dir / RUN_TXT).write_text(
        "Open a ControlMaster (type password once), then the dashboard rsyncs:\n\n"
        f"  {script}\n"
        f"  scripts/lrz/run-repair.sh {job_id}   # one-shot fallback\n"
    )
    write_status(
        job_dir,
        phase="packed",
        job_id=job_id,
        command=script,
        message=f"Packed. If SSH is down, run `{script}` and type your LRZ password.",
    )
    return job_dir


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def apply_packed_job(job_dir: Path, backend=None) -> dict[str, Any]:
    """CUDA (or injected) refine inside a packed job directory. Mutates scene on disk."""
    job_dir = Path(job_dir)
    camera = camera_from_dict(json.loads((job_dir / CAMERA_JSON).read_text()))
    params = dict(json.loads((job_dir / PARAMS_JSON).read_text()))
    if backend is None:
        write_status(
            job_dir,
            phase="cuda_import",
            message="Loading torch/gsplat (reusing prebuilt CUDA kernels when present)…",
            packed=bool(params.get("packed")),
        )
    scene = load_ply(job_dir / SCENE_PLY)
    rendered = np.asarray(Image.open(job_dir / RENDERED_PNG).convert("RGB"), dtype=np.uint8)
    repaired = np.asarray(Image.open(job_dir / REPAIRED_PNG).convert("RGB"), dtype=np.uint8)
    tight = tight_host_ram(use_cgroup=True)
    changed = False
    if tight:
        if not params.get("packed"):
            params["packed"] = True
            changed = True
        if int(params.get("train_max_edge") or 0) <= 0:
            params["train_max_edge"] = TIGHT_TRAIN_MAX_EDGE
            changed = True
    if changed:
        _write_json(job_dir / PARAMS_JSON, params)
        edge = int(params.get("train_max_edge") or TIGHT_TRAIN_MAX_EDGE)
        write_status(
            job_dir,
            phase="cuda_import",
            message=(
                "Host RAM cgroup is tight (32G-class hold). "
                f"Packed gsplat @ {edge}px so photometric refine can start…"
            ),
            n_gaussians=int(scene.num_gaussians),
            packed=True,
            train_max_edge=edge,
        )
    else:
        write_status(
            job_dir,
            phase="cuda_import",
            message="Loading torch/gsplat (reusing /workspace/python/torch_extensions if built)…",
            n_gaussians=int(scene.num_gaussians),
            packed=bool(params.get("packed")),
        )
    if backend is None:
        from .repair_gsfix3d import instantiate_cuda_repair

        backend = instantiate_cuda_repair(params)
    existing = getattr(backend, "on_progress", None)

    def on_progress(stats: dict) -> None:
        fields = {
            k: _jsonable(v) for k, v in stats.items()
            if k != "render_rgb" and v is not None
        }
        phase = str(fields.get("phase") or "refine")
        if phase == "cuda_ready":
            packed = bool(fields.get("packed"))
            fields["message"] = (
                f"GPU ready: {fields.get('gpu_name') or 'CUDA'}"
                + (" · packed rasterize" if packed else "")
                + ". Starting photometric refine…"
            )
        elif phase == "refine":
            it = fields.get("iter") or fields.get("n_iters") or 0
            l1 = fields.get("l1")
            fields["message"] = (
                f"Refine iter {it}"
                + (f" · L1 {l1:.4f}" if isinstance(l1, (int, float)) else "")
            )
        write_status(job_dir, **fields)
        if callable(existing):
            existing(stats)

    try:
        backend.on_progress = on_progress
    except Exception:
        pass

    def should_stop() -> bool:
        return job_stop_requested(job_dir)

    deadline = params.get("deadline_unix")
    try:
        deadline_f = float(deadline) if deadline not in (None, "") else None
    except (TypeError, ValueError):
        deadline_f = None
    use_until = int(params.get("max_chunks", 1) or 0) != 1 and hasattr(backend, "apply_until")

    def on_checkpoint(stats: dict) -> None:
        write_job_checkpoint(job_dir, scene, stats)
        if callable(existing):
            existing(stats)

    if use_until:
        stats = backend.apply_until(
            scene, camera, rendered, repaired,
            should_stop=should_stop,
            deadline=deadline_f,
            on_checkpoint=on_checkpoint,
        )
    else:
        stats = backend.apply(scene, camera, rendered, repaired)
    write_job_checkpoint(job_dir, scene, stats)
    metrics = {k: _jsonable(v) for k, v in stats.items() if k != "render_rgb" and v is not None}
    write_status(job_dir, phase="done", message="CUDA refine finished.", has_ply=True, **{
        k: v for k, v in metrics.items() if k not in ("job_id", "phase", "message", "has_ply")
    })
    return stats


def ingest_job_results(scene: GaussianScene, job_dir: Path) -> dict[str, Any]:
    job_dir = Path(job_dir)
    ply = job_dir / OUT_PLY
    if not ply.is_file():
        raise FileNotFoundError(f"{ply} missing — remote job did not finish.")
    _assign_scene(scene, load_ply(ply))
    metrics: dict[str, Any] = {}
    mp = job_dir / METRICS_JSON
    if mp.is_file():
        try:
            metrics = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            metrics = {}
    render_path = job_dir / OUT_RENDER
    if render_path.is_file():
        metrics["render_rgb"] = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.uint8)
    metrics.setdefault("backend", "gsfix-gsplat")
    metrics.setdefault("n_gaussians", scene.num_gaussians)
    return metrics


def job_results_ready(job_dir: Path) -> bool:
    job_dir = Path(job_dir)
    return (job_dir / OUT_PLY).is_file() and (job_dir / METRICS_JSON).is_file()


PROBE_TTL_S = 600.0  # 10 min. LRZ treats tighter squeue loops as a DoS.
PROBE_TIMEOUT_S = 25.0
SMI_TIMEOUT_S = 35.0
SESSION_CHECK_TIMEOUT_S = 2.0
SESSION_ALIVE_TTL_S = 20.0
SESSION_DOWN_TTL_S = 4.0
SSH_STAGGER_S = 0.4
SSH_CONNECT_TIMEOUT_S = 8
SINFO_TTL_S = 600.0  # Same as probe. Reloads must not re-run sinfo.
SETUP_TIMEOUT_S = 3600.0  # Sequential nvcc of gsplat CUDA can take 20–40 min.
SETUP_POLL_S = 8.0
HISTORY_DEFAULT_DAYS = 14
SACCT_FORMAT = (
    "JobID,JobName,Partition,State%50,Submit,Start,End,Elapsed,Timelimit,ExitCode,NodeList"
)
_SACCT_KEYS = (
    "job_id", "name", "partition", "state", "submit",
    "start", "end", "elapsed", "timelimit", "exit_code", "node",
)
_SACCT_HEADER = {
    "jobid": "job_id",
    "jobname": "name",
    "partition": "partition",
    "state": "state",
    "submit": "submit",
    "start": "start",
    "end": "end",
    "elapsed": "elapsed",
    "timelimit": "timelimit",
    "exitcode": "exit_code",
    "nodelist": "node",
}
REMOTE_PYTHONPATH = "/workspace/code/src:/workspace/python"
_STALE_GPU_HINTS = (
    "forbidden",
    "invalid job",
    "invalid job id",
    "already completed",
    "already completing",
    "access denied",
    "unable to create step",
    "job has been finished",
    "job/step already completing",
    "not running",
)
_SMI_FIELDS = (
    "index", "name", "memory_used_mib", "memory_total_mib",
    "utilization_gpu", "utilization_memory", "temperature_c",
    "power_w", "power_limit_w", "compute_cap",
)
_SMI_FIELDS_UUID = (
    "index", "uuid", "name", "memory_used_mib", "memory_total_mib",
    "utilization_gpu", "utilization_memory", "temperature_c",
    "power_w", "power_limit_w", "compute_cap",
)
_GPU_PROBE_MARKERS = ("GPUENV", "GPUDEVS", "GPUCSV", "GPUAPPS", "GPUPROCS")
_CUDA_ARCH_MARKER = "python/.cuda-arch"
_PROBE = {
    "lock": threading.Lock(),
    "at": 0.0,
    "started_at": 0.0,
    "body": None,
    "error": None,
    "inflight": False,
    "history_start": None,
    "history_end": None,
}
_SINFO = {
    "lock": threading.Lock(),
    "at": 0.0,
    "body": None,
    "error": None,
    "inflight": False,
}
_ALLOCATE = {"lock": threading.Lock(), "inflight": False}
_SETUP = {
    "lock": threading.Lock(),
    "inflight": False,
    "ok": False,
    "job_id": "",
    "at": 0.0,
    "message": "",
    "error": None,
    "detail": None,
}
_SESSION = {
    "lock": threading.Lock(),
    "at": 0.0,
    "alive": False,
    "checked": False,
}
_MUX = {
    "lock": threading.Lock(),
    "last_at": 0.0,
}
_GPU_WORK = {
    "lock": threading.Lock(),
    "owner": "",
}
_LIVE_ALLOC = {
    "lock": threading.Lock(),
    "job_id": "",
    "state": "",
    "mem": "",
    "partition": "",
    "node": "",
    "at": 0.0,
}
_OURS_PROCESS_HINTS = ("splat_explorer", "splat-explorer", "repair_lrz", "gsplat")


def _remember_session(alive: bool) -> None:
    with _SESSION["lock"]:
        _SESSION["alive"] = bool(alive)
        _SESSION["checked"] = True
        _SESSION["at"] = time.time()


def _cached_session(force: bool) -> bool | None:
    if force:
        return None
    now = time.time()
    with _SESSION["lock"]:
        if not _SESSION["checked"]:
            return None
        ttl = SESSION_ALIVE_TTL_S if _SESSION["alive"] else SESSION_DOWN_TTL_S
        if (now - float(_SESSION["at"] or 0)) < ttl:
            return bool(_SESSION["alive"])
    return None


def _mux_stagger_locked() -> None:
    gap = SSH_STAGGER_S - (time.time() - float(_MUX["last_at"] or 0))
    if gap > 0:
        time.sleep(gap)


def _mux_mark_locked() -> None:
    _MUX["last_at"] = time.time()


@contextmanager
def _gpu_exclusive(owner: str, *, timeout: float = 8.0) -> Iterator[None]:
    """One setup or repair on the allocated GPU at a time."""
    if not _GPU_WORK["lock"].acquire(timeout=max(0.1, float(timeout))):
        current = str(_GPU_WORK.get("owner") or "another GPU task")
        raise RuntimeError(
            f"GPU is busy with {current}. Wait for that to finish before starting {owner}."
        )
    _GPU_WORK["owner"] = owner
    try:
        yield
    finally:
        _GPU_WORK["owner"] = ""
        _GPU_WORK["lock"].release()


def gpu_work_owner() -> str:
    return str(_GPU_WORK.get("owner") or "")


def _squeue_blank(raw: str | None) -> str | None:
    text = str(raw or "").strip()
    if not text or text.lower() in ("none", "n/a", "unknown", "(null)", "null"):
        return None
    return text


def slurm_job_is_running(slurm: dict | None) -> bool:
    return bool(slurm and str(slurm.get("state") or "").upper() == "R")


def gpu_attach_error_is_stale(message: str | None) -> bool:
    text = str(message or "").lower()
    return any(hint in text for hint in _STALE_GPU_HINTS)


def parse_squeue_line(line: str) -> dict | None:
    """Parse `squeue --me -o SQUEUE_FORMAT` (name/start/mem optional)."""
    text = (line or "").strip()
    if not text:
        return None
    row = text.splitlines()[0].strip()
    if not row or row.lower().startswith("squeue") or row.lower().startswith("jobid"):
        return None
    parts = [p.strip() for p in row.split("|")]
    while len(parts) < 7:
        parts.append("")
    reason = _squeue_blank(parts[6])
    body = {
        "job_id": parts[0],
        "state": parts[1],
        "partition": parts[2],
        "node": parts[3],
        "elapsed": parts[4],
        "timelimit": parts[5],
        "reason": reason,
        "name": _squeue_blank(parts[7]) if len(parts) > 7 else None,
        "start_time": _squeue_blank(parts[8]) if len(parts) > 8 else None,
        "mem": _squeue_blank(parts[9]) if len(parts) > 9 else None,
        "sched_nodes": _squeue_blank(parts[10]) if len(parts) > 10 else None,
        "current": False,
    }
    if not body["job_id"] or not body["job_id"].isdigit():
        return None
    return body


def parse_squeue_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        row = parse_squeue_line(line)
        if row is None or row["job_id"] in seen:
            continue
        seen.add(row["job_id"])
        rows.append(row)
    return rows


def default_history_start(days: int = HISTORY_DEFAULT_DAYS) -> str:
    """Local calendar date ``days`` ago (YYYY-MM-DD), used as sacct --starttime."""
    return (datetime.now().date() - timedelta(days=int(days))).isoformat()


def normalize_sacct_time(raw: str | None, *, default: str) -> str:
    """Accept now/today, YYYY-MM-DD, or YYYY-MM-DDTHH:MM[:SS] for sacct bounds."""
    text = str(raw or "").strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in ("now", "today"):
        return lowered
    stamp = text.replace(" ", "T")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", stamp):
        return stamp
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", stamp):
        return stamp + ":00"
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", stamp):
        return stamp
    raise ValueError("History time must be now, today, YYYY-MM-DD, or YYYY-MM-DDTHH:MM.")


def normalize_history_window(
    start: str | None = None,
    end: str | None = None,
    *,
    days: int = HISTORY_DEFAULT_DAYS,
) -> tuple[str, str]:
    """Default window is the last ``days`` through now (open-ended)."""
    start_s = normalize_sacct_time(start, default=default_history_start(days))
    end_s = normalize_sacct_time(end, default="now")
    return start_s, end_s


def requested_history_window(
    start: str | None = None,
    end: str | None = None,
) -> tuple[str, str]:
    """Honor an explicit window; otherwise reuse the last probed sacct bounds."""
    if start is None and end is None:
        with _PROBE["lock"]:
            cached_s = _PROBE.get("history_start")
            cached_e = _PROBE.get("history_end")
        if cached_s and cached_e:
            return str(cached_s), str(cached_e)
    return normalize_history_window(start, end)


def sacct_command(user: str, start: str, end: str) -> str:
    """One sacct of this user's allocations, including completed jobs and restarts."""
    return (
        "sacct"
        f" --user={shlex.quote(str(user))}"
        f" --starttime={shlex.quote(start)}"
        f" --endtime={shlex.quote(end)}"
        " --allocations --duplicates --parsable2"
        f" --format={SACCT_FORMAT}"
    )


def parse_sacct_lines(text: str) -> list[dict[str, Any]]:
    """Parse ``sacct --parsable2`` (pipe-separated, optional header). Newest first.

    Job steps (``123.batch``) are dropped; allocation restarts keep the same JobID.
    """
    rows: list[dict[str, Any]] = []
    keys = list(_SACCT_KEYS)
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw or raw.lower().startswith("sacct:"):
            continue
        parts = [p.strip() for p in raw.split("|")]
        if not parts or not parts[0]:
            continue
        head = re.sub(r"[^a-z0-9]", "", parts[0].lower())
        if head == "jobid":
            mapped: list[str] = []
            for part in parts:
                token = re.sub(r"[^a-z0-9]", "", part.lower())
                mapped.append(_SACCT_HEADER.get(token, token or part.lower()))
            keys = mapped
            continue
        job_id = parts[0]
        if "." in job_id or not re.match(r"^\d+", job_id):
            continue
        body: dict[str, Any] = {}
        for i, key in enumerate(keys):
            raw_val = parts[i] if i < len(parts) else ""
            body[key] = _squeue_blank(raw_val)
        if not body.get("job_id"):
            continue
        rows.append(body)
    rows.sort(key=lambda r: str(r.get("start") or r.get("submit") or ""), reverse=True)
    return rows


def select_slurm_job(jobs: list[dict[str, Any]], job_id: str) -> dict | None:
    wanted = str(job_id or "").strip()
    if wanted.isdigit():
        for row in jobs:
            if row.get("job_id") == wanted:
                marked = dict(row)
                marked["current"] = True
                return marked
        return None
    running = [row for row in jobs if row.get("state") == "R"]
    if len(running) == 1:
        marked = dict(running[0])
        marked["current"] = False
        return marked
    return None


def parse_sinfo_lines(text: str) -> list[dict[str, Any]]:
    """Parse `sinfo -h -o '%P|%a|%l|%D|%T|%N'`."""
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw or raw.lower().startswith("partition") or raw.lower().startswith("sinfo"):
            continue
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 6:
            continue
        nodes_raw = parts[3]
        rows.append({
            "partition": parts[0].rstrip("*"),
            "avail": parts[1],
            "timelimit": parts[2],
            "nodes": int(nodes_raw) if nodes_raw.isdigit() else nodes_raw,
            "state": parts[4],
            "nodelist": parts[5],
        })
    return rows


def _tres_gpu_count(text: str | None) -> int | None:
    match = re.search(r"gres/gpu=(\d+)", text or "", re.I)
    return int(match.group(1)) if match else None


def parse_scontrol_nodes(text: str) -> list[dict[str, Any]]:
    """Parse `scontrol show node` blocks for GPU CfgTRES vs AllocTRES."""
    nodes: list[dict[str, Any]] = []
    for chunk in re.split(r"\n\s*\n", text or ""):
        if "NodeName=" not in chunk:
            continue
        name_m = re.search(r"NodeName=(\S+)", chunk)
        if not name_m:
            continue
        state_m = re.search(r"\bState=(\S+)", chunk)
        part_m = re.search(r"\bPartitions=(\S+)", chunk)
        feat_m = re.search(r"\bAvailableFeatures=(\S+)", chunk)
        cfg_m = re.search(r"\bCfgTRES=(\S+)", chunk)
        alloc_m = re.search(r"\bAllocTRES=(\S+)", chunk)
        state = (state_m.group(1) if state_m else "").rstrip(",")
        state_u = state.upper()
        down = any(flag in state_u for flag in ("INVAL", "DOWN", "DRAIN", "FAIL", "NOT_RESPONDING"))
        gpu_cfg = _tres_gpu_count(cfg_m.group(1) if cfg_m else "")
        gpu_alloc = _tres_gpu_count(alloc_m.group(1) if alloc_m else "") or 0
        gpu_free = None if gpu_cfg is None else max(0, int(gpu_cfg) - int(gpu_alloc))
        nodes.append({
            "name": name_m.group(1),
            "state": state or None,
            "partition": (part_m.group(1) if part_m else "").split(",")[0] or None,
            "features": feat_m.group(1) if feat_m else None,
            "gpu_cfg": gpu_cfg,
            "gpu_alloc": gpu_alloc if gpu_cfg is not None else None,
            "gpu_free": gpu_free,
            "down": down,
        })
    return nodes


def summarize_gpu_availability(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for spec in PARTITION_CATALOG:
        by_id[spec["id"]] = {
            **spec,
            "gpu_cfg": 0,
            "gpu_alloc": 0,
            "gpu_free": 0,
            "nodes": 0,
            "nodes_free": 0,
            "nodes_full": 0,
            "nodes_down": 0,
            "has_free": False,
        }
    for node in nodes:
        pid = node.get("partition")
        if pid not in by_id:
            continue
        row = by_id[pid]
        row["nodes"] += 1
        if node.get("down"):
            row["nodes_down"] += 1
            continue
        cfg = int(node.get("gpu_cfg") or 0)
        alloc = int(node.get("gpu_alloc") or 0)
        free = int(node.get("gpu_free") or 0)
        row["gpu_cfg"] += cfg
        row["gpu_alloc"] += alloc
        row["gpu_free"] += free
        if free > 0:
            row["nodes_free"] += 1
        elif cfg:
            row["nodes_full"] += 1
    for row in by_id.values():
        row["has_free"] = row["gpu_free"] > 0
    return list(by_id.values())


A100_PARTITION_IDS = ("lrz-hgx-a100-80x4", "lrz-dgx-a100-80x8")


def _free_gpu_nodes(nodes: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [
        n for n in (nodes or [])
        if not n.get("down") and int(n.get("gpu_free") or 0) > 0 and n.get("name")
    ]


def partitions_already_wide(partition: str | None) -> bool:
    parts = {p.strip() for p in str(partition or "").split(",") if p.strip()}
    return set(A100_PARTITION_IDS).issubset(parts)


def placement_for_reserve(
    requested: str | None,
    nodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Pick partitions + nodes that currently show a free GPU.

    Prefer the caller's selection when those partitions have idle GPUs;
    otherwise fall back to A100, then any free partition (H100/V100).
    """
    requested_ids = [
        p.strip() for p in str(requested or DEFAULT_PARTITION).split(",") if p.strip()
    ]
    free = _free_gpu_nodes(nodes)

    def pick(allowed: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
        hit = [n for n in free if n.get("partition") in allowed]
        parts = list(dict.fromkeys(str(n.get("partition")) for n in hit if n.get("partition")))
        return parts, hit

    parts, hit = pick(set(requested_ids))
    adapted = False
    if not hit:
        parts, hit = pick(set(A100_PARTITION_IDS))
        adapted = bool(hit)
    if not hit:
        parts, hit = pick({str(n.get("partition")) for n in free if n.get("partition")})
        adapted = bool(hit)
    if not parts:
        parts = requested_ids or list(A100_PARTITION_IDS)
    names = [str(n["name"]) for n in hit]
    return {
        "partition": ",".join(parts),
        "nodelist": ",".join(names) if names else None,
        "nodes": hit,
        "adapted": adapted,
        "gpu_free": sum(int(n.get("gpu_free") or 0) for n in hit),
    }


def pending_slurm_action(
    *,
    job_id: str,
    slurm: dict,
    jobs: list[dict[str, Any]] | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> str:
    reason = slurm.get("reason") or ""
    partition = slurm.get("partition") or ""
    running = [
        j for j in (jobs or [])
        if j.get("state") == "R" and str(j.get("job_id")) != str(job_id)
    ]
    free = _free_gpu_nodes(nodes)
    idle = sum(int(n.get("gpu_free") or 0) for n in free)
    if slurm.get("state") == "PD" and running:
        extra = ""
        if idle:
            extra = (
                f" {idle} GPU(s) look idle ({', '.join(n['name'] for n in free[:3])}), "
                "but LRZ often delays a second GPU behind a job you already hold. "
                "Click Place on free GPUs to pin this pending job to those nodes, "
                "or reserve with “queue after current job”."
            )
        return (
            f"Job {job_id} is PD ({reason or 'Priority'}) because "
            f"{running[0]['job_id']} is already running.{extra}"
        )
    if slurm.get("state") == "PD" and reason == "Priority" and not partitions_already_wide(partition):
        return (
            "Job is pending on one partition while another A100 pool may be free. "
            f"Widen once: scontrol update JobId={job_id} Partition={DEFAULT_PARTITION}"
        )
    if slurm.get("state") == "PD" and idle:
        names = ", ".join(n["name"] for n in free[:4])
        return (
            f"Job is PD ({reason or 'queued'}) while {names} show free GPUs. "
            "Click Place on free GPUs to retarget onto those nodes. "
            "Widening does nothing if both A100 partitions are already listed."
        )
    if slurm.get("state") == "PD":
        return (
            f"Job {job_id} is pending ({reason or 'PD'}). Wait in the queue — "
            "do not loop squeue."
        )
    return "Reserve a new GPU from /repair/gpu (or scripts/lrz/allocate.sh 2h)."


def review_command() -> str:
    """One SSH script: sinfo summary + scontrol GPU counts. Not a poll loop."""
    parts = REVIEW_PARTITIONS
    return (
        "echo SINFO\n"
        f"sinfo -p {parts} -h -o '%P|%a|%l|%D|%T|%N' || true\n"
        "echo SCONTROL\n"
        f"nodes=$(sinfo -p {parts} -N -h -o '%N' | sort -u | paste -sd, -)\n"
        'if [ -n "$nodes" ]; then scontrol show node "$nodes"; fi\n'
    )


def sinfo_command(partitions: str | None = None) -> str:
    parts = (partitions or REVIEW_PARTITIONS).strip()
    return f"sinfo -p {parts} -h -o '%P|%a|%l|%D|%T|%N'"


def widen_command(job_id: str, partition: str | None = None) -> str:
    job = str(job_id).strip()
    if not job.isdigit():
        raise ValueError("job_id must be a numeric Slurm id.")
    part = normalize_partition(partition)
    return f"scontrol update JobId={job} Partition={part}"


def _smi_row_fields(cells: list[str]) -> tuple[str, ...]:
    if len(cells) >= 11:
        uuid = cells[1]
        if uuid.upper().startswith("GPU-") or (" " not in uuid and uuid.count("-") >= 4):
            return _SMI_FIELDS_UUID
    return _SMI_FIELDS


def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""
    gpus: list[dict] = []
    reader = csv.reader(io.StringIO(text or ""))
    for row in reader:
        if not row or not any(str(c).strip() for c in row):
            continue
        cells = [str(c).strip() for c in row]
        if cells[0].lower().startswith("index") or cells[0].lower().startswith("nvidia-smi"):
            continue
        if cells[0].lower() in {"gpuenv", "gpudevs", "gpucsv", "gpuapps", "gpuprocs"}:
            continue
        fields = _smi_row_fields(cells)
        body: dict[str, Any] = {"uuid": "", "allocated": False, "processes": []}
        for i, key in enumerate(fields):
            raw = cells[i] if i < len(cells) else ""
            if key in ("name", "compute_cap", "uuid"):
                body[key] = raw
            elif key == "index":
                body[key] = int(raw) if raw.isdigit() else 0
            else:
                try:
                    body[key] = float(raw)
                except ValueError:
                    body[key] = None
        total = body.get("memory_total_mib")
        used = body.get("memory_used_mib") or 0
        if isinstance(total, (int, float)) and total:
            body["memory_pct"] = round(100.0 * float(used) / float(total), 1)
            body["memory_free_mib"] = max(0.0, float(total) - float(used or 0))
        else:
            body["memory_free_mib"] = None
        gpus.append(body)
    return gpus


def _id_list(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() in ("none", "void", "null", "n/a", "no_device"):
        return []
    return [part.strip() for part in text.replace(" ", "").split(",") if part.strip()]


def parse_env_block(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def parse_nvidia_dev_indices(text: str) -> list[int]:
    found: list[int] = []
    for line in (text or "").splitlines():
        raw = line.strip().rsplit("nvidia", 1)[-1]
        if raw.isdigit():
            found.append(int(raw))
    return found


def parse_compute_apps_csv(text: str) -> list[dict[str, Any]]:
    apps: list[dict[str, Any]] = []
    reader = csv.reader(io.StringIO(text or ""))
    for row in reader:
        if not row or not any(str(c).strip() for c in row):
            continue
        cells = [str(c).strip() for c in row]
        if cells[0].lower() in {"gpu_uuid", "uuid", "pid"}:
            continue
        if not cells[0] or cells[0].lower().startswith("nvidia-smi"):
            continue
        pid_raw = cells[1] if len(cells) > 1 else ""
        mem_raw = cells[3] if len(cells) > 3 else ""
        try:
            mem = float(mem_raw)
        except ValueError:
            mem = None
        apps.append({
            "gpu_uuid": cells[0],
            "pid": int(pid_raw) if pid_raw.isdigit() else pid_raw,
            "name": cells[2] if len(cells) > 2 else "",
            "memory_used_mib": mem,
        })
    return apps


def parse_ps_lines(text: str) -> dict[int, dict[str, str]]:
    by_pid: dict[int, dict[str, str]] = {}
    for line in (text or "").splitlines():
        match = re.match(r"^\s*(\d+)\s+(\S+)\s+(.*)$", line.strip())
        if not match:
            continue
        by_pid[int(match.group(1))] = {
            "user": match.group(2),
            "args": match.group(3).strip(),
        }
    return by_pid


def _looks_remapped_cuda_ids(cuda_ids: list[str], gpu_count: int) -> bool:
    if gpu_count <= len(cuda_ids):
        return False
    if not cuda_ids or not all(part.isdigit() for part in cuda_ids):
        return False
    return {int(part) for part in cuda_ids} == set(range(len(cuda_ids)))


def annotate_allocated_gpus(
    gpus: list[dict[str, Any]],
    *,
    env: dict[str, str] | None = None,
    device_nodes: list[int] | None = None,
    processes: list[dict[str, Any]] | None = None,
    proc_meta: dict[int, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Mark which nvidia-smi rows belong to this Slurm step.

    nvidia-smi often lists every card on a DGX. Inside ``srun --gres=gpu:1``
    some nodes instead remap the allocated card to index 0 while
    ``SLURM_STEP_GPUS`` still holds the physical index. Treat a single visible
    nvidia-smi row as this job's GPU; keep Slurm IDs for /dev/nvidia wipe.
    """
    env = env or {}
    our_user = str(env.get("USER") or "").strip()
    cuda_ids = _id_list(env.get("CUDA_VISIBLE_DEVICES"))
    slurm_ids = _id_list(env.get("SLURM_JOB_GPUS") or env.get("SLURM_STEP_GPUS"))
    nvidia_ids = _id_list(env.get("NVIDIA_VISIBLE_DEVICES"))
    allocated_idx: set[int] = set()
    allocated_uuid: set[str] = set()
    reason = ""
    warning = None
    physical_gpus: list[int] = []
    if slurm_ids and all(part.isdigit() for part in slurm_ids):
        physical_gpus = [int(part) for part in slurm_ids]
    elif device_nodes and len(device_nodes) == 1:
        physical_gpus = [int(device_nodes[0])]
    if len(gpus) == 1:
        allocated_idx = {int(gpus[0].get("index") or 0)}
        reason = "single visible GPU"
        if physical_gpus:
            reason = f"single visible GPU (slurm {','.join(str(i) for i in physical_gpus)})"
    elif device_nodes and len(device_nodes) < max(len(gpus), 2):
        allocated_idx = set(device_nodes)
        reason = "/dev/nvidia*"
    elif slurm_ids and all(part.isdigit() for part in slurm_ids):
        allocated_idx = {int(part) for part in slurm_ids}
        reason = "SLURM_JOB_GPUS"
    elif cuda_ids and all(part.isdigit() for part in cuda_ids) and not _looks_remapped_cuda_ids(cuda_ids, len(gpus)):
        allocated_idx = {int(part) for part in cuda_ids}
        reason = "CUDA_VISIBLE_DEVICES"
    elif nvidia_ids:
        for token in nvidia_ids:
            if token.isdigit():
                allocated_idx.add(int(token))
            elif token.upper().startswith("GPU-"):
                allocated_uuid.add(token)
        reason = "NVIDIA_VISIBLE_DEVICES"
    else:
        reason = "nvidia-smi is node-wide"
        warning = (
            f"nvidia-smi listed {len(gpus)} GPUs on this node and CUDA_VISIBLE_DEVICES "
            f"({','.join(cuda_ids) or 'unset'}) looks remapped. GPU 0 is often someone "
            "else's card — do not treat that VRAM as yours."
        )

    uuid_map = {str(gpu.get("uuid") or ""): gpu for gpu in gpus}
    for gpu in gpus:
        idx = int(gpu.get("index") or 0)
        uuid = str(gpu.get("uuid") or "")
        gpu["allocated"] = idx in allocated_idx or (uuid in allocated_uuid if uuid else False)
        gpu["processes"] = []

    for app in processes or []:
        gpu = uuid_map.get(str(app.get("gpu_uuid") or ""))
        if gpu is None and len(gpus) == 1:
            gpu = gpus[0]
        if gpu is None:
            continue
        pid = app.get("pid")
        meta = proc_meta.get(int(pid), {}) if isinstance(pid, int) and proc_meta else {}
        user = str(meta.get("user") or "")
        body = {
            "pid": pid,
            "name": app.get("name") or "",
            "args": meta.get("args") or app.get("name") or "",
            "user": user,
            "memory_used_mib": app.get("memory_used_mib"),
            "foreign": bool(our_user and user and user != our_user),
        }
        gpu.setdefault("processes", []).append(body)

    return {
        "gpus": gpus,
        "env": env,
        "scope": "allocated" if any(gpu.get("allocated") for gpu in gpus) else "node",
        "scope_reason": reason,
        "warning": warning,
        "user": our_user,
        "device_nodes": list(device_nodes or []),
        "physical_gpus": physical_gpus,
    }


def parse_gpu_occupancy_text(text: str) -> dict[str, Any]:
    """Parse the occupancy srun bundle, or a bare nvidia-smi GPU CSV."""
    raw = text or ""
    if any(marker in raw for marker in _GPU_PROBE_MARKERS):
        sections = parse_probe_bundle_markers(raw, _GPU_PROBE_MARKERS)
        env = parse_env_block(sections.get("gpuenv") or "")
        gpus = parse_nvidia_smi_csv(sections.get("gpucsv") or "")
        apps = parse_compute_apps_csv(sections.get("gpuapps") or "")
        procs = parse_ps_lines(sections.get("gpuprocs") or "")
        devices = parse_nvidia_dev_indices(sections.get("gpudevs") or "")
        return annotate_allocated_gpus(
            gpus, env=env, device_nodes=devices, processes=apps, proc_meta=procs,
        )
    gpus = parse_nvidia_smi_csv(raw)
    return annotate_allocated_gpus(gpus)


def parse_probe_bundle_markers(text: str, markers: tuple[str, ...]) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    wanted = set(markers)
    for line in (text or "").splitlines():
        key = line.strip()
        if key in wanted:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = key.lower()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def gpu_occupancy_summary(gpu: dict | None) -> dict[str, Any]:
    gpus = list((gpu or {}).get("gpus") or []) if isinstance(gpu, dict) else []
    allocated = [row for row in gpus if row.get("allocated")]
    others = [row for row in gpus if not row.get("allocated")]
    body_user = str((gpu or {}).get("user") or "") if isinstance(gpu, dict) else ""
    processes: list[dict[str, Any]] = []
    for row in gpus:
        for proc in row.get("processes") or []:
            processes.append({
                **proc,
                "gpu_index": row.get("index"),
                "gpu_uuid": row.get("uuid"),
                "gpu_name": row.get("name"),
                "allocated_gpu": bool(row.get("allocated")),
            })
    foreign = [proc for proc in processes if proc.get("foreign") and proc.get("allocated_gpu")]
    ours = [
        proc for proc in processes
        if proc.get("allocated_gpu") and process_is_ours(proc, str(body_user))
    ]
    allocated_procs = [proc for proc in processes if proc.get("allocated_gpu")]
    used = float((allocated[0].get("memory_used_mib") or 0) if allocated else 0)
    body = {
        "scope": (gpu or {}).get("scope") if isinstance(gpu, dict) else None,
        "scope_reason": (gpu or {}).get("scope_reason") if isinstance(gpu, dict) else None,
        "warning": (gpu or {}).get("warning") if isinstance(gpu, dict) else None,
        "env": (gpu or {}).get("env") if isinstance(gpu, dict) else {},
        "user": body_user,
        "physical_gpus": list((gpu or {}).get("physical_gpus") or []) if isinstance(gpu, dict) else [],
        "allocated": allocated,
        "others": others,
        "processes": processes,
        "foreign_on_allocated": foreign,
        "ours_on_allocated": ours,
        "high_vram_no_apps": bool(allocated and used >= 1024 and not allocated_procs),
        "needs_overwrite": None,
    }
    body["needs_overwrite"] = occupancy_needs_overwrite(body)
    return body


def process_is_ours(proc: dict | None, our_user: str = "") -> bool:
    """True when a compute app belongs to this account's splat-explorer worker."""
    proc = proc or {}
    if proc.get("foreign"):
        return False
    user = str(proc.get("user") or "")
    if our_user and user and user != our_user:
        return False
    args = str(proc.get("args") or proc.get("name") or "").lower()
    if any(hint in args for hint in _OURS_PROCESS_HINTS):
        return True
    return bool(our_user and user == our_user)


def occupancy_needs_overwrite(occupancy: dict | None) -> str | None:
    """Foreign leftover on THIS reserved GPU, or None.

    Our own splat-explorer / sleep-hold processes are the pipeline — they are
    not overwritten. Busy VRAM on other cards is ignored.
    """
    if not occupancy:
        return None
    foreign = list(occupancy.get("foreign_on_allocated") or [])
    if not foreign:
        return None
    bits = [_process_brief(proc) for proc in foreign[:4]]
    return (
        "Allocated GPU still has leftover process(es) from another user: "
        + "; ".join(bits)
        + ". Load GPU setup will clear only this reserved card, then start splat-explorer."
    )


class SetupNeedsOverwrite(RuntimeError):
    """Setup refused until the dashboard confirms overwrite of leftover GPU state."""


def occupancy_blocks_setup(occupancy: dict | None) -> str | None:
    """Back-compat alias: occupied GPUs now need overwrite, they are not hard-blocked."""
    return occupancy_needs_overwrite(occupancy)


def _cached_occupancy_raw() -> tuple[dict | None, float]:
    with _PROBE["lock"]:
        body = _PROBE["body"] if isinstance(_PROBE["body"], dict) else {}
        gpu = body.get("gpu") if isinstance(body.get("gpu"), dict) else None
        at = float(body.get("occupancy_at") or (gpu or {}).get("occupancy_at") or 0)
    return gpu, at


def occupancy_reason_for_setup(*, overwrite: bool = False, refresh_if_missing: bool = True) -> str | None:
    """Foreign leftover on this GPU, if any. Setup is no longer blocked on confirm."""
    del overwrite
    gpu, _at = _cached_occupancy_raw()
    if gpu is None and refresh_if_missing:
        try:
            gpu = probe_gpu_occupancy()
        except RuntimeError as exc:
            logger.warning("occupancy probe before setup failed: %s", exc)
            return None
    return occupancy_needs_overwrite(
        gpu_occupancy_summary(gpu if isinstance(gpu, dict) else None)
    )


def _process_brief(proc: dict) -> str:
    name = str(proc.get("args") or proc.get("name") or "").strip() or "process"
    if len(name) > 80:
        name = "…" + name[-79:]
    mem = proc.get("memory_used_mib")
    extra = f" ({int(mem)} MiB)" if isinstance(mem, (int, float)) else ""
    return f"{proc.get('user') or '?'} pid {proc.get('pid')} {name}{extra}"


def physical_indices_for_wipe(occupancy: dict | None) -> list[int]:
    """Physical /dev/nvidiaN indices for THIS job only.

    Prefer SLURM_JOB_GPUS / SLURM_STEP_GPUS. nvidia-smi index 0 on a DGX is
    often a remapped CUDA device, not physical GPU 0.
    """
    occupancy = occupancy or {}
    env = occupancy.get("env") if isinstance(occupancy.get("env"), dict) else {}
    slurm_ids = _id_list(env.get("SLURM_JOB_GPUS") or env.get("SLURM_STEP_GPUS"))
    if slurm_ids and all(part.isdigit() for part in slurm_ids):
        return [int(part) for part in slurm_ids]
    stored = occupancy.get("physical_gpus") or []
    if stored:
        out: list[int] = []
        for idx in stored:
            try:
                out.append(int(idx))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    devices = [
        int(idx) for idx in (occupancy.get("device_nodes") or [])
        if str(idx).isdigit() or isinstance(idx, int)
    ]
    if len(devices) == 1:
        return devices
    gpus = list(occupancy.get("gpus") or [])
    allocated = [gpu for gpu in gpus if gpu.get("allocated")]
    if not allocated:
        return []
    reason = str(occupancy.get("scope_reason") or "")
    if reason.startswith("single visible GPU"):
        return []
    if "CUDA_VISIBLE_DEVICES" in reason and len(gpus) > len(allocated):
        return []
    out = []
    for gpu in allocated:
        try:
            out.append(int(gpu.get("index")))
        except (TypeError, ValueError):
            continue
    return out


_WIPE_RESOLVE_PY = r"""
import ctypes
import glob
import os
import re
import sys


def nvidia_minors():
    found = []
    try:
        names = os.listdir("/dev")
    except OSError:
        return found
    for name in names:
        match = re.fullmatch(r"nvidia(\d+)", name)
        if match:
            found.append(match.group(1))
    return sorted(found, key=int)


def uuid_to_minor():
    mapping = {}
    for path in glob.glob("/proc/driver/nvidia/gpus/*/information"):
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        uuid_match = re.search(r"GPU UUID:\s*(\S+)", text, re.I)
        minor_match = re.search(r"Device Minor:\s*(\d+)", text, re.I)
        if uuid_match and minor_match:
            mapping[uuid_match.group(1).lower()] = minor_match.group(1)
    return mapping


def cuda_uuid():
    class Uuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_byte * 16)]

    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        fn = getattr(lib, "cudaDeviceGetUuid", None)
        if fn is None:
            continue
        buf = Uuid()
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        if fn(ctypes.byref(buf), 0) != 0:
            continue
        hexes = "".join(f"{(b & 0xFF):02x}" for b in buf.bytes)
        return "GPU-" + "-".join((hexes[0:8], hexes[8:12], hexes[12:16], hexes[16:20], hexes[20:32]))
    return ""


minors = nvidia_minors()
print("NDEV", len(minors))
print("DEVS", ",".join(minors))
mapping = uuid_to_minor()
allowed = []
source = ""
if len(minors) == 1:
    allowed, source = minors, "cgroup"
nv = os.environ.get("NVIDIA_VISIBLE_DEVICES") or ""
print("NVIDIA_VISIBLE_DEVICES", nv)
if not allowed and nv and nv.lower() not in ("void", "none", "all"):
    found = []
    for tok in nv.split(","):
        tok = tok.strip()
        if tok.lower().startswith("gpu-") and tok.lower() in mapping:
            found.append(mapping[tok.lower()])
            source = "nvidia_uuid"
        elif tok.isdigit() and len(minors) == 1 and tok in minors:
            found.append(tok)
            source = "nvidia_index"
    if found:
        allowed = found
if not allowed:
    slurm = (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS") or "").replace(" ", "")
    found = [p for p in slurm.split(",") if p.isdigit()]
    if found:
        allowed, source = found, "slurm"
if not allowed:
    uuid = cuda_uuid()
    if uuid:
        print("CUDA_UUID", uuid)
        minor = mapping.get(uuid.lower(), "")
        if minor:
            allowed, source = [minor], "cuda_uuid"
hint = os.environ.get("HINT") or ""
if not allowed and hint:
    parts = [p for p in hint.split(",") if p.isdigit()]
    if parts == ["0"] and len(minors) != 1:
        print("REJECT_HINT_GPU0")
    elif parts:
        allowed, source = parts, "hint"
if allowed:
    print("ALLOWED_FROM", source)
    print("ALLOWED_IDX", ",".join(allowed))
else:
    print("SKIP_PHYSICAL")
    sys.exit(0)
"""


_WIPE_SHARED_PID_PY = r"""
import os
import sys

allowed = {int(x) for x in os.environ.get("ALLOWED", "").split(",") if x.strip().isdigit()}
uuid_idx = {}
for line in (os.environ.get("MAP") or "").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2 and parts[0].isdigit():
        uuid_idx[parts[1]] = int(parts[0])
pid_gpus = {}
for line in (os.environ.get("APPS") or "").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2 or not parts[1].isdigit():
        continue
    idx = uuid_idx.get(parts[0])
    if idx is None:
        continue
    pid_gpus.setdefault(int(parts[1]), set()).add(idx)
shared = 0
for pid, gpus in pid_gpus.items():
    if gpus & allowed and gpus - allowed:
        print("SKIP_SHARED_PID", pid, "gpus", ",".join(str(g) for g in sorted(gpus)))
        shared = 1
sys.exit(0 if not shared else 2)
"""


def wipe_gpu_srun_command(cfg: dict, *, gpu_indices: list[int], pids: list[int]) -> str:
    """Reset THIS job's reserved GPU from inside srun --gres=gpu:1.

    Pins the card via cgroup / NVIDIA UUID / SLURM_JOB_GPUS / CUDA device 0
    UUID (not remapped CUDA_VISIBLE_DEVICES=0). Foreign PIDs on THIS card are
    signalled; our splat-explorer worker is kept. Other cards are not touched.
    """
    del pids
    job = shlex.quote(str(cfg["job_id"]))
    hint = ",".join(str(int(idx)) for idx in gpu_indices)
    inner = f"""
set +e
echo WIPE_START
echo GPUDEVS
ls -1 /dev/nvidia[0-9]* 2>/dev/null || true
echo CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES-}}"
echo SLURM_JOB_GPUS="${{SLURM_JOB_GPUS-}}"
echo SLURM_STEP_GPUS="${{SLURM_STEP_GPUS-}}"
echo NVIDIA_VISIBLE_DEVICES="${{NVIDIA_VISIBLE_DEVICES-}}"
export HINT={shlex.quote(hint)}
RESOLVE=$(python3 - <<'PY'
{_WIPE_RESOLVE_PY.strip()}
PY
)
echo "$RESOLVE"
ALLOWED=$(printf '%s\\n' "$RESOLVE" | awk '/^ALLOWED_IDX /{{print $2}}' | tail -n 1)
echo ALLOWED "$ALLOWED"
SHARED=0
if [ -n "$ALLOWED" ]; then
  APPS=$(timeout 15 nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
  MAP=$(timeout 15 nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)
  export APPS MAP ALLOWED
  python3 - <<'PY'
{_WIPE_SHARED_PID_PY.strip()}
PY
  rc=$?
  if [ "$rc" = "2" ]; then
    SHARED=1
    echo SKIP_FUSER_SHARED
  fi
fi
python3 -c 'import ctypes; ctypes.CDLL("libcudart.so").cudaDeviceReset()' 2>/dev/null && echo CUDA_RESET_OK || echo CUDA_RESET_SKIP
OUR=$(id -un 2>/dev/null || whoami)
if [ -n "$ALLOWED" ] && [ "$SHARED" != "1" ]; then
  IFS=,
  for i in $ALLOWED; do
    [ -e "/dev/nvidia$i" ] || continue
    echo FUSER "/dev/nvidia$i"
    fuser -v "/dev/nvidia$i" 2>&1 || true
    for pid in $(fuser "/dev/nvidia$i" 2>/dev/null | tr -s '[:space:]' '\\n' | grep -E '^[0-9]+$' || true); do
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ -n "$owner" ] && [ "$owner" != "$OUR" ]; then
        echo KILL_FOREIGN "$pid" "$owner"
        kill -TERM "$pid" 2>/dev/null || echo KILL_EPERM "$pid"
      else
        echo KEEP_OURS "$pid" "${{owner:-$OUR}}"
      fi
    done
  done
  unset IFS
fi
SMI_IDX=$(timeout 15 nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{{gsub(/ /,""); print}}')
OURS_LEFT=$(printf '%s\\n' "$(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)" | grep -cE '^[0-9]+$' || true)
if [ -n "$SMI_IDX" ] && [ "$SHARED" != "1" ] && [ "${{OURS_LEFT:-0}}" = "0" ]; then
  echo RESET_SMI "$SMI_IDX"
  timeout 30 nvidia-smi --gpu-reset -i "$SMI_IDX" && echo RESET_OK || echo RESET_FAIL
elif [ -z "$ALLOWED" ]; then
  echo SKIP_PHYSICAL
else
  echo SKIP_RESET_OURS
fi
echo APPS
timeout 15 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo MEM
timeout 15 nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits || true
echo WIPE_DONE
"""
    return (
        f"srun --jobid={job} --overlap --nodes=1 --ntasks=1 "
        f"--cpus-per-task=1 --gres=gpu:1 --mem=1G --quiet "
        f"bash -lc {shlex.quote(inner.strip())}"
    )


def _wipe_summary(out: str) -> str:
    keys = (
        "ALLOWED_FROM", "ALLOWED_IDX", "ALLOWED ", "CUDA_UUID", "FUSER",
        "FUSER_EPERM", "KILL_FOREIGN", "KEEP_OURS", "RESET ", "RESET_SMI",
        "RESET_OK", "RESET_FAIL", "SKIP_PHYSICAL", "SKIP_RESET_OURS",
        "SKIP_FUSER_SHARED", "SKIP_SHARED_PID", "REJECT_HINT_GPU0", "WIPE_DONE",
    )
    hits = []
    for line in (out or "").splitlines():
        raw = line.strip()
        if any(raw.startswith(k.strip()) or raw.startswith(k) for k in keys):
            hits.append(raw[:160])
    return "; ".join(hits[-10:]) if hits else (out or "no wipe output")[-240:]


def wipe_allocated_gpu(cfg: dict, occupancy: dict | None = None) -> str:
    """Reset this job's reserved GPU only. Never resets other cards on the node."""
    indices = physical_indices_for_wipe(occupancy)
    _set_setup_message(
        "Resetting this job's reserved GPU"
        + (f" (hint index {','.join(str(i) for i in indices)})" if indices else " (resolving CUDA/Slurm UUID on the node)")
        + "; other GPUs on the node are not touched…"
    )
    result = _ssh_run(cfg, wipe_gpu_srun_command(cfg, gpu_indices=indices, pids=[]), timeout=120)
    out = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    logger.info("GPU wipe srun rc=%s %s", result.returncode, out[-2000:])
    summary = _wipe_summary(out)
    _set_setup_message("GPU wipe: " + summary)
    if "WIPE_DONE" not in out and result.returncode != 0:
        logger.warning("GPU wipe srun returned %s: %s", result.returncode, out[-800:])
    return out


def pick_connected_gpu(gpus: list[dict] | None) -> dict | None:
    rows = list(gpus or [])
    allocated = [row for row in rows if row.get("allocated")]
    if allocated:
        return allocated[0]
    if len(rows) == 1:
        return rows[0]
    return None


def parse_probe_bundle(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    markers = {
        "SQUEUE", "CONTAINER", "NGC", "WORKSPACE", "STATUS", "SETUP",
        "DATE", "UID", "SQUEUE_LONG", "SACCT",
    }
    for line in (text or "").splitlines():
        key = line.strip()
        if key in markers:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = key.lower()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def parse_setup_marker_text(raw: str, *, job_id: str = "") -> dict[str, Any]:
    """Parse the SETUP section of a login probe (OK + JSON, or MISSING)."""
    text = (raw or "").strip()
    if not text or text == "MISSING" or text.startswith("MISSING"):
        return {"ok": False, "job_id": job_id}
    body_text = text
    if text.startswith("OK"):
        body_text = text[2:].strip()
        if not body_text:
            return {"ok": True, "job_id": job_id}
    try:
        loaded = json.loads(body_text)
    except json.JSONDecodeError:
        lines = [ln for ln in body_text.splitlines() if ln.strip()]
        loaded = None
        if lines:
            try:
                loaded = json.loads("\n".join(lines))
            except json.JSONDecodeError:
                loaded = None
        if not isinstance(loaded, dict):
            return {"ok": True, "job_id": job_id, "raw": body_text[:400]}
    if not isinstance(loaded, dict):
        return {"ok": True, "job_id": job_id}
    loaded.setdefault("ok", True)
    if job_id and not loaded.get("job_id"):
        loaded["job_id"] = job_id
    return loaded


def parse_setup_ok_output(text: str) -> dict[str, Any] | None:
    """Last SETUP_OK line from the container setup worker."""
    found = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("SETUP_OK"):
            payload = stripped[len("SETUP_OK"):].strip()
            try:
                loaded = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                found = loaded
    return found


def packed_job_summary(job_dir: Path) -> dict[str, Any]:
    job_dir = Path(job_dir)
    status: dict[str, Any] = {}
    sp = job_dir / STATUS_JSON
    if sp.is_file():
        try:
            loaded = json.loads(sp.read_text())
            if isinstance(loaded, dict):
                status = loaded
        except (OSError, json.JSONDecodeError):
            status = {}
    metrics: dict[str, Any] = {}
    mp = job_dir / METRICS_JSON
    if mp.is_file():
        try:
            loaded = json.loads(mp.read_text())
            if isinstance(loaded, dict):
                metrics = loaded
        except (OSError, json.JSONDecodeError):
            metrics = {}
    return {
        "id": job_dir.name,
        "phase": status.get("phase"),
        "message": status.get("message"),
        "updated_at": status.get("updated_at") or job_dir.stat().st_mtime,
        "ready": job_results_ready(job_dir),
        "has_ply": (job_dir / OUT_PLY).is_file(),
        "n_iters": metrics.get("n_iters") or status.get("n_iters") or status.get("iter"),
        "l1_before": metrics.get("l1_before") or status.get("l1_before"),
        "l1_after": metrics.get("l1_after") or status.get("l1_after"),
        "l1": status.get("l1"),
        "gpu_name": status.get("gpu_name"),
        "n_gaussians": metrics.get("n_gaussians") or status.get("n_gaussians"),
        "n_spawned": metrics.get("n_spawned") or status.get("n_spawned"),
    }


def list_packed_jobs(limit: int = 12) -> list[dict[str, Any]]:
    root = Path.cwd() / "outputs" / "lrz-jobs"
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [packed_job_summary(d) for d in dirs[: max(1, int(limit))]]


def build_connection_checks(
    *,
    configured: bool,
    session: bool,
    job_id: str,
    slurm: dict | None,
    container: dict | None,
    gpu: dict | None,
    gpu_error: str | None,
    probed: bool,
    setup: dict | None = None,
    jobs: list[dict[str, Any]] | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if configured and str(job_id).isdigit():
        checks.append({
            "id": "config", "ok": True, "label": "LRZ config",
            "detail": f"job_id {job_id} in configs/lrz.local.yaml",
            "action": None,
        })
    else:
        checks.append({
            "id": "config", "ok": False, "label": "LRZ config",
            "detail": "No running sbatch id in configs/lrz.local.yaml.",
            "action": "Reserve 8h or 24h from /repair/gpu (or scripts/lrz/allocate.sh 8h).",
        })
    if session:
        checks.append({
            "id": "ssh", "ok": True, "label": "SSH session",
            "detail": f"ControlMaster {control_path()}",
            "action": None,
        })
    else:
        checks.append({
            "id": "ssh", "ok": False, "label": "SSH session",
            "detail": "ControlMaster is down — the dashboard cannot reach the cluster.",
            "action": "Connect eduVPN, then run scripts/lrz/ssh-session.sh and type your LRZ password once.",
        })

    if not probed:
        checks.append({
            "id": "slurm", "ok": None, "label": "Slurm job",
            "detail": "Waiting for a one-shot squeue (not looped).",
            "action": None if session else "Open the SSH session first.",
        })
    elif slurm and slurm.get("state") == "R":
        node = slurm.get("node") or "?"
        checks.append({
            "id": "slurm", "ok": True, "label": "Slurm job",
            "detail": f"{slurm.get('job_id')} R on {node} · {slurm.get('elapsed') or '?'} / {slurm.get('timelimit') or '?'}",
            "action": None,
        })
    elif slurm and slurm.get("state"):
        state = slurm["state"]
        reason = slurm.get("reason")
        extra = f" ({reason})" if reason else ""
        start = slurm.get("start_time")
        if state == "PD" and start:
            extra = f"{extra} · est. start {start}".strip()
        action = pending_slurm_action(
            job_id=str(job_id), slurm=slurm, jobs=jobs, nodes=nodes,
        )
        checks.append({
            "id": "slurm", "ok": False, "label": "Slurm job",
            "detail": f"Job {job_id} is {state}{extra}",
            "action": action,
        })
    else:
        queued = [j for j in (jobs or []) if j.get("job_id")]
        if queued:
            detail = (
                f"Job {job_id} is not in the queue (allocation ended). "
                f"{len(queued)} other job(s) are still listed — click Use, or wait for PD → R."
            )
            action = (
                "The previous GPU hold ended. Click Use on a queued job, or reserve a new "
                "one from /repair/gpu (or scripts/lrz/allocate.sh 8h)."
            )
        elif job_id:
            detail = f"Job {job_id} is not in the queue (expired or wrong id)."
            action = "The GPU hold job ended. Reserve 8h or 24h from /repair/gpu (or scripts/lrz/allocate.sh 8h)."
        else:
            detail = "No Slurm job selected."
            action = "Reserve 8h or 24h from /repair/gpu (or scripts/lrz/allocate.sh 8h)."
        checks.append({
            "id": "slurm", "ok": False, "label": "Slurm job",
            "detail": detail,
            "action": action,
        })

    if not probed:
        checks.append({
            "id": "container", "ok": None, "label": "Enroot image",
            "detail": "Not probed yet.",
            "action": None,
        })
    elif container and container.get("ok"):
        size = container.get("bytes")
        detail = f"{size / (1024 ** 3):.1f} GB squashfs" if isinstance(size, (int, float)) and size else "pytorch.sqsh present"
        checks.append({
            "id": "container", "ok": True, "label": "Enroot image",
            "detail": detail,
            "action": None,
        })
    else:
        checks.append({
            "id": "container", "ok": False, "label": "Enroot image",
            "detail": "pytorch.sqsh is missing on DSS — CUDA repair cannot start.",
            "action": (
                "On the GPU node (scripts/lrz/gpu-shell.sh): "
                "enroot import -o containers/pytorch.sqsh "
                "docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel "
                "(or NGC with # : docker://nvcr.io#nvidia/pytorch:24.10-py3). "
                "See scripts/lrz/bootstrap.sh."
            ),
        })

    slurm_running = bool(slurm and slurm.get("state") == "R")
    container_ok = bool(container and container.get("ok"))
    if setup and setup.get("inflight"):
        checks.append({
            "id": "setup", "ok": None, "label": "GPU setup",
            "detail": setup.get("message") or "Loading PyTorch container + gsplat…",
            "action": None,
        })
    elif setup and setup.get("ok"):
        detail = setup.get("detail") or {}
        gpu_name = detail.get("gpu") or setup.get("gpu")
        extra = f" · {gpu_name}" if gpu_name else ""
        checks.append({
            "id": "setup", "ok": True, "label": "GPU setup",
            "detail": (setup.get("message") or f"Named Pyxis container ready{extra}").strip(),
            "action": None,
        })
    elif probed and slurm_running and container_ok:
        checks.append({
            "id": "setup", "ok": False, "label": "GPU setup",
            "detail": "PyTorch container is not loaded on this allocation.",
            "action": (
                "Click Load GPU setup (once per job). That starts the named "
                "Pyxis container from DSS pytorch.sqsh and installs gsplat "
                "onto the shared drive so later repairs skip this."
            ),
        })
    else:
        checks.append({
            "id": "setup", "ok": None, "label": "GPU setup",
            "detail": "Needs a running job and pytorch.sqsh on DSS.",
            "action": None,
        })

    gpus = (gpu or {}).get("gpus") if isinstance(gpu, dict) else gpu
    rows = gpus if isinstance(gpus, list) else []
    connected = pick_connected_gpu(rows)
    occupancy = gpu_occupancy_summary(gpu if isinstance(gpu, dict) else {"gpus": rows})
    if connected is not None:
        used = connected.get("memory_used_mib")
        total = connected.get("memory_total_mib")
        mem = f"{int(used)}/{int(total)} MiB" if used is not None and total else ""
        idx = connected.get("index")
        label = f"GPU {idx}" if idx is not None else None
        nproc = len(connected.get("processes") or [])
        procs = f"{nproc} compute proc" + ("s" if nproc != 1 else "")
        foreign_n = len(occupancy.get("foreign_on_allocated") or [])
        detail = " · ".join(x for x in (label, connected.get("name"), mem, procs) if x)
        if foreign_n:
            checks.append({
                "id": "gpu", "ok": None, "label": "GPU",
                "detail": f"{detail} · {foreign_n} leftover process(es) are not yours",
                "action": (
                    occupancy.get("needs_overwrite")
                    or "Load GPU setup will clear leftover processes on this reserved GPU only."
                ),
            })
        elif occupancy.get("needs_overwrite"):
            checks.append({
                "id": "gpu", "ok": None, "label": "GPU",
                "detail": detail,
                "action": occupancy.get("needs_overwrite"),
            })
        else:
            checks.append({
                "id": "gpu", "ok": True, "label": "GPU",
                "detail": detail,
                "action": None,
            })
    elif rows:
        checks.append({
            "id": "gpu", "ok": None, "label": "GPU",
            "detail": (
                occupancy.get("warning")
                or f"nvidia-smi saw {len(rows)} GPUs on the node; GPU 0 is not necessarily yours."
            ),
            "action": "Refresh occupancy on /repair/gpu before Load GPU setup.",
        })
    elif probed and slurm and slurm.get("state") == "R":
        checks.append({
            "id": "gpu", "ok": False if gpu_error else None, "label": "GPU",
            "detail": gpu_error or "Job is R; nvidia-smi via srun --overlap did not return yet.",
            "action": gpu_error,
        })
    else:
        checks.append({
            "id": "gpu", "ok": None, "label": "GPU",
            "detail": "Needs a running Slurm allocation.",
            "action": None,
        })
    return checks


def next_action_from_checks(checks: list[dict[str, Any]]) -> str | None:
    for check in checks:
        if check.get("ok") is False and check.get("action"):
            return str(check["action"])
    for check in checks:
        if check.get("ok") is None and check.get("action"):
            return str(check["action"])
    return None


def _ssh_run(cfg: dict, remote: str, *, timeout: float = PROBE_TIMEOUT_S) -> subprocess.CompletedProcess:
    """One ControlMaster channel at a time, with a short gap between commands."""
    argv = ssh_argv(cfg, multiplex=True) + [remote]
    with _MUX["lock"]:
        _mux_stagger_locked()
        try:
            return subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"SSH command timed out after {timeout:.0f}s "
                "(login node busy, or a stale GPU job blocked srun)."
            ) from exc
        finally:
            _mux_mark_locked()


def _login_probe_script(
    cfg: dict,
    packed_id: str | None,
    *,
    history_start: str | None = None,
    history_end: str | None = None,
) -> str:
    container = shlex.quote(str(cfg.get("container") or "/nonexistent"))
    workspace = shlex.quote(str(cfg["workspace"]))
    job = str(cfg.get("job_id") or "none").strip() or "none"
    setup_marker = shlex.quote(f"{cfg['workspace']}/logs/setup-{job}.json")
    start, end = normalize_history_window(history_start, history_end)
    history = sacct_command(str(cfg.get("user") or _DEFAULTS["user"]), start, end)
    if packed_id:
        status = shlex.quote(f"{cfg['workspace']}/inputs/{packed_id}/{STATUS_JSON}")
        status_block = f"if [ -f {status} ]; then cat {status}; else echo NONE; fi"
    else:
        status_block = "echo NONE"
    return (
        "echo SQUEUE\n"
        f"squeue --me -h -o '{SQUEUE_FORMAT}' || true\n"
        "echo CONTAINER\n"
        f"if [ -f {container} ]; then echo OK $(stat -c%s {container}); else echo MISSING; fi\n"
        "echo NGC\n"
        "if [ -s \"$HOME/enroot/.credentials\" ]; then echo OK; else echo MISSING; fi\n"
        "echo WORKSPACE\n"
        f"if [ -d {workspace} ]; then echo OK; else echo MISSING; fi\n"
        "echo SETUP\n"
        f"if [ -f {setup_marker} ]; then echo OK; cat {setup_marker}; else echo MISSING; fi\n"
        "echo STATUS\n"
        f"{status_block}\n"
        "echo DATE\n"
        "date '+%F %T %Z %z' || true\n"
        "echo UID\n"
        "id -u || true\n"
        "echo SACCT\n"
        f"{history} || true\n"
    )


def nvidia_smi_command(cfg: dict) -> str:
    """Occupancy probe: env + /dev/nvidia* + every card + compute apps.

    nvidia-smi may list every card on a DGX, or only the remapped allocated
    GPU as index 0 (``SLURM_STEP_GPUS`` still has the physical index).
    """
    job = shlex.quote(str(cfg["job_id"]))
    inner = """
echo GPUENV
printf 'CUDA_VISIBLE_DEVICES=%s\\n' "${CUDA_VISIBLE_DEVICES-}"
printf 'SLURM_JOB_GPUS=%s\\n' "${SLURM_JOB_GPUS-}"
printf 'SLURM_STEP_GPUS=%s\\n' "${SLURM_STEP_GPUS-}"
printf 'NVIDIA_VISIBLE_DEVICES=%s\\n' "${NVIDIA_VISIBLE_DEVICES-}"
printf 'SLURM_JOB_ID=%s\\n' "${SLURM_JOB_ID-}"
printf 'USER=%s\\n' "$(id -un 2>/dev/null || whoami)"
printf 'HOSTNAME=%s\\n' "$(hostname -s 2>/dev/null || hostname)"
echo GPUDEVS
ls -1 /dev/nvidia[0-9]* 2>/dev/null | sed 's|.*/nvidia||' || true
echo GPUCSV
timeout 20 nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,compute_cap --format=csv,noheader,nounits
echo GPUAPPS
timeout 15 nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo GPUPROCS
pids=$(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u | paste -sd, -)
if [ -n "$pids" ]; then
  ps -ww -p "$pids" -o pid=,user=,args= 2>/dev/null || true
fi
"""
    return (
        f"srun --jobid={job} --overlap --nodes=1 --ntasks=1 "
        f"--cpus-per-task=1 --gres=gpu:1 --mem=1G --quiet "
        f"bash -lc {shlex.quote(inner.strip())}"
    )


def _login_probe_body(cfg: dict, parsed: dict[str, str]) -> dict[str, Any]:
    jobs = parse_squeue_lines(parsed.get("squeue") or "")
    slurm = select_slurm_job(jobs, str(cfg.get("job_id") or ""))
    for row in jobs:
        row["current"] = bool(
            slurm and row.get("job_id") == slurm.get("job_id") and slurm.get("current")
        )
    container_raw = (parsed.get("container") or "").strip()
    size = None
    parts = container_raw.split()
    if len(parts) >= 2 and parts[0] == "OK":
        try:
            size = int(parts[1])
        except ValueError:
            size = None
    remote_status = None
    status_raw = (parsed.get("status") or "").strip()
    if status_raw and status_raw != "NONE":
        try:
            loaded = json.loads(status_raw)
            if isinstance(loaded, dict):
                remote_status = loaded
        except json.JSONDecodeError:
            remote_status = {"raw": status_raw[:500]}
    uid_raw = (parsed.get("uid") or "").strip()
    if slurm:
        remember_live_allocation(
            job_id=str(slurm.get("job_id") or ""),
            state=str(slurm.get("state") or ""),
            mem=str(slurm.get("mem") or ""),
            partition=str(slurm.get("partition") or ""),
            node=str(slurm.get("node") or ""),
        )
    return {
        "slurm": slurm,
        "jobs": jobs,
        "container": {"ok": container_raw.startswith("OK"), "bytes": size},
        "ngc": (parsed.get("ngc") or "").strip() == "OK",
        "workspace_ok": (parsed.get("workspace") or "").strip() == "OK",
        "remote_status": remote_status if slurm_job_is_running(slurm) else None,
        "setup": parse_setup_marker_text(
            parsed.get("setup") or "", job_id=str(cfg.get("job_id") or ""),
        ),
        "history_jobs": parse_sacct_lines(parsed.get("sacct") or ""),
        "cluster_time": (parsed.get("date") or "").strip() or None,
        "uid": uid_raw or None,
        "gpu": None,
        "gpu_error": None,
    }


def _store_probe_partial(body: dict[str, Any]) -> None:
    """Publish login-node data before nvidia-smi so the dashboard stays usable."""
    with _PROBE["lock"]:
        _PROBE["body"] = body
        _PROBE["error"] = None
        _PROBE["at"] = time.time()


def probe_lrz_gpu(
    cfg: dict | None = None,
    *,
    packed_id: str | None = None,
    history_start: str | None = None,
    history_end: str | None = None,
) -> dict[str, Any]:
    """One SSH login probe (squeue --me including %S start times) plus nvidia-smi only if a job is R."""
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    start, end = normalize_history_window(history_start, history_end)
    result = _ssh_run(
        cfg,
        _login_probe_script(
            cfg, packed_id, history_start=start, history_end=end,
        ),
        timeout=PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(err or "LRZ login probe failed.")
    body = _login_probe_body(cfg, parse_probe_bundle(result.stdout or ""))
    body["history_start"] = start
    body["history_end"] = end
    _store_probe_partial(body)
    slurm = body.get("slurm")
    smi_job = slurm.get("job_id") if slurm_job_is_running(slurm) else None
    if not smi_job:
        return body
    if gpu_work_owner() == "repair":
        # A second srun --gres=gpu:1 shares the hold job's host-RAM cgroup.
        body["gpu_error"] = None
        return body
    smi_cfg = dict(cfg)
    smi_cfg["job_id"] = smi_job
    try:
        smi = _ssh_run(cfg, nvidia_smi_command(smi_cfg), timeout=SMI_TIMEOUT_S)
    except RuntimeError as exc:
        body["gpu_error"] = str(exc)[:400]
        return body
    if smi.returncode == 0:
        occupancy = parse_gpu_occupancy_text(smi.stdout or "")
        occupancy["node"] = slurm.get("node") if slurm else None
        occupancy["occupancy_at"] = time.time()
        if occupancy.get("gpus"):
            body["gpu"] = occupancy
            body["occupancy_at"] = occupancy["occupancy_at"]
        else:
            body["gpu_error"] = (smi.stdout or smi.stderr or "empty nvidia-smi").strip()[:400]
    else:
        err = (smi.stderr or smi.stdout or "nvidia-smi via srun failed").strip()[:400]
        if gpu_attach_error_is_stale(err):
            body["gpu_error"] = None
            body["gpu"] = None
        else:
            body["gpu_error"] = err
    return body


def request_gpu_probe(
    *,
    force: bool = False,
    packed_id: str | None = None,
    history_start: str | None = None,
    history_end: str | None = None,
) -> dict[str, Any]:
    """Start at most one background probe. Never loops squeue."""
    cfg = load_lrz_config()
    if not lrz_session_alive(cfg):
        return {"started": False, "reason": "ssh"}
    start, end = normalize_history_window(history_start, history_end)
    now = time.time()
    inflight_limit = PROBE_TIMEOUT_S + SMI_TIMEOUT_S + 10.0
    with _PROBE["lock"]:
        if _PROBE["inflight"]:
            started = float(_PROBE.get("started_at") or 0)
            if started and (now - started) < inflight_limit:
                return {"started": False, "reason": "inflight"}
            logger.warning("LRZ GPU probe inflight watchdog reset after %.0fs", now - started)
        age = now - float(_PROBE["at"] or 0)
        cached_start = _PROBE.get("history_start")
        cached_end = _PROBE.get("history_end")
        range_changed = (
            (cached_start not in (None, start) or cached_end not in (None, end))
            if _PROBE["body"] is not None else False
        )
        if (
            not force
            and not range_changed
            and _PROBE["body"] is not None
            and age < PROBE_TTL_S
        ):
            return {"started": False, "reason": "fresh", "age_s": round(age, 1)}
        if not force and not range_changed and _PROBE["error"] and age < 8:
            return {"started": False, "reason": "backoff"}
        _PROBE["inflight"] = True
        _PROBE["started_at"] = now
        _PROBE["history_start"] = start
        _PROBE["history_end"] = end
    threading.Thread(
        target=_run_probe_thread,
        args=(cfg, packed_id, start, end),
        daemon=True,
        name="lrz-gpu-probe",
    ).start()
    return {"started": True}


def _run_probe_thread(
    cfg: dict,
    packed_id: str | None,
    history_start: str | None = None,
    history_end: str | None = None,
) -> None:
    body = None
    err = None
    try:
        body = probe_lrz_gpu(
            cfg,
            packed_id=packed_id,
            history_start=history_start,
            history_end=history_end,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("LRZ GPU probe failed: %s", err)
    with _PROBE["lock"]:
        _PROBE["at"] = time.time()
        if body is not None:
            _PROBE["body"] = body
            _PROBE["error"] = None
        else:
            _PROBE["error"] = err
        _PROBE["inflight"] = False


def reset_gpu_probe_cache() -> None:
    """Tests only."""
    with _PROBE["lock"]:
        _PROBE["at"] = 0.0
        _PROBE["started_at"] = 0.0
        _PROBE["body"] = None
        _PROBE["error"] = None
        _PROBE["inflight"] = False
        _PROBE["history_start"] = None
        _PROBE["history_end"] = None
    with _SINFO["lock"]:
        _SINFO["at"] = 0.0
        _SINFO["body"] = None
        _SINFO["error"] = None
        _SINFO["inflight"] = False
    with _SESSION["lock"]:
        _SESSION["at"] = 0.0
        _SESSION["alive"] = False
        _SESSION["checked"] = False
    _MUX["last_at"] = 0.0
    reset_live_allocation()
    reset_setup_cache()


def reset_setup_cache() -> None:
    """Drop cached setup when the selected job changes. Does not interrupt an in-flight load."""
    with _SETUP["lock"]:
        if _SETUP["inflight"]:
            return
        _SETUP["ok"] = False
        _SETUP["job_id"] = ""
        _SETUP["at"] = 0.0
        _SETUP["message"] = ""
        _SETUP["error"] = None
        _SETUP["detail"] = None


def _resolve_after_job(cfg: dict, after: bool | str | None) -> str | None:
    if after in (None, False, "", 0):
        return None
    if after is True:
        job = str(cfg.get("job_id") or "").strip()
        if not job.isdigit():
            raise RuntimeError("No current job_id to chain after. Reserve without --after, or pass a job id.")
        return job
    job = str(after).strip()
    if not job.isdigit():
        raise ValueError("after must be a numeric Slurm job id.")
    return job


def allocate_lrz_gpu(
    hours: int | str = 8,
    *,
    after: bool | str | None = False,
    begin: str | None = None,
    partition: str | None = None,
    switch: bool | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Submit one sleep hold job. Never waits on the queue."""
    hours = normalize_hold_hours(hours)
    begin_spec = slurm_begin_spec(begin)
    partition = normalize_partition(partition)
    cfg = cfg or load_lrz_config()
    if not cfg.get("user") or not cfg.get("host"):
        raise RuntimeError("Set user/host in configs/lrz.local.yaml first.")
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    after_job = _resolve_after_job(cfg, after)
    if switch is None:
        switch = after_job is None and begin_spec is None
    command = sbatch_hold_command(
        hours,
        partition=partition,
        after_job=after_job,
        begin=begin_spec,
        cpus=int(cfg.get("cpus") or 4),
        mem=str(cfg.get("mem") or "64G"),
    )
    with _ALLOCATE["lock"]:
        if _ALLOCATE["inflight"]:
            raise RuntimeError("An allocation is already in flight.")
        _ALLOCATE["inflight"] = True
    try:
        result = _ssh_run(cfg, command, timeout=PROBE_TIMEOUT_S)
        output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        if result.returncode != 0:
            raise RuntimeError(output or "sbatch failed.")
        job_id = parse_sbatch_output(output)
        switched = False
        if switch:
            write_lrz_job_id(job_id)
            switched = True
            reset_gpu_probe_cache()
        when = f" starting {begin_spec}" if begin_spec else ""
        if after_job:
            message = (
                f"Queued job {job_id} ({hours}h{when}) after {after_job} on {partition}. "
                "When it is R, click Use on that row and reconnect: scripts/lrz/gpu-shell.sh"
            )
        elif begin_spec:
            message = (
                f"Submitted job {job_id} ({hours}h) on {partition}, begin={begin_spec}. "
                "It stays PD until that time. Do not loop squeue."
            )
        else:
            message = (
                f"Submitted job {job_id} ({hours}h sleep hold) on {partition}. "
                "One squeue --me to confirm. If PD (Priority), widen partitions. "
                "Then scripts/lrz/gpu-shell.sh"
            )
        return {
            "ok": True,
            "job_id": job_id,
            "hours": hours,
            "after_job": after_job,
            "begin": begin_spec,
            "partition": partition,
            "switched": switched,
            "command": command,
            "stdout": output,
            "message": message,
        }
    finally:
        with _ALLOCATE["lock"]:
            _ALLOCATE["inflight"] = False


def scancel_command(job_id: str) -> str:
    job = str(job_id).strip()
    if not job.isdigit():
        raise ValueError("job_id must be a numeric Slurm id.")
    return f"scancel {job}"


def cancel_lrz_job(job_id: str, *, confirm: bool = False, cfg: dict | None = None) -> dict[str, Any]:
    """One ``scancel``. Running (ST=R) jobs require confirm=True."""
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    job = str(job_id).strip()
    command = scancel_command(job)
    with _PROBE["lock"]:
        cached = _PROBE["body"] if isinstance(_PROBE["body"], dict) else None
    jobs = list((cached or {}).get("jobs") or [])
    row = next((j for j in jobs if str(j.get("job_id")) == job), None)
    if row and slurm_job_is_running(row) and not confirm:
        raise RuntimeError(
            f"Job {job} is running (ST=R). Click cancel? to confirm scancel."
        )
    result = _ssh_run(cfg, command, timeout=PROBE_TIMEOUT_S)
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"scancel {job} failed.")
    cleared = False
    current = str(cfg.get("job_id") or "").strip()
    if current == job:
        write_lrz_job_id("")
        cleared = True
    reset_gpu_probe_cache()
    state = (row or {}).get("state") or "queued"
    return {
        "ok": True,
        "job_id": job,
        "command": command,
        "stdout": output,
        "cleared": cleared,
        "was_running": slurm_job_is_running(row),
        "message": (
            f"Cancelled {'running' if slurm_job_is_running(row) else state} job {job}. "
            + ("Cleared job_id in configs/lrz.local.yaml. " if cleared else "")
            + "Probe once — do not loop squeue."
        ),
    }


def use_lrz_job(job_id: str) -> dict[str, Any]:
    path = write_lrz_job_id(job_id)
    reset_gpu_probe_cache()
    return {
        "ok": True,
        "job_id": str(job_id).strip(),
        "path": str(path),
        "message": (
            f"Now using job {job_id}. /repair will attach on the next CUDA run "
            "(Load GPU setup only if the PyTorch container is missing on this allocation)."
        ),
    }


def widen_lrz_job(
    job_id: str | None = None,
    *,
    partition: str | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """One scontrol to update eligible partitions. Not a loop."""
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    job = str(job_id or cfg.get("job_id") or "").strip()
    command = widen_command(job, partition=partition)
    result = _ssh_run(cfg, command, timeout=PROBE_TIMEOUT_S)
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode != 0:
        raise RuntimeError(output or "scontrol update failed.")
    reset_gpu_probe_cache()
    part = normalize_partition(partition)
    return {
        "ok": True,
        "job_id": job,
        "command": command,
        "stdout": output,
        "partition": part,
        "message": f"Updated job {job} to {part}. Probe once — do not loop squeue.",
    }


def _review_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in (text or "").splitlines():
        key = line.strip()
        if key in {"SINFO", "SCONTROL"}:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = key.lower()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def review_lrz_partitions(*, force: bool = False, cfg: dict | None = None) -> dict[str, Any]:
    """One sinfo + one scontrol show node. Cached; never polled in a loop."""
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    now = time.time()
    with _SINFO["lock"]:
        age = now - float(_SINFO["at"] or 0)
        cached = _SINFO["body"]
        inflight = bool(_SINFO.get("inflight"))
        if inflight:
            if isinstance(cached, dict):
                return {
                    "ok": True,
                    **cached,
                    "cached": True,
                    "inflight": True,
                    "age_s": round(age, 1) if _SINFO["at"] else None,
                    "message": "Availability review already running.",
                }
            return {
                "ok": True,
                "partitions": [],
                "nodes": [],
                "summary": [],
                "default_free": None,
                "cached": True,
                "inflight": True,
                "message": "Availability review already running.",
            }
        if not force and isinstance(cached, dict) and age < SINFO_TTL_S:
            return {
                "ok": True,
                **cached,
                "cached": True,
                "age_s": round(age, 1),
                "message": "Using cached sinfo (LRZ forbids sinfo loops).",
            }
        _SINFO["inflight"] = True
    try:
        command = review_command()
        result = _ssh_run(cfg, command, timeout=max(PROBE_TIMEOUT_S, 60.0))
        output = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0:
            message = err or output or "sinfo/scontrol failed."
            with _SINFO["lock"]:
                _SINFO["at"] = time.time()
                _SINFO["error"] = message
            raise RuntimeError(message)
        sections = _review_sections(output)
        partitions = parse_sinfo_lines(sections.get("sinfo") or "")
        nodes = parse_scontrol_nodes(sections.get("scontrol") or "")
        summary = summarize_gpu_availability(nodes)
        default_free = sum(row["gpu_free"] for row in summary if row.get("default"))
        other_free = [row for row in summary if not row.get("default") and row.get("has_free")]
        if default_free == 0 and other_free:
            hint = (
                "Default A100 partitions look full. "
                + ", ".join(f"{r['label']} has {r['gpu_free']} free" for r in other_free)
                + " — select those below."
            )
        elif default_free:
            hint = f"{default_free} A100 GPU(s) free on the default partitions. MIXED nodes can still have a slot."
        else:
            hint = "No free GPUs counted on reviewed nodes. Queue anyway or pick another start time."
        body = {
            "partitions": partitions,
            "nodes": nodes,
            "summary": summary,
            "default_free": default_free,
        }
        with _SINFO["lock"]:
            _SINFO["at"] = time.time()
            _SINFO["body"] = body
            _SINFO["error"] = None
        return {
            "ok": True,
            **body,
            "cached": False,
            "command": command,
            "message": hint,
        }
    finally:
        with _SINFO["lock"]:
            _SINFO["inflight"] = False


def lrz_dashboard_snapshot(
    repair_job: dict | None = None,
    *,
    request_probe: bool = False,
    force_probe: bool = False,
    history_start: str | None = None,
    history_end: str | None = None,
) -> dict[str, Any]:
    """Local connection + current repair, plus a cached GPU probe."""
    status = lrz_status()
    packed = list_packed_jobs()
    packed_id = packed[0]["id"] if packed else None
    hist_start, hist_end = requested_history_window(history_start, history_end)
    if request_probe or force_probe:
        request_gpu_probe(
            force=force_probe, packed_id=packed_id,
            history_start=hist_start, history_end=hist_end,
        )
    with _PROBE["lock"]:
        inflight = bool(_PROBE["inflight"])
        cached = _PROBE["body"]
        error = _PROBE["error"]
        probed_at = float(_PROBE["at"] or 0)
    if (
        status["session"]
        and cached is None
        and not inflight
        and not error
    ):
        request_gpu_probe(
            force=False, packed_id=packed_id,
            history_start=hist_start, history_end=hist_end,
        )
        with _PROBE["lock"]:
            inflight = bool(_PROBE["inflight"])
            cached = _PROBE["body"]
            error = _PROBE["error"]
            probed_at = float(_PROBE["at"] or 0)
    slurm = (cached or {}).get("slurm")
    jobs = list((cached or {}).get("jobs") or [])
    container = (cached or {}).get("container")
    gpu = (cached or {}).get("gpu")
    gpu_error = (cached or {}).get("gpu_error")
    occupancy_at = None
    if isinstance(gpu, dict):
        occupancy_at = gpu.get("occupancy_at")
    if occupancy_at is None and isinstance(cached, dict):
        occupancy_at = cached.get("occupancy_at")
    gpu_running = slurm_job_is_running(slurm)
    occupancy_fresh = (
        isinstance(gpu, dict)
        and bool(gpu.get("gpus"))
        and occupancy_at
        and not gpu_attach_error_is_stale(gpu_error)
    )
    show_gpu = gpu_running or occupancy_fresh
    if not show_gpu:
        gpu = None
        gpu_error = None
    elif gpu_attach_error_is_stale(gpu_error):
        gpu = None
        gpu_error = (
            "The previous GPU allocation ended (srun forbidden / job gone). "
            "Dashboard login data is still available — reserve or wait for a job in ST=R."
        )
    probe_error = None if cached is not None else error
    sinfo_nodes: list[dict[str, Any]] = []
    with _SINFO["lock"]:
        sinfo_body = _SINFO["body"]
        partitions_at = float(_SINFO["at"] or 0)
        partitions_error = _SINFO["error"]
        reviewing = bool(_SINFO.get("inflight"))
    if isinstance(sinfo_body, dict):
        sinfo_nodes = list(sinfo_body.get("nodes") or [])
    setup = lrz_setup_status(probe_setup=(cached or {}).get("setup"))
    checks = build_connection_checks(
        configured=bool(status["configured"]),
        session=bool(status["session"]),
        job_id=str(status.get("job_id") or ""),
        slurm=slurm,
        container=container,
        gpu=gpu,
        gpu_error=gpu_error,
        probed=cached is not None,
        setup=setup,
        jobs=jobs,
        nodes=sinfo_nodes,
    )
    required = ["config", "ssh", "slurm", "container"]
    if cached is not None:
        required.append("setup")
    ready = all(
        c.get("ok") is True
        for c in checks
        if c["id"] in required
    )
    latest = packed[0] if packed else None
    gpus = (gpu or {}).get("gpus") if isinstance(gpu, dict) else None
    occupancy = gpu_occupancy_summary(gpu if isinstance(gpu, dict) else None)
    connected = pick_connected_gpu(gpus if isinstance(gpus, list) else None)
    free_mib = connected.get("memory_free_mib") if connected else None
    occupancy_age_s = (
        round(time.time() - float(occupancy_at), 1)
        if occupancy_at else None
    )
    if isinstance(sinfo_body, dict):
        partitions = sinfo_body.get("partitions")
        nodes = sinfo_body.get("nodes") or []
        summary = sinfo_body.get("summary") or []
        default_free = sinfo_body.get("default_free")
    else:
        partitions = sinfo_body
        nodes = []
        summary = []
        default_free = None
    gpu_target = gpu_target_snapshot(
        {"job_id": status.get("job_id"), "mem": status.get("mem")},
        slurm=slurm,
        gpu=gpu if show_gpu else None,
        setup=setup,
    )
    status = dict(status)
    status["gpu_target"] = gpu_target
    if gpu_target.get("needs_reload") and not (setup or {}).get("inflight"):
        ready = False
    return {
        "connection": status,
        "checks": checks,
        "ready": ready,
        "next_action": next_action_from_checks(checks),
        "slurm": slurm,
        "jobs": jobs,
        "partitions": partitions,
        "nodes": nodes,
        "summary": summary,
        "default_free": default_free,
        "catalog": [dict(p) for p in PARTITION_CATALOG],
        "partitions_age_s": round(time.time() - partitions_at, 1) if partitions_at else None,
        "partitions_error": partitions_error,
        "container": container,
        "ngc": (cached or {}).get("ngc"),
        "workspace_ok": (cached or {}).get("workspace_ok"),
        "setup": setup,
        "gpu_target": gpu_target,
        "gpu": gpus if show_gpu else None,
        "gpu_node": (
            (gpu or {}).get("node") if isinstance(gpu, dict) and show_gpu
            else (slurm or {}).get("node") if show_gpu else None
        ),
        "gpu_error": gpu_error,
        "gpu_free_mib": free_mib if show_gpu else None,
        "gpu_running": bool(gpu_running or occupancy_fresh),
        "gpu_scope": (gpu or {}).get("scope") if isinstance(gpu, dict) and show_gpu else None,
        "gpu_warning": (gpu or {}).get("warning") if isinstance(gpu, dict) and show_gpu else None,
        "gpu_env": (gpu or {}).get("env") if isinstance(gpu, dict) and show_gpu else None,
        "gpu_occupancy": occupancy if show_gpu else None,
        "occupancy_at": occupancy_at if show_gpu else None,
        "occupancy_age_s": occupancy_age_s if show_gpu else None,
        "remote_status": (cached or {}).get("remote_status") if gpu_running else None,
        "repair": repair_job,
        "probe_error": probe_error,
        "packed_jobs": packed,
        "current_packed": latest,
        "probing": inflight,
        "reviewing": reviewing,
        "probed_at": probed_at or None,
        "probe_age_s": round(time.time() - probed_at, 1) if probed_at else None,
        "scripts": lrz_scripts(),
        "hold_hours": list(HOLD_HOURS),
        "max_hold_hours": MAX_HOLD_HOURS,
        "partition": status.get("partition") or DEFAULT_PARTITION,
        "history": {
            "jobs": list((cached or {}).get("history_jobs") or []),
            "start": (cached or {}).get("history_start") or hist_start,
            "end": (cached or {}).get("history_end") or hist_end,
            "cluster_time": (cached or {}).get("cluster_time"),
            "uid": (cached or {}).get("uid"),
            "squeue_long": (cached or {}).get("squeue_long"),
            "default_days": HISTORY_DEFAULT_DAYS,
        },
        "hint": (
            "Slurm (one squeue --me + one sacct) is queried at most once per 10 min "
            "while this page is visible. Hidden tabs send nothing. sinfo only on "
            "Refresh availability, and never at the same time as a probe."
        ),
    }


def _ssh_bin() -> str:
    return os.environ.get("LRZ_SSH_BIN") or "/usr/bin/ssh"


def ssh_argv(cfg: dict | None = None, *, multiplex: bool | None = None) -> list[str]:
    """SSH argv. Default: reuse ControlMaster at ``control_path()``."""
    cfg = cfg or load_lrz_config()
    target = f"{cfg['user']}@{cfg['host']}"
    ssh = _ssh_bin()
    if multiplex is None:
        multiplex = lrz_session_alive(cfg)
    if multiplex:
        return [
            ssh, "-4", "-F", "/dev/null",
            "-o", "ControlMaster=no",
            "-o", f"ControlPath={control_path()}",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={int(SSH_CONNECT_TIMEOUT_S)}",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            target,
        ]
    return [
        ssh, "-4", "-F", "/dev/null",
        "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "KbdInteractiveAuthentication=no",
        target,
    ]


def rsync_ssh_cmd(cfg: dict | None = None) -> str:
    cfg = cfg or load_lrz_config()
    argv = ssh_argv(cfg, multiplex=True)[:-1]  # drop user@host; rsync -e adds dest
    return " ".join(shlex.quote(p) for p in argv)


def remote_job_dir(cfg: dict, job_id: str) -> str:
    return f"{cfg['workspace']}/inputs/{job_id}"


def push_job_params(cfg: dict, job_id: str, job_dir: Path) -> None:
    """Overwrite remote params.json after a packed-rasterize retry."""
    remote = f"{cfg['user']}@{cfg['host']}"
    _mux_run([
        "rsync", "-az", "-e", rsync_ssh_cmd(cfg),
        str(Path(job_dir) / PARAMS_JSON),
        f"{remote}:{remote_job_dir(cfg, job_id)}/{PARAMS_JSON}",
    ])


def request_remote_stop(cfg: dict, job_id: str) -> None:
    """Create STOP on DSS so the GPU loop checkpoints instead of dying mid-iter."""
    path = f"{remote_job_dir(cfg, job_id)}/{STOP_NAME}"
    _ssh_run(cfg, f"touch {shlex.quote(path)}", timeout=15)


def pull_remote_job_artifacts(cfg: dict, job_id: str, job_dir: Path) -> None:
    """Download repaired ply/render/metrics only — never re-upload the scene."""
    job_dir = Path(job_dir)
    remote = f"{cfg['user']}@{cfg['host']}:{remote_job_dir(cfg, job_id)}/"
    argv = [
        "rsync", "-az", "-e", rsync_ssh_cmd(cfg),
        "--include", OUT_PLY,
        "--include", OUT_RENDER,
        "--include", METRICS_JSON,
        "--exclude", "*",
        remote,
        f"{job_dir}/",
    ]
    _mux_run(argv)


def _code_src() -> Path:
    code_src = Path(__file__).resolve().parents[2] / "src"
    if not code_src.is_dir():
        code_src = Path.cwd() / "src"
    return code_src


def _remote_pythonpath_exports(cfg: dict | None = None) -> str:
    # Parallel nvcc of gsplat kernels OOMs a 56G step cgroup. One job also
    # reuses TORCH_EXTENSIONS_DIR on DSS so later srun steps skip compile.
    return (
        f"export PYTHONPATH={REMOTE_PYTHONPATH}${{PYTHONPATH:+:$PYTHONPATH}}; "
        "export PATH=/workspace/python/bin:$PATH; "
        "if [ -z \"${LRZ_CUDA_ARCH:-}\" ] && [ -f /workspace/python/.cuda-arch ]; then "
        "LRZ_CUDA_ARCH=$(cat /workspace/python/.cuda-arch); fi; "
        "export TORCH_CUDA_ARCH_LIST=\"${LRZ_CUDA_ARCH:-8.0}\"; "
        "export OMP_NUM_THREADS=1; export TORCH_NUM_THREADS=1; "
        "export MALLOC_ARENA_MAX=1; export PYTHONMALLOC=malloc; "
        "export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 NINJAFLAGS=-j1 MAKEFLAGS=-j1; "
        "export FAST_COMPILE=1 VERBOSE=1; "
        "export NVCC_APPEND_FLAGS='--threads=1'; "
        "export TORCH_EXTENSIONS_DIR=/workspace/python/torch_extensions; "
        "export TMPDIR=/workspace/tmp TMP=/workspace/tmp TEMP=/workspace/tmp; "
        "export HOME=/workspace/python/home XDG_CACHE_HOME=/workspace/python/cache; "
        "mkdir -p /workspace/python/torch_extensions /workspace/tmp /workspace/python/home /workspace/python/cache /workspace/python/bin; "
    )


def container_name_for_job(cfg: dict) -> str:
    """Per-allocation Pyxis name so A100/H100 jobs do not reuse a stale container."""
    base = str(cfg.get("container_name") or "splat-repair").strip() or "splat-repair"
    job = str(cfg.get("job_id") or "").strip()
    return f"{base}-{job}" if job.isdigit() else base


def container_srun_prefix(cfg: dict, *, mem_flag: str | None = None) -> str:
    image = cfg.get("container") or f"{cfg['workspace']}/containers/pytorch.sqsh"
    name = container_name_for_job(cfg)
    mem = mem_flag or srun_mem_flag(cfg)
    return (
        f"srun --jobid={shlex.quote(str(cfg['job_id']))} --overlap "
        f"--nodes=1 --ntasks=1 --cpus-per-task={int(cfg['cpus'])} --gres=gpu:1 "
        f"{mem} "
        f"--container-image={shlex.quote(str(image))} "
        f"--container-name={shlex.quote(str(name))} "
        f"--container-mounts={shlex.quote(cfg['workspace'] + ':/workspace')} "
    )


def srun_worker_command(cfg: dict, job_id: str, *, mem_flag: str | None = None) -> str:
    inner = (
        _remote_pythonpath_exports(cfg)
        + f"python -u -m splat_explorer.repair_lrz --job-dir /workspace/inputs/{job_id}"
    )
    return container_srun_prefix(cfg, mem_flag=mem_flag) + f"bash -lc {shlex.quote(inner)}"


def srun_setup_command(cfg: dict) -> str:
    inner = _remote_pythonpath_exports(cfg) + "python -m splat_explorer.repair_lrz --setup"
    setup_cfg = dict(cfg)
    setup_cfg["cpus"] = 1  # nvcc thread pool tracks CPU count; 1 keeps host RAM down
    return container_srun_prefix(setup_cfg) + f"bash -lc {shlex.quote(inner)}"


def sync_code_to_dss(cfg: dict | None = None) -> None:
    """rsync local src/ onto DSS. ControlMaster must already be up."""
    cfg = cfg or load_lrz_config()
    remote = f"{cfg['user']}@{cfg['host']}"
    ssh_e = rsync_ssh_cmd(cfg)
    code_src = _code_src()
    ws = str(cfg["workspace"])
    _mux_run(ssh_argv(cfg, multiplex=True) + [
        "mkdir -p "
        f"{shlex.quote(ws + '/code')} "
        f"{shlex.quote(ws + '/outputs')} "
        f"{shlex.quote(ws + '/logs')} "
        f"{shlex.quote(ws + '/containers')} "
        f"{shlex.quote(ws + '/python')} "
        f"{shlex.quote(ws + '/inputs')}"
    ])
    _mux_run([
        "rsync", "-az", "--delete", "-e", ssh_e,
        f"{code_src}/", f"{remote}:{ws}/code/src/",
    ])
    pyproject = code_src.parent / "pyproject.toml"
    if pyproject.is_file():
        _mux_run([
            "rsync", "-az", "-e", ssh_e,
            str(pyproject), f"{remote}:{ws}/code/pyproject.toml",
        ])


def lrz_setup_status(cfg: dict | None = None, *, probe_setup: dict | None = None) -> dict[str, Any]:
    """Local (and optional probe-marker) GPU setup status for the current job."""
    cfg = cfg or load_lrz_config()
    job = str(cfg.get("job_id") or "").strip()
    if probe_setup is None:
        with _PROBE["lock"]:
            cached = _PROBE["body"]
        if isinstance(cached, dict):
            probe_setup = cached.get("setup")
    with _SETUP["lock"]:
        local_job = str(_SETUP.get("job_id") or "")
        body = {
            "inflight": bool(_SETUP["inflight"]),
            "ok": bool(_SETUP["ok"]) and (not job or local_job == job),
            "job_id": local_job or job,
            "message": str(_SETUP.get("message") or ""),
            "error": _SETUP.get("error"),
            "detail": _SETUP.get("detail"),
            "at": _SETUP.get("at") or None,
        }
    started = float(_SETUP.get("at") or 0)
    if body["inflight"] and started:
        body["elapsed_s"] = round(time.time() - started, 1)
    marker = probe_setup if isinstance(probe_setup, dict) else None
    gpu, _at = _cached_occupancy_raw()
    connected = pick_connected_gpu(
        list((gpu or {}).get("gpus") or []) if isinstance(gpu, dict) else None
    )
    slurm = live_allocation(job)
    if slurm.get("job_id") != job:
        slurm = {"job_id": job, "partition": slurm.get("partition"), "mem": slurm.get("mem"), "node": slurm.get("node")}
    matches = False
    reason = ""
    if marker and marker.get("ok") and not body["inflight"]:
        matches, reason = setup_matches_allocation(
            marker, job_id=job, slurm=slurm, connected=connected,
        )
        if matches:
            body["ok"] = True
            body["detail"] = marker
            if not body["message"]:
                gpu_name = marker.get("gpu") or "CUDA"
                body["message"] = (
                    f"Pyxis container ready: {gpu_name} "
                    f"(torch {marker.get('torch') or '?'}, gsplat {marker.get('gsplat') or '?'}"
                    f"{', sm ' + str(marker.get('cuda_arch')) if marker.get('cuda_arch') else ''})"
                )
            with _SETUP["lock"]:
                if not _SETUP["inflight"]:
                    _SETUP["ok"] = True
                    _SETUP["job_id"] = job or str(marker.get("job_id") or "")
                    _SETUP["detail"] = marker
                    if not _SETUP.get("message"):
                        _SETUP["message"] = body["message"]
                    _SETUP["error"] = None
        else:
            body["ok"] = False
            body["needs_reload"] = True
            if reason and not body["inflight"]:
                body["message"] = reason
    if body.get("ok") and not body["inflight"] and body.get("detail"):
        still_ok, reason = setup_matches_allocation(
            body.get("detail"), job_id=job, slurm=slurm, connected=connected,
        )
        if not still_ok:
            body["ok"] = False
            body["needs_reload"] = True
            body["message"] = reason
            with _SETUP["lock"]:
                if not _SETUP["inflight"]:
                    _SETUP["ok"] = False
    if not body["ok"] and not body["inflight"] and not body.get("needs_reload"):
        summary = gpu_occupancy_summary(gpu if isinstance(gpu, dict) else None)
        ours = list(summary.get("ours_on_allocated") or [])
        if ours and matches:
            body["ok"] = True
            if not body["message"]:
                gpu_name = ""
                allocated = summary.get("allocated") or []
                if allocated:
                    gpu_name = str(allocated[0].get("name") or "")
                body["message"] = (
                    "splat-explorer is already on this reserved GPU"
                    + (f" ({gpu_name})" if gpu_name else "")
                    + ". /repair can use this allocation."
                )
            with _SETUP["lock"]:
                if not _SETUP["inflight"] and not _SETUP["ok"]:
                    _SETUP["ok"] = True
                    _SETUP["job_id"] = job
                    if not _SETUP.get("message"):
                        _SETUP["message"] = body["message"]
                    _SETUP["error"] = None
    return body


def _set_setup_message(message: str) -> None:
    with _SETUP["lock"]:
        _SETUP["message"] = message


def detect_cuda_arch_list(*, device: int = 0) -> str:
    """sm_80 A100, sm_90 H100, sm_70 V100 — native arch of CUDA device 0."""
    forced = (os.environ.get("LRZ_CUDA_ARCH") or "").strip()
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(device)
            return f"{major}.{minor}"
    except Exception:
        pass
    return "8.0"


def _find_gsplat_cuda_so() -> Path | None:
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or "")
    if root.is_dir():
        hits = sorted(root.rglob("gsplat*.so"))
        if hits:
            return hits[0]
    return None


def _patch_gsplat_nvcc_single_thread(site: Path) -> None:
    """Ask nvcc for one thread so CUDA 13 JIT fits a 64G Slurm cgroup."""
    path = Path(site) / "gsplat" / "cuda" / "_backend.py"
    if not path.is_file():
        return
    text = path.read_text()
    if '"--threads"' in text or "'--threads'" in text:
        return
    old = '            extra_cuda_cflags += ["-use_fast_math"]\n'
    new = old + '        extra_cuda_cflags += ["--threads", "1"]\n'
    if old not in text:
        return
    path.write_text(text.replace(old, new, 1))


def _prepare_gsplat_compile_env(*, site: str | None = None) -> None:
    os.environ["MAX_JOBS"] = "1"
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    os.environ["NINJAFLAGS"] = "-j1"
    os.environ["MAKEFLAGS"] = "-j1"
    os.environ["FAST_COMPILE"] = "1"
    os.environ["VERBOSE"] = "1"
    os.environ["NVCC_APPEND_FLAGS"] = "--threads=1"
    if site:
        site_path = Path(site)
        _patch_gsplat_nvcc_single_thread(site_path)
        bin_dir = site_path / "bin"
        if bin_dir.is_dir():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _gsplat_cuda_dir(site: str | None = None) -> Path:
    """Locate gsplat/cuda without importing gsplat (import JIT-compiles)."""
    import importlib.util
    import sys

    if site and site not in sys.path:
        sys.path.insert(0, site)
    spec = importlib.util.find_spec("gsplat")
    if spec is None or not spec.origin:
        raise RuntimeError("gsplat is not installed (needed to compile gsplat_cuda.so).")
    return Path(spec.origin).resolve().parent / "cuda"


def _gsplat_cuda_sources(site: str | None = None) -> tuple[list[str], list[str]]:
    cuda_dir = _gsplat_cuda_dir(site)
    sources = (
        sorted(str(p) for p in cuda_dir.glob("csrc/*.cu"))
        + sorted(str(p) for p in cuda_dir.glob("csrc/*.cpp"))
        + [str(cuda_dir / "ext.cpp")]
    )
    includes = [
        str(cuda_dir / "include"),
        str(cuda_dir / "csrc" / "third_party" / "glm"),
    ]
    missing = [p for p in sources if not Path(p).is_file()]
    if missing or not any(p.endswith(".cu") for p in sources):
        raise RuntimeError(f"gsplat CUDA sources missing under {cuda_dir}: {missing[:8]}")
    return sources, includes


def gsplat_ninja_build_dir() -> Path:
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or "")
    if not str(root):
        raise RuntimeError("TORCH_EXTENSIONS_DIR is not set.")
    return root / "gsplat_cuda"


def gsplat_ninja_command(build_dir: Path | None = None) -> list[str]:
    return ["ninja", "-j1", "-v", "-C", str(build_dir or gsplat_ninja_build_dir())]


def _gsplat_version_from_site(site: Path) -> str | None:
    for meta in sorted(site.glob("gsplat-*.dist-info/METADATA")):
        for line in meta.read_text().splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    return None


def _probe_torch_cuda_subprocess() -> dict[str, Any]:
    """Short-lived CUDA probe so the setup process never keeps a torch RSS."""
    import sys

    script = (
        "import json, torch\n"
        "ok = bool(torch.cuda.is_available())\n"
        "cap = list(torch.cuda.get_device_capability(0)) if ok else [0, 0]\n"
        "name = torch.cuda.get_device_name(0) if ok else ''\n"
        "print(json.dumps({"
        "'available': ok, 'gpu': name, 'cap': cap, "
        "'torch': torch.__version__, 'cuda': torch.version.cuda"
        "}))\n"
    )
    out = subprocess.check_output([sys.executable, "-c", script], text=True, timeout=120)
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError("torch CUDA probe produced no JSON.")
    return json.loads(lines[-1])


def write_gsplat_ninja_build(*, site: str | None = None) -> Path:
    """Import torch only long enough to emit build.ninja — do not run nvcc here."""
    import inspect
    import sys

    _prepare_gsplat_compile_env(site=site)
    if site and site not in sys.path:
        sys.path.insert(0, site)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch.utils.cpp_extension as cpp

    cpp._run_ninja_build = lambda *a, **k: None
    if hasattr(cpp, "_run_ninja"):
        cpp._run_ninja = lambda *a, **k: None
    writer = getattr(cpp, "_write_ninja_file_and_build_library", None)
    if writer is None:
        raise RuntimeError("this torch build cannot emit a ninja file for gsplat CUDA")
    name = "gsplat_cuda"
    get_build = getattr(cpp, "_get_build_directory")
    build_dir = Path(get_build(name, verbose=True))
    build_dir.mkdir(parents=True, exist_ok=True)
    lock = build_dir / "lock"
    try:
        lock.unlink()
    except FileNotFoundError:
        pass
    sources, includes = _gsplat_cuda_sources(site)
    extra_cflags = ["-O0", "-Wno-attributes"]
    extra_cuda_cflags = ["-O0", "-use_fast_math", "--threads", "1"]
    sig = inspect.signature(writer)
    kwargs = {
        "name": name,
        "sources": sources,
        "extra_cflags": extra_cflags,
        "extra_cuda_cflags": extra_cuda_cflags,
        "extra_sycl_cflags": None,
        "extra_ldflags": [],
        "extra_include_paths": includes,
        "build_directory": str(build_dir),
        "verbose": True,
        "with_cuda": True,
        "with_sycl": False,
        "is_standalone": False,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    logger.info("Writing gsplat CUDA ninja for %d sources → %s", len(sources), build_dir)
    writer(**kwargs)
    ninja = build_dir / "build.ninja"
    if not ninja.is_file():
        raise RuntimeError(f"gsplat ninja was not written to {ninja}")
    logger.info("Wrote %s", ninja)
    return build_dir


def run_gsplat_ninja_build(build_dir: Path | None = None) -> Path:
    """Run ninja -j1 in this process. Caller must not have imported torch."""
    import shutil
    import sys

    if "torch" in sys.modules:
        raise RuntimeError(
            "nvcc must not share RSS with a loaded torch; run ninja in a fresh process."
        )
    build_dir = Path(build_dir or gsplat_ninja_build_dir())
    ninja_file = build_dir / "build.ninja"
    if not ninja_file.is_file():
        raise RuntimeError(f"missing {ninja_file}")
    ninja_bin = shutil.which("ninja")
    if not ninja_bin:
        raise RuntimeError("ninja is not on PATH (pip install ninja into /workspace/python).")
    argv = gsplat_ninja_command(build_dir)
    argv[0] = ninja_bin
    env = os.environ.copy()
    env["MAX_JOBS"] = "1"
    env["NINJAFLAGS"] = "-j1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    logger.info("Compiling gsplat CUDA with %s (torch not loaded)", " ".join(argv))
    subprocess.check_call(argv, env=env)
    so = _find_gsplat_cuda_so()
    if so is None:
        cand = build_dir / "gsplat_cuda.so"
        if cand.is_file():
            return cand
        raise RuntimeError(
            "gsplat CUDA ninja finished but gsplat_cuda.so was not found under "
            f"TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR')!r}."
        )
    return so


def compile_gsplat_cuda_extension(*, site: str | None = None) -> dict[str, Any]:
    """Build fused gsplat CUDA into TORCH_EXTENSIONS_DIR without a torch splat stand-in.

    In-process JIT keeps a 10–20G torch RSS while nvcc runs, which OOMs a 62G
    Slurm step. Emit ninja in a short torch process, then nvcc with torch gone.
    """
    import sys

    _prepare_gsplat_compile_env(site=site)
    so = _find_gsplat_cuda_so()
    if so is not None:
        logger.info("Reusing prebuilt gsplat CUDA extension %s", so)
        return {"ok": True, "so": str(so), "reused": True}
    if "torch" in sys.modules:
        argv = [sys.executable, "-m", "splat_explorer.repair_lrz", "--compile-gsplat-cuda"]
        if site:
            argv += ["--site", site]
        logger.info("Spawning torch-free gsplat CUDA compile: %s", " ".join(argv))
        subprocess.check_call(argv)
        so = _find_gsplat_cuda_so()
        if so is None:
            raise RuntimeError(
                "out-of-process gsplat CUDA compile finished without gsplat_cuda.so under "
                f"TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR')!r}."
            )
        return {"ok": True, "so": str(so), "reused": False, "out_of_process": True}
    argv = [sys.executable, "-m", "splat_explorer.repair_lrz", "--write-gsplat-ninja"]
    if site:
        argv += ["--site", site]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    logger.info("Emitting gsplat CUDA ninja (short torch process)…")
    subprocess.check_call(argv, env=env)
    so = run_gsplat_ninja_build()
    return {"ok": True, "so": str(so), "reused": False, "out_of_process": True}


def apply_gpu_setup(*, workspace: str = "/workspace") -> dict[str, Any]:
    """Inside the Pyxis container: verify torch/CUDA and install gsplat onto DSS."""
    import sys

    root = Path(workspace)
    python_dir = root / "python"
    logs = root / "logs"
    python_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    site = str(python_dir)
    if site not in sys.path:
        sys.path.insert(0, site)
    existing = os.environ.get("PYTHONPATH") or ""
    os.environ["PYTHONPATH"] = site + (os.pathsep + existing if existing else "")
    os.environ["PATH"] = str(python_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

    probe = _probe_torch_cuda_subprocess()
    if not probe.get("available"):
        raise RuntimeError("torch.cuda is not available in this container.")
    cap = probe.get("cap") or [8, 0]
    arch = f"{int(cap[0])}.{int(cap[1])}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    gpu = str(probe.get("gpu") or "")
    torch_ver = str(probe.get("torch") or "")
    cuda_ver = str(probe.get("cuda") or "")
    logger.info("torch %s cuda %s gpu %s arch %s", torch_ver, cuda_ver, gpu, arch)

    arch_file = root / _CUDA_ARCH_MARKER
    prev_arch = arch_file.read_text().strip() if arch_file.is_file() else ""
    gsplat_pkg = python_dir / "gsplat"
    need_gsplat = (not gsplat_pkg.is_dir()) or (bool(prev_arch) and prev_arch != arch)

    installed = False
    if need_gsplat:
        installed = True
        cmd = [sys.executable, "-m", "pip", "install", "--target", site]
        if prev_arch and prev_arch != arch:
            logger.info("gsplat was built for sm_%s, rebuilding for sm_%s", prev_arch, arch)
            cmd += ["--upgrade", "--force-reinstall", "gsplat>=1.4", "ninja"]
        else:
            cmd += [
                "ninja",
                "numpy>=1.26", "pillow>=10.0", "pyyaml>=6.0", "scipy>=1.11",
                "gsplat>=1.4",
            ]
        logger.info("pip install --target %s gsplat …", site)
        subprocess.check_call(cmd)
        import importlib
        importlib.invalidate_caches()
        if site not in sys.path:
            sys.path.insert(0, site)
    gsplat_ver = _gsplat_version_from_site(python_dir)
    arch_file.write_text(arch + "\n")

    ext_dir = python_dir / "torch_extensions"
    tmp_dir = root / "tmp"
    ext_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    cuda_ext = compile_gsplat_cuda_extension(site=site)

    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("LRZ_JOB_ID") or "unknown"
    family = gpu_family_from_name(gpu)
    body = {
        "ok": True,
        "job_id": str(job_id),
        "gpu": gpu,
        "family": family,
        "partition": os.environ.get("SLURM_JOB_PARTITION") or "",
        "torch": torch_ver,
        "cuda": cuda_ver,
        "gsplat": str(gsplat_ver or "?"),
        "installed": installed,
        "python": site,
        "cuda_arch": arch,
        "compute_cap": f"{int(cap[0])}.{int(cap[1])}",
        "gsplat_cuda": cuda_ext,
    }
    (logs / f"setup-{job_id}.json").write_text(json.dumps(body, indent=2) + "\n")
    print("SETUP_OK " + json.dumps(body), flush=True)
    return body


def probe_gpu_occupancy(cfg: dict | None = None) -> dict[str, Any]:
    """nvidia-smi occupancy only — does not re-run squeue."""
    cfg = cfg or load_lrz_config()
    if gpu_work_owner() == "repair":
        gpu, _at = _cached_occupancy_raw()
        if isinstance(gpu, dict) and gpu.get("gpus"):
            skipped = dict(gpu)
            skipped["deferred"] = True
            skipped["warning"] = (
                "Occupancy probe skipped while a CUDA repair is using the GPU "
                "(a second srun would share the hold job's host-RAM cgroup)."
            )
            return skipped
        raise RuntimeError(
            "Occupancy probe skipped: a CUDA repair is using this GPU. "
            "A second nvidia-smi srun would share the hold job's host RAM "
            "and can OOM a 32G allocation."
        )
    smi = _ssh_run(cfg, nvidia_smi_command(cfg), timeout=SMI_TIMEOUT_S)
    if smi.returncode != 0:
        err = (smi.stderr or smi.stdout or "nvidia-smi occupancy probe failed").strip()[:400]
        raise RuntimeError(err)
    occupancy = parse_gpu_occupancy_text(smi.stdout or "")
    now = time.time()
    with _PROBE["lock"]:
        body = dict(_PROBE["body"]) if isinstance(_PROBE["body"], dict) else {}
        slurm = body.get("slurm") if isinstance(body.get("slurm"), dict) else None
        occupancy["node"] = (slurm or {}).get("node")
        occupancy["occupancy_at"] = now
        body["gpu"] = occupancy
        body["gpu_error"] = None
        body["occupancy_at"] = now
        _PROBE["body"] = body
        _PROBE["error"] = None
    return occupancy


def request_occupancy_probe() -> dict[str, Any]:
    """One nvidia-smi occupancy refresh. Isolated from the 10 min squeue probe."""
    cfg = load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    job = str(cfg.get("job_id") or "").strip()
    if not job.isdigit():
        raise RuntimeError("No job_id. Click Use on a running reserved job first.")
    return probe_gpu_occupancy(cfg)


def _remote_setup_paths(cfg: dict) -> tuple[str, str]:
    job = str(cfg["job_id"])
    logs = f"{cfg['workspace']}/logs"
    return f"{logs}/setup-{job}.json", f"{logs}/setup-{job}.log"


def launch_detached_setup(cfg: dict) -> str:
    """Start srun setup on the login node so the SSH mux is not held for 30 min."""
    marker, log = _remote_setup_paths(cfg)
    srun = srun_setup_command(cfg)
    inner = f"{srun} >{shlex.quote(log)} 2>&1; echo SETUP_EXIT:$? >>{shlex.quote(log)}"
    remote = (
        f"mkdir -p {shlex.quote(str(Path(log).parent))} && "
        f"rm -f {shlex.quote(marker)} && "
        f"nohup sh -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & "
        "echo SETUP_PID $!"
    )
    result = _ssh_run(cfg, remote, timeout=120)
    out = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode != 0:
        raise RuntimeError(out[-1500:] or "Failed to start detached GPU setup.")
    pid = ""
    for line in out.splitlines():
        if "SETUP_PID" in line:
            pid = line.strip().split()[-1]
    logger.info("detached GPU setup pid %s job %s", pid, cfg.get("job_id"))
    return pid


def _setup_message_from_log(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    interesting = [
        ln for ln in lines
        if not ln.startswith("SETUP_EXIT") and not ln.startswith("SETUP_PID")
    ]
    tokens = ("error", "install", "gsplat", "torch", "pyxis", "enroot", "cuda", "compiling", "building")
    for ln in reversed(interesting):
        if any(tok in ln.lower() for tok in tokens):
            return ln[-240:]
    return (interesting[-1] if interesting else "Starting named Pyxis container…")[-240:]


def poll_detached_setup(cfg: dict, *, timeout: float = SETUP_TIMEOUT_S) -> dict[str, Any]:
    marker, log = _remote_setup_paths(cfg)
    t0 = time.time()
    while time.time() - t0 < timeout:
        elapsed = int(time.time() - t0)
        script = (
            f"if [ -f {shlex.quote(marker)} ]; then echo MARKER; cat {shlex.quote(marker)}; "
            f"elif grep -q '^SETUP_EXIT:' {shlex.quote(log)} 2>/dev/null; then echo FAILED; "
            f"tail -n 100 {shlex.quote(log)}; "
            f"else echo RUNNING; tail -n 16 {shlex.quote(log)} 2>/dev/null; fi"
        )
        result = _ssh_run(cfg, script, timeout=25)
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        stripped = text.lstrip()
        if stripped.startswith("MARKER"):
            payload = text.split("MARKER", 1)[-1].strip()
            try:
                loaded = json.loads(payload)
            except json.JSONDecodeError:
                loaded = parse_setup_ok_output(payload) or {}
            if isinstance(loaded, dict) and loaded.get("ok"):
                return loaded
            raise RuntimeError(payload[-1800:] or "Setup marker was not OK.")
        if stripped.startswith("FAILED"):
            rest = text.split("FAILED", 1)[-1]
            detail = parse_setup_ok_output(rest)
            if detail and detail.get("ok"):
                return detail
            raise RuntimeError(rest[-2500:] or "GPU setup srun failed.")
        rest = text.split("RUNNING", 1)[-1] if "RUNNING" in text else text
        msg = _setup_message_from_log(rest)
        mins, secs = divmod(elapsed, 60)
        _set_setup_message(
            f"{msg} ({mins}m {secs}s; first container start/compile on a new node "
            "can take 10–20 min)"
        )
        time.sleep(SETUP_POLL_S)
    raise RuntimeError(
        "GPU setup timed out waiting for the Pyxis container. "
        "gsplat CUDA compile can take 10–20 min — click Load GPU setup again."
    )


def read_remote_setup_marker(cfg: dict | None = None) -> dict[str, Any]:
    """One SSH cat of logs/setup-<job>.json on DSS (survives dashboard restarts)."""
    cfg = cfg or load_lrz_config()
    marker, _log = _remote_setup_paths(cfg)
    result = _ssh_run(
        cfg,
        f"if [ -f {shlex.quote(marker)} ]; then cat {shlex.quote(marker)}; else echo MISSING; fi",
        timeout=20,
    )
    return parse_setup_marker_text(
        result.stdout or "", job_id=str(cfg.get("job_id") or ""),
    )


def wait_for_lrz_setup(cfg: dict | None = None, *, timeout: float = SETUP_TIMEOUT_S) -> dict[str, Any]:
    cfg = cfg or load_lrz_config()
    deadline = time.time() + max(30.0, float(timeout))
    last = lrz_setup_status(cfg)
    while time.time() < deadline:
        last = lrz_setup_status(cfg)
        if last.get("ok") and not last.get("inflight"):
            return last
        if not last.get("inflight") and last.get("error"):
            raise RuntimeError(str(last["error"]))
        time.sleep(2.0)
    raise RuntimeError(
        last.get("message")
        or "Timed out waiting for GPU setup. Open /repair/gpu and check Load GPU setup."
    )


def remember_setup_ok(cfg: dict, detail: dict[str, Any], message: str) -> dict[str, Any]:
    job = str(cfg.get("job_id") or detail.get("job_id") or "")
    with _SETUP["lock"]:
        _SETUP["ok"] = True
        _SETUP["inflight"] = False
        _SETUP["error"] = None
        _SETUP["detail"] = detail
        _SETUP["job_id"] = job
        _SETUP["message"] = message
        _SETUP["at"] = time.time()
    return lrz_setup_status(cfg)


def ensure_lrz_gpu_ready(cfg: dict | None = None) -> dict[str, Any]:
    """Bind this allocation for CUDA repair: reuse marker if GPU family matches.

    Called from the /repair pipeline so a dashboard restart or a new A100/H100
    job does not require a second-click overwrite before the first refine.
    """
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    probe_job(cfg)
    status = lrz_setup_status(cfg)
    if status.get("inflight"):
        _set_setup_message("Waiting for GPU setup to finish before starting the repair…")
        return wait_for_lrz_setup(cfg)
    slurm = live_allocation(str(cfg.get("job_id") or ""))
    gpu, _at = _cached_occupancy_raw()
    connected = pick_connected_gpu(
        list((gpu or {}).get("gpus") or []) if isinstance(gpu, dict) else None
    )
    try:
        marker = read_remote_setup_marker(cfg)
    except RuntimeError as exc:
        logger.warning("setup marker read failed: %s", exc)
        marker = status.get("detail") if status.get("ok") else {"ok": False}
    if not isinstance(marker, dict):
        marker = {"ok": False}
    matches, reason = setup_matches_allocation(
        marker, job_id=str(cfg.get("job_id") or ""), slurm=slurm, connected=connected,
    )
    if matches:
        gpu_name = marker.get("gpu") or (connected or {}).get("name") or "CUDA"
        family = gpu_family_from_name(gpu_name) or ""
        extra = f" ({family})" if family else ""
        return remember_setup_ok(
            cfg, marker,
            f"Reusing GPU setup on {gpu_name}{extra} "
            f"(DSS marker for job {cfg.get('job_id')}).",
        )
    if reason:
        _set_setup_message(reason + " Loading GPU setup…")
    request_lrz_setup(force=True, overwrite=False)
    return wait_for_lrz_setup(cfg)


def setup_lrz_gpu(cfg: dict | None = None, *, overwrite: bool = False) -> dict[str, Any]:
    """Rsync code, start named Pyxis container, verify torch/gsplat.

    Foreign leftovers on THIS reserved GPU are cleared automatically. Our own
    splat-explorer process is never killed — that is the /repair pipeline.
    """
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    if not str(cfg.get("job_id") or "").isdigit():
        raise RuntimeError(
            "No job_id. Click Use on a running reserved job first "
            "(or scripts/lrz/allocate.sh --use <id>)."
        )
    if gpu_work_owner() == "repair":
        raise RuntimeError(
            "A CUDA repair is using this GPU. Stop it on /repair before reloading setup."
        )
    probe_job(cfg)
    with _gpu_exclusive("setup", timeout=8.0):
        return _setup_lrz_gpu_locked(cfg, overwrite=overwrite)


def _setup_lrz_gpu_locked(cfg: dict, *, overwrite: bool = False) -> dict[str, Any]:
    _set_setup_message("Checking who is using the allocated GPU…")
    occ = None
    try:
        occ = probe_gpu_occupancy(cfg)
    except RuntimeError as exc:
        logger.warning("occupancy probe before setup failed: %s", exc)
        occ = None
    summary = gpu_occupancy_summary(occ) if occ else {}
    foreign = bool(summary.get("foreign_on_allocated"))
    ours = bool(summary.get("ours_on_allocated"))
    leftover_note = ""
    if ours and not foreign:
        _set_setup_message(
            "splat-explorer is already on this reserved GPU — uploading latest "
            "code, not resetting the card."
        )
        sync_code_to_dss(cfg)
        try:
            marker = read_remote_setup_marker(cfg)
        except RuntimeError:
            marker = {"ok": False}
        if marker.get("ok"):
            marker = dict(marker)
            marker["reused"] = True
            return marker
        connected = pick_connected_gpu((occ or {}).get("gpus") if occ else None)
        return {
            "ok": True,
            "reused": True,
            "job_id": str(cfg.get("job_id") or ""),
            "gpu": (connected or {}).get("name") or "CUDA",
            "compute_cap": (connected or {}).get("compute_cap"),
        }
    if foreign:
        _set_setup_message(
            "Clearing leftover process(es) on THIS reserved GPU only, then loading splat-explorer…"
        )
        wipe_out = wipe_allocated_gpu(cfg, occ)
        leftover_note = " Wipe: " + _wipe_summary(wipe_out) + "."
        try:
            occ = probe_gpu_occupancy(cfg)
            leftover = occupancy_needs_overwrite(gpu_occupancy_summary(occ))
            if leftover:
                leftover_note += " After wipe: " + leftover
        except RuntimeError as exc:
            logger.warning("occupancy probe after wipe failed: %s", exc)
            leftover_note += " Could not re-read occupancy after wipe (" + str(exc)[:160] + ")."
            occ = None
    elif overwrite:
        _set_setup_message("Reloading GPU setup on this reserved card…")
    connected = pick_connected_gpu((occ or {}).get("gpus") if occ else None)
    bits = []
    if connected:
        bits.append(
            f"Allocated GPU {connected.get('index')} "
            f"{connected.get('name')} sm {connected.get('compute_cap') or '?'}"
        )
        if foreign:
            bits.append("cleared leftover occupant")
    elif occ and gpu_occupancy_summary(occ).get("warning"):
        bits.append(str(gpu_occupancy_summary(occ)["warning"]))
    bits.append("Uploading splat-explorer code to DSS…" + leftover_note)
    _set_setup_message(". ".join(bits))
    sync_code_to_dss(cfg)
    name = container_name_for_job(cfg)
    _set_setup_message(
        f"Starting Pyxis container {name} from pytorch.sqsh on this allocation "
        "(first extract/compile on a new A100/H100 node can take 10–20 min)…"
    )
    launch_detached_setup(cfg)
    return poll_detached_setup(cfg)


def request_lrz_setup(*, force: bool = True, overwrite: bool = False) -> dict[str, Any]:
    """Start GPU setup in a background thread. Safe to click once per allocation."""
    cfg = load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    job = str(cfg.get("job_id") or "").strip()
    if not job.isdigit():
        raise RuntimeError("No job_id. Click Use on a running reserved job first.")
    with _PROBE["lock"]:
        cached = _PROBE["body"] if isinstance(_PROBE["body"], dict) else None
    slurm = (cached or {}).get("slurm") if cached else None
    jobs = list((cached or {}).get("jobs") or []) if cached else []
    selected = select_slurm_job(jobs, job) if jobs else slurm
    if selected is not None and not slurm_job_is_running(selected):
        state = selected.get("state") or "PD"
        raise RuntimeError(
            f"Job {job} is {state}, not running. Load GPU setup after the allocation "
            "is ST=R (or click Use on a running row)."
        )
    owner = gpu_work_owner()
    if owner == "repair":
        raise RuntimeError(
            "A CUDA repair is using this GPU. Stop it on /repair before reloading setup."
        )
    reason = occupancy_reason_for_setup(overwrite=overwrite, refresh_if_missing=True)
    with _SETUP["lock"]:
        if _SETUP["inflight"]:
            return lrz_setup_status(cfg)
        if _SETUP["ok"] and _SETUP.get("job_id") == job and not force:
            return lrz_setup_status(cfg)
        _SETUP["inflight"] = True
        _SETUP["ok"] = False
        _SETUP["error"] = None
        _SETUP["detail"] = None
        _SETUP["job_id"] = job
        _SETUP["at"] = time.time()
        _SETUP["message"] = (
            "Clearing leftover occupant on this reserved GPU, then loading splat-explorer…"
            if reason
            else "Uploading code to DSS, then starting the PyTorch container…"
        )
    threading.Thread(
        target=_run_setup_thread, args=(cfg, overwrite), daemon=True, name="lrz-gpu-setup",
    ).start()
    return lrz_setup_status(cfg)


def _run_setup_thread(cfg: dict, overwrite: bool = False) -> None:
    try:
        detail = setup_lrz_gpu(cfg, overwrite=overwrite)
        gpu = detail.get("gpu") or "CUDA"
        extra = " (installed gsplat onto DSS)" if detail.get("installed") else ""
        with _SETUP["lock"]:
            _SETUP["ok"] = True
            _SETUP["detail"] = detail
            _SETUP["error"] = None
            _SETUP["job_id"] = str(cfg.get("job_id") or detail.get("job_id") or "")
            _SETUP["message"] = f"GPU setup ready on {gpu}{extra}."
            _SETUP["at"] = time.time()
            _SETUP["inflight"] = False
        try:
            request_gpu_probe(force=True)
        except Exception:
            logger.warning("post-setup GPU probe failed", exc_info=True)
    except Exception as exc:
        logger.warning("LRZ GPU setup failed: %s", exc)
        with _SETUP["lock"]:
            _SETUP["ok"] = False
            _SETUP["error"] = str(exc)
            _SETUP["message"] = f"GPU setup failed: {exc}"
            _SETUP["at"] = time.time()
            _SETUP["inflight"] = False


def _askpass_env(password: str) -> tuple[dict[str, str], Path]:
    fd, name = tempfile.mkstemp(prefix="lrz-askpass-", suffix=".sh")
    os.close(fd)
    helper = Path(name)
    helper.write_text('#!/bin/sh\nprintf "%s\\n" "$LRZ_SSH_PASSWORD"\n')
    helper.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    env = os.environ.copy()
    env["LRZ_SSH_PASSWORD"] = password
    env["SSH_ASKPASS"] = str(helper)
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = env.get("DISPLAY") or ":0"
    env["SSH_ASKPASS_REQUIRE"] = "force"
    return env, helper


def probe_job(cfg: dict | None = None, *, password: str | None = None) -> str:
    """One `squeue` call. Returns the Slurm state (e.g. R) or raises."""
    cfg = cfg or load_lrz_config()
    if not str(cfg.get("job_id") or "").isdigit():
        raise RuntimeError(
            "No LRZ job id. Put the current sbatch id in configs/lrz.local.yaml "
            "(job_id) or export LRZ_JOB_ID."
        )
    mux = lrz_session_alive(cfg)
    if not mux and not password:
        raise RuntimeError(session_required_message())
    argv = ssh_argv(cfg, multiplex=mux) + [
        f"squeue --me --job={cfg['job_id']} -h -o '%t|%m|%P|%N'"
    ]
    env = os.environ.copy()
    helper = None
    if password and not mux:
        env, helper = _askpass_env(password)
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    finally:
        if helper is not None:
            try:
                helper.unlink()
            except OSError:
                pass
            env.pop("LRZ_SSH_PASSWORD", None)
    line = (result.stdout or "").strip().splitlines()[0] if result.stdout else ""
    parts = [p.strip() for p in line.split("|")] if line else []
    state = parts[0] if parts else ""
    if result.returncode != 0 or not state:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Could not query Slurm job {cfg['job_id']}: {err or 'empty squeue'}. "
            "Is eduVPN up? Is the allocation still running?"
        )
    remember_live_allocation(
        job_id=str(cfg["job_id"]),
        state=state,
        mem=parts[1] if len(parts) > 1 else "",
        partition=parts[2] if len(parts) > 2 else "",
        node=parts[3] if len(parts) > 3 else "",
    )
    if state != "R":
        raise RuntimeError(
            f"LRZ job {cfg['job_id']} is {state}, not running. "
            "Allocate a GPU with sbatch and update configs/lrz.local.yaml."
        )
    return state


def _mux_run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(format_remote_command_error(argv, result))
    return result


def pull_remote_job_status(cfg: dict, job_id: str, job_dir: Path) -> dict[str, Any] | None:
    """Login-node `cat` of DSS status.json — never a second GPU srun."""
    path = f"{remote_job_dir(cfg, job_id)}/{STATUS_JSON}"
    result = _ssh_run(cfg, f"cat {shlex.quote(path)}", timeout=12)
    if result.returncode != 0:
        return None
    try:
        body = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict) or not body:
        return None
    write_status(job_dir, **{k: v for k, v in body.items() if k != "updated_at"})
    return body


def run_srun_worker(
    cfg: dict,
    job_id: str,
    job_dir: Path,
    *,
    on_progress: Callable[[dict], None] | None = None,
    should_stop=None,
    mem_flag: str | None = None,
    attempt: int = 0,
) -> None:
    """Stream CUDA srun logs and poll DSS status.json while the GPU step runs."""
    flag = mem_flag or srun_mem_flag(cfg)
    argv = ssh_argv(cfg, multiplex=True) + [
        srun_worker_command(cfg, job_id, mem_flag=flag)
    ]
    log_path = Path(job_dir) / "srun.log"
    rc: int | None = None
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        try:
            last_ckpt = -1
            stop_at: float | None = None
            while True:
                if should_stop is not None and should_stop():
                    if stop_at is None:
                        try:
                            request_remote_stop(cfg, job_id)
                        except Exception as exc:
                            logger.warning("could not write remote STOP: %s", exc)
                        stop_at = time.time()
                    if time.time() - stop_at >= STOP_GRACE_S:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        try:
                            cancel_overlapping_repair_steps(cfg)
                        except Exception as exc:
                            logger.warning("could not cancel leftover CUDA steps: %s", exc)
                        raise RepairStopped("Stopped during CUDA srun.")
                rc = proc.poll()
                try:
                    pull_remote_job_status(cfg, job_id, job_dir)
                except RuntimeError as exc:
                    logger.warning("remote status poll failed: %s", exc)
                body = read_status_file(job_dir)
                ckpt = int(body.get("checkpoint_iters") or 0)
                if ckpt > last_ckpt:
                    try:
                        pull_remote_job_artifacts(cfg, job_id, job_dir)
                        last_ckpt = ckpt
                    except Exception as exc:
                        logger.warning("artifact download failed: %s", exc)
                if on_progress is not None:
                    on_progress(progress_from_status(read_status_file(job_dir)))
                if rc is not None:
                    break
                time.sleep(2.0)
        except Exception:
            if proc.poll() is None:
                proc.terminate()
            raise
    if rc:
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-2500:]
        except OSError:
            pass
        failed = subprocess.CompletedProcess(argv, rc, stdout=tail, stderr=tail)
        error = format_remote_command_error(argv, failed, log_tail=tail)
        blob = error.lower()
        refused = "memory required by task is not available" in blob
        oom = "oom_kill" in blob or "out of memory" in blob
        if attempt < 2 and refused and flag != "--mem=16G":
            logger.warning("CUDA srun refused at %s; retrying with --mem=16G", flag)
            msg = f"Hold RAM too small for {flag}; retrying CUDA srun at 16G…"
            write_status(job_dir, phase="srun", message=msg)
            if on_progress is not None:
                on_progress({"phase": "srun", "message": msg})
            run_srun_worker(
                cfg, job_id, job_dir,
                on_progress=on_progress, should_stop=should_stop,
                mem_flag="--mem=16G", attempt=attempt + 1,
            )
            return
        if attempt < 2 and oom:
            tighter = enable_tighter_job_params(job_dir)
            if tighter:
                try:
                    push_job_params(cfg, job_id, job_dir)
                except Exception as exc:
                    logger.warning("could not push packed params.json: %s", exc)
            retry_flag = flag
            logger.warning(
                "CUDA srun OOM at %s tighter=%s; retrying %s",
                flag, tighter, retry_flag,
            )
            params = {}
            try:
                params = json.loads((Path(job_dir) / PARAMS_JSON).read_text())
            except (OSError, json.JSONDecodeError):
                params = {}
            edge = params.get("train_max_edge") or TIGHT_TRAIN_MAX_EDGE
            msg = (
                "CUDA step OOM-killed on the host cgroup after cuda_ready. "
                f"Retrying packed gsplat at {edge}px / {retry_flag}…"
            )
            write_status(job_dir, phase="srun", message=msg, packed=True)
            if on_progress is not None:
                on_progress({"phase": "srun", "message": msg, "packed": True})
            if tighter or attempt == 0:
                run_srun_worker(
                    cfg, job_id, job_dir,
                    on_progress=on_progress, should_stop=should_stop,
                    mem_flag=retry_flag, attempt=attempt + 1,
                )
                return
        write_status(job_dir, phase="error", message=error[:400])
        raise RuntimeError(error)


def sync_code_and_job(
    job_dir: Path,
    *,
    password: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
    should_stop=None,
) -> None:
    """rsync + srun + rsync over ControlMaster, or the interactive shell script."""
    if not lrz_configured():
        raise RuntimeError(
            "LRZ is not configured. Copy configs/lrz.example.yaml to "
            "configs/lrz.local.yaml and set job_id to the current sbatch id."
        )
    job_dir = Path(job_dir)
    if password and not lrz_session_alive():
        _run_repair_script(job_dir, password=password)
        return
    if not lrz_session_alive():
        raise RuntimeError(session_required_message())
    cfg = load_lrz_config()
    probe_job(cfg)
    live = live_allocation(str(cfg.get("job_id") or ""))
    if live.get("mem"):
        cfg = dict(cfg)
        cfg["job_mem"] = live["mem"]

    def note(phase: str, message: str) -> None:
        write_status(job_dir, phase=phase, message=message)
        if on_progress is not None:
            on_progress(progress_from_status(read_status_file(job_dir)))

    note("gpu_ready", "Connecting this repair to the reserved GPU…")
    ensure_lrz_gpu_ready(cfg)
    job_id = job_dir.name
    remote = f"{cfg['user']}@{cfg['host']}"
    ssh_e = rsync_ssh_cmd(cfg)
    note("rsync_up", "Uploading scene + code to DSS (once for this repair)…")
    with _gpu_exclusive("repair", timeout=SETUP_TIMEOUT_S):
        sync_code_to_dss(cfg)
        _mux_run(ssh_argv(cfg, multiplex=True) + [
            f"mkdir -p {shlex.quote(remote_job_dir(cfg, job_id))}"
        ])
        _mux_run([
            "rsync", "-az", "-e", ssh_e, "--exclude", STATUS_JSON,
            f"{job_dir}/", f"{remote}:{remote_job_dir(cfg, job_id)}/",
        ])
        note("srun", "Running GSFix CUDA refine on the reserved GPU…")
        try:
            cancelled = cancel_overlapping_repair_steps(cfg)
        except Exception as exc:
            logger.warning("could not cancel leftover CUDA steps: %s", exc)
            cancelled = []
        if cancelled:
            time.sleep(3.0)
        run_srun_worker(
            cfg, job_id, job_dir, on_progress=on_progress, should_stop=should_stop,
        )
        note("rsync_down", "Downloading repaired splat…")
        _mux_run([
            "rsync", "-az", "-e", ssh_e,
            f"{remote}:{remote_job_dir(cfg, job_id)}/",
            f"{job_dir}/",
        ])
    if not job_results_ready(job_dir):
        raise RuntimeError(
            f"Remote job {job_id} finished without {OUT_PLY} / {METRICS_JSON}."
        )
    write_status(job_dir, phase="done", message="Results are local.")


def _run_repair_script(job_dir: Path, *, password: str | None = None) -> None:
    """Invoke scripts/lrz/run-repair.sh. Password, if any, is env-only."""
    job_dir = Path(job_dir)
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "lrz" / "run-repair.sh"
    if not script.is_file():
        script = Path.cwd() / "scripts" / "lrz" / "run-repair.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")
    env = os.environ.copy()
    if password:
        env["LRZ_SSH_PASSWORD"] = password
    write_status(job_dir, phase="ssh", message="Running scripts/lrz/run-repair.sh…")
    result = subprocess.run(
        [str(script), job_dir.name],
        cwd=str(root if (root / "src").is_dir() else Path.cwd()),
        env=env,
        check=False,
        stdin=subprocess.DEVNULL if password else None,
        start_new_session=bool(password),
    )
    env.pop("LRZ_SSH_PASSWORD", None)
    if result.returncode != 0:
        raise RuntimeError(
            f"scripts/lrz/run-repair.sh {job_dir.name} exited {result.returncode}. "
            "Check eduVPN, the sbatch job id, and the terminal output."
        )
    if not job_results_ready(job_dir):
        raise RuntimeError(
            f"Remote job {job_dir.name} finished without {OUT_PLY} / {METRICS_JSON}."
        )
    write_status(job_dir, phase="done", message="Results are local.")


def wait_for_job_results(
    job_dir: Path,
    *,
    should_stop=None,
    poll_s: float = 2.0,
    timeout_s: float = 4 * 3600,
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Block until the interactive shell script has rsynced results back."""
    job_dir = Path(job_dir)
    command = f"scripts/lrz/run-repair.sh {job_dir.name}"
    write_status(
        job_dir,
        phase="awaiting_ssh",
        command=command,
        message=(
            f"Waiting for `{command}`. Run it in a project terminal and type "
            "your LRZ password when ssh asks."
        ),
    )
    if on_progress is not None:
        on_progress({
            "phase": "awaiting_ssh",
            "n_iters": 0,
            "n_updated": 0,
            "command": command,
        })
    deadline = time.time() + float(timeout_s)
    while True:
        if job_results_ready(job_dir):
            return
        if should_stop is not None and should_stop():
            raise RuntimeError("Stopped before the LRZ job finished.")
        if time.time() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for `{command}`. "
                "Run the script (eduVPN + password) or paste the password on the repair page."
            )
        if on_progress is not None:
            phase = "awaiting_ssh"
            sp = job_dir / STATUS_JSON
            if sp.is_file():
                try:
                    phase = json.loads(sp.read_text()).get("phase") or phase
                except (OSError, json.JSONDecodeError):
                    pass
            on_progress({
                "phase": phase,
                "n_iters": 0,
                "n_updated": 0,
                "command": command,
            })
        time.sleep(float(poll_s))


@dataclass
class LrzRemoteRepair:
    """Same stats contract as the local CUDA lift, executed on LRZ."""

    method: str = "gsfix-gsplat"
    iters: int = 20
    kf_iters: int = 50
    lambda_dssim: float = 0.2
    densify: bool = False
    densify_every: int = 5
    densify_grad_thresh: float = 0.0002
    prune_opacity: float = 0.005
    split_scale: float = 0.1
    max_clone: int = 2048
    max_gaussians: int = 2_500_000
    lr_means: float = 1.6e-4
    lr_colors: float = 0.0025
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    near: float = 0.05
    packed: bool = False
    train_max_edge: int = 0
    sparse_grad: bool = False
    white_background: bool = False
    max_chunks: int = 1
    on_progress: Callable[[dict], None] | None = None
    should_stop: Callable[[], bool] | None = None

    def _params(self, cfg: dict | None = None) -> dict:
        tight = tight_host_ram(cfg)
        params = {
            "method": str(self.method),
            "iters": int(self.iters),
            "kf_iters": int(self.kf_iters),
            "lambda_dssim": float(self.lambda_dssim),
            "densify": bool(self.densify),
            "densify_every": int(self.densify_every),
            "densify_grad_thresh": float(self.densify_grad_thresh),
            "prune_opacity": float(self.prune_opacity),
            "split_scale": float(self.split_scale),
            "max_clone": int(self.max_clone),
            "max_gaussians": int(self.max_gaussians),
            "lr_means": float(self.lr_means),
            "lr_colors": float(self.lr_colors),
            "lr_opacities": float(self.lr_opacities),
            "lr_scales": float(self.lr_scales),
            "lr_quats": float(self.lr_quats),
            "near": float(self.near),
            "packed": bool(self.packed) or tight,
            "train_max_edge": int(
                self.train_max_edge or (TIGHT_TRAIN_MAX_EDGE if tight else 0)
            ),
            "sparse_grad": bool(self.sparse_grad),
            "white_background": bool(self.white_background),
            "max_chunks": int(self.max_chunks),
        }
        deadline = getattr(self, "_deadline", None)
        if deadline:
            params["deadline_unix"] = float(deadline)
        if str(self.method) in ("gsfix-gsplat-visprune", "visprune"):
            params.update(
                freeze_occluded=False,
                error_prune=True,
                error_thresh=0.12,
                prune_max_frac=0.02,
                prune_min_keep=32,
                depth_margin=0.05,
                contrib_thresh=0.05,
                anchor_weight=0.3,
                yaw_offset_deg=12.0,
                pitch_offset_deg=8.0,
            )
        return params

    def apply(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
    ) -> dict[str, Any]:
        if not lrz_configured():
            raise RuntimeError(
                "gsplat CUDA refine needs an NVIDIA GPU, or LRZ (configs/lrz.local.yaml "
                "with a running sbatch job_id)."
            )
        cfg = load_lrz_config()
        if lrz_session_alive(cfg):
            try:
                probe_job(cfg)
            except Exception as exc:
                logger.warning("squeue before pack failed: %s", exc)
            live = live_allocation(str(cfg.get("job_id") or ""))
            if live.get("mem"):
                cfg = dict(cfg)
                cfg["job_mem"] = live["mem"]
        job_dir = pack_refine_job(
            scene, camera, rendered_rgb, repaired_rgb, params=self._params(cfg),
        )
        password = get_ssh_password() or os.environ.get("LRZ_SSH_PASSWORD")
        last_ply_mtime = 0.0
        user_progress = self.on_progress

        def wrapped_progress(stats: dict | None) -> None:
            nonlocal last_ply_mtime
            merged = dict(stats or {})
            ply = job_dir / OUT_PLY
            try:
                mtime = ply.stat().st_mtime if ply.is_file() else 0.0
            except OSError:
                mtime = 0.0
            if mtime > last_ply_mtime:
                last_ply_mtime = mtime
                try:
                    ingested = ingest_job_results(scene, job_dir)
                    merged = {**ingested, **merged}
                    if ingested.get("render_rgb") is not None:
                        merged["render_rgb"] = ingested["render_rgb"]
                    if ingested.get("l1_after") is not None:
                        merged["l1_after"] = ingested["l1_after"]
                    merged.setdefault("phase", "refine")
                    merged.setdefault(
                        "checkpoint_iters",
                        int(merged.get("n_iters") or ingested.get("n_iters") or 0),
                    )
                except FileNotFoundError:
                    pass
            if user_progress is not None:
                user_progress(merged)

        if user_progress is not None:
            user_progress({"phase": "rsync_up", "n_iters": 0, "n_updated": 0})
        kwargs = {
            "on_progress": wrapped_progress,
            "should_stop": getattr(self, "should_stop", None),
        }
        try:
            if lrz_session_alive():
                sync_code_and_job(job_dir, **kwargs)
            elif password:
                sync_code_and_job(job_dir, password=password, **kwargs)
            else:
                raise RuntimeError(session_required_message())
        except RepairStopped:
            if job_results_ready(job_dir):
                stats = ingest_job_results(scene, job_dir)
                stats["lrz_job_dir"] = str(job_dir)
                stats["lrz_job_id"] = job_dir.name
                stats["stopped"] = True
                return stats
            raise
        stats = ingest_job_results(scene, job_dir)
        stats["lrz_job_dir"] = str(job_dir)
        stats["lrz_job_id"] = job_dir.name
        return stats

    def apply_until(
        self,
        scene: GaussianScene,
        camera: Camera,
        rendered_rgb: np.ndarray,
        repaired_rgb: np.ndarray,
        *,
        should_stop=None,
        deadline: float | None = None,
        on_checkpoint=None,
    ) -> dict[str, Any]:
        """One remote job: upload the scene once, GPU loops, dashboard only downloads."""
        empty = {
            "backend": str(self.method),
            "n_visible": scene.num_gaussians,
            "n_updated": 0,
            "n_stamped": 0,
            "n_spawned": 0,
            "n_gaussians": scene.num_gaussians,
            "n_iters": 0,
            "l1_before": 0.0,
            "l1_after": None,
        }
        if should_stop is not None and should_stop():
            return empty
        if deadline is not None and time.time() >= deadline:
            return empty
        self.should_stop = should_stop
        self.on_progress = on_checkpoint
        self._deadline = deadline
        try:
            last = self.apply(scene, camera, rendered_rgb, repaired_rgb)
        except RepairStopped:
            return empty
        finally:
            self._deadline = None
        last = dict(last)
        last["n_stamped"] = int(last.get("n_stamped") or 0)
        last["phase"] = last.get("phase") or "refine"
        if on_checkpoint is not None:
            on_checkpoint(last)
        return last


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="splat_explorer.repair_lrz")
    parser.add_argument("--job-dir", help="Packed job directory (local or /workspace/inputs/id)")
    parser.add_argument(
        "--setup", action="store_true",
        help="Inside the Pyxis container: verify torch/CUDA and install gsplat onto DSS",
    )
    parser.add_argument(
        "--write-gsplat-ninja", action="store_true",
        help="Emit gsplat CUDA build.ninja then exit (no nvcc; used by --setup)",
    )
    parser.add_argument(
        "--compile-gsplat-cuda", action="store_true",
        help="Build gsplat_cuda.so with nvcc in a torch-free process",
    )
    parser.add_argument("--site", help="pip --target dir that contains gsplat (e.g. /workspace/python)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.write_gsplat_ninja:
        path = write_gsplat_ninja_build(site=args.site)
        print("NINJA_OK " + str(path), flush=True)
        return
    if args.compile_gsplat_cuda:
        stats = compile_gsplat_cuda_extension(site=args.site)
        print("COMPILE_OK " + json.dumps(stats), flush=True)
        return
    if args.setup:
        stats = apply_gpu_setup()
        logger.info("gpu setup done: %s", stats)
        return
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("LRZ_CUDA_ARCH") or "8.0"
    if not args.job_dir:
        parser.error("one of --job-dir or --setup is required")
    stats = apply_packed_job(Path(args.job_dir))
    logger.info("repair-job done: %s", {k: v for k, v in stats.items() if k != "render_rgb"})


if __name__ == "__main__":
    main()
