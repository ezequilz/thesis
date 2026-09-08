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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .rendering.base import Camera
from .scene import GaussianScene, load_ply, save_ply

logger = logging.getLogger(__name__)

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

_DEFAULTS = {
    "user": "go73kaf2",
    "host": "login.ai.lrz.de",
    "job_id": "",
    "workspace": "/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer",
    "container": "/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/go73kaf2/splat-explorer/containers/pytorch.sqsh",
    "cpus": 4,
    "mem": "32G",
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


def lrz_session_alive(cfg: dict | None = None) -> bool:
    """True when ``scripts/lrz/ssh-session.sh`` left a working ControlMaster."""
    sock = control_path()
    if not sock.exists():
        return False
    cfg = cfg or load_lrz_config()
    result = subprocess.run(
        [
            _ssh_bin(), "-o", f"ControlPath={sock}", "-O", "check",
            f"{cfg['user']}@{cfg['host']}",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


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
        "status_script": "scripts/lrz/status.sh",
        "gpu_url": "/repair/gpu",
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
        "status": "scripts/lrz/status.sh",
        "status_sinfo": "scripts/lrz/status.sh --sinfo",
        "gpu_shell": "scripts/lrz/gpu-shell.sh",
        "bootstrap": "scripts/lrz/bootstrap.sh",
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
    mem: str = "32G",
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
    if not job_id.isdigit():
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


def _write_json(path: Path, body: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(path)


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
    params = json.loads((job_dir / PARAMS_JSON).read_text())
    scene = load_ply(job_dir / SCENE_PLY)
    rendered = np.asarray(Image.open(job_dir / RENDERED_PNG).convert("RGB"), dtype=np.uint8)
    repaired = np.asarray(Image.open(job_dir / REPAIRED_PNG).convert("RGB"), dtype=np.uint8)
    write_status(
        job_dir,
        phase="cuda_import",
        message="Loading torch/gsplat (first run on this node compiles CUDA kernels)…",
        n_gaussians=int(scene.num_gaussians),
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
            fields["message"] = (
                f"GPU ready: {fields.get('gpu_name') or 'CUDA'}. "
                "Starting photometric refine…"
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
    stats = backend.apply(scene, camera, rendered, repaired)
    save_ply(scene, job_dir / OUT_PLY)
    render_rgb = stats.get("render_rgb")
    if render_rgb is not None:
        Image.fromarray(np.asarray(render_rgb, dtype=np.uint8)).save(job_dir / OUT_RENDER)
    metrics = {k: v for k, v in stats.items() if k != "render_rgb"}
    metrics["job_id"] = job_dir.name
    _write_json(job_dir / METRICS_JSON, metrics)
    write_status(job_dir, phase="done", message="CUDA refine finished.", **{
        k: _jsonable(v) for k, v in metrics.items() if k not in ("job_id",)
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


PROBE_TTL_S = 30.0  # LRZ treats automated squeue loops as a DoS.
PROBE_TIMEOUT_S = 25.0
SMI_TIMEOUT_S = 35.0
_SMI_FIELDS = (
    "index", "name", "memory_used_mib", "memory_total_mib",
    "utilization_gpu", "utilization_memory", "temperature_c",
    "power_w", "power_limit_w", "compute_cap",
)
_PROBE = {
    "lock": threading.Lock(),
    "at": 0.0,
    "body": None,
    "error": None,
    "inflight": False,
}
_SINFO = {
    "lock": threading.Lock(),
    "at": 0.0,
    "body": None,
    "error": None,
}
_ALLOCATE = {"lock": threading.Lock(), "inflight": False}


def parse_squeue_line(line: str) -> dict | None:
    """Parse `squeue -o '%i|%t|%P|%N|%M|%l|%r'` or with optional `|%j` name."""
    text = (line or "").strip()
    if not text:
        return None
    row = text.splitlines()[0].strip()
    if not row or row.lower().startswith("squeue") or row.lower().startswith("jobid"):
        return None
    parts = [p.strip() for p in row.split("|")]
    while len(parts) < 7:
        parts.append("")
    reason = parts[6] if parts[6] not in ("", "None", "N/A") else None
    body = {
        "job_id": parts[0],
        "state": parts[1],
        "partition": parts[2],
        "node": parts[3],
        "elapsed": parts[4],
        "timelimit": parts[5],
        "reason": reason,
        "name": parts[7] if len(parts) > 7 and parts[7] else None,
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
        body: dict[str, Any] = {}
        for i, key in enumerate(_SMI_FIELDS):
            raw = cells[i] if i < len(cells) else ""
            if key in ("name", "compute_cap"):
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


def parse_probe_bundle(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    markers = {"SQUEUE", "CONTAINER", "NGC", "WORKSPACE", "STATUS"}
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
        action = pending_slurm_action(
            job_id=str(job_id), slurm=slurm, jobs=jobs, nodes=nodes,
        )
        checks.append({
            "id": "slurm", "ok": False, "label": "Slurm job",
            "detail": f"Job {job_id} is {state}{extra}",
            "action": action,
        })
    else:
        checks.append({
            "id": "slurm", "ok": False, "label": "Slurm job",
            "detail": f"Job {job_id} is not in the queue (expired or wrong id).",
            "action": "The GPU hold job ended. Reserve 8h or 24h from /repair/gpu (or scripts/lrz/allocate.sh 8h).",
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

    gpus = (gpu or {}).get("gpus") if isinstance(gpu, dict) else gpu
    if isinstance(gpus, list) and gpus:
        g0 = gpus[0]
        used = g0.get("memory_used_mib")
        total = g0.get("memory_total_mib")
        mem = f"{int(used)}/{int(total)} MiB" if used is not None and total else ""
        checks.append({
            "id": "gpu", "ok": True, "label": "GPU",
            "detail": " · ".join(x for x in (g0.get("name"), mem) if x),
            "action": None,
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
    argv = ssh_argv(cfg, multiplex=True) + [remote]
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def _login_probe_script(cfg: dict, packed_id: str | None) -> str:
    container = shlex.quote(str(cfg.get("container") or "/nonexistent"))
    workspace = shlex.quote(str(cfg["workspace"]))
    if packed_id:
        status = shlex.quote(f"{cfg['workspace']}/inputs/{packed_id}/{STATUS_JSON}")
        status_block = f"if [ -f {status} ]; then cat {status}; else echo NONE; fi"
    else:
        status_block = "echo NONE"
    return (
        "echo SQUEUE\n"
        "squeue --me -h -o '%i|%t|%P|%N|%M|%l|%r|%j' || true\n"
        "echo CONTAINER\n"
        f"if [ -f {container} ]; then echo OK $(stat -c%s {container}); else echo MISSING; fi\n"
        "echo NGC\n"
        "if [ -s \"$HOME/enroot/.credentials\" ]; then echo OK; else echo MISSING; fi\n"
        "echo WORKSPACE\n"
        f"if [ -d {workspace} ]; then echo OK; else echo MISSING; fi\n"
        "echo STATUS\n"
        f"{status_block}\n"
    )


def nvidia_smi_command(cfg: dict) -> str:
    job = shlex.quote(str(cfg["job_id"]))
    inner = (
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,compute_cap "
        "--format=csv,noheader,nounits"
    )
    return (
        f"srun --jobid={job} --overlap --nodes=1 --ntasks=1 "
        f"--cpus-per-task=1 --gres=gpu:1 --quiet "
        f"bash -lc {shlex.quote(inner)}"
    )


def probe_lrz_gpu(cfg: dict | None = None, *, packed_id: str | None = None) -> dict[str, Any]:
    """One SSH login probe (includes one squeue --me) plus nvidia-smi if the job is R."""
    cfg = cfg or load_lrz_config()
    if not lrz_session_alive(cfg):
        raise RuntimeError(session_required_message())
    result = _ssh_run(cfg, _login_probe_script(cfg, packed_id), timeout=PROBE_TIMEOUT_S)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(err or "LRZ login probe failed.")
    parsed = parse_probe_bundle(result.stdout or "")
    jobs = parse_squeue_lines(parsed.get("squeue") or "")
    slurm = select_slurm_job(jobs, str(cfg.get("job_id") or ""))
    for row in jobs:
        row["current"] = bool(slurm and row.get("job_id") == slurm.get("job_id") and slurm.get("current"))
    container_raw = (parsed.get("container") or "").strip()
    size = None
    parts = container_raw.split()
    if len(parts) >= 2 and parts[0] == "OK":
        try:
            size = int(parts[1])
        except ValueError:
            size = None
    container = {"ok": container_raw.startswith("OK"), "bytes": size}
    remote_status = None
    status_raw = (parsed.get("status") or "").strip()
    if status_raw and status_raw != "NONE":
        try:
            loaded = json.loads(status_raw)
            if isinstance(loaded, dict):
                remote_status = loaded
        except json.JSONDecodeError:
            remote_status = {"raw": status_raw[:500]}
    gpu_body: dict[str, Any] | None = None
    gpu_error = None
    smi_job = (slurm or {}).get("job_id") if slurm and slurm.get("state") == "R" else None
    if smi_job:
        smi_cfg = dict(cfg)
        smi_cfg["job_id"] = smi_job
        smi = _ssh_run(cfg, nvidia_smi_command(smi_cfg), timeout=SMI_TIMEOUT_S)
        if smi.returncode == 0:
            gpu_body = {
                "gpus": parse_nvidia_smi_csv(smi.stdout or ""),
                "node": slurm.get("node") if slurm else None,
            }
            if not gpu_body["gpus"]:
                gpu_error = (smi.stdout or smi.stderr or "empty nvidia-smi").strip()[:400]
        else:
            gpu_error = (smi.stderr or smi.stdout or "nvidia-smi via srun failed").strip()[:400]
    return {
        "slurm": slurm,
        "jobs": jobs,
        "container": container,
        "ngc": (parsed.get("ngc") or "").strip() == "OK",
        "workspace_ok": (parsed.get("workspace") or "").strip() == "OK",
        "remote_status": remote_status,
        "gpu": gpu_body,
        "gpu_error": gpu_error,
    }


def request_gpu_probe(*, force: bool = False, packed_id: str | None = None) -> dict[str, Any]:
    """Start at most one background probe. Never loops squeue."""
    cfg = load_lrz_config()
    if not lrz_session_alive(cfg):
        return {"started": False, "reason": "ssh"}
    now = time.time()
    with _PROBE["lock"]:
        if _PROBE["inflight"]:
            return {"started": False, "reason": "inflight"}
        age = now - float(_PROBE["at"] or 0)
        if not force and _PROBE["body"] is not None and age < PROBE_TTL_S:
            return {"started": False, "reason": "fresh", "age_s": round(age, 1)}
        if not force and _PROBE["error"] and age < 8:
            return {"started": False, "reason": "backoff"}
        _PROBE["inflight"] = True
    threading.Thread(
        target=_run_probe_thread, args=(cfg, packed_id), daemon=True, name="lrz-gpu-probe",
    ).start()
    return {"started": True}


def _run_probe_thread(cfg: dict, packed_id: str | None) -> None:
    body = None
    err = None
    try:
        body = probe_lrz_gpu(cfg, packed_id=packed_id)
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
        _PROBE["body"] = None
        _PROBE["error"] = None
        _PROBE["inflight"] = False
    with _SINFO["lock"]:
        _SINFO["at"] = 0.0
        _SINFO["body"] = None
        _SINFO["error"] = None


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
        mem=str(cfg.get("mem") or "32G"),
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


def use_lrz_job(job_id: str) -> dict[str, Any]:
    path = write_lrz_job_id(job_id)
    reset_gpu_probe_cache()
    return {
        "ok": True,
        "job_id": str(job_id).strip(),
        "path": str(path),
        "message": f"Now using job {job_id}. Probe GPU, then scripts/lrz/gpu-shell.sh",
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
        if not force and isinstance(cached, dict) and age < 60:
            return {
                "ok": True,
                **cached,
                "cached": True,
                "age_s": round(age, 1),
                "message": "Using cached sinfo (LRZ forbids sinfo loops).",
            }
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


def lrz_dashboard_snapshot(
    repair_job: dict | None = None,
    *,
    request_probe: bool = False,
    force_probe: bool = False,
) -> dict[str, Any]:
    """Local connection + current repair, plus a cached GPU probe."""
    status = lrz_status()
    packed = list_packed_jobs()
    packed_id = packed[0]["id"] if packed else None
    if request_probe or force_probe:
        request_gpu_probe(force=force_probe, packed_id=packed_id)
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
        request_gpu_probe(force=False, packed_id=packed_id)
        with _PROBE["lock"]:
            inflight = bool(_PROBE["inflight"])
            cached = _PROBE["body"]
            error = _PROBE["error"]
            probed_at = float(_PROBE["at"] or 0)
    slurm = (cached or {}).get("slurm")
    jobs = list((cached or {}).get("jobs") or [])
    container = (cached or {}).get("container")
    gpu = (cached or {}).get("gpu")
    gpu_error = (cached or {}).get("gpu_error") or error
    checks = build_connection_checks(
        configured=bool(status["configured"]),
        session=bool(status["session"]),
        job_id=str(status.get("job_id") or ""),
        slurm=slurm,
        container=container,
        gpu=gpu,
        gpu_error=gpu_error,
        probed=cached is not None,
    )
    ready = all(
        c.get("ok") is True
        for c in checks
        if c["id"] in ("config", "ssh", "slurm", "container")
    )
    latest = packed[0] if packed else None
    gpus = (gpu or {}).get("gpus") if isinstance(gpu, dict) else None
    free_mib = None
    if isinstance(gpus, list) and gpus:
        free_mib = gpus[0].get("memory_free_mib")
    with _SINFO["lock"]:
        sinfo_body = _SINFO["body"]
        partitions_at = float(_SINFO["at"] or 0)
        partitions_error = _SINFO["error"]
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
        "gpu": gpus,
        "gpu_node": (gpu or {}).get("node") if isinstance(gpu, dict) else (slurm or {}).get("node"),
        "gpu_error": gpu_error,
        "gpu_free_mib": free_mib,
        "remote_status": (cached or {}).get("remote_status"),
        "repair": repair_job,
        "packed_jobs": packed,
        "current_packed": latest,
        "probing": inflight,
        "probed_at": probed_at or None,
        "probe_age_s": round(time.time() - probed_at, 1) if probed_at else None,
        "scripts": lrz_scripts(),
        "hold_hours": list(HOLD_HOURS),
        "max_hold_hours": MAX_HOLD_HOURS,
        "partition": status.get("partition") or DEFAULT_PARTITION,
        "hint": "Slurm is queried at most once per 30s while this page is visible. sinfo only on Review partitions.",
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


def srun_worker_command(cfg: dict, job_id: str) -> str:
    remote = remote_job_dir(cfg, job_id)
    image = cfg.get("container") or f"{cfg['workspace']}/containers/pytorch.sqsh"
    name = cfg.get("container_name") or "splat-repair"
    inner = (
        "export PYTHONPATH=/workspace/code/src${PYTHONPATH:+:$PYTHONPATH}; "
        # NGC PyTorch images pre-set a wide TORCH_CUDA_ARCH_LIST (sm_52–sm_90).
        # gsplat 1.5 uses labeled_partition, which will not compile for sm_52.
        "export TORCH_CUDA_ARCH_LIST=8.0; "
        "export MAX_JOBS=4; "
        f"python -m splat_explorer.repair_lrz --job-dir /workspace/inputs/{job_id}"
    )
    return (
        f"srun --jobid={shlex.quote(str(cfg['job_id']))} --overlap "
        f"--nodes=1 --ntasks=1 --cpus-per-task={int(cfg['cpus'])} --gres=gpu:1 "
        f"--container-image={shlex.quote(str(image))} "
        f"--container-name={shlex.quote(str(name))} "
        f"--container-mounts={shlex.quote(cfg['workspace'] + ':/workspace')} "
        f"bash -lc {shlex.quote(inner)}"
    )


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
    argv = ssh_argv(cfg, multiplex=mux) + [f"squeue --me --job={cfg['job_id']} -h -o %t"]
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
    state = (result.stdout or "").strip().split()[0] if result.stdout else ""
    if result.returncode != 0 or not state:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Could not query Slurm job {cfg['job_id']}: {err or 'empty squeue'}. "
            "Is eduVPN up? Is the allocation still running?"
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
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(argv[:8])}… {err}")
    return result


def sync_code_and_job(job_dir: Path, *, password: str | None = None) -> None:
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
    job_id = job_dir.name
    remote = f"{cfg['user']}@{cfg['host']}"
    ssh_e = rsync_ssh_cmd(cfg)
    code_src = Path(__file__).resolve().parents[2] / "src"
    if not code_src.is_dir():
        code_src = Path.cwd() / "src"
    write_status(job_dir, phase="rsync_up", message="Uploading scene + code to DSS…")
    _mux_run(ssh_argv(cfg, multiplex=True) + [
        f"mkdir -p {shlex.quote(remote_job_dir(cfg, job_id))} "
        f"{shlex.quote(cfg['workspace'] + '/code')} "
        f"{shlex.quote(cfg['workspace'] + '/outputs')} "
        f"{shlex.quote(cfg['workspace'] + '/logs')} "
        f"{shlex.quote(cfg['workspace'] + '/containers')}"
    ])
    _mux_run([
        "rsync", "-az", "--delete", "-e", ssh_e,
        f"{code_src}/", f"{remote}:{cfg['workspace']}/code/src/",
    ])
    pyproject = code_src.parent / "pyproject.toml"
    if pyproject.is_file():
        _mux_run([
            "rsync", "-az", "-e", ssh_e,
            str(pyproject), f"{remote}:{cfg['workspace']}/code/pyproject.toml",
        ])
    _mux_run([
        "rsync", "-az", "-e", ssh_e, "--exclude", STATUS_JSON,
        f"{job_dir}/", f"{remote}:{remote_job_dir(cfg, job_id)}/",
    ])
    write_status(job_dir, phase="srun", message="Running GSFix CUDA refine on the A100…")
    _mux_run(ssh_argv(cfg, multiplex=True) + [srun_worker_command(cfg, job_id)])
    write_status(job_dir, phase="rsync_down", message="Downloading repaired splat…")
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
    densify: bool = True
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
    white_background: bool = False
    max_chunks: int = 1
    on_progress: Callable[[dict], None] | None = None
    should_stop: Callable[[], bool] | None = None

    def _params(self) -> dict:
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
            "packed": bool(self.packed),
            "white_background": bool(self.white_background),
            "max_chunks": int(self.max_chunks),
        }
        if str(self.method) in ("gsfix-gsplat-visprune", "visprune"):
            params.update(
                freeze_occluded=True,
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
        job_dir = pack_refine_job(
            scene, camera, rendered_rgb, repaired_rgb, params=self._params(),
        )
        password = get_ssh_password() or os.environ.get("LRZ_SSH_PASSWORD")
        if self.on_progress is not None:
            self.on_progress({"phase": "rsync_up", "n_iters": 0, "n_updated": 0})
        if lrz_session_alive():
            sync_code_and_job(job_dir)
        elif password:
            sync_code_and_job(job_dir, password=password)
        else:
            raise RuntimeError(session_required_message())
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
        last: dict[str, Any] | None = None
        total_iters = 0
        l1_before = None
        chunk = 0
        self.should_stop = should_stop
        self.on_progress = on_checkpoint
        password = get_ssh_password() or os.environ.get("LRZ_SSH_PASSWORD")
        once = not lrz_session_alive() and not bool(password)
        limit = int(self.max_chunks)
        saved_densify = self.densify
        visprune = str(self.method) in ("gsfix-gsplat-visprune", "visprune")
        try:
            while True:
                if should_stop is not None and should_stop():
                    break
                if deadline is not None and time.time() >= deadline:
                    break
                if limit > 0 and chunk >= limit:
                    break
                last = self.apply(scene, camera, rendered_rgb, repaired_rgb)
                chunk += 1
                if visprune:
                    self.densify = False
                if l1_before is None:
                    l1_before = last.get("l1_before")
                total_iters += int(last.get("n_iters") or 0)
                last = dict(last)
                last["n_iters"] = total_iters
                last["n_chunks"] = chunk
                last["n_stamped"] = int(last.get("n_stamped") or 0)
                last["l1_before"] = l1_before
                if on_checkpoint is not None:
                    on_checkpoint(last)
                if once:
                    break
        finally:
            self.densify = saved_densify
        if last is None:
            return {
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
        return last


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="splat_explorer.repair_lrz")
    parser.add_argument("--job-dir", required=True, help="Packed job directory (local or /workspace/inputs/id)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("LRZ_CUDA_ARCH") or "8.0"
    stats = apply_packed_job(Path(args.job_dir))
    logger.info("repair-job done: %s", {k: v for k, v in stats.items() if k != "render_rgb"})


if __name__ == "__main__":
    main()
