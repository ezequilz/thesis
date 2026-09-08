"""Ship a GSFix3D CUDA refine job to an allocated LRZ GPU.

Open ``scripts/lrz/ssh-session.sh`` once (type the LRZ password). Later rsync
and srun reuse ``~/.ssh/cm-lrz``. If that socket is missing, apply() errors
with “open the LRZ SSH session first”.

Optional fallbacks: repair-page password field (SSH_ASKPASS for that run) or
``scripts/lrz/run-repair.sh <job-id>``.

The A100 worker is :func:`apply_packed_job` (``python -m splat_explorer.repair_lrz``).
"""

from __future__ import annotations

import json
import logging
import os
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
            "ssh", "-o", f"ControlPath={sock}", "-O", "check",
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
    alive = lrz_session_alive(cfg) if lrz_configured() else False
    return {
        "configured": lrz_configured(),
        "session": alive,
        "user": cfg["user"],
        "host": cfg["host"],
        "job_id": cfg["job_id"],
        "workspace": cfg["workspace"],
        "container": cfg.get("container"),
        "control_path": str(control_path()),
        "session_script": "scripts/lrz/ssh-session.sh",
        "run_script": "scripts/lrz/run-repair.sh",
    }


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


def apply_packed_job(job_dir: Path, backend=None) -> dict[str, Any]:
    """CUDA (or injected) refine inside a packed job directory. Mutates scene on disk."""
    job_dir = Path(job_dir)
    camera = camera_from_dict(json.loads((job_dir / CAMERA_JSON).read_text()))
    params = json.loads((job_dir / PARAMS_JSON).read_text())
    scene = load_ply(job_dir / SCENE_PLY)
    rendered = np.asarray(Image.open(job_dir / RENDERED_PNG).convert("RGB"), dtype=np.uint8)
    repaired = np.asarray(Image.open(job_dir / REPAIRED_PNG).convert("RGB"), dtype=np.uint8)
    if backend is None:
        from .repair_gsfix import GsplatPhotometricRepair

        allowed = {
            "iters", "lambda_dssim", "densify", "densify_every",
            "densify_grad_thresh", "prune_opacity", "max_clone", "max_gaussians",
            "lr_means", "lr_colors", "lr_opacities", "lr_scales", "lr_quats",
            "near", "packed",
        }
        kwargs = {k: params[k] for k in allowed if k in params}
        backend = GsplatPhotometricRepair(**kwargs)
    stats = backend.apply(scene, camera, rendered, repaired)
    save_ply(scene, job_dir / OUT_PLY)
    render_rgb = stats.get("render_rgb")
    if render_rgb is not None:
        Image.fromarray(np.asarray(render_rgb, dtype=np.uint8)).save(job_dir / OUT_RENDER)
    metrics = {k: v for k, v in stats.items() if k != "render_rgb"}
    metrics["job_id"] = job_dir.name
    _write_json(job_dir / METRICS_JSON, metrics)
    write_status(job_dir, phase="done", message="CUDA refine finished.")
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


def ssh_argv(cfg: dict | None = None, *, multiplex: bool | None = None) -> list[str]:
    """SSH argv. Default: reuse ControlMaster at ``control_path()``."""
    cfg = cfg or load_lrz_config()
    target = f"{cfg['user']}@{cfg['host']}"
    if multiplex is None:
        multiplex = lrz_session_alive(cfg)
    if multiplex:
        return [
            "ssh", "-4", "-F", "/dev/null",
            "-o", "ControlMaster=no",
            "-o", f"ControlPath={control_path()}",
            target,
        ]
    return [
        "ssh", "-4", "-F", "/dev/null",
        "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password",
        "-o", "NumberOfPasswordPrompts=3",
        "-o", "StrictHostKeyChecking=accept-new",
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
        f"python -m splat_explorer.repair_lrz --job-dir /workspace/inputs/{job_id}"
    )
    return (
        f"srun --jobid={shlex.quote(str(cfg['job_id']))} "
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
    argv = ssh_argv(cfg, multiplex=mux) + [f"squeue --me --job={cfg['job_id']} -h -o %T"]
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
    """Same stats contract as GsplatPhotometricRepair, executed on LRZ."""

    iters: int = 20
    lambda_dssim: float = 0.2
    densify: bool = True
    densify_every: int = 5
    densify_grad_thresh: float = 0.0002
    prune_opacity: float = 0.005
    max_clone: int = 2048
    max_gaussians: int = 2_500_000
    lr_means: float = 1.6e-4
    lr_colors: float = 0.0025
    lr_opacities: float = 0.05
    lr_scales: float = 0.005
    lr_quats: float = 0.001
    near: float = 0.05
    packed: bool = False
    on_progress: Callable[[dict], None] | None = None
    should_stop: Callable[[], bool] | None = None

    def _params(self) -> dict:
        return {
            "iters": int(self.iters),
            "lambda_dssim": float(self.lambda_dssim),
            "densify": bool(self.densify),
            "densify_every": int(self.densify_every),
            "densify_grad_thresh": float(self.densify_grad_thresh),
            "prune_opacity": float(self.prune_opacity),
            "max_clone": int(self.max_clone),
            "max_gaussians": int(self.max_gaussians),
            "lr_means": float(self.lr_means),
            "lr_colors": float(self.lr_colors),
            "lr_opacities": float(self.lr_opacities),
            "lr_scales": float(self.lr_scales),
            "lr_quats": float(self.lr_quats),
            "near": float(self.near),
            "packed": bool(self.packed),
        }

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
        self.should_stop = should_stop
        self.on_progress = on_checkpoint
        password = get_ssh_password() or os.environ.get("LRZ_SSH_PASSWORD")
        once = not lrz_session_alive() and not bool(password)
        while True:
            if should_stop is not None and should_stop():
                break
            if deadline is not None and time.time() >= deadline:
                break
            last = self.apply(scene, camera, rendered_rgb, repaired_rgb)
            total_iters += int(last.get("n_iters") or 0)
            last = dict(last)
            last["n_iters"] = total_iters
            if on_checkpoint is not None:
                on_checkpoint(last)
            if once:
                break
        if last is None:
            return {
                "backend": "gsfix-gsplat",
                "n_visible": scene.num_gaussians,
                "n_updated": 0,
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
    stats = apply_packed_job(Path(args.job_dir))
    logger.info("repair-job done: %s", {k: v for k, v in stats.items() if k != "render_rgb"})


if __name__ == "__main__":
    main()
