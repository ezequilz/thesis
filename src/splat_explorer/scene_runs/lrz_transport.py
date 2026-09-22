"""Local DSS/SSH transport for the persistent LRZ scene-run GPU worker."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .. import repair_lrz
from .gpu_worker import (
    CHECKPOINT_NAME,
    DEPTH_NAME,
    HEARTBEAT_NAME,
    METRICS_NAME,
    REGENERATED_NAME,
    RENDERED_NAME,
    REQUEST_NAME,
    RESPONSE_NAME,
    STOP_NAME,
    WORKER_NAME,
    atomic_write_bytes,
    atomic_write_json,
)
from .models import parse_slurm_duration, parse_slurm_end

REMOTE_RUNS_NAME = "scene-runs"
STARTUP_TIMEOUT_SECONDS = 300.0
RESPONSE_GRACE_SECONDS = 45.0
ALLOCATION_SAFETY_SECONDS = 300.0
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _safe_id(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-'")
    return text


def remote_scene_run_dir(cfg: dict[str, Any], run_id: str) -> str:
    run = _safe_id(run_id, label="run_id")
    return f"{str(cfg['workspace']).rstrip('/')}/{REMOTE_RUNS_NAME}/{run}"


def active_job_squeue_command(job_id: str) -> str:
    job = str(job_id or "").strip()
    if not job.isdigit():
        raise ValueError("LRZ job_id must be numeric")
    return f"squeue --me --job={job} -h -o {shlex.quote(repair_lrz.SQUEUE_FORMAT)}"


def scene_worker_srun_command(
    cfg: dict[str, Any],
    run_id: str,
    *,
    overall_deadline: float | None = None,
) -> str:
    """Generate the long-lived srun command inside the existing Pyxis image."""
    run = _safe_id(run_id, label="run_id")
    args = (
        "python -u -m splat_explorer.scene_runs.gpu_worker "
        f"--run-dir /workspace/{REMOTE_RUNS_NAME}/{run}"
    )
    if overall_deadline not in (None, 0, 0.0):
        args += f" --overall-deadline {float(overall_deadline):.6f}"
    inner = repair_lrz._remote_pythonpath_exports(cfg) + args
    return repair_lrz.container_srun_prefix(cfg) + f"bash -lc {shlex.quote(inner)}"


def scene_worker_launch_command(
    cfg: dict[str, Any],
    run_id: str,
    *,
    overall_deadline: float | None = None,
) -> str:
    """Generate a detached login-node command; stdout is the launcher PID."""
    remote_dir = remote_scene_run_dir(cfg, run_id)
    log = f"{remote_dir}/worker.log"
    pid = f"{remote_dir}/launcher.pid"
    srun = scene_worker_srun_command(
        cfg, run_id, overall_deadline=overall_deadline,
    )
    return (
        f"mkdir -p {shlex.quote(remote_dir + '/requests')}; "
        f"if [ -f {shlex.quote(pid)} ] && "
        f"kill -0 \"$(cat {shlex.quote(pid)})\" 2>/dev/null; then "
        f"cat {shlex.quote(pid)}; else "
        f"rm -f {shlex.quote(remote_dir + '/' + STOP_NAME)} "
        f"{shlex.quote(remote_dir + '/' + WORKER_NAME)} "
        f"{shlex.quote(remote_dir + '/' + HEARTBEAT_NAME)}; "
        f"nohup bash -lc {shlex.quote(srun)} "
        f"> {shlex.quote(log)} 2>&1 < /dev/null & "
        f"launcher=$!; printf '%s\\n' \"$launcher\" > {shlex.quote(pid)}; "
        "printf '%s\\n' \"$launcher\"; fi"
    )


def request_protocol_body(
    *,
    request_id: str,
    step: int,
    camera,
    repair_seconds: float,
    deadline: float,
    prompt: str | None = None,
    repair: dict[str, Any] | None = None,
    operation: str = "repair",
) -> dict[str, Any]:
    """Build the JSON request consumed by :mod:`gpu_worker`."""
    from ..repair_lrz import camera_to_dict
    from .gpu_worker import DEFAULT_PROMPT

    camera_body = camera_to_dict(camera) if not isinstance(camera, dict) else dict(camera)
    body: dict[str, Any] = {
        "protocol": 1,
        "operation": str(operation),
        "request_id": _safe_id(request_id, label="request_id"),
        "step": int(step),
        "camera": camera_body,
        "repair_seconds": float(repair_seconds),
        "deadline_unix": float(deadline),
        "created_at": time.time(),
    }
    if str(operation) == "repair":
        body["prompt"] = str(prompt or DEFAULT_PROMPT)
    if repair:
        body["repair"] = dict(repair)
    return body


def allocation_deadline_unix(
    row: dict[str, Any],
    *,
    now: float | None = None,
    safety_seconds: float = ALLOCATION_SAFETY_SECONDS,
) -> float | None:
    """Convert Slurm expected-end/time-left into a conservative Unix deadline."""
    current = time.time() if now is None else float(now)
    expected = parse_slurm_end(row.get("expected_end"))
    if expected is not None:
        return expected.timestamp() - float(safety_seconds)
    remaining = parse_slurm_duration(row.get("time_left"))
    if remaining is not None:
        return current + remaining.total_seconds() - float(safety_seconds)
    return None


class LrzSceneRunTransport:
    """Stage once, then exchange atomic request directories through DSS."""

    def __init__(
        self,
        app_cfg,
        run_id: str,
        run_dir: Path,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        poll_seconds: float = 2.0,
    ):
        self.app_cfg = app_cfg
        self.run_id = _safe_id(run_id, label="run_id")
        self.run_dir = Path(run_dir)
        self.on_progress = on_progress
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.cfg = repair_lrz.load_lrz_config()
        self.remote_dir = remote_scene_run_dir(self.cfg, self.run_id)
        self._effective_deadline: float | None = None
        self._started = False
        self._run_config: dict[str, Any] = {}

    def _progress(self, phase: str, message: str, **fields: Any) -> None:
        if self.on_progress is not None:
            self.on_progress({"phase": phase, "message": message, **fields})

    def _require_session(self) -> None:
        if not repair_lrz.lrz_configured():
            raise RuntimeError("LRZ is not configured for this scene-run")
        if not repair_lrz.lrz_session_alive(self.cfg):
            raise RuntimeError(repair_lrz.session_required_message())

    def _remote_json(self, name: str) -> dict[str, Any] | None:
        path = f"{self.remote_dir}/{name}"
        result = repair_lrz._ssh_run(
            self.cfg,
            f"if [ -f {shlex.quote(path)} ]; then cat {shlex.quote(path)}; fi",
            timeout=20,
        )
        if result.returncode != 0 or not (result.stdout or "").strip():
            return None
        try:
            body = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return body if isinstance(body, dict) else None

    def validate(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate one selected running allocation and its setup marker."""
        self._require_session()
        job_id = str(self.cfg.get("job_id") or "").strip()
        result = repair_lrz._ssh_run(
            self.cfg, active_job_squeue_command(job_id), timeout=25,
        )
        row = repair_lrz.parse_squeue_line(result.stdout or "")
        if result.returncode != 0 or row is None:
            detail = (result.stderr or result.stdout or "empty squeue").strip()
            raise RuntimeError(f"Could not query selected LRZ job {job_id}: {detail}")
        if row.get("state") != "R":
            raise RuntimeError(
                f"Selected LRZ job {job_id} is {row.get('state') or 'missing'}, not R"
            )
        repair_lrz.remember_live_allocation(
            job_id=job_id,
            state=str(row.get("state") or ""),
            mem=str(row.get("mem") or ""),
            partition=str(row.get("partition") or ""),
            node=str(row.get("node") or ""),
        )
        marker = repair_lrz.read_remote_setup_marker(self.cfg)
        matches, reason = repair_lrz.setup_matches_allocation(
            marker, job_id=job_id, slurm=row,
        )
        if not matches:
            raise RuntimeError(reason or "LRZ GPU setup marker is missing")
        if (
            isinstance(config, dict)
            and self._qwen_required(config)
            and marker.get("image_edit_ready") is not True
        ):
            raise RuntimeError(
                "This scene-run uses Qwen-Image-Edit, but GPU setup was loaded "
                "with QWEN_required=false. Turn QWEN_required on and click "
                "Reload GPU setup."
            )
        if isinstance(config, dict):
            wants_extended = config.get("pipeline") == "extended"
            mode = marker.get("setup_mode", "baseline")
            if wants_extended and (mode != "artifixer" or not marker.get("artifixer_ready")):
                raise RuntimeError("Reload GPU setup with ArtiFixer enabled on /scene-runs-ext or /repair/gpu first")
            if not wants_extended and mode != "baseline":
                raise RuntimeError("Reload GPU setup with ArtiFixer disabled before starting a baseline run")
        return {"allocation": row, "setup": marker}

    def _image_backend(self, config: dict[str, Any] | None = None) -> str:
        from ..image_edit import SCENE_RUN_GPT_IMAGE_MODEL, canonical_image_edit_choice

        body = config if isinstance(config, dict) else self._run_config
        raw = body.get("image_edit_backend") or SCENE_RUN_GPT_IMAGE_MODEL
        return canonical_image_edit_choice(str(raw))

    def _qwen_required(self, config: dict[str, Any] | None = None) -> bool:
        from ..image_edit import is_qwen_backend, resolve_qwen_required

        body = config if isinstance(config, dict) else self._run_config
        backend = self._image_backend(body)
        explicit = body.get("QWEN_required", body.get("qwen_required"))
        if explicit is None:
            explicit = True if is_qwen_backend(backend) else False
        return resolve_qwen_required(explicit, backend=backend)

    def _worker_config(self, config: dict[str, Any]) -> dict[str, Any]:
        from ..image_edit import BACKEND_QWEN, concrete_gpt_image_model, is_qwen_backend

        image_edit: dict[str, Any] = {}
        try:
            image_edit = dict(self.app_cfg.get("image_edit") or {})
        except (AttributeError, TypeError, ValueError):
            pass
        choice = self._image_backend(config)
        qwen = is_qwen_backend(choice)
        if qwen:
            image_edit["backend"] = BACKEND_QWEN
            # Starting a Qwen scene-run explicitly authorizes the one-time model
            # fetch. Subsequent runs reuse the DSS-backed Hugging Face cache.
            image_edit["download"] = True
        else:
            image_edit["backend"] = "gpt-image-2"
            image_edit["model"] = concrete_gpt_image_model(choice) or choice
            image_edit["download"] = False
        image_edit["QWEN_required"] = qwen
        repair = dict(config.get("repair") or {})
        repair_type = str(
            repair.get("repair_type")
            or config.get("repair_type")
            or "original"
        ).strip().lower()
        if repair_type in {"looped", "loop"}:
            repair["repair_type"] = "looped"
            repair["densify"] = True
            repair["max_chunks"] = 0
        else:
            repair["repair_type"] = "original"
            repair["iters"] = 20
            repair["max_chunks"] = 1
            repair["densify"] = True
            repair["upstream_gsfix3d"] = True
        return {
            "protocol": 1,
            "run_id": self.run_id,
            "scene_run": dict(config),
            "extended_runtime": dict(self.app_cfg.get("scene_runs_ext") or {}),
            "image_edit": image_edit,
            "image_edit_prompt": config.get("image_edit_prompt"),
            "QWEN_required": qwen,
            # LRZ steps commonly have 62 GiB host memory. Process exit is the
            # only reliable way to reclaim Qwen loading buffers before GSFix.
            # CliRelay edits never enter this process.
            "image_edit_subprocess": qwen,
            "repair": repair,
        }

    def _stage(self, source_ply: Path, config: dict[str, Any]) -> None:
        source = Path(source_ply)
        if not source.is_file():
            raise FileNotFoundError(f"scene-run source PLY missing: {source}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.run_dir / "worker_config.json", self._worker_config(config))
        repair_lrz.sync_code_to_dss(self.cfg)
        remote = f"{self.cfg['user']}@{self.cfg['host']}"
        ssh_e = repair_lrz.rsync_ssh_cmd(self.cfg)
        repair_lrz._mux_run(
            repair_lrz.ssh_argv(self.cfg, multiplex=True)
            + [f"mkdir -p {shlex.quote(self.remote_dir + '/requests')}"]
        )
        repair_lrz._mux_run([
            "rsync", "-az", "-e", ssh_e,
            "--exclude", f"/{STOP_NAME}",
            "--exclude", "/requests/",
            "--exclude", f"/{HEARTBEAT_NAME}",
            "--exclude", f"/{WORKER_NAME}",
            "--exclude", "/launcher.pid",
            "--exclude", "/worker.log",
            f"{self.run_dir}/",
            f"{remote}:{self.remote_dir}/",
        ])
        repair_lrz._mux_run([
            "rsync", "-az", "-e", ssh_e,
            str(source),
            f"{remote}:{self.remote_dir}/scene.ply",
        ])

    def _wait_ready(self, timeout: float = STARTUP_TIMEOUT_SECONDS) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            last = self._remote_json(WORKER_NAME)
            if last and last.get("status") == "ready":
                return last
            if last and last.get("status") == "error":
                raise RuntimeError(last.get("error") or "LRZ scene-run worker failed")
            time.sleep(self.poll_seconds)
        raise RuntimeError(
            f"Timed out waiting for LRZ scene-run worker {self.run_id}: {last}"
        )

    def start(
        self,
        *,
        source_ply: Path,
        config: dict[str, Any],
        deadline: float,
    ) -> dict[str, Any]:
        self._progress("gpu_validate", "Validating reserved LRZ GPU")
        validated = self.validate(config)
        allocation_deadline = allocation_deadline_unix(validated["allocation"])
        effective = float(deadline)
        if allocation_deadline is not None:
            effective = min(effective, allocation_deadline)
        if effective <= time.time():
            raise RuntimeError("Selected LRZ allocation has no safe scene-run time left")
        self._effective_deadline = effective
        self._run_config = dict(config)
        self._progress("gpu_stage", "Staging scene-run on DSS")
        self._stage(Path(source_ply), self._run_config)
        command = scene_worker_launch_command(
            self.cfg, self.run_id, overall_deadline=effective,
        )
        result = repair_lrz._ssh_run(self.cfg, command, timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Could not launch LRZ scene-run worker: {detail}")
        self._progress("gpu_start", "Waiting for persistent GPU worker")
        ready = self._wait_ready()
        self._started = True
        return {
            "status": "ready",
            "job_id": str(self.cfg.get("job_id") or ""),
            "run_id": self.run_id,
            "launcher_pid": (result.stdout or "").strip() or None,
            "effective_deadline": effective,
            "allocation": validated["allocation"],
            "setup": validated["setup"],
            "worker": ready,
        }

    def _push_request(self, request_id: str, local_dir: Path) -> None:
        remote = f"{self.cfg['user']}@{self.cfg['host']}"
        ssh_e = repair_lrz.rsync_ssh_cmd(self.cfg)
        incoming = f"{self.remote_dir}/requests/.incoming-{request_id}"
        target = f"{self.remote_dir}/requests/{request_id}"
        repair_lrz._ssh_run(
            self.cfg,
            f"rm -rf {shlex.quote(incoming)}; mkdir -p {shlex.quote(incoming)}",
            timeout=20,
        )
        repair_lrz._mux_run([
            "rsync", "-az", "-e", ssh_e,
            f"{local_dir}/", f"{remote}:{incoming}/",
        ])
        result = repair_lrz._ssh_run(
            self.cfg,
            (
                f"if [ -d {shlex.quote(target)} ]; then "
                f"rm -rf {shlex.quote(incoming)}; "
                f"else mv {shlex.quote(incoming)} {shlex.quote(target)}; fi"
            ),
            timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "request publish failed").strip()
            )

    def _pull_request(
        self, request_id: str, local_dir: Path, *, checkpoint: bool = True,
    ) -> None:
        remote = f"{self.cfg['user']}@{self.cfg['host']}"
        ssh_e = repair_lrz.rsync_ssh_cmd(self.cfg)
        request_remote = f"{remote}:{self.remote_dir}/requests/{request_id}/"
        repair_lrz._mux_run([
            "rsync", "-az", "-e", ssh_e,
            "--include", "extended/",
            "--include", "extended/bundle.json",
            "--include", "extended/inference.json",
            "--include", "extended/artifixer.log",
            "--include", "extended/anchor.png",
            "--include", "extended/targets/***",
            "--include", REGENERATED_NAME,
            "--include", RENDERED_NAME,
            "--include", DEPTH_NAME,
            "--include", METRICS_NAME,
            "--include", RESPONSE_NAME,
            "--exclude", "*",
            request_remote,
            f"{local_dir}/",
        ])
        if checkpoint:
            self.pull_checkpoint()

    def _wait_response(
        self,
        request_id: str,
        *,
        deadline: float,
        should_stop: Callable[[], bool],
    ) -> dict[str, Any]:
        response_path = f"{self.remote_dir}/requests/{request_id}/{RESPONSE_NAME}"
        wait_until = float(deadline) + RESPONSE_GRACE_SECONDS
        stop_sent = False
        while time.time() < wait_until:
            if should_stop() and not stop_sent:
                self.stop()
                stop_sent = True
            result = repair_lrz._ssh_run(
                self.cfg,
                (
                    f"if [ -f {shlex.quote(response_path)} ]; "
                    f"then cat {shlex.quote(response_path)}; fi"
                ),
                timeout=20,
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                try:
                    parsed = json.loads(result.stdout)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
            heartbeat = self._remote_json(HEARTBEAT_NAME)
            if heartbeat:
                if heartbeat.get("status") == "stopped":
                    detail = heartbeat.get("error") or heartbeat.get("phase") or "stopped"
                    raise RuntimeError(f"LRZ scene-run worker stopped: {detail}")
                self._progress(
                    str(heartbeat.get("phase") or "gpu"),
                    f"GPU worker: {heartbeat.get('phase') or 'running'}",
                    request_id=request_id,
                    heartbeat=heartbeat,
                )
            time.sleep(self.poll_seconds)
        raise RuntimeError(f"Timed out waiting for LRZ GPU response for {request_id}")

    def render(
        self,
        *,
        step: int,
        camera,
        deadline: float,
        should_stop: Callable[[], bool],
    ):
        """Render on LRZ so scene-runs continue without an open browser tab."""
        if not self._started:
            raise RuntimeError("LRZ scene-run worker has not been started")
        request_id = f"render-{int(step):05d}"
        local_dir = self.run_dir / "requests" / request_id
        local_dir.mkdir(parents=True, exist_ok=True)
        request_deadline = min(
            float(deadline),
            self._effective_deadline if self._effective_deadline is not None else float(deadline),
        )
        body = request_protocol_body(
            request_id=request_id,
            step=step,
            camera=camera,
            repair_seconds=1.0,
            deadline=request_deadline,
            operation="render",
        )
        atomic_write_json(local_dir / REQUEST_NAME, body)
        self._progress("render_upload", f"Requesting GPU render for step {step}")
        self._push_request(request_id, local_dir)
        response = self._wait_response(
            request_id,
            deadline=request_deadline,
            should_stop=should_stop,
        )
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error") or "GPU render failed")
        self._pull_request(request_id, local_dir, checkpoint=False)
        from PIL import Image
        import numpy as np

        rgb = np.asarray(
            Image.open(local_dir / RENDERED_NAME).convert("RGB"), dtype=np.uint8,
        )
        depth = np.load(local_dir / DEPTH_NAME).astype(np.float32)
        return rgb, depth

    def pull_checkpoint(self) -> Path:
        """Pull only the cumulative PLY, never re-uploading the scene."""
        remote = f"{self.cfg['user']}@{self.cfg['host']}"
        repair_lrz._mux_run([
            "rsync", "-az", "-e", repair_lrz.rsync_ssh_cmd(self.cfg),
            f"{remote}:{self.remote_dir}/{CHECKPOINT_NAME}",
            str(self.run_dir / CHECKPOINT_NAME),
        ])
        return self.run_dir / CHECKPOINT_NAME

    def repair(
        self,
        *,
        step: int,
        camera,
        rendered_path: Path,
        repair_seconds: float,
        deadline: float,
        should_stop: Callable[[], bool],
        prompt: str | None = None,
        proposal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("LRZ scene-run worker has not been started")
        request_id = f"repair-{int(step):05d}"
        local_dir = self.run_dir / "requests" / request_id
        local_dir.mkdir(parents=True, exist_ok=True)
        source = Path(rendered_path)
        if not source.is_file():
            raise FileNotFoundError(f"rendered scene-run frame missing: {source}")
        destination = local_dir / RENDERED_NAME
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        prompt_text = str(
            prompt or self._run_config.get("image_edit_prompt") or ""
        )
        image_edit_result = None
        if not self._qwen_required(self._run_config):
            self._progress(
                "image_edit",
                f"CliRelay image edit for step {step}",
            )
            image_edit_result = self._edit_with_clirelay(
                destination,
                local_dir / REGENERATED_NAME,
                prompt_text,
            )
        request_deadline = min(
            float(deadline),
            self._effective_deadline if self._effective_deadline is not None else float(deadline),
        )
        repair_cfg = dict(self._run_config.get("repair") or {})
        body = request_protocol_body(
            request_id=request_id,
            step=step,
            camera=camera,
            repair_seconds=repair_seconds,
            deadline=request_deadline,
            prompt=prompt_text or None,
            repair=repair_cfg,
            operation="repair",
        )
        if proposal is not None:
            body["proposal"] = dict(proposal)
        if image_edit_result is not None:
            body["skip_image_edit"] = True
            body["image_model"] = image_edit_result.get("model")
            body["image_edit_result"] = image_edit_result
        atomic_write_json(local_dir / REQUEST_NAME, body)
        self._progress("request_upload", f"Uploading GPU repair request for step {step}")
        self._push_request(request_id, local_dir)

        response = self._wait_response(
            request_id,
            deadline=request_deadline,
            should_stop=should_stop,
        )
        self._pull_request(request_id, local_dir)
        local_response = local_dir / RESPONSE_NAME
        if local_response.is_file():
            try:
                response = json.loads(local_response.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        response = dict(response)
        response["checkpoint"] = str(self.run_dir / CHECKPOINT_NAME)
        response["repaired_ply"] = response["checkpoint"]
        response["regenerated_path"] = str(local_dir / REGENERATED_NAME)
        metrics_path = local_dir / METRICS_NAME
        if metrics_path.is_file():
            try:
                response["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return response

    def _edit_with_clirelay(
        self,
        image_path: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        """Run gpt-image-2 / 2.5 on the machine that can reach CliRelay."""
        from ..image_edit import make_image_edit_backend

        backend = self._image_backend(self._run_config)
        editor = make_image_edit_backend(backend, cfg=self.app_cfg)
        result = editor.edit(Path(image_path), prompt)
        images = list(getattr(result, "images", None) or [])
        error = getattr(result, "error", None)
        if error or not images:
            raise RuntimeError(error or f"{backend} image edit returned no PNG")
        atomic_write_bytes(output_path, images[0])
        payload = dict(getattr(result, "payload", None) or {})
        payload["external"] = True
        payload["backend"] = getattr(editor, "name", backend)
        payload["model"] = getattr(editor, "model", backend)
        payload["prompt"] = prompt
        return payload

    def stop(self) -> None:
        """Request graceful checkpoint/exit through the DSS STOP marker."""
        result = repair_lrz._ssh_run(
            self.cfg,
            f"touch {shlex.quote(self.remote_dir + '/' + STOP_NAME)}",
            timeout=20,
        )
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "could not stop LRZ worker").strip()
            )

    def close(self) -> None:
        """Best-effort final checkpoint pull; the worker owns its deadline."""
        if not self._started:
            return
        # Release the GPU worker before the manager releases its shared lease.
        # The next setup/run must not overlap an idle worker retaining tensors.
        self.stop()
        until = time.monotonic() + 30.0
        while time.monotonic() < until:
            worker = self._remote_json(WORKER_NAME) or {}
            if worker.get("status") in {"stopped", "error"}:
                break
            time.sleep(self.poll_seconds)
        else:
            raise RuntimeError("GPU worker has not exited yet; wait before reloading setup")
        try:
            self.pull_checkpoint()
        except Exception:
            # A run can close before the first repair, so no checkpoint may exist.
            pass
