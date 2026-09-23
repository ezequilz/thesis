"""Persistent file-queue GPU worker for one LRZ scene-run.

The worker is intentionally transport-agnostic: DSS directories are the
protocol, and the login-node process only copies files and writes STOP markers.
CUDA, Qwen, and gsplat imports stay lazy so protocol tests run without a GPU.
CliRelay image edits happen off this GPU; Qwen loads only when QWEN_required
is true.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

SCENE_NAME = "scene.ply"
CHECKPOINT_NAME = "scene_repaired.ply"
REQUESTS_NAME = "requests"
REQUEST_NAME = "request.json"
RENDERED_NAME = "rendered.png"
DEPTH_NAME = "depth.npy"
REGENERATED_NAME = "regenerated.png"
METRICS_NAME = "metrics.json"
IMAGE_EDIT_RESULT_NAME = "image_edit.json"
RESPONSE_NAME = "response.json"
STOP_NAME = "STOP"
HEARTBEAT_NAME = "heartbeat.json"
WORKER_NAME = "worker.json"
DEFAULT_REPAIR_SECONDS = 180.0
ORIGINAL_REPAIR_TYPE = "original"
LOOPED_REPAIR_TYPE = "looped"
REPAIR_TYPE_ALIASES = {
    "original": ORIGINAL_REPAIR_TYPE,
    "original_gsfix3d": ORIGINAL_REPAIR_TYPE,
    "gsfix3d": ORIGINAL_REPAIR_TYPE,
    "paper": ORIGINAL_REPAIR_TYPE,
    "looped": LOOPED_REPAIR_TYPE,
    "loop": LOOPED_REPAIR_TYPE,
}
# https://github.com/GSFix3D/GSFix3D/blob/main/scripts/gsfix3d/refine_gs.py
# --iters 20, 0.8 L1 + 0.2 SSIM, densify every 5, prune last iter.
# LRs from gs/arguments.py OptimizationParams (position_lr_init=0.00032).
UPSTREAM_GSFIX3D = {
    "iters": 20,
    "max_chunks": 1,
    "densify": True,
    "densify_every": 5,
    "densify_grad_thresh": 0.005,
    "prune_opacity": 0.005,
    "split_scale": 0.1,
    "lr_means": 0.00032,
    "lr_colors": 0.0025,
    "lr_opacities": 0.025,
    "lr_scales": 0.002,
    "lr_quats": 0.001,
    "lambda_dssim": 0.2,
    "lambda_preserve": 0.0,
    "lambda_color_bound": 0.0,
    "lambda_color_reg": 0.0,
    "min_fix_weight": 1.0,
    "sh_clip": 0.0,
    "freeze_geometry_after_first_chunk": False,
    "upstream_gsfix3d": True,
}
DEFAULT_PROMPT = (
    "Regenerate and fix this image. Repair artifacts, reconstruct plausible "
    "geometry and upscale to higher resolution."
)

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def atomic_write_json(path: Path, body: dict[str, Any]) -> None:
    """Write a JSON completion marker without exposing partial contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(body), indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def camera_payload(request: dict[str, Any]) -> dict[str, Any]:
    """Return camera fields from nested or flat request protocol versions."""
    nested = request.get("camera")
    if isinstance(nested, dict):
        return nested
    required = ("position", "rotation", "width", "height", "fov_deg")
    missing = [key for key in required if key not in request]
    if missing:
        raise ValueError(f"request camera is missing: {', '.join(missing)}")
    return {key: request[key] for key in required}


def bounded_repair_deadline(
    request: dict[str, Any],
    *,
    now: float | None = None,
    overall_deadline: float | None = None,
) -> float:
    """Bound GSFix time by repair_seconds and all absolute deadlines."""
    current = time.time() if now is None else float(now)
    seconds = float(request.get("repair_seconds") or DEFAULT_REPAIR_SECONDS)
    candidates = [current + max(0.0, seconds)]
    request_deadline = request.get("deadline_unix", request.get("deadline"))
    if request_deadline not in (None, ""):
        candidates.append(float(request_deadline))
    if overall_deadline not in (None, 0, 0.0):
        candidates.append(float(overall_deadline))
    return min(candidates)


def pending_request_dirs(run_dir: Path) -> list[Path]:
    """Ready, unanswered requests in deterministic directory-name order."""
    root = Path(run_dir) / REQUESTS_NAME
    if not root.is_dir():
        return []
    return [
        path for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
        and not path.name.startswith(".")
        and (path / REQUEST_NAME).is_file()
        and not (path / RESPONSE_NAME).is_file()
    ]


def _default_scene_loader(path: Path):
    from ..scene import load_ply

    return load_ply(path)


def _default_scene_saver(scene, path: Path) -> None:
    from ..scene import save_ply

    save_ply(scene, path)


def qwen_required_for_worker(config: dict[str, Any] | None) -> bool:
    """True when this worker must run Qwen-Image-Edit on the GPU."""
    from ..image_edit import resolve_qwen_required

    body = config if isinstance(config, dict) else {}
    image_edit = body.get("image_edit") if isinstance(body.get("image_edit"), dict) else {}
    scene_run = body.get("scene_run") if isinstance(body.get("scene_run"), dict) else {}
    explicit = body.get("QWEN_required", body.get("qwen_required"))
    if explicit is None and isinstance(image_edit, dict):
        explicit = image_edit.get("QWEN_required", image_edit.get("qwen_required"))
    backend = (
        image_edit.get("backend")
        or image_edit.get("model")
        or scene_run.get("image_edit_backend")
        or body.get("image_edit_backend")
    )
    return resolve_qwen_required(explicit, cfg=body, backend=None if explicit is not None else backend)


def _default_image_edit_factory(config: dict[str, Any]):
    """Qwen when QWEN_required, otherwise the CliRelay GPT image backend."""
    if qwen_required_for_worker(config):
        from ..image_edit_qwen import QwenImageEditBackend

        return QwenImageEditBackend.from_config(config)
    from ..image_edit import make_image_edit_backend

    image_edit = config.get("image_edit") if isinstance(config.get("image_edit"), dict) else {}
    scene_run = config.get("scene_run") if isinstance(config.get("scene_run"), dict) else {}
    name = (
        image_edit.get("model")
        or scene_run.get("image_edit_backend")
        or image_edit.get("backend")
        or config.get("image_edit_backend")
    )
    return make_image_edit_backend(name, cfg=config)


def normalize_repair_type(value: Any) -> str:
    """Map dashboard / request aliases onto original vs looped refine."""
    key = str(value or ORIGINAL_REPAIR_TYPE).strip().lower()
    return REPAIR_TYPE_ALIASES.get(key, ORIGINAL_REPAIR_TYPE)


def _default_repair_factory(params: dict[str, Any]):
    # Scene-runs share the working-backup ADC implementation. "original"
    # locks the GitHub refine_gs.py 20-iter schedule; "looped" keeps the
    # time-budget repeat.
    from ..repair_gsfix3d_working_backup import GsplatGsfix3dRepair

    body = dict(params or {})
    kind = normalize_repair_type(body.get("repair_type"))
    allowed = {field.name for field in fields(GsplatGsfix3dRepair)}
    if kind == LOOPED_REPAIR_TYPE:
        kwargs = {key: value for key, value in body.items() if key in allowed}
        kwargs["densify"] = True
        kwargs["max_chunks"] = 0
        kwargs["upstream_gsfix3d"] = False
        return GsplatGsfix3dRepair(**kwargs)
    kwargs = {key: value for key, value in UPSTREAM_GSFIX3D.items() if key in allowed}
    return GsplatGsfix3dRepair(**kwargs)


def run_image_edit_once(request_dir: Path, config_path: Path) -> dict[str, Any]:
    """Run one image edit in a disposable process and persist its result."""
    request_dir = Path(request_dir)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    request = json.loads((request_dir / REQUEST_NAME).read_text(encoding="utf-8"))
    prompt = str(
        request.get("prompt")
        or config.get("image_edit_prompt")
        or DEFAULT_PROMPT
    )
    edit = _default_image_edit_factory(config).edit(
        request_dir / RENDERED_NAME, prompt,
    )
    images = list(getattr(edit, "images", None) or [])
    error = getattr(edit, "error", None)
    result = {
        "status": "error" if error or not images else "ok",
        "error": error or (None if images else "image edit returned no PNG"),
        "payload": _jsonable(getattr(edit, "payload", {}) or {}),
        "prompt": prompt,
    }
    if images:
        atomic_write_bytes(request_dir / REGENERATED_NAME, images[0])
    atomic_write_json(request_dir / IMAGE_EDIT_RESULT_NAME, result)
    return result


class SceneRunGpuWorker:
    """Own one cumulative scene and process its requests sequentially."""

    def __init__(
        self,
        run_dir: Path,
        *,
        overall_deadline: float | None = None,
        heartbeat_seconds: float = 10.0,
        scene_loader: Callable[[Path], Any] = _default_scene_loader,
        scene_saver: Callable[[Any, Path], None] = _default_scene_saver,
        image_edit_factory: Callable[[dict[str, Any]], Any] = _default_image_edit_factory,
        repair_factory: Callable[[dict[str, Any]], Any] = _default_repair_factory,
        clock: Callable[[], float] = time.time,
    ):
        self.run_dir = Path(run_dir)
        self.overall_deadline = (
            float(overall_deadline) if overall_deadline not in (None, 0, 0.0) else None
        )
        self.heartbeat_seconds = max(0.2, float(heartbeat_seconds))
        self.scene_loader = scene_loader
        self.scene_saver = scene_saver
        self.image_edit_factory = image_edit_factory
        self.repair_factory = repair_factory
        self.clock = clock
        self.scene = None
        self.extended_scale_ceiling = None
        self.renderer = None
        self.image_editor = None
        self._stop = threading.Event()
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._active_request: str | None = None
        self._phase = "starting"
        self.config = self._read_config()

    def _read_config(self) -> dict[str, Any]:
        path = self.run_dir / "worker_config.json"
        if not path.is_file():
            return {}
        body = json.loads(path.read_text(encoding="utf-8"))
        return body if isinstance(body, dict) else {}

    def _global_stop(self) -> bool:
        return (
            self._stop.is_set()
            or (self.run_dir / STOP_NAME).is_file()
            or (
                self.overall_deadline is not None
                and self.clock() >= self.overall_deadline
            )
        )

    def _request_stop(self, request_dir: Path, deadline: float) -> bool:
        return (
            self._global_stop()
            or (request_dir / STOP_NAME).is_file()
            or self.clock() >= deadline
        )

    def _heartbeat(self, *, final: bool = False, error: str | None = None) -> None:
        body = {
            "status": "stopped" if final else "running",
            "phase": self._phase,
            "request_id": self._active_request,
            "updated_at": self.clock(),
            "overall_deadline": self.overall_deadline,
        }
        if error:
            body["error"] = error
        with self._heartbeat_lock:
            atomic_write_json(self.run_dir / HEARTBEAT_NAME, body)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self._heartbeat()

    def _load_scene_once(self) -> None:
        if self.scene is not None:
            return
        checkpoint = self.run_dir / CHECKPOINT_NAME
        source = checkpoint if checkpoint.is_file() else self.run_dir / SCENE_NAME
        if not source.is_file():
            raise FileNotFoundError(f"scene-run source PLY missing: {source}")
        self.scene = self.scene_loader(source)
        if (self.config.get("scene_run") or {}).get("pipeline") == "extended":
            from ..scene_runs_ext.pipeline import validate_runtime
            from ..scene_runs_ext.fitting import initial_scale_ceiling
            # A restart may load a repaired checkpoint. Always derive the bound
            # from the immutable run input, not from already-grown Gaussians.
            original = self.scene if source == self.run_dir / SCENE_NAME else self.scene_loader(self.run_dir / SCENE_NAME)
            self.extended_scale_ceiling = initial_scale_ceiling(original)
            validate_runtime(self.config.get("extended_runtime"))

    def _editor(self):
        if self.image_editor is None:
            self.image_editor = self.image_edit_factory(self.config)
        return self.image_editor

    def _renderer(self):
        """CUDA renderer over the current cumulative scene."""
        if self.renderer is None:
            from ..rendering.gsplat_renderer import GsplatRenderer

            self.renderer = GsplatRenderer(self.scene)
        return self.renderer

    def _checkpoint(self, request_dir: Path | None = None, stats: dict | None = None) -> None:
        if self.scene is None:
            return
        self.scene_saver(self.scene, self.run_dir / CHECKPOINT_NAME)
        if request_dir is not None and stats is not None:
            atomic_write_json(request_dir / METRICS_NAME, stats)

    def process_request(self, request_dir: Path) -> dict[str, Any]:
        """Run image edit (unless CliRelay already did) then GSFix3D."""
        request_dir = Path(request_dir)
        request = json.loads((request_dir / REQUEST_NAME).read_text(encoding="utf-8"))
        request_id = str(request.get("request_id") or request_dir.name)
        started = self.clock()
        response: dict[str, Any] = {
            "request_id": request_id,
            "step": request.get("step"),
            "status": "error",
            "started_at": started,
        }
        self._active_request = request_id
        try:
            if self._global_stop():
                raise RuntimeError("scene-run worker stopped before request")
            operation = str(request.get("operation") or "repair")
            camera_body = camera_payload(request)
            from ..repair_lrz import camera_from_dict

            camera = camera_from_dict(camera_body)
            if operation == "render":
                self._phase = "render"
                self._heartbeat()
                rgb, depth = self._renderer().render_with_depth(camera)
                Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
                    request_dir / RENDERED_NAME,
                )
                np.save(request_dir / DEPTH_NAME, np.asarray(depth, dtype=np.float32))
                response.update(
                    status="ok",
                    rendered=RENDERED_NAME,
                    depth=DEPTH_NAME,
                )
                return response
            if operation != "repair":
                raise ValueError(f"Unknown scene-run GPU operation {operation!r}")
            rendered_path = request_dir / RENDERED_NAME
            if not rendered_path.is_file():
                raise FileNotFoundError(f"{RENDERED_NAME} missing for {request_id}")
            self._phase = "image_edit"
            self._heartbeat()
            prompt = str(
                request.get("prompt")
                or self.config.get("image_edit_prompt")
                or DEFAULT_PROMPT
            )
            regenerated_path = request_dir / REGENERATED_NAME
            external = bool(request.get("skip_image_edit")) or (
                regenerated_path.is_file()
                and not qwen_required_for_worker(self.config)
            )
            if external:
                if not regenerated_path.is_file():
                    raise FileNotFoundError(
                        f"{REGENERATED_NAME} missing for external CliRelay image edit"
                    )
                supplied = request.get("image_edit_result")
                edit_payload = _jsonable(supplied if isinstance(supplied, dict) else {})
                edit_payload["external"] = True
                edit_payload.setdefault(
                    "model", request.get("image_model") or request.get("image_edit_backend"),
                )
                edit_payload["prompt"] = prompt
            elif self.config.get("image_edit_subprocess", False):
                config_path = self.run_dir / "worker_config.json"
                timeout = None
                if self.overall_deadline is not None:
                    timeout = max(1.0, self.overall_deadline - self.clock())
                completed = subprocess.run(
                    [
                        sys.executable, "-m",
                        "splat_explorer.scene_runs.gpu_worker",
                        "--image-edit-request", str(request_dir),
                        "--image-edit-config", str(config_path),
                    ],
                    check=False,
                    timeout=timeout,
                )
                result_path = request_dir / IMAGE_EDIT_RESULT_NAME
                edit_result = (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.is_file()
                    else {}
                )
                if completed.returncode != 0 or edit_result.get("status") != "ok":
                    raise RuntimeError(
                        edit_result.get("error")
                        or f"Qwen image-edit subprocess exited {completed.returncode}"
                    )
                edit_payload = _jsonable(edit_result.get("payload") or {})
                edit_payload["prompt"] = prompt
            else:
                edit = self._editor().edit(rendered_path, prompt)
                images = list(getattr(edit, "images", None) or [])
                edit_error = getattr(edit, "error", None)
                if edit_error or not images:
                    raise RuntimeError(edit_error or "image edit returned no PNG")
                atomic_write_bytes(request_dir / REGENERATED_NAME, images[0])
                edit_payload = _jsonable(getattr(edit, "payload", {}) or {})
                edit_payload["prompt"] = prompt
            if (
                not external
                and not self.config.get("image_edit_subprocess", False)
                and self.config.get("release_image_editor_before_repair", False)
            ):
                # The common 64 GiB LRZ step cannot retain Qwen's host-side
                # buffers while GSFix allocates optimizer state. Weights remain
                # cached on DSS and reload for the next image-edit request.
                self.image_editor = None
                del edit
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass

            rendered = np.asarray(Image.open(rendered_path).convert("RGB"), dtype=np.uint8)
            regenerated = np.asarray(
                Image.open(request_dir / REGENERATED_NAME).convert("RGB"), dtype=np.uint8,
            )
            if (self.config.get("scene_run") or {}).get("pipeline") == "extended":
                from ..scene_runs_ext.pipeline import repair as extended_repair
                from ..scene_runs_ext.bundle import selected_views
                views = selected_views(request_dir, request.get("proposal") or {}, request.get("step"))
                self.renderer = None
                gc.collect()
                ext_deadline = min(float(request.get("deadline_unix") or float("inf")),
                                   self.overall_deadline or float("inf"))
                def ext_progress(body):
                    self._phase = str(body.get("phase") or "extended")
                    self._heartbeat()
                    atomic_write_json(request_dir / METRICS_NAME, body)
                candidate, metrics = extended_repair(
                    self.scene, camera, regenerated_path, request_dir,
                    options=(self.config.get("scene_run") or {}).get("extended"),
                    runtime=self.config.get("extended_runtime"),
                    proposal=request.get("proposal") or {},
                    should_stop=lambda: self._request_stop(request_dir, ext_deadline),
                    on_progress=ext_progress,
                    selected_views=views,
                    scale_ceiling=self.extended_scale_ceiling,
                )
                metrics.update(image_edit=edit_payload, request_id=request_id, step=request.get("step"))
                # Only complete candidates reach disk and the explorer.
                pending = self.run_dir / "scene_candidate.ply"
                self.scene_saver(candidate, pending)
                pending.replace(self.run_dir / CHECKPOINT_NAME)
                self.scene = candidate
                atomic_write_json(request_dir / METRICS_NAME, metrics)
                response.update(status="ok", regenerated=REGENERATED_NAME,
                                checkpoint=CHECKPOINT_NAME, metrics=metrics)
                return response
            repair_params = dict(self.config.get("repair") or {})
            repair_params.update(dict(request.get("repair") or {}))
            scene_run = self.config.get("scene_run") or {}
            if not repair_params.get("repair_type"):
                repair_params["repair_type"] = (
                    scene_run.get("repair_type")
                    or self.config.get("repair_type")
                    or ORIGINAL_REPAIR_TYPE
                )
            kind = normalize_repair_type(repair_params.get("repair_type"))
            backend = self.repair_factory(repair_params)
            deadline = None
            if kind == LOOPED_REPAIR_TYPE:
                deadline = bounded_repair_deadline(
                    request, now=self.clock(), overall_deadline=self.overall_deadline,
                )
            elif self.overall_deadline not in (None, 0, 0.0):
                deadline = float(self.overall_deadline)
            self._phase = "repair"
            self._heartbeat()

            def should_stop() -> bool:
                if deadline is None:
                    return self._global_stop() or (request_dir / STOP_NAME).is_file()
                return self._request_stop(request_dir, deadline)

            def checkpoint(stats: dict[str, Any]) -> None:
                metrics = {
                    key: _jsonable(value)
                    for key, value in stats.items()
                    if key != "render_rgb"
                }
                metrics.update(
                    request_id=request_id,
                    step=request.get("step"),
                    phase=str(metrics.get("phase") or "repair"),
                    updated_at=self.clock(),
                )
                self._checkpoint(request_dir, metrics)

            stats = backend.apply_until(
                self.scene,
                camera,
                rendered,
                regenerated,
                should_stop=should_stop,
                deadline=deadline,
                on_checkpoint=checkpoint,
            )
            # The prior CUDA renderer owns tensors from the pre-repair scene.
            self.renderer = None
            metrics = {
                key: _jsonable(value)
                for key, value in dict(stats or {}).items()
                if key != "render_rgb"
            }
            metrics.update(
                request_id=request_id,
                step=request.get("step"),
                image_edit=edit_payload,
                repair_deadline=deadline,
                finished_at=self.clock(),
            )
            self._checkpoint(request_dir, metrics)
            stopped = (
                (request_dir / STOP_NAME).is_file()
                or (self.run_dir / STOP_NAME).is_file()
                or self._stop.is_set()
            )
            response.update(
                # Hitting Stop is the only early-exit for original 20-iter
                # refine; looped still treats repair_seconds as completion.
                status="stopped" if stopped else "ok",
                regenerated=REGENERATED_NAME,
                checkpoint=CHECKPOINT_NAME,
                metrics=metrics,
            )
        except Exception as exc:
            logger.exception("Scene-run request %s failed", request_id)
            response["error"] = f"{type(exc).__name__}: {exc}"
            if self.scene is not None:
                self._checkpoint()
        finally:
            response["finished_at"] = self.clock()
            atomic_write_json(request_dir / RESPONSE_NAME, response)
            self._active_request = None
            self._phase = "idle"
            self._heartbeat()
        return response

    def run(self, *, poll_seconds: float = 1.0) -> None:
        """Serve the DSS request queue until STOP or the overall deadline."""
        error: str | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / REQUESTS_NAME).mkdir(parents=True, exist_ok=True)
        try:
            self._load_scene_once()
            self._phase = "idle"
            atomic_write_json(
                self.run_dir / WORKER_NAME,
                {
                    "status": "ready",
                    "pid": __import__("os").getpid(),
                    "started_at": self.clock(),
                    "overall_deadline": self.overall_deadline,
                },
            )
            self._heartbeat()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="scene-run-heartbeat",
            )
            self._heartbeat_thread.start()
            while not self._global_stop():
                requests = pending_request_dirs(self.run_dir)
                if not requests:
                    self._stop.wait(max(0.1, float(poll_seconds)))
                    continue
                self.process_request(requests[0])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Persistent scene-run GPU worker failed")
            raise
        finally:
            self._stop.set()
            self._phase = "checkpoint"
            try:
                self._checkpoint()
            finally:
                atomic_write_json(
                    self.run_dir / WORKER_NAME,
                    {
                        "status": "error" if error else "stopped",
                        "pid": __import__("os").getpid(),
                        "finished_at": self.clock(),
                        "error": error,
                    },
                )
                self._heartbeat(final=True, error=error)

    def request_stop(self) -> None:
        self._stop.set()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="splat_explorer.scene_runs.gpu_worker")
    parser.add_argument("--run-dir")
    parser.add_argument("--image-edit-request")
    parser.add_argument("--image-edit-config")
    parser.add_argument("--overall-deadline", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.image_edit_request:
        if not args.image_edit_config:
            parser.error("--image-edit-config is required with --image-edit-request")
        result = run_image_edit_once(
            Path(args.image_edit_request), Path(args.image_edit_config),
        )
        raise SystemExit(0 if result.get("status") == "ok" else 1)
    if not args.run_dir:
        parser.error("--run-dir is required")
    worker = SceneRunGpuWorker(
        Path(args.run_dir),
        overall_deadline=args.overall_deadline or None,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    def stop(_signum, _frame) -> None:
        worker.request_stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker.run(poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
