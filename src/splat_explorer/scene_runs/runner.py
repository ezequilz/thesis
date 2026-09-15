"""Deadline-driven scene exploration with synchronous image/3D repair barriers.

This is intentionally separate from :mod:`splat_explorer.agent.loop`: the
episode loop remains the small debugging harness, while a scene-run owns an
independent PLY lineage and can continue until a wall-clock/GPU deadline.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from ..agent.actions import Action
from ..agent.camera_rig import CameraRig
from ..agent.loop import _ensure_depth, _motion_note, _render_observation
from ..agent.vlm import make_policy
from ..cli import _build_navigation, _resolve_start
from ..config import Config
from ..navigation import MotionContext
from ..rendering import make_renderer
from ..rendering.annotate import depth_to_image
from ..rendering.birdseye import ExplorationMap
from ..scene import load_ply, load_scene, save_ply
from ..scene.catalog import SceneSpec, apply_spec, publish_live_scene, spec_by_id

logger = logging.getLogger(__name__)


class SceneRunGpu(Protocol):
    """Transport contract implemented by the LRZ scene-run client."""

    def start(
        self, *, source_ply: Path, config: dict[str, Any], deadline: float,
    ) -> dict[str, Any] | None: ...

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
    ) -> dict[str, Any]: ...

    def render(
        self, *, step: int, camera, deadline: float,
        should_stop: Callable[[], bool],
    ): ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _append_jsonl(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(body)) + "\n")
        stream.flush()


def _triggered(mode: str, action: Action) -> bool:
    """Fallback trigger predicate; the store exposes the same public behavior."""
    from ..agent.actions import wants_regenerate

    key = str(mode or "regenerate_yes").strip().lower().replace("-", "_")
    if key == "every_step":
        return True
    if action.name != "report_artifact":
        return False
    return key == "every_artifact" or wants_regenerate(action.args)


def _image_edit_prompt(action: Action, params: dict[str, Any] | None = None) -> str:
    """Build the Qwen instruction from the run default plus any artifact report."""
    from .gpu_worker import DEFAULT_PROMPT

    configured = ""
    if params:
        configured = str(params.get("image_edit_prompt") or "").strip()
    prompt = configured or DEFAULT_PROMPT
    if action.name != "report_artifact":
        return prompt
    args = dict(action.args or {})
    details = []
    description = str(args.get("description") or "").strip()
    region = str(args.get("image_region") or "").strip()
    severity = str(args.get("severity") or "").strip()
    if description:
        details.append(description)
    if region:
        details.append(f"Focus on: {region}.")
    if severity:
        details.append(f"Severity: {severity}.")
    if not details:
        return prompt
    return f"{prompt} Specific artifact to repair: {' '.join(details)}"


def _repair_trigger_state(
    mode: str, action: Action, every_step_armed: bool,
) -> tuple[bool, bool]:
    """Return ``(trigger_now, armed_after_action)`` for a scene-run action."""
    key = str(mode or "regenerate_yes").strip().lower().replace("-", "_")
    if key != "every_step":
        return _triggered(key, action), every_step_armed
    armed = every_step_armed or action.name == "report_artifact"
    return armed, armed


def _require_vlm_response(policy: Any) -> None:
    """Fail a run when CliRelay never produced a response for this decision."""
    debug = getattr(policy, "last_debug", None)
    if not isinstance(debug, dict):
        return
    if debug.get("backend") != "cli_relay" or not debug.get("fallback"):
        return
    attempts = debug.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return
    errors = [attempt.get("error") for attempt in attempts if isinstance(attempt, dict)]
    if len(errors) == len(attempts) and all(errors):
        raise RuntimeError(
            "CliRelay did not return a VLM decision; refusing scripted rotation "
            f"fallback: {errors[-1]}"
        )


class SceneRunExecutor:
    """Execute one run created by ``SceneRunStore``."""

    def __init__(
        self,
        cfg: Config,
        store,
        *,
        gpu_factory: Callable[..., SceneRunGpu] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.cfg = cfg
        self.store = store
        self.gpu_factory = gpu_factory
        self.clock = clock
        self._run_id = ""
        self._run_dir = Path()

    def _detail(self, run_id: str) -> dict[str, Any]:
        for name in ("detail", "get", "read"):
            fn = getattr(self.store, name, None)
            if callable(fn):
                body = fn(run_id)
                if body:
                    dump = getattr(body, "to_dict", None)
                    return dict(dump() if callable(dump) else body)
        raise FileNotFoundError(f"Scene-run {run_id} not found")

    def _status(self, **fields: Any) -> None:
        fields.setdefault("updated_at", self.clock())
        fn = getattr(self.store, "update_status", None)
        if callable(fn):
            try:
                fn(self._run_id, **fields)
            except TypeError:
                fn(self._run_id, fields)
            return
        path = self._run_dir / "status.json"
        current: dict[str, Any] = {}
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        current.update(_jsonable(fields))
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2))
        tmp.replace(path)

    def _event(self, event: str, **fields: Any) -> None:
        body = {"event": event, "at": self.clock(), **fields}
        fn = getattr(self.store, "append_event", None)
        if callable(fn):
            try:
                fn(self._run_id, body)
            except TypeError:
                fn(self._run_id, event, **fields)
        else:
            _append_jsonl(self._run_dir / "events.jsonl", body)

    def _stop_requested(self) -> bool:
        fn = getattr(self.store, "stop_requested", None)
        if callable(fn):
            return bool(fn(self._run_id))
        return (self._run_dir / "STOP").is_file()

    def _new_gpu(self, progress: Callable[[dict[str, Any]], None]) -> SceneRunGpu:
        if self.gpu_factory is not None:
            return self.gpu_factory(
                self.cfg, self._run_id, self._run_dir, progress,
            )
        from .lrz_transport import LrzSceneRunTransport

        return LrzSceneRunTransport(
            self.cfg, self._run_id, self._run_dir, on_progress=progress,
        )

    def execute(self, run_id: str) -> dict[str, Any]:
        self._run_id = str(run_id)
        self._run_dir = (
            Path(self.cfg.output.dir) / "scene-runs" / self._run_id
        ).resolve()
        detail = self._detail(self._run_id)
        params = dict(detail.get("config") or detail.get("params") or {})
        state = dict(detail.get("state") or detail.get("status") or {})
        state_details = dict(state.get("details") or {})
        started = self.clock()
        requested_deadline = float(
            params.get("requested_deadline")
            or state_details.get("requested_deadline")
            or (started + float(params.get("duration_seconds") or 3600))
        )
        effective_deadline = float(
            params.get("effective_deadline")
            or state_details.get("effective_deadline")
            or requested_deadline
        )
        gpu: SceneRunGpu | None = None
        repairs = 0
        artifacts = 0
        steps_done = 0
        terminal = "completed"
        error: str | None = None

        def progress(body: dict[str, Any]) -> None:
            phase = str(body.get("phase") or "gpu")
            self._status(
                phase=phase,
                message=body.get("message") or phase,
                gpu=body,
            )

        try:
            self._status(
                status="starting",
                phase="scene_load",
                started_at=started,
                requested_deadline=requested_deadline,
                effective_deadline=effective_deadline,
                pid=__import__("os").getpid(),
                error=None,
            )
            run_cfg, spec, scene = self._load_source(params)
            original_ply = self._run_dir / "scene_original.ply"
            repaired_ply = self._run_dir / "scene_repaired.ply"
            if not original_ply.is_file():
                save_ply(scene, original_ply)
            if not repaired_ply.is_file():
                save_ply(scene.copy(), repaired_ply)

            generation = int(self.clock() * 1000)
            self._publish_and_wait(spec, repaired_ply, generation)
            renderer = make_renderer(scene, run_cfg.renderer)
            nav, spawn = _build_navigation(run_cfg, scene)
            start = spawn.points[0].position if spawn else _resolve_start(run_cfg, scene)
            rig = CameraRig(
                start,
                up_axis=run_cfg.camera.up_axis,
                yaw_deg=run_cfg.camera.start_yaw_deg,
            )
            policy = make_policy(run_cfg.agent)
            exploration = self._exploration_map(spawn, rig, run_cfg)
            pose_history: list[dict[str, Any]] = []
            motion_note: str | None = None
            send_depth_once = False
            send_coverage_once = False

            if spawn is not None and spawn.points:
                self._status(
                    phase="choose_start",
                    message="Asking VLM to choose a starting position",
                )
                Image.fromarray(spawn.image).save(self._run_dir / "birdseye.png")
                choice = int(getattr(policy, "choose_start")(spawn.image, spawn))
                _require_vlm_response(policy)
                choice = int(np.clip(choice, 0, len(spawn.points) - 1))
                rig.position = np.asarray(spawn.points[choice].position).copy()
                start_record = {
                    "step": -1,
                    "pose": rig.state_description(),
                    "position": rig.position.tolist(),
                    "yaw_deg": rig.yaw_deg,
                    "pitch_deg": rig.pitch_deg,
                    "action": {"name": "choose_start", "args": {"point": choice}},
                    "frame": "birdseye.png",
                    "vlm": getattr(policy, "last_debug", None),
                }
                _append_jsonl(self._run_dir / "actions.jsonl", start_record)
                self._event("start_selected", point=choice, frame="birdseye.png")

            gpu = self._new_gpu(progress)
            gpu_info = gpu.start(
                source_ply=repaired_ply,
                config=params,
                deadline=effective_deadline,
            ) or {}
            if gpu_info.get("effective_deadline"):
                effective_deadline = min(
                    effective_deadline, float(gpu_info["effective_deadline"]),
                )
            self._status(
                status="running",
                phase="explore",
                effective_deadline=effective_deadline,
                gpu=gpu_info,
                step=0,
                repairs=0,
                artifacts=0,
            )

            step = 0
            every_step_armed = False
            while self.clock() < effective_deadline and not self._stop_requested():
                camera = rig.camera(
                    int(params.get("width") or 960),
                    int(params.get("height") or 720),
                    float(run_cfg.renderer.fov_deg),
                )
                want_depth = bool(send_depth_once)
                gpu_render = getattr(gpu, "render", None)
                if callable(gpu_render):
                    observation, depth = gpu_render(
                        step=step,
                        camera=camera,
                        deadline=effective_deadline,
                        should_stop=self._stop_requested,
                    )
                else:
                    observation, depth = _render_observation(renderer, camera, want_depth)
                frame_path = self._run_dir / f"step_{step:05d}.png"
                Image.fromarray(observation).save(frame_path)

                map_image = None
                map_name = None
                coverage_image = None
                coverage_name = None
                coverage = None
                if exploration is not None:
                    exploration.add_pose(rig.position, rig.heading(), step)
                    map_image = exploration.render()
                    map_name = f"step_{step:05d}_map.png"
                    Image.fromarray(map_image).save(self._run_dir / map_name)
                    coverage = exploration.coverage_fraction
                    if send_coverage_once:
                        coverage_image = exploration.render_coverage()
                        coverage_name = f"step_{step:05d}_coverage.png"
                        Image.fromarray(coverage_image).save(
                            self._run_dir / coverage_name,
                        )
                self._event(
                    "observation_rendered",
                    step=step,
                    frame=frame_path.name,
                    map_frame=map_name,
                )

                pose = rig.state_description()
                if coverage is not None:
                    pose += f" | viewed-area coverage {coverage:.0%}"
                if motion_note:
                    pose += f" | {motion_note}"
                depth_image = depth_to_image(depth) if depth is not None else None
                action = policy.decide(
                    observation,
                    pose,
                    step,
                    depth_image=depth_image if send_depth_once else None,
                    map_image=map_image,  # fixed on for scene-runs
                    coverage_image=coverage_image if send_coverage_once else None,
                )
                _require_vlm_response(policy)
                action = action.clamped(
                    float(run_cfg.agent.max_move_distance),
                    float(run_cfg.agent.max_rotate_degrees),
                )
                is_artifact = action.name == "report_artifact"
                if is_artifact:
                    artifacts += 1
                trigger, every_step_armed = _repair_trigger_state(
                    str(params.get("repair_trigger") or "regenerate_yes"),
                    action,
                    every_step_armed,
                )
                record: dict[str, Any] = {
                    "step": step,
                    "pose": rig.state_description(),
                    "position": rig.position.tolist(),
                    "yaw_deg": rig.yaw_deg,
                    "pitch_deg": rig.pitch_deg,
                    "action": {"name": action.name, "args": action.args},
                    "frame": frame_path.name,
                    "map_frame": map_name,
                    "map_sent": map_image is not None,
                    "coverage_frame": coverage_name,
                    "repair_triggered": trigger,
                    "every_step_armed": every_step_armed,
                    "vlm": getattr(policy, "last_debug", None),
                }
                self._event(
                    "vlm_action",
                    step=step,
                    action={"name": action.name, "args": action.args},
                    repair_triggered=trigger,
                    every_step_armed=every_step_armed,
                )

                if trigger:
                    if self.clock() >= effective_deadline or self._stop_requested():
                        break
                    self._status(
                        phase="image_edit",
                        message=f"Repairing step {step}",
                        step=step,
                    )
                    prompt = _image_edit_prompt(action, params)
                    record["image_edit_prompt"] = prompt
                    result = gpu.repair(
                        step=step,
                        camera=camera,
                        rendered_path=frame_path,
                        repair_seconds=float(params.get("repair_seconds") or 180),
                        deadline=effective_deadline,
                        should_stop=self._stop_requested,
                        prompt=prompt,
                    )
                    record["repair"] = _jsonable(result)
                    if str(result.get("status") or "") != "ok":
                        raise RuntimeError(
                            result.get("error")
                            or f"GPU repair step {step} returned {result.get('status')!r}"
                        )
                    checkpoint = result.get("checkpoint") or result.get("repaired_ply")
                    if checkpoint:
                        source = Path(checkpoint)
                        if source.resolve() != repaired_ply.resolve():
                            import shutil

                            shutil.copy2(source, repaired_ply)
                    regen = result.get("regenerated_path") or result.get("regenerated")
                    if regen and Path(regen).is_file():
                        destination = self._run_dir / f"step_{step:05d}_regen.png"
                        if Path(regen).resolve() != destination.resolve():
                            import shutil

                            shutil.copy2(regen, destination)
                        record["regenerate_frame"] = destination.name
                    repairs += 1
                    generation += 1
                    self._publish_and_wait(spec, repaired_ply, generation)
                    # The policy, rig, maps and navigation remain intact. Only
                    # rendering/depth get the cumulative repaired scene.
                    if not callable(gpu_render):
                        scene = load_ply(repaired_ply)
                        renderer = make_renderer(scene, run_cfg.renderer)
                    self._event(
                        "scene_reloaded",
                        step=step,
                        generation=generation,
                        repairs=repairs,
                        metrics=result.get("metrics"),
                    )

                outcome = None
                if action.name != "done":
                    pose_history.append({
                        "step": step,
                        "position": rig.position.copy(),
                        "yaw_deg": rig.yaw_deg,
                        "pitch_deg": rig.pitch_deg,
                    })
                    if action.name == "move_toward" and depth is None:
                        depth = _ensure_depth(renderer, camera, depth)
                    outcome = rig.apply(
                        action,
                        MotionContext(
                            world=nav,
                            camera=camera,
                            depth=depth,
                            waypoints=getattr(spawn, "waypoints", None),
                            pose_history=pose_history,
                        ),
                    )
                else:
                    # Scene-runs are deadline controlled; the production v3
                    # prompt hides done, but alternate prompts cannot end early.
                    outcome = {"kind": "done", "ignored": True}
                motion_note = _motion_note(outcome)
                if outcome is not None:
                    record["motion"] = _jsonable(outcome)
                send_depth_once = action.name == "view_depth"
                send_coverage_once = action.name == "view_coverage_map"
                _append_jsonl(self._run_dir / "actions.jsonl", record)
                if is_artifact:
                    _append_jsonl(
                        self._run_dir / "artifacts.jsonl",
                        {"step": step, **action.args},
                    )
                steps_done = step + 1
                self._status(
                    status="running",
                    phase="explore",
                    step=steps_done,
                    repairs=repairs,
                    artifacts=artifacts,
                    message=f"Exploring step {steps_done}",
                )
                step += 1

            if self._stop_requested():
                terminal = "stopped"
            elif self.clock() >= effective_deadline:
                terminal = (
                    "gpu_expired"
                    if effective_deadline + 1 < requested_deadline
                    else "completed"
                )
        except Exception as exc:
            if self._stop_requested():
                terminal = "stopped"
            elif self.clock() >= effective_deadline:
                terminal = (
                    "gpu_expired"
                    if effective_deadline + 1 < requested_deadline
                    else "completed"
                )
            else:
                terminal = "error"
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("Scene-run %s failed", self._run_id)
        finally:
            if gpu is not None:
                try:
                    if terminal in ("stopped", "error", "gpu_expired"):
                        gpu.stop()
                except Exception:
                    logger.exception("Could not stop scene-run GPU worker")
                try:
                    gpu.close()
                except Exception:
                    logger.exception("Could not close scene-run GPU transport")
            finished = self.clock()
            self._status(
                status=terminal,
                phase="finished",
                message=(
                    error
                    or f"{terminal.replace('_', ' ')} after {steps_done} steps and {repairs} repairs"
                ),
                error=error,
                finished_at=finished,
                steps=steps_done,
                repairs=repairs,
                artifacts=artifacts,
            )
            self._event(
                "run_finished",
                status=terminal,
                error=error,
                steps=steps_done,
                repairs=repairs,
                artifacts=artifacts,
            )
        return self._detail(self._run_id)

    def _load_source(
        self, params: dict[str, Any],
    ) -> tuple[Config, SceneSpec, Any]:
        run_cfg = Config(copy.deepcopy(dict(self.cfg)))
        scene_id = str(params.get("scene_id") or params.get("scene") or "venetian-balcony")
        spec = spec_by_id(run_cfg, scene_id)
        if spec is None:
            raise ValueError(f"Unknown scene {scene_id!r}")
        apply_spec(run_cfg, spec)
        run_cfg["renderer"]["width"] = int(params.get("width") or 960)
        run_cfg["renderer"]["height"] = int(params.get("height") or 720)
        backend = str(params.get("backend") or "cli_relay")
        if backend != "cli_relay":
            raise ValueError(
                "Automated scene-runs require the cli_relay VLM backend; "
                "scripted policies are only available in the episode debugger."
            )
        run_cfg["agent"]["vlm_backend"] = backend
        run_cfg["agent"]["model"] = str(
            params.get("model") or run_cfg["agent"].get("model") or "gpt-5.6-luna"
        )
        run_cfg["agent"]["send_map"] = True
        run_cfg["agent"]["send_depth"] = False
        run_cfg["agent"]["send_coverage"] = False
        scene = load_scene(
            spec.path,
            min_opacity=float(run_cfg.scene.min_opacity),
            lod_level=int(spec.lod_level),
        )
        return run_cfg, spec, scene

    @staticmethod
    def _exploration_map(spawn, rig: CameraRig, cfg: Config):
        if (
            spawn is None
            or getattr(spawn, "base_image", None) is None
            or getattr(spawn, "camera", None) is None
        ):
            return None
        waypoints = list(getattr(spawn, "waypoints", None) or [])
        xyz = np.stack([w.position for w in waypoints]) if waypoints else None
        return ExplorationMap(
            spawn.base_image,
            spawn.camera,
            float(cfg.renderer.fov_deg),
            rig.up,
            waypoints=xyz,
        )

    def _publish_and_wait(
        self,
        catalog: SceneSpec,
        repaired_ply: Path,
        generation: int,
        timeout_s: float = 300.0,
    ) -> None:
        preview = SceneSpec(
            id=f"scene-run-{self._run_id}",
            label=f"{catalog.label} · {self._run_id}",
            path=repaired_ply,
            up_axis=catalog.up_axis,
        )
        publish_live_scene(
            preview,
            generation,
            reload=True,
            catalog_id=catalog.id,
        )
        if str(self.cfg.renderer.backend) != "viser":
            return
        base = (
            self.cfg.renderer.get("viser_url")
            or __import__("os").environ.get("VISER_RENDER_URL")
            or "http://localhost:8081"
        ).rstrip("/")
        deadline = self.clock() + float(timeout_s)
        last = "viewer not reached"
        while self.clock() < deadline:
            if self._stop_requested():
                raise RuntimeError("Scene-run stopped while waiting for viewer reload")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2.0) as resp:
                    body = json.loads(resp.read().decode())
                scene = body.get("scene") or {}
                if (
                    scene.get("status") == "ready"
                    and int(scene.get("generation") or 0) >= int(generation)
                ):
                    return
                last = f"viewer scene={scene}"
            except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                last = str(exc)
            time.sleep(0.5)
        raise RuntimeError(f"Timed out waiting for repaired scene reload: {last}")
