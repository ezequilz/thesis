"""The observe -> decide -> act episode loop.

Episode flow:
  1. Optional start selection: when a SpawnSelection is provided, the policy
     is shown the annotated bird's-eye view (ceiling removed, numbered spawn
     markers) and picks the starting point; the rig teleports there.
  2. Each step: render RGB (+ depth for navigation), overlay the walked path
     onto the cached bird's-eye map, hand the observation to the policy
     (depth / path map / coverage map only if send_depth / send_map /
     send_coverage is on or the previous action was view_depth / view_map /
     view_coverage_map),
     clamp and apply the returned action (optionally path-clamped through the
     MotionContext when navigation.collision is full or low), and
     log everything to outputs/episodes/<timestamp>/ as frames plus an
     actions.jsonl trace. The outcome of the previous motion (e.g. "move cut
     short by an obstacle") is fed back to the policy with the next prompt.

Every step record includes render/decide wall times and, when the policy
exposes a `last_debug` dict (see CliRelayPolicy), the raw VLM exchange —
so traces double as debugging material for the dashboard. Depth, path-map,
and coverage-map PNGs are always written (dashboard tiles). Depth is attached
after view_depth or when send_depth is on; the bird's-eye path map after
view_map or, when send_map is on, at full resolution on every prompt; the
coverage map after view_coverage_map or, when send_coverage is on, as a
freshly rendered low-res overview on every prompt (RGB first). RGB is always
in the prompt.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..logging_utils import log_to_file
from ..navigation import CollisionWorld, MotionContext, SpawnSelection
from ..rendering import Renderer
from ..rendering.annotate import depth_to_image
from ..rendering.birdseye import COVERAGE_PROMPT_MAX_SIDE, ExplorationMap
from .actions import Action
from .camera_rig import CameraRig
from .vlm import VLMPolicy

logger = logging.getLogger(__name__)


def _write_meta(episode_dir: Path, meta: dict) -> None:
    """Atomically (re)write the episode's meta.json (run history metadata)."""
    tmp = episode_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(episode_dir / "meta.json")


def _motion_note(outcome: dict | None) -> str | None:
    """One-line feedback about the previous action for the next prompt."""
    if not outcome:
        return None
    if outcome.get("error"):
        return f"previous {outcome['kind']} FAILED: {outcome['error']}"
    kind = outcome.get("kind")
    if kind == "move":
        note = f"previous move {outcome['direction']} travelled {outcome['travelled']} of {outcome['requested']} units"
        if outcome.get("blocked"):
            note += " (stopped early by an obstacle)"
        return note
    if kind == "move_toward":
        if outcome.get("travelled", 1) == 0 and outcome.get("blocked"):
            note = (
                f"previous move_toward travelled 0 units (blocked immediately) toward the "
                f"surface {outcome['target_distance']} units away at pixel {outcome['pixel']}; "
                "pick a more open direction or rotate"
            )
            return note
        note = (
            f"previous move_toward travelled {outcome['travelled']} units toward the surface "
            f"{outcome['target_distance']} units away at pixel {outcome['pixel']}"
        )
        if outcome.get("blocked"):
            note += " (stopped early by an obstacle)"
        return note
    if kind == "view_map":
        return (
            "previous action requested the map — this observation includes the "
            "bird's-eye path map next to the RGB view"
        )
    if kind == "view_coverage_map":
        return (
            "previous action requested the coverage map — this observation includes "
            "the viewed-area coverage map next to the RGB view"
        )
    if kind == "view_depth":
        return (
            "previous action requested the depth map — this observation includes "
            "the depth map next to the RGB view"
        )
    return None


def _select_start(
    policy: VLMPolicy,
    spawn: SpawnSelection,
    rig: CameraRig,
    episode_dir: Path,
    trace,
    on_step: Callable[[dict, Path, CameraRig], None] | None,
) -> None:
    """Initial prompt: show the bird's-eye view, let the policy pick a start."""
    birdseye_path = episode_dir / "birdseye.png"
    Image.fromarray(spawn.image).save(birdseye_path)

    chooser = getattr(policy, "choose_start", None)
    choice = chooser(spawn.image, spawn) if chooser else 0
    choice = int(np.clip(choice, 0, len(spawn.points) - 1))
    point = spawn.points[choice]
    rig.position = np.asarray(point.position, dtype=np.float64).copy()
    logger.info("Start selection: point %d at %s (clearance %.2f)",
                choice, np.round(point.position, 2), point.clearance)

    record = {
        "step": -1,
        "pose": rig.state_description(),
        "position": rig.position.tolist(),
        "yaw_deg": rig.yaw_deg,
        "pitch_deg": rig.pitch_deg,
        "action": {"name": "choose_start", "args": {"point": choice}},
        "frame": birdseye_path.name,
        "spawn_points": [
            {"index": p.index, "position": np.asarray(p.position).tolist(),
             "clearance": p.clearance}
            for p in spawn.points
        ],
        "vlm": getattr(policy, "last_debug", None),
    }
    trace.write(json.dumps(record) + "\n")
    trace.flush()
    if on_step:
        on_step(record, birdseye_path, rig)


def run_episode(
    renderer: Renderer,
    rig: CameraRig,
    policy: VLMPolicy,
    output_dir: Path,
    width: int,
    height: int,
    fov_deg: float,
    max_steps: int = 40,
    max_move_distance: float = 2.0,
    max_rotate_degrees: float = 90.0,
    nav: CollisionWorld | None = None,
    spawn: SpawnSelection | None = None,
    send_depth: bool = False,
    send_map: bool = False,
    send_coverage: bool = False,
    run_meta: dict | None = None,
    on_step: Callable[[dict, Path, CameraRig], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Run one episode. Returns the episode output directory.

    nav is the collision world (path clamp follows navigation.collision);
    spawn triggers the
    bird's-eye start-selection prompt before the first step. send_depth
    forces the depth map onto every VLM prompt; otherwise it is attached
    only on the observation after a view_depth action (it is always
    rendered, saved, and used for move_toward). send_map forces the
    full-resolution bird's-eye path map onto every VLM prompt (no downsize);
    otherwise it is attached only after view_map. send_coverage forces a
    low-res coverage map (redrawn from the original buffers) onto every
    VLM prompt after RGB; otherwise the coverage map is attached only after
    view_coverage_map. The path map and full-res coverage map are always
    updated and saved. RGB is always in the prompt.
    run_meta is merged into the episode's meta.json.

    on_step(record, frame_path, rig) fires after each decision — `record`
    matches the actions.jsonl line, `rig` still holds the pose the frame was
    rendered from. should_stop() is checked before each step for cooperative
    cancellation (e.g. from the dashboard).

    Every run leaves in its episode directory: actions.jsonl (per-step
    trace), meta.json (status/params/error), episode.log (all log output,
    crashes included), artifacts.json, and the PNG frames.
    """
    episode_dir = output_dir / "episodes" / time.strftime("%Y%m%d_%H%M%S")
    episode_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    status = "completed"
    summary: str | None = None
    steps_done = 0
    motion_note: str | None = None
    send_map_once = False
    send_coverage_once = False
    send_depth_once = False
    expl_map: ExplorationMap | None = None
    if (
        spawn is not None
        and getattr(spawn, "base_image", None) is not None
        and getattr(spawn, "camera", None) is not None
    ):
        expl_map = ExplorationMap(spawn.base_image, spawn.camera, fov_deg, rig.up)

    meta = {
        "episode": episode_dir.name,
        "status": "running",
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
        "steps": 0,
        "max_steps": max_steps,
        "send_depth": send_depth,
        "send_map": send_map,
        "send_coverage": send_coverage,
        "artifact_count": 0,
        "summary": None,
        **(run_meta or {}),
    }
    _write_meta(episode_dir, meta)

    with log_to_file(episode_dir / "episode.log"):
        try:
            with open(episode_dir / "actions.jsonl", "w") as trace:
                if spawn is not None and spawn.points:
                    _select_start(policy, spawn, rig, episode_dir, trace, on_step)

                for step in range(max_steps):
                    if should_stop and should_stop():
                        status = "stopped"
                        logger.info("Episode stop requested at step %d", step)
                        break

                    t0 = time.perf_counter()
                    camera = rig.camera(width, height, fov_deg)
                    render_depth = getattr(renderer, "render_with_depth", None)
                    if render_depth is not None:
                        observation, depth = render_depth(camera)
                    else:
                        observation, depth = renderer.render(camera), None
                    render_s = time.perf_counter() - t0
                    render_backend = getattr(renderer, "last_backend", None)

                    frame_path = episode_dir / f"step_{step:03d}.png"
                    Image.fromarray(observation).save(frame_path)
                    depth_image = None
                    depth_frame_name = None
                    if depth is not None:
                        depth_image = depth_to_image(depth)
                        depth_frame_name = f"step_{step:03d}_depth.png"
                        Image.fromarray(depth_image).save(episode_dir / depth_frame_name)

                    map_image = None
                    map_frame_name = None
                    coverage_image = None
                    coverage_frame_name = None
                    coverage_frac = None
                    if expl_map is not None:
                        expl_map.add_pose(rig.position, rig.heading(), step)
                        map_image = expl_map.render()
                        map_frame_name = f"step_{step:03d}_map.png"
                        Image.fromarray(map_image).save(episode_dir / map_frame_name)
                        coverage_image = expl_map.render_coverage()
                        coverage_frame_name = f"step_{step:03d}_coverage.png"
                        Image.fromarray(coverage_image).save(episode_dir / coverage_frame_name)
                        coverage_frac = expl_map.coverage_fraction

                    pose = rig.state_description()
                    if coverage_frac is not None:
                        pose += f" | viewed-area coverage {coverage_frac:.0%}"
                    if motion_note:
                        pose += f" | {motion_note}"

                    t1 = time.perf_counter()
                    attach_depth = (send_depth or send_depth_once) and depth_image is not None
                    attach_map = (send_map or send_map_once) and map_image is not None
                    attach_coverage = (
                        (send_coverage or send_coverage_once) and coverage_image is not None
                    )
                    prompt_coverage = None
                    coverage_lowres = False
                    if attach_coverage:
                        # Always-on: redraw a compact map from the original
                        # buffers so labels stay readable at low res.
                        # view_coverage_map (without always-on) keeps full-res.
                        if send_coverage and expl_map is not None:
                            prompt_coverage = expl_map.render_coverage(
                                max_side=COVERAGE_PROMPT_MAX_SIDE,
                            )
                            coverage_lowres = True
                        else:
                            prompt_coverage = coverage_image
                    action = policy.decide(
                        observation, pose, step,
                        depth_image=depth_image if attach_depth else None,
                        map_image=map_image if attach_map else None,
                        coverage_image=prompt_coverage,
                    )
                    decide_s = time.perf_counter() - t1
                    action = action.clamped(max_move_distance, max_rotate_degrees)
                    logger.info(
                        "step %03d | %s | %s %s | render %.2fs decide %.2fs",
                        step, rig.state_description(), action.name, action.args, render_s, decide_s,
                    )

                    record = {
                        "step": step,
                        "pose": rig.state_description(),
                        "position": rig.position.tolist(),
                        "yaw_deg": rig.yaw_deg,
                        "pitch_deg": rig.pitch_deg,
                        "action": {"name": action.name, "args": action.args},
                        "frame": frame_path.name,
                        "depth_frame": depth_frame_name,
                        "depth_sent": attach_depth,
                        "map_frame": map_frame_name,
                        "map_sent": attach_map,
                        "coverage_frame": coverage_frame_name,
                        "coverage_sent": attach_coverage,
                        "coverage_lowres": coverage_lowres,
                        "coverage": None if coverage_frac is None else round(coverage_frac, 4),
                        "render_backend": render_backend,
                        "timing": {"render_s": round(render_s, 3), "decide_s": round(decide_s, 3)},
                        "vlm": getattr(policy, "last_debug", None),
                    }

                    if action.name == "report_artifact":
                        artifacts.append({"step": step, **action.args})
                    # Some prompts hide `done`; the run still ends at max_steps.
                    done = action.name == "done" and getattr(policy, "allow_done", True)
                    if action.name == "done" and not done:
                        logger.warning("Ignoring done at step %d (prompt does not allow it)", step)
                    if done:
                        summary = action.args.get("summary")
                    outcome = None
                    if not done:
                        ctx = MotionContext(world=nav, camera=camera, depth=depth)
                        outcome = rig.apply(action, ctx)
                    motion_note = _motion_note(outcome)
                    if outcome is not None:
                        record["motion"] = outcome
                        if motion_note:
                            logger.info("step %03d | %s", step, motion_note)
                    send_map_once = action.name == "view_map"
                    send_coverage_once = action.name == "view_coverage_map"
                    send_depth_once = action.name == "view_depth"

                    trace.write(json.dumps(record) + "\n")
                    trace.flush()
                    steps_done = step + 1

                    if on_step:
                        on_step(record, frame_path, rig)
                    if done:
                        break
        except Exception as exc:
            status = "error"
            meta["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Episode crashed at step %d", steps_done)
            raise
        finally:
            meta.update(
                status=status,
                finished_at=time.time(),
                steps=steps_done,
                artifact_count=len(artifacts),
                summary=summary,
            )
            _write_meta(episode_dir, meta)
            with open(episode_dir / "artifacts.json", "w") as f:
                json.dump(artifacts, f, indent=2)
            logger.info("Episode %s: %d artifact report(s) -> %s",
                        status, len(artifacts), episode_dir)
    return episode_dir
