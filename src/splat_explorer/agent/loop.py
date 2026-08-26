"""The observe -> decide -> act episode loop.

Episode flow:
  1. Optional start selection: when a SpawnSelection is provided, the policy
     is shown the annotated bird's-eye view (ceiling removed, numbered spawn
     markers) and picks the starting point; the rig teleports there.
  2. Each step: render the current view as RGB + depth, hand both to the
     policy, clamp and apply the returned action (collision-checked through
     the MotionContext), and log everything to outputs/episodes/<timestamp>/
     as step_NNN.png / step_NNN_depth.png frames plus an actions.jsonl trace.
     The outcome of the previous motion (e.g. "move cut short by an obstacle")
     is fed back to the policy with the next prompt.

Every step record includes render/decide wall times and, when the policy
exposes a `last_debug` dict (see CliRelayPolicy), the raw VLM exchange —
so traces double as debugging material for the dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..navigation import CollisionWorld, MotionContext, SpawnSelection
from ..rendering import Renderer
from ..rendering.annotate import depth_to_image
from .actions import Action
from .camera_rig import CameraRig
from .vlm import VLMPolicy

logger = logging.getLogger(__name__)


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
        note = (
            f"previous move_toward travelled {outcome['travelled']} units toward the surface "
            f"{outcome['target_distance']} units away at pixel {outcome['pixel']}"
        )
        if outcome.get("blocked"):
            note += " (stopped early by an obstacle)"
        return note
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
    on_step: Callable[[dict, Path, CameraRig], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Run one episode. Returns the episode output directory.

    nav enables collision clamping for all movement; spawn triggers the
    bird's-eye start-selection prompt before the first step.

    on_step(record, frame_path, rig) fires after each decision — `record`
    matches the actions.jsonl line, `rig` still holds the pose the frame was
    rendered from. should_stop() is checked before each step for cooperative
    cancellation (e.g. from the dashboard).
    """
    episode_dir = output_dir / "episodes" / time.strftime("%Y%m%d_%H%M%S")
    episode_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    status = "completed"
    motion_note: str | None = None

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

            frame_path = episode_dir / f"step_{step:03d}.png"
            Image.fromarray(observation).save(frame_path)
            depth_image = None
            if depth is not None:
                depth_image = depth_to_image(depth)
                Image.fromarray(depth_image).save(episode_dir / f"step_{step:03d}_depth.png")

            pose = rig.state_description()
            if motion_note:
                pose += f" | {motion_note}"

            t1 = time.perf_counter()
            action = policy.decide(observation, pose, step, depth_image=depth_image)
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
                "timing": {"render_s": round(render_s, 3), "decide_s": round(decide_s, 3)},
                "vlm": getattr(policy, "last_debug", None),
            }

            if action.name == "report_artifact":
                artifacts.append({"step": step, **action.args})
            done = action.name == "done"
            outcome = None
            if not done:
                ctx = MotionContext(world=nav, camera=camera, depth=depth)
                outcome = rig.apply(action, ctx)
            motion_note = _motion_note(outcome)
            if outcome is not None:
                record["motion"] = outcome
                if motion_note:
                    logger.info("step %03d | %s", step, motion_note)

            trace.write(json.dumps(record) + "\n")
            trace.flush()

            if on_step:
                on_step(record, frame_path, rig)
            if done:
                break

    with open(episode_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    logger.info("Episode %s: %d artifact report(s) -> %s", status, len(artifacts), episode_dir)
    return episode_dir
