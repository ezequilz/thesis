"""The observe -> decide -> act episode loop.

Each step: render the current view, hand it to the policy, clamp and apply the
returned action, and log everything to outputs/episodes/<timestamp>/ as
step_NNN.png frames plus an actions.jsonl trace. Artifact reports are
collected into artifacts.json at the end.

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

from ..rendering import Renderer
from .actions import Action
from .camera_rig import CameraRig
from .vlm import VLMPolicy

logger = logging.getLogger(__name__)


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
    on_step: Callable[[dict, Path, CameraRig], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Run one episode. Returns the episode output directory.

    on_step(record, frame_path, rig) fires after each decision, before the
    action is applied — `record` matches the actions.jsonl line, `rig` still
    holds the pose the frame was rendered from. should_stop() is checked
    before each step for cooperative cancellation (e.g. from the dashboard).
    """
    episode_dir = output_dir / "episodes" / time.strftime("%Y%m%d_%H%M%S")
    episode_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    status = "completed"

    with open(episode_dir / "actions.jsonl", "w") as trace:
        for step in range(max_steps):
            if should_stop and should_stop():
                status = "stopped"
                logger.info("Episode stop requested at step %d", step)
                break

            t0 = time.perf_counter()
            observation = renderer.render(rig.camera(width, height, fov_deg))
            render_s = time.perf_counter() - t0
            frame_path = episode_dir / f"step_{step:03d}.png"
            Image.fromarray(observation).save(frame_path)

            t1 = time.perf_counter()
            action = policy.decide(observation, rig.state_description(), step)
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
            trace.write(json.dumps(record) + "\n")
            trace.flush()

            if on_step:
                on_step(record, frame_path, rig)

            if action.name == "report_artifact":
                artifacts.append({"step": step, **action.args})
            if action.name == "done":
                break
            rig.apply(action)

    with open(episode_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    logger.info("Episode %s: %d artifact report(s) -> %s", status, len(artifacts), episode_dir)
    return episode_dir
