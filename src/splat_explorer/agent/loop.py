"""The observe -> decide -> act episode loop.

Each step: render the current view, hand it to the policy, clamp and apply the
returned action, and log everything to outputs/episodes/<timestamp>/ as
step_NNN.png frames plus an actions.jsonl trace. Artifact reports are
collected into artifacts.json at the end.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

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
) -> Path:
    episode_dir = output_dir / "episodes" / time.strftime("%Y%m%d_%H%M%S")
    episode_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []

    with open(episode_dir / "actions.jsonl", "w") as trace:
        for step in range(max_steps):
            observation = renderer.render(rig.camera(width, height, fov_deg))
            frame_path = episode_dir / f"step_{step:03d}.png"
            Image.fromarray(observation).save(frame_path)

            action = policy.decide(observation, rig.state_description(), step)
            action = action.clamped(max_move_distance, max_rotate_degrees)
            logger.info("step %03d | %s | %s %s", step, rig.state_description(), action.name, action.args)

            trace.write(json.dumps({
                "step": step,
                "pose": rig.state_description(),
                "position": rig.position.tolist(),
                "yaw_deg": rig.yaw_deg,
                "pitch_deg": rig.pitch_deg,
                "action": {"name": action.name, "args": action.args},
                "frame": frame_path.name,
            }) + "\n")

            if action.name == "report_artifact":
                artifacts.append({"step": step, **action.args})
            if action.name == "done":
                break
            rig.apply(action)

    with open(episode_dir / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)
    logger.info("Episode finished: %d artifact report(s) -> %s", len(artifacts), episode_dir)
    return episode_dir
