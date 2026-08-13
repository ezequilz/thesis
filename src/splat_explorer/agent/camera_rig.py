"""Embodied camera state: position + yaw/pitch on a gravity-aligned rig.

The rig owns the mapping from discrete agent actions to camera poses. Yaw
rotates around the world up axis; pitch tilts the view. Movement directions
are yaw-relative but stay in the horizontal plane (except up/down), which
matches how a person walks through a room.

No collision handling yet — the agent can currently walk through walls.
TODO: add occupancy checks against the gaussian density field.
"""

from __future__ import annotations

import numpy as np

from ..rendering.base import Camera, up_vector
from .actions import Action


class CameraRig:
    def __init__(
        self,
        position: np.ndarray,
        up_axis: str = "+y",
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ):
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.up = up_vector(up_axis).astype(np.float64)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)
        # Build a horizontal basis: pick any vector not parallel to up.
        seed = np.array([0.0, 0.0, -1.0]) if abs(self.up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        self._fwd0 = seed - self.up * np.dot(seed, self.up)
        self._fwd0 /= np.linalg.norm(self._fwd0)
        self._right0 = np.cross(self._fwd0, self.up)

    # --- direction helpers ---------------------------------------------------
    def _heading(self) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal forward and right vectors for the current yaw."""
        yaw = np.radians(self.yaw_deg)
        forward = np.cos(yaw) * self._fwd0 + np.sin(yaw) * self._right0
        right = np.cross(forward, self.up)
        return forward, right

    def view_direction(self) -> np.ndarray:
        forward, _ = self._heading()
        pitch = np.radians(self.pitch_deg)
        return np.cos(pitch) * forward + np.sin(pitch) * self.up

    # --- action application ----------------------------------------------------
    def apply(self, action: Action) -> None:
        if action.name == "move":
            forward, right = self._heading()
            d = float(action.args["distance"])
            direction = {
                "forward": forward, "back": -forward,
                "right": right, "left": -right,
                "up": self.up, "down": -self.up,
            }[action.args["direction"]]
            self.position += direction * d
        elif action.name == "rotate":
            self.yaw_deg = (self.yaw_deg + float(action.args["yaw_degrees"])) % 360.0
        elif action.name == "look":
            self.pitch_deg = float(action.args["pitch_degrees"])
        # report_artifact / done don't change the pose.

    def camera(self, width: int, height: int, fov_deg: float) -> Camera:
        target = self.position + self.view_direction()
        return Camera.look_at(
            self.position, target, self.up,
            width=width, height=height, fov_deg=fov_deg,
        )

    def state_description(self) -> str:
        """Human/VLM-readable pose summary included in prompts."""
        p = self.position
        return (
            f"position=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}), "
            f"yaw={self.yaw_deg:.0f} deg, pitch={self.pitch_deg:.0f} deg"
        )
