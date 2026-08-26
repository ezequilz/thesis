"""Embodied camera state: position + yaw/pitch on a gravity-aligned rig.

The rig owns the mapping from discrete agent actions to camera poses. Yaw
rotates around the world up axis; pitch tilts the view. Movement directions
are yaw-relative but stay in the horizontal plane (except up/down), which
matches how a person walks through a room.

apply() takes an optional MotionContext (collision world + the camera/depth
the current observation was rendered from): with it, `move` is clamped so the
camera never enters geometry, and `move_toward` resolves a VLM-picked pixel
through the depth map into a collision-safe travel toward that surface.
apply() returns a small outcome dict (travelled distance, whether the move was
cut short, ...) that the loop logs and feeds back into the next prompt.
"""

from __future__ import annotations

import numpy as np

from ..navigation import MotionContext, resolve_move_toward
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
    def apply(self, action: Action, ctx: MotionContext | None = None) -> dict | None:
        """Apply an action; returns an outcome dict for motion actions."""
        if action.name == "move":
            return self._apply_move(action, ctx)
        if action.name == "move_toward":
            return self._apply_move_toward(action, ctx)
        if action.name == "rotate":
            outcome = {"kind": "rotate"}
            yaw = action.args.get("yaw_degrees")
            if yaw is not None:
                self.yaw_deg = (self.yaw_deg + float(yaw)) % 360.0
                outcome["yaw_degrees"] = float(yaw)
            pitch = action.args.get("pitch_degrees")
            if pitch is not None:
                self.pitch_deg = float(pitch)
                outcome["pitch_degrees"] = float(pitch)
            return outcome
        # report_artifact / done don't change the pose.
        return None

    def _apply_move(self, action: Action, ctx: MotionContext | None) -> dict:
        forward, right = self._heading()
        requested = float(action.args["distance"])
        direction = {
            "forward": forward, "back": -forward,
            "right": right, "left": -right,
            "up": self.up, "down": -self.up,
        }[action.args["direction"]]
        if ctx is not None and ctx.world is not None:
            travelled, blocked = ctx.world.clamp_motion(self.position, direction, requested)
        else:
            travelled, blocked = requested, False
        self.position += direction * travelled
        return {
            "kind": "move",
            "direction": action.args["direction"],
            "requested": round(requested, 3),
            "travelled": round(travelled, 3),
            "blocked": blocked,
        }

    def _apply_move_toward(self, action: Action, ctx: MotionContext | None) -> dict:
        if ctx is None or ctx.camera is None or ctx.depth is None:
            return {"kind": "move_toward", "error": "no depth map available for this view"}
        result = resolve_move_toward(
            ctx.camera, ctx.depth,
            action.args.get("pixel_x", 0),
            action.args.get("pixel_y", 0),
            action.args.get("amount", 0.0),
            world=ctx.world,
            up=self.up,
        )
        if result is None:
            return {
                "kind": "move_toward",
                "error": "no geometry at the picked pixel (empty depth); pick a non-black depth pixel",
            }
        if result.get("error"):
            return {
                "kind": "move_toward",
                "pixel": [int(action.args.get("pixel_x", 0)), int(action.args.get("pixel_y", 0))],
                "amount": float(action.args.get("amount", 0.0)),
                "error": result["error"],
            }
        self.position = np.asarray(result["new_position"], dtype=np.float64)
        return {
            "kind": "move_toward",
            "pixel": [int(action.args.get("pixel_x", 0)), int(action.args.get("pixel_y", 0))],
            "amount": float(action.args.get("amount", 0.0)),
            "target_distance": result["target_distance"],
            "travelled": result["travelled"],
            "blocked": result["blocked"],
        }

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
