"""Embodied camera state: position + yaw/pitch on a gravity-aligned rig.

The rig owns the mapping from discrete agent actions to camera poses. Yaw
rotates around the world up axis; pitch tilts the view. Movement directions
are yaw-relative but stay in the horizontal plane (except up/down), which
matches how a person walks through a room.

apply() takes an optional MotionContext (collision world + the camera/depth
the current observation was rendered from): with it, `move` is path-clamped
when navigation.collision is full or low, and `move_toward` resolves a
VLM-picked pixel through the depth map into travel toward that surface
(always stopping a margin short of the target; path-clamped only in full/low).
apply() returns a small outcome dict (travelled distance, whether the move was
cut short, ...) that the loop logs and feeds back into the next prompt.
"""

from __future__ import annotations

import numpy as np

from ..navigation import MotionContext, resolve_move_toward
from ..rendering.base import Camera, up_vector
from .actions import Action, parse_jump_target, parse_look_direction


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

    def heading(self) -> np.ndarray:
        """Unit look direction in the ground plane (yaw only, ignores pitch)."""
        forward, _ = self._heading()
        return forward

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
        if action.name == "view_map":
            return {"kind": "view_map"}
        if action.name == "view_coverage_map":
            return {"kind": "view_coverage_map"}
        if action.name == "view_depth":
            return {"kind": "view_depth"}
        if action.name == "jump_to_waypoint":
            return self._apply_jump(action, ctx)
        # report_artifact / done don't change the pose.
        return None

    def _apply_jump(self, action: Action, ctx: MotionContext | None) -> dict:
        parsed = parse_jump_target(action.args)
        if parsed is None:
            return {
                "kind": "jump_to_waypoint",
                "error": "could not parse target; use 'waypoint N' / 'W N' or 'step N'",
            }
        kind, index = parsed
        if kind == "waypoint":
            waypoints = list(ctx.waypoints) if ctx is not None and ctx.waypoints else []
            if not waypoints:
                return {"kind": "jump_to_waypoint", "error": "no waypoints available"}
            match = next((w for w in waypoints if int(w.index) == index), None)
            if match is None:
                available = ", ".join(f"W{w.index}" for w in waypoints)
                return {
                    "kind": "jump_to_waypoint",
                    "error": f"waypoint {index} out of range (available {available})",
                }
            self.position = np.asarray(match.position, dtype=np.float64).copy()
            outcome = {
                "kind": "jump_to_waypoint",
                "target_kind": "waypoint",
                "index": index,
                "destination": f"waypoint {index} (W{index})",
            }
            outcome.update(self._apply_optional_look(action))
            return outcome
        history = list(ctx.pose_history) if ctx is not None and ctx.pose_history else []
        pose = next((p for p in history if int(p.get("step", -999)) == index), None)
        if pose is None:
            steps = [int(p.get("step", -1)) for p in history]
            lo, hi = (min(steps), max(steps)) if steps else (None, None)
            span = f"{lo}..{hi}" if steps else "none recorded yet"
            return {
                "kind": "jump_to_waypoint",
                "error": f"step {index} is not a recorded pose (available {span})",
            }
        self.position = np.asarray(pose["position"], dtype=np.float64).copy()
        if "yaw_deg" in pose:
            self.yaw_deg = float(pose["yaw_deg"])
        if "pitch_deg" in pose:
            self.pitch_deg = float(pose["pitch_deg"])
        outcome = {
            "kind": "jump_to_waypoint",
            "target_kind": "step",
            "index": index,
            "destination": f"step {index}",
        }
        outcome.update(self._apply_optional_look(action))
        return outcome

    def _apply_optional_look(self, action: Action) -> dict:
        """Set yaw from a map-clock / compass look; pitch is leveled to 0.

        12 / north is the top of the bird's-eye map (same ground basis as
        the map renderer). Omitted or unparseable look leaves heading as-is.
        """
        parsed, error = parse_look_direction(action.args)
        if error:
            return {"look_error": error}
        if parsed is None:
            return {}
        theta = np.radians(parsed.degrees_cw)
        # Map north = top of image = -_right0; map east = right of image = _fwd0.
        world_dir = np.sin(theta) * self._fwd0 - np.cos(theta) * self._right0
        self._set_horizontal_look(world_dir)
        return {"look": parsed.label, "look_degrees_cw": parsed.degrees_cw}

    def _set_horizontal_look(self, world_dir: np.ndarray) -> None:
        d = np.asarray(world_dir, dtype=np.float64)
        d = d - self.up * np.dot(d, self.up)
        n = float(np.linalg.norm(d))
        if n < 1e-8:
            return
        d /= n
        yaw = np.degrees(np.arctan2(np.dot(d, self._right0), np.dot(d, self._fwd0)))
        self.yaw_deg = float(yaw % 360.0)
        self.pitch_deg = 0.0

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
