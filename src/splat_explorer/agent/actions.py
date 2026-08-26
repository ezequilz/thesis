"""The agent's action space, exposed to the VLM as OpenAI-style tool schemas.

Movement model:
  - move_toward is the primary travel tool: the VLM picks a pixel in its
    current RGB view plus an amount in [0, 1]. The harness ray-casts through
    that pixel via the depth map, then walks that fraction of the ground-plane
    distance toward the surface (eye height is held, so a floor pixel walks
    you there instead of diving into it). amount = 1 lands right at (never
    inside) the geometry.
  - move remains for small body-relative correction steps.
  - view_map / view_depth request an extra image on the next observation
    (RGB stays in the prompt). Depth is always rendered for navigation.
  - rotate handles both yaw and (optional, absolute) pitch; the former
    separate `look` action was folded into it.
All movement is collision-clamped by the harness before being applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MOVE_DIRECTIONS = ("forward", "back", "left", "right", "up", "down")

MAX_PITCH_DEGREES = 85.0


@dataclass
class Action:
    """A single parsed tool call from the VLM."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def clamped(self, max_move: float, max_rotate: float) -> "Action":
        args = dict(self.args)
        if self.name == "move" and "distance" in args:
            args["distance"] = float(min(max(args["distance"], 0.0), max_move))
        if self.name == "move_toward":
            if "amount" in args:
                args["amount"] = float(min(max(args["amount"], 0.0), 1.0))
            for key in ("pixel_x", "pixel_y"):
                if key in args:
                    args[key] = int(args[key])
        if self.name == "rotate":
            if "yaw_degrees" in args and args["yaw_degrees"] is not None:
                args["yaw_degrees"] = float(max(-max_rotate, min(args["yaw_degrees"], max_rotate)))
            if "pitch_degrees" in args and args["pitch_degrees"] is not None:
                args["pitch_degrees"] = float(
                    max(-MAX_PITCH_DEGREES, min(args["pitch_degrees"], MAX_PITCH_DEGREES))
                )
        return Action(self.name, args)


# OpenAI chat-completions tool definitions. Kept as plain data so any
# tools-capable API (vLLM, OpenRouter, Anthropic via adapter) can consume them.
ACTION_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "move_toward",
            "description": (
                "PRIMARY movement tool. Pick a pixel in the CURRENT RGB view and walk toward "
                "the 3D surface visible at that pixel, staying at the current eye height "
                "(a point on the floor walks you there, not down into it). amount is the "
                "fraction of the ground distance to travel: 0 = stay, 1.0 = walk right up "
                "to that location (a safety margin is enforced, you can never enter "
                "geometry). If unsure what is solid, call view_depth: black depth pixels "
                "have no geometry to move toward. Use this for all larger moves."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pixel_x": {"type": "integer", "description": "Pixel column in the RGB view, 0 = left edge."},
                    "pixel_y": {"type": "integer", "description": "Pixel row in the RGB view, 0 = top edge."},
                    "amount": {"type": "number", "description": "Fraction of the distance to the surface to travel, 0..1 (e.g. 0.8)."},
                },
                "required": ["pixel_x", "pixel_y", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": (
                "Small body-relative correction step by a distance in scene units (roughly "
                "meters). Use only for fine adjustments (about 0.3-1.0 units); prefer "
                "move_toward for larger moves. Movement stops early if it would hit geometry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": list(MOVE_DIRECTIONS)},
                    "distance": {"type": "number", "description": "Distance in scene units, e.g. 0.5"},
                },
                "required": ["direction", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": (
                "Rotate the camera. yaw_degrees turns around the vertical axis (positive = "
                "right / clockwise from above, relative). pitch_degrees tilts the view and is "
                "ABSOLUTE relative to horizontal (positive = up, -85..85; 0 = level). Provide "
                "one or both."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "yaw_degrees": {"type": "number", "description": "Degrees to turn, e.g. 30 or -45."},
                    "pitch_degrees": {"type": "number", "description": "Absolute pitch in degrees, -85..85."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_map",
            "description": (
                "Look at a top-down bird's-eye MAP of the scene (ceiling removed) showing "
                "the path you have walked, every past camera position, and a small camera "
                "frustum at each step for viewing direction. Does not move the camera. The "
                "NEXT observation will include that map alongside the usual RGB view "
                "(move_toward pixel coordinates still refer to RGB, not the map). Use this "
                "to check coverage and avoid retracing."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_depth",
            "description": (
                "Look at a DEPTH MAP of the current camera view (bright = near, dark = far, "
                "black = no geometry). Does not move the camera. The NEXT observation will "
                "include that depth map alongside the usual RGB view (move_toward pixel "
                "coordinates still refer to RGB). Use this to judge distance, find holes/"
                "floaters, or check whether a pixel has geometry before move_toward."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_artifact",
            "description": "Report a rendering artifact visible in the CURRENT view (floaters, holes, blur blobs, stretched gaussians, ghosting).",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What the artifact looks like and where it is in the image."},
                    "image_region": {"type": "string", "description": "Rough location, e.g. 'upper-left', 'center'."},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["description", "image_region", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish the episode when the area has been sufficiently explored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of exploration coverage and findings."},
                },
                "required": ["summary"],
            },
        },
    },
]
