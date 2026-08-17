"""The agent's action space, exposed to the VLM as OpenAI-style tool schemas.

Movement is body-relative (forward = current viewing direction projected onto
the ground plane), so the VLM reasons in "walk 1.5m forward, turn 30 degrees
right" terms. The harness clamps parameters to configured limits before
applying them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MOVE_DIRECTIONS = ("forward", "back", "left", "right", "up", "down")


@dataclass
class Action:
    """A single parsed tool call from the VLM."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def clamped(self, max_move: float, max_rotate: float) -> "Action":
        args = dict(self.args)
        if self.name == "move" and "distance" in args:
            args["distance"] = float(min(max(args["distance"], 0.0), max_move))
        if self.name == "rotate" and "yaw_degrees" in args:
            args["yaw_degrees"] = float(max(-max_rotate, min(args["yaw_degrees"], max_rotate)))
        # if self.name == "look" and "pitch_degrees" in args:
        #     args["pitch_degrees"] = float(max(-89.0, min(args["pitch_degrees"], 89.0)))
        return Action(self.name, args)


# OpenAI chat-completions tool definitions. Kept as plain data so any
# tools-capable API (vLLM, OpenRouter, Anthropic via adapter) can consume them.
ACTION_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the camera in a body-relative direction by a distance in scene units (roughly meters).",
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
            "description": "Rotate the camera around the vertical axis. Positive = right (clockwise from above).",
            "parameters": {
                "type": "object",
                "properties": {
                    "yaw_degrees": {"type": "number", "description": "Degrees to turn, e.g. 30 or -45"},
                },
                "required": ["yaw_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Tilt the camera up or down. Positive = up. Sets absolute pitch relative to horizontal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pitch_degrees": {"type": "number", "description": "Absolute pitch in degrees, -89..89"},
                },
                "required": ["pitch_degrees"],
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
