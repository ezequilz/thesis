"""The agent's action space, exposed to the VLM as OpenAI-style tool schemas.

Movement model:
  - move_toward is the primary travel tool: the VLM picks a pixel in its
    current RGB view plus an amount in [0, 1]. The harness ray-casts through
    that pixel via the depth map, then walks that fraction of the ground-plane
    distance toward the surface (eye height is held, so a floor pixel walks
    you there instead of diving into it). amount = 1 lands a margin short of
    the picked surface.
  - jump_to_waypoint teleports to a precomputed vantage (gold W# on the
    bird's-eye map) or to a past step's camera pose (e.g. "step 3"). An
    optional look/facing argument sets yaw from a clock hour or compass
    direction on that map (12 / north = top of the image); omitted, the
    current heading is kept.
  - move remains for small body-relative correction steps.
  - view_map / view_coverage_map / view_depth request an extra image on the
    next observation (RGB stays in the prompt). Depth is always rendered for
    navigation.
  - rotate handles both yaw and (optional, absolute) pitch; the former
    separate `look` action was folded into it.
Path collision-clamping is optional (navigation.collision: full / low / off).
"""

from __future__ import annotations

import re
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


_STEP_TARGET = re.compile(r"^\s*(?:step|s)\s*[:#\-]?\s*(\d+)\s*$", re.IGNORECASE)
_WAYPOINT_TARGET = re.compile(
    r"^\s*(?:waypoint|way\s*point|wp|w)\s*[:#\-]?\s*(\d+)\s*$", re.IGNORECASE,
)
_BARE_INDEX = re.compile(r"^\s*(\d+)\s*$")

# Clock hour on the bird's-eye map: 12 = top, 3 = right, 6 = bottom, 9 = left.
_CLOCK_HOUR = re.compile(
    r"^\s*(\d{1,2})\s*(?::00)?\s*(?:o\s*['’]?\s*clock)?\s*(?:[ap]\s*\.?\s*m\.?)?\s*$",
    re.IGNORECASE,
)
_LOOK_KEYS = ("look", "facing", "look_direction", "look_toward", "look_towards")
_KEEP_LOOK = frozenset({
    "", "forward", "same", "keep", "current", "none", "unchanged", "default",
})
# Degrees clockwise from map-north (top of the bird's-eye image).
_COMPASS_CW = {
    "n": 0.0, "north": 0.0, "up": 0.0, "top": 0.0,
    "ne": 45.0, "northeast": 45.0,
    "e": 90.0, "east": 90.0, "right": 90.0,
    "se": 135.0, "southeast": 135.0,
    "s": 180.0, "south": 180.0, "down": 180.0, "bottom": 180.0,
    "sw": 225.0, "southwest": 225.0,
    "w": 270.0, "west": 270.0, "left": 270.0,
    "nw": 315.0, "northwest": 315.0,
}
_CARDINAL_AT_HOUR = {12: "north", 3: "east", 6: "south", 9: "west"}


@dataclass(frozen=True)
class LookDirection:
    """Horizontal facing after a jump, in map-clock degrees (0 = top / north)."""

    degrees_cw: float
    label: str


def parse_jump_target(args: dict[str, Any] | None) -> tuple[str, int] | None:
    """Parse jump_to_waypoint args into ('waypoint'|'step', index).

    Accepts target='waypoint 2' / 'W2' / '2', target='step 3', or the
    alternate keys waypoint= / step=. Bare numbers default to a waypoint.
    """
    args = args or {}
    if args.get("step") is not None and args.get("waypoint") is None:
        try:
            return "step", int(args["step"])
        except (TypeError, ValueError):
            return None
    if args.get("waypoint") is not None:
        try:
            return "waypoint", int(args["waypoint"])
        except (TypeError, ValueError):
            return None
    target = args.get("target", args.get("to", args.get("name")))
    if target is None:
        return None
    if isinstance(target, bool):
        return None
    if isinstance(target, (int, float)):
        return "waypoint", int(target)
    text = str(target).strip()
    match = _STEP_TARGET.match(text)
    if match:
        return "step", int(match.group(1))
    match = _WAYPOINT_TARGET.match(text) or _BARE_INDEX.match(text)
    if match:
        return "waypoint", int(match.group(1))
    return None


def _clock_label(hour: int) -> str:
    cardinal = _CARDINAL_AT_HOUR.get(hour)
    if cardinal:
        return f"{hour} o'clock ({cardinal})"
    return f"{hour} o'clock"


def _compass_label(degrees_cw: float) -> str:
    for name, deg in _COMPASS_CW.items():
        if name in ("n", "ne", "e", "se", "s", "sw", "w", "nw",
                    "up", "top", "right", "down", "bottom", "left"):
            continue
        if abs(deg - degrees_cw) < 1e-6:
            return name
    return f"{degrees_cw:.0f} deg"


_REGENERATE_KEYS = ("regenerate", "fix")
_REGENERATE_YES = frozenset({"yes", "y", "true", "on", "1"})


def wants_regenerate(args: dict[str, Any] | None) -> bool:
    """True when report_artifact asked to regenerate/fix the current RGB view.

    Accepts regenerate= or fix=, with yes/no, 1/0, or booleans. Omitted or
    unrecognised values mean no (report only).
    """
    args = args or {}
    raw = None
    for key in _REGENERATE_KEYS:
        if key in args and args[key] is not None:
            raw = args[key]
            break
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw != 0
    return str(raw).strip().lower() in _REGENERATE_YES


def parse_look_direction(args: dict[str, Any] | None) -> tuple[LookDirection | None, str | None]:
    """Parse an optional jump facing into a map-aligned look direction.

    Returns (parsed, error). Both None means the argument was omitted (keep
    the current heading). The clock face and compass are the bird's-eye MAP:
    12 / north = top of the image, 3 / east = right, 6 / south = bottom,
    9 / west = left. Accepts look= / facing= / look_direction=.
    """
    args = args or {}
    raw = None
    for key in _LOOK_KEYS:
        if key in args and args[key] is not None:
            raw = args[key]
            break
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, f"{raw!r} is not a look direction"
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if float(raw) != int(raw):
            return None, f"{raw!r} is not a clock hour (use 1-12) or compass name"
        hour = int(raw)
        if hour == 0:
            hour = 12
        if not 1 <= hour <= 12:
            return None, f"{raw!r} is not a clock hour (use 1-12) or compass name"
        return LookDirection((hour % 12) * 30.0, _clock_label(hour)), None
    text = str(raw).strip()
    if text.lower() in _KEEP_LOOK:
        return None, None
    key = re.sub(r"[^a-z]", "", text.lower())
    if key in _COMPASS_CW:
        deg = _COMPASS_CW[key]
        return LookDirection(deg, _compass_label(deg)), None
    match = _CLOCK_HOUR.match(text)
    if match:
        hour = int(match.group(1))
        if hour == 0:
            hour = 12
        if 1 <= hour <= 12:
            return LookDirection((hour % 12) * 30.0, _clock_label(hour)), None
    return None, f"{text!r} not understood; use north/east/south/west or a clock hour 1-12 (12 = top of the map)"


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
                "to that location (stops a small margin short of the picked surface). "
                "To enter another room, pick a pixel on the floor through a doorway. If "
                "unsure what is solid, call view_depth: black depth pixels have no "
                "geometry to move toward. Use this for all larger moves."
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
                "move_toward for larger moves."
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
            "name": "jump_to_waypoint",
            "description": (
                "Teleport to a numbered vantage waypoint or a past camera pose. "
                "target is 'waypoint N' / 'W N' / just 'N' for a gold W# marker "
                "on the bird's-eye map (precomputed open-space vantages covering "
                "the rooms), or 'step N' to return to the camera pose of that "
                "earlier step. By default heading is kept when jumping to a "
                "waypoint (a past step restores that step's yaw and pitch). "
                "Pass optional look to face a chosen direction on landing: a "
                "clock hour or compass name on the bird's-eye MAP "
                "(12 / north = top of the map, 3 / east = right, 6 / south = "
                "bottom, 9 / west = left). The view stays level (first-person). "
                "Use look when you already know which way the room opens, e.g. "
                "jump to a bedroom waypoint and look south toward the bed. Omit "
                "look to keep your current heading. Use this to reach another "
                "room or revisit a previous view without walking there. Budget "
                "the use of jumping and make sure to explore areas fully before "
                "moving on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": (
                            "Where to jump: 'waypoint 2' / 'W2' / '2', or 'step 3'."
                        ),
                    },
                    "look": {
                        "type": "string",
                        "description": (
                            "Optional. Where to face after the jump, as a clock "
                            "hour or compass direction on the bird's-eye MAP "
                            "(not the RGB view). 12 or north = toward the TOP of "
                            "the map; 3 or east = RIGHT; 6 or south = BOTTOM; "
                            "9 or west = LEFT. Also northeast/southeast/"
                            "southwest/northwest, or hours 1-12 "
                            "('7', '7 o'clock'). Omit to keep your current "
                            "heading. The camera stays horizontally level."
                        ),
                    },
                },
                "required": ["target"],
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
                "the path you have walked, every past camera position, a small camera "
                "frustum at each step for viewing direction, and pale gold W# markers "
                "for jump_to_waypoint vantages. Does not move the camera. The "
                "NEXT observation will include that map alongside the usual RGB view "
                "(move_toward pixel coordinates still refer to RGB, not the map). Use this "
                "to check where you have walked, pick a waypoint, and avoid retracing."
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
            "name": "view_coverage_map",
            "description": (
                "Look at a COVERAGE MAP: the same top-down bird's-eye backdrop as view_map, "
                "with a larger yellow-green cone painted for every direction you have looked. "
                "Close floor is marked more strongly; strength falls off with distance so "
                "the far side of a room / a distant wall stays faint or unshaded. Overlapping "
                "views stack (still translucent). A coverage percentage (0-100%) is shown on "
                "the image. Does not move the camera. The NEXT observation includes that map "
                "alongside the usual RGB view (move_toward pixels still refer to RGB). Use "
                "this to find unshaded rooms and see which floor has actually been looked at."
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
            "description": (
                "Report a rendering artifact visible in the CURRENT view (floaters, "
                "holes, blur blobs, stretched gaussians, ghosting). Optional "
                "regenerate=yes queues a background image-to-image repair of this "
                "RGB view; exploration continues immediately. Default is no."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What the artifact looks like and where it is in the image."},
                    "image_region": {"type": "string", "description": "Rough location, e.g. 'upper-left', 'center'."},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "regenerate": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": (
                            "Optional. 'yes' (or 1) to regenerate/fix the current RGB "
                            "view in the background; 'no' (or 0, default) only reports "
                            "the artifact."
                        ),
                    },
                },
                "required": ["description", "image_region", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish the episode when the area has been sufficiently explored (check the coverage map: unshaded rooms mean you have not been there).",
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


def filter_tools(hidden: tuple[str, ...] | list[str] = ()) -> list[dict]:
    """ACTION_TOOLS with named tools removed (e.g. hide `done` for a prompt)."""
    hide = set(hidden)
    return [tool for tool in ACTION_TOOLS if tool["function"]["name"] not in hide]
