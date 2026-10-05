"""One view-finding loop; selected RGBs are written only by the checkpoint.

Navigation/rendering reuse the scene explorer primitives. Policy inputs and
termination belong to this pipeline and never enter the old artifact loop.
"""
from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from ..agent.actions import ACTION_TOOLS, Action
from ..agent.cli_relay import (
    CliRelayPolicy, RESPONSE_FORMAT, _png_data_url, parse_action, render_tool_catalog,
)
from ..navigation import MotionContext

ACTION_NAMES = (
    "move_toward", "rotate_around", "move", "rotate", "finished", "view_map",
    "view_coverage_map",
)


def view_tools():
    """Independent tool schemas, in the order presented to the view finder."""
    by_name = {tool["function"]["name"]: copy.deepcopy(tool) for tool in ACTION_TOOLS}
    by_name["finished"] = {"type": "function", "function": {
        "name": "finished", "description": (
            "Select the CURRENT RGB and camera as the next reconstruction view. "
            "This fixes one tile; continue finding the next complementary view."
        ), "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "What this view adds to the set."},
        }, "required": [], "additionalProperties": False},
    }}
    by_name["move_toward"]["function"]["description"] = (
        "Pick integer pixel_x/y in CURRENT RGB and move amount (0..1) of the "
        "ground-plane distance toward its surface, preserving eye height. "
        "Stops a margin short of the surface; use for larger moves."
    )
    by_name["view_map"]["function"]["description"] = (
        "Inspect the bird's-eye path map supplied beside RGB. Does not move the camera."
    )
    by_name["view_coverage_map"]["function"]["description"] = (
        "Request a viewed-floor coverage map on the next observation. Search-history "
        "coverage is a navigation hint, not proof of multi-angle Gaussian coverage."
    )
    return [by_name[name] for name in ACTION_NAMES]


SYSTEM_PROMPT = """Choose the viewpoints users would want to see in a repaired 3D Gaussian scene.
Build a complementary reconstruction view set that constrains visible Gaussians
from multiple angles: cover scene regions, opposite sides, useful camera heights,
and overlapping content. Translate to create parallax; rotation alone does not
show a surface from a new position. Seek ordinary useful, well-framed views, not
only artifacts or extreme close-ups. Prefer unexplored surfaces and angles to
near-duplicates. A finite set cannot guarantee all-angle coverage; use your
remaining slots to reduce the biggest gaps.
Image 1 is CURRENT RGB; all pixel targeting refers to Image 1 only. Image 2 is a
selected-view contact sheet: FINISHED tiles stay fixed, ACTIVE updates each step,
and EMPTY slots remain to be found. Other images are labelled navigation maps.
Call finished only when the CURRENT view adds useful coverage to that set. It
selects one view, then the same loop continues for the next slot. There is no
report_artifact, look, or done tool. rotate combines relative yaw and absolute
pitch. move_toward uses INTEGER current-image pixels; rotate_around uses NORMALIZED
0..1 current-image pixels and re-aims while translating. Inspect movement feedback
and actual RGB after every move; blocked travel is not a new viewpoint. The path
coverage map describes exploration history, not reconstruction coverage of the
selected views. Call exactly one tool per turn.
"""


class ViewFinderPolicy(CliRelayPolicy):
    """Same relay transport, with explicitly labelled selected-view input."""

    def __init__(self, model, base_url="", api_key=""):
        super().__init__(model=model, base_url=base_url, api_key=api_key)
        self._tools = view_tools()
        self._task = SimpleNamespace(image_detail="high")
        self.allow_done = False

    def decide(self, observation, pose_description, step, *, selected_views_image,
               map_image, coverage_image=None):
        history = "\n".join(self._history[-self.MAX_HISTORY_LINES:])
        prompt = (SYSTEM_PROMPT + "\nAvailable tools:\n" + render_tool_catalog(self._tools)
                  + f"\nStep {step}. {pose_description}\nRecent actions:\n{history}\n"
                  + RESPONSE_FORMAT)
        images = [
            ("Image 1 — CURRENT RGB (all pixel targets refer here):", _png_data_url(observation)),
            ("Image 2 — selected views (FINISHED fixed, ACTIVE current, EMPTY remaining):",
             _png_data_url(selected_views_image)),
            ("Image 3 — bird's-eye path MAP:", _png_data_url(map_image)),
        ]
        if coverage_image is not None:
            images.append(("Image 4 — viewed-floor search-history COVERAGE MAP:",
                           _png_data_url(coverage_image)))
        attempts = []
        for _ in range(self.MAX_ATTEMPTS):
            reply, error = self._ask(prompt, images)
            action = parse_action(reply, set(ACTION_NAMES)) if reply else None
            attempts.append({"reply": reply, "error": error, "parsed_ok": action is not None})
            if action is not None:
                self.last_debug = {"prompt": prompt, "attempts": attempts,
                                   "parsed_action": {"name": action.name, "args": action.args}}
                self._history.append(f"{step}: {action.name} {json.dumps(action.args)}")
                return action
        self.last_debug = {"prompt": prompt, "attempts": attempts}
        raise RuntimeError("View finder received no valid action after three attempts; checkpoint retained")


class ScriptedViewPolicy:
    """Offline smoke-test policy; its sweep is not a semantic view selector."""

    def decide(self, observation, pose_description, step, **kwargs):
        return (Action("finished", {"reason": "Scripted smoke-test view"}) if step % 2 == 0
                else Action("rotate", {"yaw_degrees": 60.0}))


def make_view_policy(agent_cfg):
    backend = agent_cfg.get("vlm_backend", "cli_relay")
    if backend == "scripted":
        return ScriptedViewPolicy()
    if backend not in ("cli_relay", "openai"):
        raise ValueError(f"Unsupported splatfix VLM backend: {backend!r}")
    if backend == "openai":
        key = agent_cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY for the openai view finder")
        return ViewFinderPolicy(agent_cfg["model"],
                                agent_cfg.get("base_url", "") or "https://api.openai.com/v1", key)
    return ViewFinderPolicy(agent_cfg["model"], agent_cfg.get("relay_base_url", ""),
                            agent_cfg.get("relay_api_key", ""))


def selected_view_sheet(finished, active, total=6, tile_width=320):
    """Only ACTIVE changes; finished RGBs are never overwritten or re-rendered."""
    if total < 1 or len(finished) >= total:
        raise ValueError("A contact sheet requires at least one unfinished slot")
    active = Image.fromarray(np.asarray(active, dtype=np.uint8)).convert("RGB")
    height = max(1, round(active.height * tile_width / active.width))
    label_height = 24
    columns = min(3, total)
    sheet = Image.new("RGB", (columns * tile_width,
                              math.ceil(total / columns) * (height + label_height)), (22, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for index in range(total):
        x, y = index % columns * tile_width, index // columns * (height + label_height)
        status = "FINISHED" if index < len(finished) else "ACTIVE" if index == len(finished) else "EMPTY"
        if index <= len(finished):
            frame = Image.fromarray(np.asarray(finished[index], dtype=np.uint8)).convert("RGB") if index < len(finished) else active
            sheet.paste(frame.resize((tile_width, height)), (x, y + label_height))
        color = (70, 225, 170) if status == "ACTIVE" else (225, 225, 225)
        draw.text((x + 6, y + 5), f"{index + 1}  {status}", fill=color)
        if status == "ACTIVE":
            draw.rectangle((x, y, x + tile_width - 1, y + height + label_height - 1), outline=color, width=3)
    return np.asarray(sheet)


def _make_map(scene, rig, width, height, fov_deg):
    from ..navigation import strip_ceiling
    from ..rendering.birdseye import ExplorationMap, render_birdseye

    axis = ("+" if rig.up[np.argmax(np.abs(rig.up))] > 0 else "-") + "xyz"[np.argmax(np.abs(rig.up))]
    stripped, _ = strip_ceiling(scene, axis, 25.0)
    image, camera = render_birdseye(stripped, up_axis=axis, width=width, height=height)
    return ExplorationMap(image, camera, fov_deg, rig.up)


def _action_error(action):
    """Treat malformed model arguments as feedback, not an interrupted run."""
    if action.name not in ACTION_NAMES:
        return f"Unsupported action {action.name!r}; use only the listed tools"
    if not isinstance(action.args, dict):
        return "Tool arguments must be an object"
    schema = next(tool['function']['parameters'] for tool in view_tools()
                  if tool['function']['name'] == action.name)
    for name in schema.get('required', []):
        if name not in action.args:
            return f"Missing required argument {name}"
    for name, value in action.args.items():
        spec = schema['properties'].get(name)
        if spec is None:
            return f"Unknown argument {name}"
        if spec['type'] in ('integer', 'number'):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or (spec['type'] == 'integer' and not isinstance(value, int))):
                return f"{name} must be a finite {spec['type']}"
            if ('minimum' in spec and value < spec['minimum']) or (
                    'maximum' in spec and value > spec['maximum']):
                return f"{name} is outside its allowed range"
        if spec['type'] == 'string' and not isinstance(value, str):
            return f"{name} must be a string"
        if 'enum' in spec and value not in spec['enum']:
            return f"Invalid {name}: {value!r}"
    return None


def run_view_finding(renderer, rig, policy, checkpoint, *, width=960, height=720,
                     fov_deg=75.0, views=6, max_steps_per_view=40, world=None,
                     scene=None, exploration_map=None, max_move=1.0, max_rotate=90.0):
    """Select ``views`` cameras, persisting each immediately via ``add_view``.

    A per-view budget failure raises with the partial checkpoint intact; it
    never silently fills missing views. Completed checkpoints are a no-op.
    Existing checkpoints resume with their fixed RGB tiles. Map construction
    needs ``scene`` unless an ``ExplorationMap`` is supplied by the caller.
    """
    if views < 1 or max_steps_per_view < 1:
        raise ValueError("views and max_steps_per_view must be positive")
    if checkpoint.target_views != views:
        raise ValueError("Requested views must match the checkpoint target_views")
    if len(checkpoint.views) >= views:
        return checkpoint.views
    if exploration_map is None:
        if scene is None:
            raise ValueError("Provide scene or exploration_map for the required map input")
        exploration_map = _make_map(scene, rig, width, height, fov_deg)
    finished = []
    for record in checkpoint.views:
        with Image.open(checkpoint.root / record["original_rgb"]) as image:
            finished.append(np.asarray(image.convert("RGB")).copy())
    if checkpoint.views:
        from .checkpoint import camera_from_record
        previous = camera_from_record(checkpoint.views[-1])
        rig.position = previous.position.astype(np.float64).copy()
        rig.aim_at(rig.position + previous.rotation[:, 2])
    step = max((int(v.get("metadata", {}).get("step", -1)) for v in checkpoint.views), default=-1) + 1
    turns = 0
    feedback = ""
    coverage_requested = False
    with (checkpoint.root / "actions.jsonl").open("a", encoding="utf-8") as trace:
        while len(finished) < views:
            camera = rig.camera(width, height, fov_deg)
            rgb = np.asarray(renderer.render(camera), dtype=np.uint8)
            exploration_map.add_pose(rig.position, rig.heading(), step)
            description = (f"Selecting view {len(finished) + 1}/{views}; "
                           f"{turns}/{max_steps_per_view} turns used for this view. "
                           f"{rig.state_description()}. Previous result: {feedback}")
            action = policy.decide(rgb, description, step,
                selected_views_image=selected_view_sheet(finished, rgb, views),
                map_image=exploration_map.render(),
                coverage_image=exploration_map.render_coverage() if coverage_requested else None)
            turns += 1
            coverage_requested = action.name == "view_coverage_map"
            error = _action_error(action)
            if error:
                outcome = {"error": error}
            elif action.name == "finished":
                # The camera corresponds exactly to the RGB supplied for this decision.
                checkpoint.add_view(rgb, camera, metadata={"step": step, "yaw_deg": rig.yaw_deg,
                    "pitch_deg": rig.pitch_deg, "reason": str(action.args.get("reason", ""))})
                finished.append(rgb.copy())
                turns = 0
                outcome = {"selected": len(finished), "remaining": views - len(finished)}
            else:
                action = action.clamped(max_move, max_rotate)
                depth = None
                if action.name == "move_toward":
                    depth_renderer = getattr(renderer, "render_depth", None)
                    render_both = getattr(renderer, "render_with_depth", None)
                    if depth_renderer:
                        depth = depth_renderer(camera)
                    elif render_both:
                        depth = render_both(camera)[1]
                if action.name == "move_toward" and depth is None:
                    outcome = {"error": "This renderer has no depth for move_toward; use move or rotate"}
                else:
                    outcome = rig.apply(action, MotionContext(world=world, camera=camera, depth=depth, scene=scene))
            feedback = json.dumps(outcome, default=lambda value: np.asarray(value).tolist())
            trace.write(json.dumps({"step": step, "action": {"name": action.name, "args": action.args},
                                    "outcome": json.loads(feedback)}) + "\n")
            trace.flush()
            step += 1
            if turns >= max_steps_per_view:
                raise RuntimeError(f"View {len(finished) + 1}/{views} exhausted its step budget; partial checkpoint saved")
    return checkpoint.views
