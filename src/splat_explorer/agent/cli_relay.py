"""CliRelay VLM backend: one endpoint for Gemini / Claude / OpenAI and friends.

CliRelay (https://github.com/kittors/CliRelay, an enhanced CLIProxyAPI fork)
is a self-hosted proxy that turns AI CLI subscriptions and OAuth credentials
into a single OpenAI-compatible API — default http://localhost:8317/v1 — with
routing, failover, and request logging handled server-side. This backend
speaks the standard Chat Completions protocol to that endpoint, so the same
code drives any vision model the relay routes to (Gemini, Claude, GPT, ...).

Prompting protocol: although CliRelay supports function calling, tool-call
translation fidelity varies across the relayed upstream providers. To stay
provider-agnostic we reuse the stateless text protocol proven with the old
gemini_web backend — every turn sends one self-contained prompt (task, tool
catalog rendered as text, compact action history) plus the current screenshot
as an image part, and parses a single JSON action object from the reply.

Configuration (configs/*.yaml, agent section):
  vlm_backend: cli_relay
  model:          model ID the relay routes, e.g. gpt-5.3-codex
  prompt:         task prompt variant (v1 / v2); see tasks.registry
  relay_base_url: relay endpoint; falls back to the CLIRELAY_BASE_URL env var
                  (used in Docker to reach a host-local relay), then to
                  http://localhost:8317/v1
API key resolution, first match wins: agent.relay_api_key config key,
CLIRELAY_API_KEY env, OPENAI_API_KEY env. If none is set a placeholder is
sent, which only works when the relay runs with allow-unauthenticated: true.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time

import numpy as np

from ..tasks.registry import load_prompt
from .actions import ACTION_TOOLS, Action, filter_tools

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8317/v1"

# Matches a ```json ... ``` fence, else the first {...} blob in the reply.
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def render_tool_catalog(tools: list[dict] | None = None) -> str:
    """Render tool schemas as plain text for the prompt-based action protocol."""
    lines = []
    for tool in tools if tools is not None else ACTION_TOOLS:
        fn = tool["function"]
        args = []
        for name, spec in fn["parameters"]["properties"].items():
            desc = spec.get("type", "any")
            if "enum" in spec:
                desc += " (one of: " + ", ".join(map(str, spec["enum"])) + ")"
            args.append(f'"{name}": {desc}')
        lines.append(f"- {fn['name']}: {fn['description']} Args: {{{', '.join(args)}}}")
    return "\n".join(lines)


RESPONSE_FORMAT = """\
Respond with ONLY one JSON object choosing your next action, no prose:
{"action": "<tool name>", "args": {<tool args>}}

Example: {"action": "move_toward", "args": {"pixel_x": 480, "pixel_y": 300, "amount": 0.7}}"""


CHOOSE_START_FORMAT = """\
Respond with ONLY one JSON object choosing your starting point, no prose:
{"action": "choose_start", "args": {"point": <point number>}}"""


def parse_start_choice(text: str, num_points: int) -> int | None:
    """Extract the chosen spawn point index from a free-text model reply."""
    match = _FENCED_JSON.search(text) or _BARE_JSON.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(1) if match.re is _FENCED_JSON else match.group(0))
    except json.JSONDecodeError:
        return None
    args = obj.get("args") or obj.get("arguments") or obj
    for key in ("point", "point_index", "index", "start"):
        value = args.get(key) if isinstance(args, dict) else None
        if value is not None:
            try:
                index = int(value)
            except (TypeError, ValueError):
                return None
            return index if 0 <= index < num_points else None
    return None


def parse_action(text: str, allowed: set[str] | None = None) -> Action | None:
    """Extract the first JSON action object from a free-text model reply."""
    match = _FENCED_JSON.search(text) or _BARE_JSON.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(1) if match.re is _FENCED_JSON else match.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("action") or obj.get("name") or obj.get("tool")
    if not isinstance(name, str):
        return None
    known = allowed if allowed is not None else {tool["function"]["name"] for tool in ACTION_TOOLS}
    if name not in known:
        return None
    args = obj.get("args") or obj.get("arguments") or {}
    return Action(name, args if isinstance(args, dict) else {})


def _png_data_url(image: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolve_api_key(configured: str) -> str:
    key = configured or os.environ.get("CLIRELAY_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        logger.warning(
            "No CliRelay API key set (agent.relay_api_key / CLIRELAY_API_KEY / "
            "OPENAI_API_KEY) — sending a placeholder; this only works if the "
            "relay has allow-unauthenticated: true."
        )
        key = "sk-unauthenticated"
    return key


class CliRelayPolicy:
    """Drives the explore loop through a CliRelay OpenAI-compatible endpoint.

    Stateless prompting: CliRelay's OpenAI-compatible /v1/chat/completions
    endpoint has no server-side discussion session — each request is
    independent. Every turn therefore resends the full task + tool catalog +
    a compact text history of prior actions, plus the current screenshot
    (and, when enabled, depth / bird's-eye path map / coverage). Context
    growth stays bounded regardless of episode length; spatial memory of
    where the agent has been comes from the optional coverage map, not a
    rolling chat transcript.
    """

    # Retries per step for empty/unparseable replies.
    MAX_ATTEMPTS = 3
    # Cap the action history included in the prompt.
    MAX_HISTORY_LINES = 20
    # Hard cap per request, seconds.
    REQUEST_TIMEOUT_S = 150

    def __init__(self, model: str, base_url: str = "", api_key: str = "",
                 prompt: str = ""):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package missing — pip install '.[vlm]'") from exc
        if not model:
            raise RuntimeError("Set agent.model to a model ID your CliRelay instance routes.")

        self.model = model
        self.base_url = base_url or os.environ.get("CLIRELAY_BASE_URL", "") or DEFAULT_BASE_URL
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=_resolve_api_key(api_key),
            timeout=self.REQUEST_TIMEOUT_S,
        )
        self._history: list[str] = []
        self._task = load_prompt(prompt or None)
        self._tools = filter_tools(getattr(self._task, "HIDDEN_TOOLS", ()))
        self.allow_done = "done" not in getattr(self._task, "HIDDEN_TOOLS", ())
        # Full record of the most recent decide(): prompt, per-attempt raw
        # replies + latencies, parse outcome. Consumed by the episode loop for
        # traces and by the debug dashboard.
        self.last_debug: dict | None = None
        logger.info("CliRelay backend: %s via %s (prompt %s)",
                    self.model, self.base_url, prompt or "default")

    def choose_start(self, birdseye_image: np.ndarray, spawn) -> int:
        """Initial prompt: bird's-eye view + numbered spawn points, returns the
        index the model picked (falls back to the top-ranked point 0)."""
        prompt = (
            f"{self._task.SPAWN_PROMPT}\n"
            f"Candidate starting points:\n{spawn.describe_points()}\n\n"
            f"{CHOOSE_START_FORMAT}"
        )
        images = [("Bird's-eye view (numbered dots = starting points):", _png_data_url(birdseye_image))]

        choice = None
        attempts: list[dict] = []
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            t0 = time.perf_counter()
            text, error = self._ask(prompt, images)
            seconds = time.perf_counter() - t0
            parsed = parse_start_choice(text, len(spawn.points)) if text else None
            attempts.append({
                "attempt": attempt,
                "seconds": round(seconds, 3),
                "reply": text,
                "error": error,
                "parsed_ok": parsed is not None,
            })
            if parsed is not None:
                choice = parsed
                break
            logger.warning("CliRelay: no parseable start choice (attempt %d)", attempt)

        fallback = choice is None
        if fallback:
            choice = 0
            logger.error("CliRelay: falling back to spawn point 0 after %d attempts", self.MAX_ATTEMPTS)

        self.last_debug = {
            "backend": "cli_relay",
            "model": self.model,
            "base_url": self.base_url,
            "prompt": prompt,
            "attempts": attempts,
            "raw_response": attempts[-1]["reply"] if attempts else "",
            "parsed_action": {"name": "choose_start", "args": {"point": choice}},
            "fallback": fallback,
        }
        self._history.append(f"start: chose spawn point {choice} from the bird's-eye view")
        logger.info("CliRelay chose start point: %d", choice)
        return choice

    def decide(
        self,
        observation: np.ndarray,
        pose_description: str,
        step: int,
        depth_image: np.ndarray | None = None,
        map_image: np.ndarray | None = None,
        coverage_image: np.ndarray | None = None,
    ) -> Action:
        prompt = self._build_prompt(
            pose_description, step,
            with_depth=depth_image is not None,
            with_map=map_image is not None,
            with_coverage=coverage_image is not None,
        )
        images = [("Image 1 - RGB view from your current pose:", _png_data_url(observation))]
        n = 2
        if depth_image is not None:
            images.append((
                f"Image {n} - DEPTH MAP of the same view (bright = near, dark = far, black = nothing):",
                _png_data_url(depth_image),
            ))
            n += 1
        if map_image is not None:
            images.append((
                f"Image {n} - BIRD'S-EYE MAP (ceiling removed; red line = path, "
                "triangles = camera view at each step, cyan = current pose, "
                "pale gold W# = jump_to_waypoint vantages). "
                "For jump look: 12/north is the TOP of this map, 3/east the right, "
                "6/south the bottom, 9/west the left. "
                "move_toward pixels still refer to the RGB view:",
                _png_data_url(map_image),
            ))
            n += 1
        if coverage_image is not None:
            images.append((
                f"Image {n} - COVERAGE MAP (yellow-green = viewed floor, stronger "
                "nearby, fades with distance, overlaps stack; coverage % on the label). "
                "This is your search history so far — unshaded rooms have not been "
                "looked at. move_toward pixels still refer to the RGB view:",
                _png_data_url(coverage_image),
            ))

        action = None
        attempts: list[dict] = []
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            t0 = time.perf_counter()
            text, error = self._ask(prompt, images)
            seconds = time.perf_counter() - t0
            parsed = parse_action(text, {t["function"]["name"] for t in self._tools}) if text else None
            attempts.append({
                "attempt": attempt,
                "seconds": round(seconds, 3),
                "reply": text,
                "error": error,
                "parsed_ok": parsed is not None,
            })
            if parsed:
                logger.debug("CliRelay reply (step %d, attempt %d): %s", step, attempt, text)
                action = parsed
                break
            logger.warning("CliRelay: no parseable action (step %d, attempt %d)", step, attempt)

        fallback = action is None
        if fallback:
            # Keep the episode alive rather than crash mid-run; the trace will
            # show the fallback so failures stay visible.
            action = Action("rotate", {"yaw_degrees": 45.0})
            logger.error("CliRelay: falling back to %s after %d attempts", action, self.MAX_ATTEMPTS)

        self.last_debug = {
            "backend": "cli_relay",
            "model": self.model,
            "base_url": self.base_url,
            "prompt": prompt,
            "attempts": attempts,
            "raw_response": attempts[-1]["reply"] if attempts else "",
            "parsed_action": {"name": action.name, "args": action.args},
            "fallback": fallback,
        }
        self._history.append(f"step {step}: {action.name} {json.dumps(action.args)}")
        logger.info("CliRelay chose: %s %s", action.name, action.args)
        return action

    def _ask(self, prompt: str, images: list[tuple[str, str]]) -> tuple[str, str | None]:
        """Send one prompt with labelled images. Returns (reply_text, error);
        exactly one of the two is meaningful."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for label, url in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": url}})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            logger.exception("CliRelay request failed")
            return "", f"{type(exc).__name__}: {exc}"
        choices = response.choices or []
        if not choices:
            return "", "relay returned no choices"
        return (choices[0].message.content or "").strip(), None

    def _build_prompt(self, pose_description: str, step: int,
                      with_depth: bool, with_map: bool = False,
                      with_coverage: bool = False) -> str:
        history = self._history[-self.MAX_HISTORY_LINES:]
        history_block = (
            "Your previous actions:\n" + "\n".join(history)
            if history
            else "This is your first step; no actions taken yet."
        )
        attached = ["your current RGB view"]
        if with_depth:
            attached.append("its depth map")
        if with_map:
            attached.append(
                "a bird's-eye MAP of your path so far (ceiling removed; "
                "frustums = viewing direction, pale gold W# = jump waypoints; "
                "12/north = top of the map for jump look)"
            )
        if with_coverage:
            attached.append(
                "a COVERAGE MAP of the floor you have looked at (yellow-green cones)"
            )
        extras = (
            (["the depth map"] if with_depth else [])
            + (["the path map"] if with_map else [])
            + (["the coverage map"] if with_coverage else [])
        )
        if len(attached) == 1:
            images_note = (
                "The attached image is your current RGB view. Pixel coordinates "
                "for move_toward refer to it."
            )
        else:
            listed = (
                " and ".join([", ".join(attached[:-1]), attached[-1]])
                if len(attached) > 2 else " and ".join(attached)
            )
            images_note = (
                f"The attached images are {listed} (labelled). "
                "Pixel coordinates for move_toward refer to the RGB view"
                + (f", not {' or '.join(extras)}." if extras else ".")
            )
        return (
            f"{self._task.system_prompt(with_depth, with_map, with_coverage)}\n"
            f"Available tools:\n{render_tool_catalog(self._tools)}\n\n"
            f"{history_block}\n\n"
            f"Step {step}. Current pose: {pose_description}. "
            f"{images_note}\n\n"
            f"{RESPONSE_FORMAT}"
        )
