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
  model:          model ID the relay routes, e.g. gemini-2.5-pro
  relay_base_url: relay endpoint, default http://localhost:8317/v1
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

import numpy as np

from ..tasks.artifact_hunt import SYSTEM_PROMPT
from .actions import ACTION_TOOLS, Action

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8317/v1"

# Matches a ```json ... ``` fence, else the first {...} blob in the reply.
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def render_tool_catalog() -> str:
    """Render ACTION_TOOLS as plain text for the prompt-based action protocol."""
    lines = []
    for tool in ACTION_TOOLS:
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

Example: {"action": "move", "args": {"direction": "forward", "distance": 1.0}}"""


def parse_action(text: str) -> Action | None:
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
    known = {tool["function"]["name"] for tool in ACTION_TOOLS}
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

    Stateless prompting: full task + tool catalog + action history is resent
    every turn, so no server-side conversation state is required and context
    growth stays bounded regardless of episode length.
    """

    # Retries per step for empty/unparseable replies.
    MAX_ATTEMPTS = 3
    # Cap the action history included in the prompt.
    MAX_HISTORY_LINES = 20
    # Hard cap per request, seconds.
    REQUEST_TIMEOUT_S = 150

    def __init__(self, model: str, base_url: str = "", api_key: str = ""):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package missing — pip install '.[vlm]'") from exc
        if not model:
            raise RuntimeError("Set agent.model to a model ID your CliRelay instance routes.")

        self.model = model
        self.base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=_resolve_api_key(api_key),
            timeout=self.REQUEST_TIMEOUT_S,
        )
        self._history: list[str] = []
        logger.info("CliRelay backend: %s via %s", self.model, self.base_url)

    def decide(self, observation: np.ndarray, pose_description: str, step: int) -> Action:
        prompt = self._build_prompt(pose_description, step)
        image_url = _png_data_url(observation)

        action = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            text = self._ask(prompt, image_url)
            if text:
                logger.debug("CliRelay reply (step %d, attempt %d): %s", step, attempt, text)
                action = parse_action(text)
                if action:
                    break
            logger.warning("CliRelay: no parseable action (step %d, attempt %d)", step, attempt)

        if action is None:
            # Keep the episode alive rather than crash mid-run; the trace will
            # show the fallback so failures stay visible.
            action = Action("rotate", {"yaw_degrees": 45.0})
            logger.error("CliRelay: falling back to %s after %d attempts", action, self.MAX_ATTEMPTS)

        self._history.append(f"step {step}: {action.name} {json.dumps(action.args)}")
        logger.info("CliRelay chose: %s %s", action.name, action.args)
        return action

    def _ask(self, prompt: str, image_url: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )
        except Exception:
            logger.exception("CliRelay request failed")
            return ""
        choices = response.choices or []
        if not choices:
            return ""
        return (choices[0].message.content or "").strip()

    def _build_prompt(self, pose_description: str, step: int) -> str:
        history = self._history[-self.MAX_HISTORY_LINES:]
        history_block = (
            "Your previous actions:\n" + "\n".join(history)
            if history
            else "This is your first step; no actions taken yet."
        )
        return (
            f"{SYSTEM_PROMPT}\n"
            f"Available tools:\n{render_tool_catalog()}\n\n"
            f"{history_block}\n\n"
            f"Step {step}. Current pose: {pose_description}. "
            f"The attached image is your current view.\n\n"
            f"{RESPONSE_FORMAT}"
        )
