"""VLM policy backends: given the latest observation, pick the next action.

Backends:
  - ScriptedPolicy: canned action sequence, no network (loop smoke-testing).
  - GeminiWebPolicy (gemini_web.py): Gemini through browser-session cookies
    via gemini_webapi. Free stand-in VLM for research/testing.
  - OpenAIVLMPolicy: STUB — any OpenAI-compatible vision endpoint with tool
    calling. Written but not yet exercised against a live API.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Protocol

import numpy as np

from ..tasks.artifact_hunt import SYSTEM_PROMPT
from .actions import ACTION_TOOLS, Action

logger = logging.getLogger(__name__)


class VLMPolicy(Protocol):
    def decide(self, observation: np.ndarray, pose_description: str, step: int) -> Action:
        """Return the next action given the current rendered view."""
        ...


class ScriptedPolicy:
    """Rotates in place, then walks a small square. Useful for smoke-testing
    the render->act->render loop and episode logging without a VLM."""

    def __init__(self):
        self._script = (
            [Action("rotate", {"yaw_degrees": 45.0})] * 8
            + [
                Action("move", {"direction": "forward", "distance": 1.0}),
                Action("rotate", {"yaw_degrees": 90.0}),
            ] * 4
            + [Action("done", {"summary": "Scripted run complete."})]
        )

    def decide(self, observation: np.ndarray, pose_description: str, step: int) -> Action:
        if step < len(self._script):
            return self._script[step]
        return Action("done", {"summary": "Script exhausted."})


def _encode_png_b64(image: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class OpenAIVLMPolicy:
    """STUB: drives any OpenAI-compatible chat completions endpoint.

    Maintains a rolling conversation: system prompt + (image, pose) user turns
    + the model's tool calls. Not yet exercised against a live API; validate
    tool-call parsing and context-window trimming before real runs.
    """

    # Keep only the most recent N images in context to bound token usage.
    MAX_IMAGES_IN_CONTEXT = 6

    def __init__(self, model: str, base_url: str = ""):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package missing — pip install '.[vlm]'") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use the openai VLM backend.")
        self.client = OpenAI(base_url=base_url or None)
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def decide(self, observation: np.ndarray, pose_description: str, step: int) -> Action:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Step {step}. Current pose: {pose_description}. Choose exactly one tool call."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png_b64(observation)}"}},
                ],
            }
        )
        self._trim_old_images()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=ACTION_TOOLS,
            tool_choice="required",
        )
        message = response.choices[0].message
        # TODO: append the assistant message + a tool result message so the
        # model sees its own history; handle refusals / malformed calls.
        call = message.tool_calls[0]
        import json

        action = Action(call.function.name, json.loads(call.function.arguments))
        logger.info("VLM chose: %s %s", action.name, action.args)
        return action

    def _trim_old_images(self) -> None:
        # TODO: replace old image parts with "[image removed]" placeholders
        # instead of relying on unbounded context.
        pass


def make_policy(agent_cfg) -> VLMPolicy:
    backend = agent_cfg.vlm_backend
    if backend == "scripted":
        return ScriptedPolicy()
    if backend == "gemini_web":
        from .gemini_web import GeminiWebPolicy

        return GeminiWebPolicy(
            cookie_file=agent_cfg.get("cookie_file", ""),
            chrome_profile=agent_cfg.get("chrome_profile", ""),
            auto_cookies=bool(agent_cfg.get("auto_cookies", False)),
        )
    if backend == "openai":
        return OpenAIVLMPolicy(model=agent_cfg.model, base_url=agent_cfg.get("base_url", ""))
    raise ValueError(f"Unknown vlm_backend: {backend!r}")
