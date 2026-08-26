"""VLM policy backends: given the latest observation, pick the next action.

Backends:
  - ScriptedPolicy: canned action sequence, no network (loop smoke-testing).
  - CliRelayPolicy (cli_relay.py): Gemini/Claude/OpenAI through a self-hosted
    CliRelay proxy (OpenAI-compatible endpoint, prompt-based action protocol).
  - OpenAIVLMPolicy: STUB — any OpenAI-compatible vision endpoint with native
    tool calling. Written but not yet exercised against a live API.
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
    def decide(
        self,
        observation: np.ndarray,
        pose_description: str,
        step: int,
        depth_image: np.ndarray | None = None,
        map_image: np.ndarray | None = None,
    ) -> Action:
        """Return the next action given the current rendered RGB view and,
        when attached, the labelled depth-map and/or bird's-eye path map."""
        ...


class ScriptedPolicy:
    """Canned sequence covering every motion action (rotate with yaw/pitch,
    move, move_toward at the image center). Useful for smoke-testing the
    render->act->render loop, collision clamping, and episode logging
    without a VLM."""

    def __init__(self):
        self._script = (
            [Action("rotate", {"yaw_degrees": 45.0})] * 4
            + [Action("view_map", {})]
            + [Action("rotate", {"yaw_degrees": 45.0})] * 4
            + [
                Action("rotate", {"pitch_degrees": -20.0}),
                Action("rotate", {"yaw_degrees": 30.0, "pitch_degrees": 0.0}),
            ]
            + [
                # pixel -1/-1 is a placeholder resolved to the image center in
                # decide(), where the actual render resolution is known.
                Action("move_toward", {"pixel_x": -1, "pixel_y": -1, "amount": 0.5}),
                Action("move", {"direction": "forward", "distance": 1.0}),
                Action("rotate", {"yaw_degrees": 90.0}),
            ] * 3
            + [Action("done", {"summary": "Scripted run complete."})]
        )
        self.last_debug: dict | None = None

    def choose_start(self, birdseye_image: np.ndarray, spawn) -> int:
        """Scripted runs always take the top-ranked spawn point."""
        self.last_debug = {
            "backend": "scripted",
            "raw_response": "(scripted: picked spawn point 0)",
            "parsed_action": {"name": "choose_start", "args": {"point": 0}},
            "fallback": False,
        }
        return 0

    def decide(
        self,
        observation: np.ndarray,
        pose_description: str,
        step: int,
        depth_image: np.ndarray | None = None,
        map_image: np.ndarray | None = None,
    ) -> Action:
        action = self._script[step] if step < len(self._script) else Action("done", {"summary": "Script exhausted."})
        if action.name == "move_toward" and (
            action.args.get("pixel_x", 0) < 0 or action.args.get("pixel_y", 0) < 0
        ):
            # Script entries are shared objects; build a fresh Action instead
            # of mutating the placeholder in place.
            height, width = observation.shape[:2]
            action = Action("move_toward", {
                **action.args,
                "pixel_x": width // 2,
                "pixel_y": height // 2,
            })
        self.last_debug = {
            "backend": "scripted",
            "raw_response": f"(scripted step {step}/{len(self._script)})",
            "parsed_action": {"name": action.name, "args": action.args},
            "fallback": False,
        }
        return action


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

    def decide(
        self,
        observation: np.ndarray,
        pose_description: str,
        step: int,
        depth_image: np.ndarray | None = None,
        map_image: np.ndarray | None = None,
    ) -> Action:
        content = [
            {"type": "text", "text": f"Step {step}. Current pose: {pose_description}. Choose exactly one tool call."},
            {"type": "text", "text": "Image 1 — RGB view:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png_b64(observation)}"}},
        ]
        if depth_image is not None:
            content += [
                {"type": "text", "text": "Image 2 — depth map of the same view (bright = near, black = nothing):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png_b64(depth_image)}"}},
            ]
        n = 3 if depth_image is not None else 2
        if map_image is not None:
            content += [
                {"type": "text", "text": (
                    f"Image {n} — bird's-eye MAP (ceiling removed; red line = path, "
                    "triangles = camera view, cyan = current pose):"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png_b64(map_image)}"}},
            ]
        self.messages.append({"role": "user", "content": content})
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
    if backend == "cli_relay":
        from .cli_relay import CliRelayPolicy

        return CliRelayPolicy(
            model=agent_cfg.model,
            base_url=agent_cfg.get("relay_base_url", ""),
            api_key=agent_cfg.get("relay_api_key", ""),
        )
    if backend == "openai":
        return OpenAIVLMPolicy(model=agent_cfg.model, base_url=agent_cfg.get("base_url", ""))
    raise ValueError(f"Unknown vlm_backend: {backend!r}")
