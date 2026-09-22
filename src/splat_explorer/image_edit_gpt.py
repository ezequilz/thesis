"""gpt-image-2 RGB repair via CliRelay ``/v1/images/edits``.

Keeps the existing paid path so Qwen can be swapped out without deleting
the API backend. Callers go through ``image_edit.make_image_edit_backend``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .image_edit import BACKEND_GPT, ImageEditResult, _cfg_get


class GptImageEditBackend:
    """Thin wrapper around ``agent.regenerate.ask_regenerate``."""

    name = BACKEND_GPT

    def __init__(self, client, model: str = BACKEND_GPT, timeout_s: float | None = None):
        self.client = client
        self.model = model or BACKEND_GPT
        self.name = self.model
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, cfg=None, client=None, model: str | None = None) -> "GptImageEditBackend":
        from .agent.regenerate import REQUEST_TIMEOUT_S
        from .image_edit import resolve_gpt_image_model

        if client is None:
            client = _cli_relay_client(cfg)
        timeout = _cfg_get(cfg, "image_edit", "timeout_s", default=REQUEST_TIMEOUT_S)
        try:
            timeout_s = float(timeout)
        except (TypeError, ValueError):
            timeout_s = REQUEST_TIMEOUT_S
        chosen = model or resolve_gpt_image_model(cfg=cfg)
        return cls(client=client, model=chosen, timeout_s=timeout_s)

    def edit(self, image_path: Path, prompt: str) -> ImageEditResult:
        from .agent.regenerate import REQUEST_TIMEOUT_S, ask_regenerate, extract_images

        timeout = self.timeout_s if self.timeout_s is not None else REQUEST_TIMEOUT_S
        payload, error = ask_regenerate(
            self.client, self.model, Path(image_path),
            timeout_s=timeout, prompt=prompt,
        )
        images = extract_images(payload) if not error else []
        body: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
        body.setdefault("backend", BACKEND_GPT)
        body.setdefault("model", self.model)
        return ImageEditResult(images=images, payload=body, error=error)


def _cli_relay_client(cfg=None):
    from .agent.cli_relay import CliRelayPolicy
    from .agent.regenerate import IMAGE_MODEL

    agent = _cfg_get(cfg, "agent", default={}) or {}
    if not isinstance(agent, dict):
        agent = dict(agent)
    policy = CliRelayPolicy(
        model=IMAGE_MODEL,
        base_url=str(agent.get("relay_base_url") or ""),
        api_key=str(agent.get("relay_api_key") or ""),
    )
    return policy.client
