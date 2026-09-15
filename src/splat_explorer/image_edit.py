"""RGB image-edit backends for regenerate=yes views.

Photometric 3DGS repair (`repair.py` / `repair_gsfix3d.py`) consumes a
repaired PNG. That PNG used to come only from gpt-image-2 via CliRelay
(`agent.regenerate`). This module is the swap point: the same
RGB-in / instruction-in / PNG-out contract, either CliRelay or a
self-hosted Qwen-Image-Edit-2511 pipeline on the connected GPU.

The VLM agent is unchanged. It still flags views; the local Regenerator
queue sends those frames here one by one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Same visible device photometric CUDA uses (`torch.device("cuda")` after
# CUDA_VISIBLE_DEVICES remap). LRZ `srun --gres=gpu:1` already pins one
# card on HGX A100 80GB ×4, DGX A100 80GB ×8, or HGX H100 94GB ×4.
PHOTOMETRIC_CUDA_DEVICE = "cuda:0"

BACKEND_GPT = "gpt-image-2"
BACKEND_QWEN = "qwen-image-edit"

_GPT_ALIASES = {
    "gpt-image-2",
    "gpt",
    "gptimage2",
    "gpt-image",
    "clirelay",
    "cli_relay",
    "openai",
}
_QWEN_ALIASES = {
    "qwen-image-edit",
    "qwen",
    "qwen-image-edit-2511",
    "qwen-image-edit-plus",
    "qwenimageedit",
    "qwenimage",
}

_ENV_BACKEND = ("SPLAT_IMAGE_EDIT_BACKEND", "IMAGE_EDIT_BACKEND")
_ENV_DEVICE = ("SPLAT_IMAGE_EDIT_DEVICE", "IMAGE_EDIT_DEVICE")

# Partitions this Qwen path is sized for (bf16, ~40GB weights + headroom
# so photometric's <3% VRAM can stay on the same card).
LRZ_QWEN_PARTITIONS = (
    "lrz-hgx-a100-80x4",
    "lrz-dgx-a100-80x8",
    "lrz-hgx-h100-94x4",
)


@dataclass
class ImageEditResult:
    """One RGB repair: PNG bytes plus a JSON-safe payload for regen meta."""

    images: list[bytes] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ImageEditBackend(Protocol):
    """RGB image + instruction → repaired RGB image(s)."""

    name: str

    def edit(self, image_path: Path, prompt: str) -> ImageEditResult:
        ...


def photometric_cuda_device() -> str:
    """Device string photometric CUDA repair uses in-process."""
    return PHOTOMETRIC_CUDA_DEVICE


def is_qwen_backend(name: str | None) -> bool:
    return _normalize_backend_name(name) == BACKEND_QWEN


def is_gpt_backend(name: str | None) -> bool:
    return _normalize_backend_name(name) == BACKEND_GPT


def _normalize_backend_name(name: str | None) -> str:
    key = str(name or "").strip().lower().replace("_", "-")
    compact = key.replace("-", "")
    if not key:
        return BACKEND_GPT
    if key in _QWEN_ALIASES or compact in {a.replace("-", "") for a in _QWEN_ALIASES}:
        return BACKEND_QWEN
    if key in _GPT_ALIASES or compact in {a.replace("-", "") for a in _GPT_ALIASES}:
        return BACKEND_GPT
    known = ", ".join((BACKEND_GPT, BACKEND_QWEN))
    raise ValueError(f"Unknown image-edit backend {name!r}. Choose one of: {known}")


def _cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is default and key != keys[-1]:
            return default
    return default if cur is None else cur


def _env_first(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def resolve_image_edit_backend(name: str | None = None, cfg=None) -> str:
    """Pick gpt-image-2 or qwen-image-edit.

    Order: explicit ``name``, ``SPLAT_IMAGE_EDIT_BACKEND`` /
    ``IMAGE_EDIT_BACKEND``, ``image_edit.backend``,
    ``agent.image_edit_backend``, default gpt-image-2.
    """
    if name not in (None, ""):
        return _normalize_backend_name(name)
    env = _env_first(_ENV_BACKEND)
    if env:
        return _normalize_backend_name(env)
    configured = _cfg_get(cfg, "image_edit", "backend")
    if not configured:
        configured = _cfg_get(cfg, "agent", "image_edit_backend")
    return _normalize_backend_name(configured or BACKEND_GPT)


def resolve_image_edit_device(cfg=None, device: int | str | None = None) -> str:
    """CUDA device for Qwen. Default ``cuda:0`` — same GPU as photometric.

    Pass an index (``1``) or ``cuda:1`` for a dedicated card that is already
    visible in this process. Do not fan out across all 4/8 LRZ GPUs; Slurm
    ``--gres=gpu:1`` plus this index is enough. Never calls
    ``torch.cuda.set_device`` (that would steal photometric's current device).
    """
    if device is None or device == "":
        env = _env_first(_ENV_DEVICE)
        if env:
            device = env
        else:
            device = _cfg_get(cfg, "image_edit", "device", default=0)
    if device is None or device == "":
        device = 0
    text = str(device).strip().lower()
    if text.startswith("cuda"):
        return text if ":" in text else "cuda:0"
    if text in ("cpu", "mps"):
        return text
    try:
        index = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"image_edit.device must be a CUDA index or cuda:N, not {device!r}"
        ) from exc
    if index < 0:
        return "cpu"
    return f"cuda:{index}"


def image_edit_shares_photometric_gpu(cfg=None, device: int | str | None = None) -> bool:
    return resolve_image_edit_device(cfg, device) == photometric_cuda_device()


def overlay_image_edit_cfg(cfg=None, *, backend: str | None = None,
                           device: int | str | None = None):
    """Shallow copy of ``cfg`` with image_edit.backend / device overridden."""
    from .config import Config

    if isinstance(cfg, dict):
        base = dict(cfg)
        image_edit = dict(cfg.get("image_edit") or {})
    else:
        base = {}
        image_edit = {}
    if backend not in (None, ""):
        image_edit["backend"] = _normalize_backend_name(backend)
    if device is not None and device != "":
        image_edit["device"] = device
    if image_edit:
        base["image_edit"] = image_edit
    return Config(base)


def apply_image_edit_overrides(cfg, *, backend: str | None = None,
                               device: int | str | None = None):
    """Mutate a loaded Config in place from CLI flags."""
    if backend in (None, "") and device in (None, ""):
        return cfg
    image_edit = dict(cfg.get("image_edit") or {}) if isinstance(cfg, dict) else {}
    if backend not in (None, ""):
        image_edit["backend"] = _normalize_backend_name(backend)
    if device is not None and device != "":
        image_edit["device"] = device
    cfg["image_edit"] = image_edit
    return cfg


def list_image_edit_backends(cfg=None) -> dict[str, Any]:
    """Catalog for the dashboard / repair-studio RGB-repair dropdown."""
    selected = resolve_image_edit_backend(cfg=cfg)
    device = resolve_image_edit_device(cfg)
    photo = photometric_cuda_device()
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    qwen_detail = (
        f"Qwen-Image-Edit-2511 (Apache 2.0) on {device}, bf16, resident "
        f"weights, one view at a time. Default {device} is the same visible "
        f"GPU as photometric CUDA ({photo}). LRZ A100 80GB / H100 94GB with "
        f"--gres=gpu:1 already share that card. Dedicated GPU: set "
        f"image_edit.device or SPLAT_IMAGE_EDIT_DEVICE to another index. "
        f"pip install -e '.[image-edit]'."
    )
    if not cuda:
        qwen_detail = (
            "Needs a connected NVIDIA GPU (same card as photometric when "
            "feasible). " + qwen_detail
        )
    return {
        "selected": selected,
        "device": device,
        "photometric_device": photo,
        "same_gpu_as_photometric": device == photo,
        "lrz_partitions": list(LRZ_QWEN_PARTITIONS),
        "backends": [
            {
                "id": BACKEND_GPT,
                "label": "gpt-image-2 (CliRelay)",
                "available": True,
                "detail": (
                    "Paid OpenAI-compatible /v1/images/edits through CliRelay. "
                    "Default; no local GPU."
                ),
            },
            {
                "id": BACKEND_QWEN,
                "label": "Qwen-Image-Edit-2511 (GPU)",
                "available": cuda,
                "detail": qwen_detail,
            },
        ],
    }


def make_image_edit_backend(
    name: str | None = None,
    *,
    cfg=None,
    client=None,
) -> ImageEditBackend:
    """Instantiate the selected RGB-edit backend. Qwen loads weights lazily."""
    key = resolve_image_edit_backend(name, cfg=cfg)
    if key == BACKEND_QWEN:
        from .image_edit_qwen import QwenImageEditBackend

        return QwenImageEditBackend.from_config(cfg)
    from .image_edit_gpt import GptImageEditBackend

    return GptImageEditBackend.from_config(cfg, client=client)
