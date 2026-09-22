"""RGB image-edit backends for regenerate=yes views.

Photometric 3DGS repair (`repair.py` / `repair_gsfix3d.py`) consumes a
repaired PNG. That PNG comes from CliRelay ``/v1/images/edits``
(gpt-image-2, or gpt-image-2.5 sunburst/flare) or, as a backup, a
self-hosted Qwen-Image-Edit-2511 pipeline on the connected GPU.

``QWEN_required`` defaults to false. GPU setup then skips the Qwen
package install; set it true (env, config, or the GPU loader buttons)
only when a run actually edits with Qwen.

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
# Scene-runs default. Sunburst is the GPT Image 2.5 edit model aimed at
# precise edits; flare is the faster 2.5 variant. gpt-image-2 stays available.
SCENE_RUN_GPT_IMAGE_MODEL = "gpt-image-2.5-sunburst"
GPT_IMAGE_FLARE = "gpt-image-2.5-flare"

# False: GPU setup does not install or load Qwen-Image-Edit.
# Assign this, or set env QWEN_required / image_edit.QWEN_required, anywhere
# the GPU loader can see it.
QWEN_required = False

_GPT_MODEL_ALIASES = {
    "gpt-image-2.5": SCENE_RUN_GPT_IMAGE_MODEL,
    "gpt-image-2.5-sunburst": SCENE_RUN_GPT_IMAGE_MODEL,
    "sunburst": SCENE_RUN_GPT_IMAGE_MODEL,
    "gpt-image-2.5-flare": GPT_IMAGE_FLARE,
    "flare": GPT_IMAGE_FLARE,
    "gpt-image-2": BACKEND_GPT,
    "gpt": BACKEND_GPT,
    "gptimage2": BACKEND_GPT,
    "gpt-image": BACKEND_GPT,
    "clirelay": BACKEND_GPT,
    "cli-relay": BACKEND_GPT,
    "openai": BACKEND_GPT,
}
_GPT_ALIASES = set(_GPT_MODEL_ALIASES)
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


def concrete_gpt_image_model(name: str | None) -> str | None:
    """Map a GPT image alias onto the CliRelay ``images.edit`` model id."""
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return None
    low = key.lower().replace("_", "-")
    if low in _GPT_MODEL_ALIASES:
        return _GPT_MODEL_ALIASES[low]
    compact = low.replace("-", "")
    for alias, model in _GPT_MODEL_ALIASES.items():
        if compact == alias.replace("-", ""):
            return model
    if low.startswith("gpt-image-"):
        return low
    return None


def canonical_image_edit_choice(name: str | None) -> str:
    """Scene-run selection: a concrete GPT model id, or ``qwen-image-edit``."""
    model = concrete_gpt_image_model(name)
    if model:
        return model
    key = str(name or "").strip()
    if not key:
        raise ValueError("image_edit_backend is required")
    try:
        if is_qwen_backend(key):
            return BACKEND_QWEN
    except ValueError:
        pass
    known = ", ".join((
        SCENE_RUN_GPT_IMAGE_MODEL, GPT_IMAGE_FLARE, BACKEND_GPT, BACKEND_QWEN,
    ))
    raise ValueError(f"Unknown image-edit backend {name!r}. Choose one of: {known}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "required"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"QWEN_required must be true or false, not {value!r}")


def resolve_qwen_required(
    value: Any = None,
    *,
    cfg=None,
    backend: str | None = None,
) -> bool:
    """Whether GPU setup should install and load Qwen-Image-Edit.

    An explicit ``value`` wins, then ``QWEN_required`` / ``QWEN_REQUIRED`` /
    ``SPLAT_QWEN_REQUIRED``, then ``image_edit.QWEN_required``, then the
    selected image-edit backend, then the module flag ``QWEN_required``.
    The default is false so CliRelay image edits stay off the GPU.
    """
    if value is not None:
        return _as_bool(value)
    for env_name in ("QWEN_required", "QWEN_REQUIRED", "SPLAT_QWEN_REQUIRED"):
        raw = os.environ.get(env_name)
        if raw is not None and str(raw).strip() != "":
            return _as_bool(raw)
    for key in ("QWEN_required", "qwen_required"):
        configured = _cfg_get(cfg, "image_edit", key)
        if configured is None:
            configured = _cfg_get(cfg, key)
        if configured is not None and configured != "":
            return _as_bool(configured)
    name = backend if backend not in (None, "") else _cfg_get(cfg, "image_edit", "backend")
    if name not in (None, ""):
        if concrete_gpt_image_model(str(name)):
            return False
        return is_qwen_backend(str(name))
    return bool(QWEN_required)


def resolve_gpt_image_model(name: str | None = None, cfg=None) -> str:
    """CliRelay image model. Scene-runs pass a 2.5 id; episodes stay on gpt-image-2."""
    for raw in (
        name,
        _cfg_get(cfg, "image_edit", "model"),
        _cfg_get(cfg, "image_edit", "backend"),
        _cfg_get(cfg, "agent", "image_edit_backend"),
    ):
        model = concrete_gpt_image_model(None if raw in (None, "") else str(raw))
        if model:
            return model
    return BACKEND_GPT


def _normalize_backend_name(name: str | None) -> str:
    key = str(name or "").strip().lower().replace("_", "-")
    compact = key.replace("-", "")
    if not key:
        return BACKEND_GPT
    if key in _QWEN_ALIASES or compact in {a.replace("-", "") for a in _QWEN_ALIASES}:
        return BACKEND_QWEN
    if concrete_gpt_image_model(key):
        return BACKEND_GPT
    known = ", ".join((
        SCENE_RUN_GPT_IMAGE_MODEL, GPT_IMAGE_FLARE, BACKEND_GPT, BACKEND_QWEN,
    ))
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
    """Pick the GPT family or qwen-image-edit.

    GPT Image 2.5 ids collapse to the gpt-image-2 family here; the concrete
    model id is :func:`resolve_gpt_image_model`. Order: explicit ``name``,
    ``SPLAT_IMAGE_EDIT_BACKEND`` / ``IMAGE_EDIT_BACKEND``,
    ``image_edit.backend``, ``agent.image_edit_backend``, default gpt-image-2.
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
                           device: int | str | None = None,
                           model: str | None = None):
    """Shallow copy of ``cfg`` with image_edit.backend / device / model overridden."""
    from .config import Config

    if isinstance(cfg, dict):
        base = dict(cfg)
        image_edit = dict(cfg.get("image_edit") or {})
    else:
        base = {}
        image_edit = {}
    if backend not in (None, ""):
        image_edit["backend"] = _normalize_backend_name(backend)
    if model not in (None, ""):
        image_edit["model"] = concrete_gpt_image_model(str(model)) or str(model)
    elif backend not in (None, ""):
        chosen = concrete_gpt_image_model(str(backend))
        if chosen:
            image_edit["model"] = chosen
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
    qwen_on = resolve_qwen_required(cfg=cfg)
    model = None if selected == BACKEND_QWEN else resolve_gpt_image_model(cfg=cfg)
    return {
        "selected": selected,
        "model": model,
        "QWEN_required": qwen_on,
        "device": device,
        "photometric_device": photo,
        "same_gpu_as_photometric": device == photo,
        "lrz_partitions": list(LRZ_QWEN_PARTITIONS),
        "backends": [
            {
                "id": SCENE_RUN_GPT_IMAGE_MODEL,
                "label": "gpt-image-2.5 sunburst (CliRelay)",
                "available": True,
                "detail": (
                    "GPT Image 2.5 sunburst via CliRelay /v1/images/edits. "
                    "Scene-run default. Precise edits; no local GPU."
                ),
            },
            {
                "id": GPT_IMAGE_FLARE,
                "label": "gpt-image-2.5 flare (CliRelay)",
                "available": True,
                "detail": (
                    "GPT Image 2.5 flare via CliRelay /v1/images/edits. "
                    "Faster 2.5 variant; no local GPU."
                ),
            },
            {
                "id": BACKEND_GPT,
                "label": "gpt-image-2 (CliRelay)",
                "available": True,
                "detail": (
                    "Paid OpenAI-compatible /v1/images/edits through CliRelay. "
                    "No local GPU."
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

    model = concrete_gpt_image_model(name) if name not in (None, "") else None
    return GptImageEditBackend.from_config(cfg, client=client, model=model)
