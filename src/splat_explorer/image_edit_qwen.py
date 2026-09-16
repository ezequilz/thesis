"""Self-hosted Qwen-Image-Edit-2511 RGB repair (Apache 2.0).

Loads ``Qwen/Qwen-Image-Edit-2511`` via Diffusers
``QwenImageEditPlusPipeline`` (the class the model card specifies) on a
single CUDA device. Default placement is ``cuda:0`` — the same visible
GPU photometric GSFix3D uses after ``CUDA_VISIBLE_DEVICES`` remap.

Memory (A100 80GB / H100 94GB, one Slurm GPU):

- Weights stay resident in bf16 (~20B params, ~40GB). Photometric refine
  is <3% VRAM, so co-residency on the allocated card is the default.
- ``from_pretrained(..., low_cpu_mem_usage=True)`` avoids duplicating the
  20B weights on the CPU. Scene-runs still need a 128G/256G *host* RAM
  cgroup; 64G holds isolate Qwen in a subprocess.
- VAE tiling on; attention slicing / sequential CPU offload only when
  configured (OOM path).
- Never ``torch.cuda.set_device`` during a live photometric kernel.
  ``empty_cache`` is OK *between* Qwen edit and GSFix, not during either.
  Dedicated GPU: ``image_edit.device: 1`` (or ``SPLAT_IMAGE_EDIT_DEVICE``)
  without changing the current CUDA device.

Weights are not reloaded per view. First ``edit()`` instantiates the
pipeline; later one-by-one Regenerator jobs reuse it.

Install: ``pip install -e '.[image-edit]'``. First-time fetch is ~20GB
and is refused unless the Hugging Face cache already has the repo or
``image_edit.download`` / ``SPLAT_IMAGE_EDIT_DOWNLOAD=1`` is set.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from .image_edit import (
    BACKEND_QWEN,
    ImageEditResult,
    _cfg_get,
    resolve_image_edit_device,
)

logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
# Model-card defaults for QwenImageEditPlusPipeline.
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_NEGATIVE_PROMPT = " "
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SEED = 0


def qwen_hub_dir_name(model_id: str = QWEN_MODEL_ID) -> str:
    return "models--" + str(model_id).replace("/", "--")


def huggingface_hub_roots() -> list[Path]:
    roots: list[Path] = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env, "").strip()
        if value:
            roots.append(Path(value))
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        roots.append(Path(home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def qwen_weights_cached(model_id: str = QWEN_MODEL_ID) -> bool:
    """True when a local Hugging Face snapshot of the edit model exists."""
    name = qwen_hub_dir_name(model_id)
    for root in huggingface_hub_roots():
        snapshots = root / name / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in snapshots.iterdir():
            if snap.is_dir() and any(snap.iterdir()):
                return True
    return False


def resolve_qwen_local_files_only(cfg=None) -> bool:
    """Refuse a 20GB download unless explicitly allowed or already cached."""
    env = os.environ.get("SPLAT_IMAGE_EDIT_DOWNLOAD", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return True
    if env in ("1", "true", "yes", "on"):
        return False
    configured = _cfg_get(cfg, "image_edit", "download")
    if configured is False:
        return True
    if configured is True:
        return False
    return True


def resolve_torch_dtype(name: str | None, torch, device: str):
    """bf16 on A100/H100; fp16 if this CUDA device cannot do bf16."""
    key = str(name or "bfloat16").lower().replace("-", "").replace("_", "")
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("fp32", "float32"):
        return torch.float32
    bf16 = getattr(torch, "bfloat16", None)
    if bf16 is None:
        return torch.float16
    if str(device).startswith("cuda"):
        check = getattr(getattr(torch, "cuda", None), "is_bf16_supported", None)
        if callable(check):
            try:
                if not check():
                    return torch.float16
            except Exception:
                pass
    return bf16


def build_qwen_pipeline(
    *,
    model_id: str,
    torch_dtype,
    device: str,
    vae_tiling: bool,
    attention_slicing: bool,
    offload: str,
    local_files_only: bool,
):
    """Official Diffusers load. Isolated so tests can mock without 20GB weights."""
    from diffusers import QwenImageEditPlusPipeline

    kwargs = dict(
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("low_cpu_mem_usage", None)
        pipeline = QwenImageEditPlusPipeline.from_pretrained(model_id, **kwargs)
    offload_key = str(offload or "none").strip().lower()
    if offload_key in ("sequential", "sequential_cpu", "cpu"):
        pipeline.enable_sequential_cpu_offload()
    elif offload_key in ("model", "model_cpu", "cpu_offload"):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    if vae_tiling:
        for attr in ("enable_vae_tiling", "enable_tiling"):
            fn = getattr(pipeline, attr, None)
            if callable(fn):
                fn()
                break
    if attention_slicing:
        fn = getattr(pipeline, "enable_attention_slicing", None)
        if callable(fn):
            fn()
    progress = getattr(pipeline, "set_progress_bar_config", None)
    if callable(progress):
        progress(disable=True)
    return pipeline


class QwenImageEditBackend:
    """Lazy, process-resident Qwen image-edit pipeline."""

    name = BACKEND_QWEN

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        model_id: str = QWEN_MODEL_ID,
        vae_tiling: bool = True,
        attention_slicing: bool = False,
        offload: str = "none",
        local_files_only: bool = True,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int | None = DEFAULT_SEED,
        pipeline=None,
        pipeline_builder=None,
    ):
        self.device = device
        self.dtype_name = dtype
        self.model_id = model_id
        self.vae_tiling = bool(vae_tiling)
        self.attention_slicing = bool(attention_slicing)
        self.offload = str(offload or "none")
        self.local_files_only = bool(local_files_only)
        self.num_inference_steps = int(num_inference_steps)
        self.true_cfg_scale = float(true_cfg_scale)
        self.negative_prompt = negative_prompt
        self.guidance_scale = float(guidance_scale)
        self.seed = seed
        self._pipeline = pipeline
        self._pipeline_builder = pipeline_builder or build_qwen_pipeline
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg=None) -> "QwenImageEditBackend":
        device = resolve_image_edit_device(cfg)
        dtype = str(_cfg_get(cfg, "image_edit", "dtype", default="bfloat16") or "bfloat16")
        model_id = str(
            _cfg_get(cfg, "image_edit", "model_id", default=QWEN_MODEL_ID) or QWEN_MODEL_ID
        )
        offload = str(_cfg_get(cfg, "image_edit", "offload", default="none") or "none")
        steps = _cfg_get(
            cfg, "image_edit", "num_inference_steps",
            default=DEFAULT_NUM_INFERENCE_STEPS,
        )
        cfg_scale = _cfg_get(
            cfg, "image_edit", "true_cfg_scale", default=DEFAULT_TRUE_CFG_SCALE,
        )
        neg = _cfg_get(
            cfg, "image_edit", "negative_prompt", default=DEFAULT_NEGATIVE_PROMPT,
        )
        guidance = _cfg_get(
            cfg, "image_edit", "guidance_scale", default=DEFAULT_GUIDANCE_SCALE,
        )
        seed = _cfg_get(cfg, "image_edit", "seed", default=DEFAULT_SEED)
        vae_tiling = bool(_cfg_get(cfg, "image_edit", "vae_tiling", default=True))
        attention = bool(_cfg_get(cfg, "image_edit", "attention_slicing", default=False))
        return cls(
            device=device,
            dtype=dtype,
            model_id=model_id,
            vae_tiling=vae_tiling,
            attention_slicing=attention,
            offload=offload,
            local_files_only=resolve_qwen_local_files_only(cfg),
            num_inference_steps=int(steps),
            true_cfg_scale=float(cfg_scale),
            negative_prompt=" " if neg is None else str(neg),
            guidance_scale=float(guidance),
            seed=None if seed in (None, "") else int(seed),
        )

    def ensure_loaded(self):
        """Instantiate the Diffusers pipeline once; reuse for later views."""
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            using_default_loader = self._pipeline_builder is build_qwen_pipeline
            torch_dtype: Any = self.dtype_name
            if using_default_loader:
                if self.local_files_only and not qwen_weights_cached(self.model_id):
                    raise RuntimeError(
                        f"Qwen weights for {self.model_id} are not in the Hugging Face "
                        "cache. On the GPU node run once with image_edit.download: true "
                        "or SPLAT_IMAGE_EDIT_DOWNLOAD=1 (about 20GB), then keep the "
                        "pipeline resident. Tests must mock the pipeline."
                    )
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError(
                        "Qwen image-edit needs PyTorch with CUDA. "
                        "pip install -e '.[gpu]' and '.[image-edit]'."
                    ) from exc
                torch_dtype = resolve_torch_dtype(self.dtype_name, torch, self.device)
            logger.info(
                "Loading %s on %s (%s, offload=%s, vae_tiling=%s) — same GPU as "
                "photometric if device is cuda:0",
                self.model_id, self.device, torch_dtype, self.offload, self.vae_tiling,
            )
            self._pipeline = self._pipeline_builder(
                model_id=self.model_id,
                torch_dtype=torch_dtype,
                device=self.device,
                vae_tiling=self.vae_tiling,
                attention_slicing=self.attention_slicing,
                offload=self.offload,
                local_files_only=self.local_files_only,
            )
            return self._pipeline

    def edit(self, image_path: Path, prompt: str) -> ImageEditResult:
        path = Path(image_path)
        payload: dict[str, Any] = {
            "backend": BACKEND_QWEN,
            "model": self.model_id,
            "device": self.device,
            "dtype": self.dtype_name,
            "offload": self.offload,
            "vae_tiling": self.vae_tiling,
            "attention_slicing": self.attention_slicing,
            "num_inference_steps": self.num_inference_steps,
            "true_cfg_scale": self.true_cfg_scale,
            "guidance_scale": self.guidance_scale,
        }
        try:
            pipeline = self.ensure_loaded()
            source = Image.open(path).convert("RGB")
            size = source.size
            kwargs: dict[str, Any] = {
                "image": source,
                "prompt": prompt,
                "true_cfg_scale": self.true_cfg_scale,
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "num_images_per_prompt": 1,
            }
            try:
                import torch
            except ImportError:
                torch = None
            if self.seed is not None and torch is not None:
                try:
                    kwargs["generator"] = torch.manual_seed(int(self.seed))
                except Exception:
                    pass
            if torch is not None and callable(getattr(torch, "inference_mode", None)):
                with torch.inference_mode():
                    output = pipeline(**kwargs)
            else:
                output = pipeline(**kwargs)
            images = getattr(output, "images", None) or []
            if not images:
                return ImageEditResult(
                    images=[], payload=payload, error="Qwen pipeline returned no image",
                )
            repaired = images[0]
            if not isinstance(repaired, Image.Image):
                repaired = Image.fromarray(repaired)
            repaired = repaired.convert("RGB")
            if repaired.size != size:
                repaired = repaired.resize(size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            repaired.save(buf, format="PNG")
            return ImageEditResult(images=[buf.getvalue()], payload=payload)
        except Exception as exc:
            logger.exception("Qwen image-edit failed for %s", path.name)
            return ImageEditResult(
                images=[],
                payload=payload,
                error=f"{type(exc).__name__}: {exc}",
            )
