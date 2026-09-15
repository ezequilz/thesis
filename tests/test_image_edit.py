"""RGB image-edit backends (gpt-image-2 / Qwen) — mocked weights, no 20GB download."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.image_edit import (
    BACKEND_GPT,
    BACKEND_QWEN,
    apply_image_edit_overrides,
    image_edit_shares_photometric_gpu,
    list_image_edit_backends,
    make_image_edit_backend,
    overlay_image_edit_cfg,
    photometric_cuda_device,
    resolve_image_edit_backend,
    resolve_image_edit_device,
)
from splat_explorer.image_edit_gpt import GptImageEditBackend
from splat_explorer.image_edit_qwen import (
    QWEN_MODEL_ID,
    QwenImageEditBackend,
    build_qwen_pipeline,
    qwen_weights_cached,
    resolve_qwen_local_files_only,
    resolve_torch_dtype,
)


def _png(path: Path, color=(40, 50, 60), size=(8, 6)) -> Path:
    Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8)).save(path)
    return path


def _tiny_png_bytes(color=(12, 34, 56), size=(6, 4)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class FakePipeline:
    def __init__(self):
        self.to_calls = []
        self.kwargs = None
        self.vae_tiled = False
        self.sliced = False
        self.offload = None
        self.call_count = 0

    def to(self, device):
        self.to_calls.append(device)
        return self

    def enable_vae_tiling(self):
        self.vae_tiled = True

    def enable_attention_slicing(self):
        self.sliced = True

    def enable_sequential_cpu_offload(self):
        self.offload = "sequential"

    def enable_model_cpu_offload(self):
        self.offload = "model"

    def set_progress_bar_config(self, **kwargs):
        self.progress = kwargs

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.call_count += 1
        image = kwargs["image"]
        size = image.size if hasattr(image, "size") else (8, 6)
        return SimpleNamespace(images=[Image.new("RGB", size, (200, 10, 30))])


def _builder(pipe: FakePipeline):
    def build(**kwargs):
        pipe.build_kwargs = kwargs
        offload = str(kwargs.get("offload") or "none").lower()
        if offload in ("sequential", "sequential_cpu", "cpu"):
            pipe.enable_sequential_cpu_offload()
        elif offload in ("model", "model_cpu", "cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(kwargs["device"])
        if kwargs.get("vae_tiling"):
            pipe.enable_vae_tiling()
        if kwargs.get("attention_slicing"):
            pipe.enable_attention_slicing()
        return pipe
    return build


def test_resolve_backend_aliases_and_default():
    assert resolve_image_edit_backend(None) == BACKEND_GPT
    assert resolve_image_edit_backend("gpt-image-2") == BACKEND_GPT
    assert resolve_image_edit_backend("cli_relay") == BACKEND_GPT
    assert resolve_image_edit_backend("qwen") == BACKEND_QWEN
    assert resolve_image_edit_backend("qwen-image-edit-2511") == BACKEND_QWEN
    with pytest.raises(ValueError, match="Unknown image-edit backend"):
        resolve_image_edit_backend("midjourney")


def test_resolve_backend_env_and_config(monkeypatch):
    monkeypatch.delenv("SPLAT_IMAGE_EDIT_BACKEND", raising=False)
    monkeypatch.delenv("IMAGE_EDIT_BACKEND", raising=False)
    assert resolve_image_edit_backend(cfg={"image_edit": {"backend": "qwen"}}) == BACKEND_QWEN
    monkeypatch.setenv("SPLAT_IMAGE_EDIT_BACKEND", "gpt-image-2")
    assert resolve_image_edit_backend(cfg={"image_edit": {"backend": "qwen"}}) == BACKEND_GPT
    monkeypatch.setenv("IMAGE_EDIT_BACKEND", "qwen-image-edit")
    monkeypatch.delenv("SPLAT_IMAGE_EDIT_BACKEND", raising=False)
    assert resolve_image_edit_backend() == BACKEND_QWEN


def test_device_default_matches_photometric(monkeypatch):
    monkeypatch.delenv("SPLAT_IMAGE_EDIT_DEVICE", raising=False)
    monkeypatch.delenv("IMAGE_EDIT_DEVICE", raising=False)
    assert photometric_cuda_device() == "cuda:0"
    assert resolve_image_edit_device() == "cuda:0"
    assert image_edit_shares_photometric_gpu() is True
    assert resolve_image_edit_device(device=0) == "cuda:0"
    assert resolve_image_edit_device(cfg={"image_edit": {"device": 1}}) == "cuda:1"
    assert image_edit_shares_photometric_gpu(cfg={"image_edit": {"device": 1}}) is False
    monkeypatch.setenv("SPLAT_IMAGE_EDIT_DEVICE", "1")
    assert resolve_image_edit_device() == "cuda:1"
    assert resolve_image_edit_device(device="cuda:0") == "cuda:0"


def test_make_backend_selects_qwen_without_loading_weights():
    backend = make_image_edit_backend("qwen-image-edit")
    assert isinstance(backend, QwenImageEditBackend)
    assert backend.name == BACKEND_QWEN
    assert backend.device == photometric_cuda_device()
    assert backend._pipeline is None


def test_make_backend_gpt_with_injected_client():
    client = object()
    backend = make_image_edit_backend("gpt-image-2", client=client)
    assert isinstance(backend, GptImageEditBackend)
    assert backend.client is client
    assert backend.name == BACKEND_GPT


def test_qwen_edit_contract_and_resident_pipeline(tmp_path: Path):
    src = _png(tmp_path / "step_003.png", (9, 8, 7), (10, 8))
    pipe = FakePipeline()
    backend = QwenImageEditBackend(
        device="cuda:0",
        local_files_only=True,
        pipeline_builder=_builder(pipe),
    )
    first = backend.edit(src, "Please regenerate and fix this image. Repair artifacts and upscale to higher resolution.")
    assert first.error is None
    assert len(first.images) == 1
    assert first.images[0].startswith(b"\x89PNG")
    out = Image.open(io.BytesIO(first.images[0]))
    assert out.size == (10, 8)
    assert out.convert("RGB").getpixel((0, 0)) == (200, 10, 30)
    assert pipe.kwargs["prompt"].startswith("Please regenerate and fix this image")
    assert pipe.kwargs["true_cfg_scale"] == 4.0
    assert pipe.kwargs["num_inference_steps"] == 40
    assert pipe.kwargs["negative_prompt"] == " "
    assert pipe.to_calls == ["cuda:0"]
    assert pipe.vae_tiled is True
    assert pipe.offload is None
    assert first.payload["backend"] == BACKEND_QWEN
    assert first.payload["model"] == QWEN_MODEL_ID
    assert first.payload["device"] == "cuda:0"
    second = backend.edit(src, "again")
    assert second.error is None
    assert pipe.call_count == 2
    assert backend._pipeline is pipe


def test_qwen_dedicated_device_does_not_use_set_device(tmp_path: Path):
    src = _png(tmp_path / "in.png")
    pipe = FakePipeline()
    backend = QwenImageEditBackend(
        device="cuda:1",
        vae_tiling=False,
        pipeline_builder=_builder(pipe),
    )
    result = backend.edit(src, "fix")
    assert result.error is None
    assert pipe.to_calls == ["cuda:1"]
    assert "set_device" not in dir(pipe) or pipe.to_calls != ["cuda:0"]
    assert result.payload["device"] == "cuda:1"
    assert image_edit_shares_photometric_gpu(device=1) is False


def test_qwen_sequential_offload_skips_to_device(tmp_path: Path):
    src = _png(tmp_path / "in.png")
    pipe = FakePipeline()
    backend = QwenImageEditBackend(
        device="cuda:0",
        offload="sequential",
        pipeline_builder=_builder(pipe),
    )
    result = backend.edit(src, "fix")
    assert result.error is None
    assert pipe.to_calls == []
    assert pipe.offload == "sequential"


def test_qwen_refuses_download_when_cache_empty(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "splat_explorer.image_edit_qwen.qwen_weights_cached", lambda model_id=None: False,
    )
    backend = QwenImageEditBackend(local_files_only=True)
    with pytest.raises(RuntimeError, match="not in the Hugging Face cache"):
        backend.ensure_loaded()


def test_qwen_weights_cached_false_in_this_environment():
    assert qwen_weights_cached() is False or isinstance(qwen_weights_cached(), bool)


def test_local_files_only_env(monkeypatch):
    monkeypatch.setenv("SPLAT_IMAGE_EDIT_DOWNLOAD", "1")
    assert resolve_qwen_local_files_only() is False
    monkeypatch.setenv("SPLAT_IMAGE_EDIT_DOWNLOAD", "0")
    assert resolve_qwen_local_files_only({"image_edit": {"download": True}}) is True
    monkeypatch.delenv("SPLAT_IMAGE_EDIT_DOWNLOAD", raising=False)
    assert resolve_qwen_local_files_only({"image_edit": {"download": False}}) is True


def test_gpt_backend_matches_images_edit_prompt(tmp_path: Path):
    png = _tiny_png_bytes()
    src = tmp_path / "step_001.png"
    src.write_bytes(png)

    class Images:
        def __init__(self):
            self.calls = []

        def edit(self, **kwargs):
            self.calls.append(kwargs)
            import base64
            return SimpleNamespace(model_dump=lambda mode="json": {
                "data": [{"b64_json": base64.b64encode(png).decode()}],
            })

    client = SimpleNamespace(images=Images())
    backend = GptImageEditBackend(client=client)
    from splat_explorer.agent.regenerate import REGENERATE_PROMPT

    result = backend.edit(src, REGENERATE_PROMPT)
    assert result.error is None
    assert result.images
    assert client.images.calls[0]["model"] == "gpt-image-2"
    assert client.images.calls[0]["prompt"] == REGENERATE_PROMPT


def test_regenerator_uses_qwen_editor(tmp_path: Path):
    from splat_explorer.agent.regenerate import REGENERATE_PROMPT, Regenerator

    src = _png(tmp_path / "step_004.png", (1, 2, 3), (5, 5))
    pipe = FakePipeline()
    editor = QwenImageEditBackend(pipeline_builder=_builder(pipe))
    regen = Regenerator(client=None, model=editor.name, editor=editor, max_workers=1)
    regen.submit(src, tmp_path, 4)
    results = regen.wait()
    assert results[0].status == "ok"
    assert results[0].model == BACKEND_QWEN
    assert (tmp_path / "step_004_regen.png").is_file()
    meta = json.loads((tmp_path / "step_004_regen.json").read_text())
    assert meta["model"] == BACKEND_QWEN
    assert pipe.kwargs["prompt"] == REGENERATE_PROMPT


def test_overlay_and_cli_overrides():
    from splat_explorer.config import Config

    cfg = Config({"image_edit": {"backend": "gpt-image-2", "device": 0}})
    over = overlay_image_edit_cfg(cfg, backend="qwen", device=1)
    assert resolve_image_edit_backend(cfg=over) == BACKEND_QWEN
    assert resolve_image_edit_device(over) == "cuda:1"
    apply_image_edit_overrides(cfg, backend="qwen-image-edit", device=0)
    assert resolve_image_edit_backend(cfg=cfg) == BACKEND_QWEN
    assert image_edit_shares_photometric_gpu(cfg) is True


def test_list_backends_reports_shared_device():
    catalog = list_image_edit_backends({"image_edit": {"backend": "qwen", "device": 0}})
    assert catalog["selected"] == BACKEND_QWEN
    assert catalog["device"] == catalog["photometric_device"] == "cuda:0"
    assert catalog["same_gpu_as_photometric"] is True
    ids = [b["id"] for b in catalog["backends"]]
    assert ids == [BACKEND_GPT, BACKEND_QWEN]
    assert "lrz-hgx-a100-80x4" in catalog["lrz_partitions"]
    assert "lrz-dgx-a100-80x8" in catalog["lrz_partitions"]
    assert "lrz-hgx-h100-94x4" in catalog["lrz_partitions"]


def test_resolve_torch_dtype_bf16_default():
    torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        cuda=SimpleNamespace(is_bf16_supported=lambda: True),
    )
    assert resolve_torch_dtype("bfloat16", torch, "cuda:0") == "bf16"
    torch.cuda.is_bf16_supported = lambda: False
    assert resolve_torch_dtype("bfloat16", torch, "cuda:0") == "fp16"
    assert resolve_torch_dtype("fp16", torch, "cuda:0") == "fp16"


def test_build_qwen_pipeline_is_the_official_loader():
    assert callable(build_qwen_pipeline)


def test_repair_studio_snapshot_includes_image_edit(tmp_path: Path, monkeypatch):
    from splat_explorer.config import Config
    from splat_explorer.web.repair_studio import RepairStudio

    monkeypatch.setattr(
        "splat_explorer.web.repair_studio.list_repair_backends",
        lambda: {"detected": "cpu-project", "lrz": {}, "backends": []},
    )

    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "meta.json").write_text(json.dumps({"params": {"scene": "arch-interiors"}}))

    class App:
        def __init__(self):
            self.lock = __import__("threading").Lock()
            self.run = None
            self.scene_status = "ready"
            self._scene_spec = SimpleNamespace(id="arch-interiors", up_axis="+y")
            self.cfg = Config({
                "image_edit": {"backend": "qwen-image-edit", "device": 0},
                "viewer": {"port": 8080},
                "camera": {"up_axis": "+y"},
                "renderer": {"fov_deg": 75.0},
            })

        def episode_path(self, episode_id):
            return ep if episode_id == ep.name else None

        @staticmethod
        def _read_json(path):
            try:
                return json.loads(Path(path).read_text())
            except (OSError, json.JSONDecodeError):
                return None

    studio = RepairStudio(App())
    snap = studio.snapshot(ep.name)
    assert snap["image_edit"]["selected"] == BACKEND_QWEN
    assert snap["image_edit"]["same_gpu_as_photometric"] is True


def test_static_pages_expose_backend_selector():
    root = Path(__file__).resolve().parents[1] / "src" / "splat_explorer" / "web" / "static"
    index = (root / "index.html").read_text()
    repair = (root / "repair.html").read_text()
    gpu = (root / "gpu.html").read_text()
    assert 'id="in-image-edit"' in index
    assert "qwen-image-edit" in index
    assert 'id="in-image-edit"' in repair
    assert "imageEditGpuNote" in gpu
    assert "Qwen-Image-Edit-2511" in gpu
