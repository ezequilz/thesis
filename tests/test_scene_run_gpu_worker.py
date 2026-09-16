from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splat_explorer.scene_runs.gpu_worker import (
    DEPTH_NAME,
    RENDERED_NAME,
    REQUEST_NAME,
    RESPONSE_NAME,
    SCENE_NAME,
    SceneRunGpuWorker,
    _default_repair_factory,
    memory_snapshot,
    pending_request_dirs,
)


def test_memory_snapshot_is_json_safe():
    snap = memory_snapshot()
    assert isinstance(snap, dict)
    json.dumps(snap)


class _Scene:
    pass


def test_scene_run_repair_factory_defaults_to_github_refine_gs():
    backend = _default_repair_factory({})
    assert backend.iters == 20
    assert backend.max_chunks == 1
    assert backend.densify is True
    assert backend.upstream_gsfix3d is True
    assert backend.densify_grad_thresh == 0.005
    assert backend.lr_means == 0.00032
    assert backend.lr_opacities == 0.025
    assert backend.lr_scales == 0.002
    assert backend.sh_clip == 0.0
    assert backend.lambda_preserve == 0.0
    assert backend.freeze_geometry_after_first_chunk is False


def test_scene_run_repair_factory_looped_keeps_time_budget_adc():
    backend = _default_repair_factory({"repair_type": "looped", "densify": False, "max_chunks": 3})
    assert backend.densify is True
    assert backend.max_chunks == 0
    assert backend.upstream_gsfix3d is False
    assert backend.freeze_geometry_after_first_chunk is True


def test_original_repair_ignores_time_budget_deadline(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import io

    (tmp_path / SCENE_NAME).write_bytes(b"scene")
    request_dir = tmp_path / "requests" / "repair-00000"
    request_dir.mkdir(parents=True)
    image = Image.new("RGB", (8, 6), (20, 30, 40))
    png = io.BytesIO()
    image.save(png, format="PNG")
    (request_dir / RENDERED_NAME).write_bytes(png.getvalue())
    (request_dir / REQUEST_NAME).write_text(json.dumps({
        "request_id": "repair-00000",
        "operation": "repair",
        "step": 1,
        "camera": {
            "position": [0, 0, 0],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "width": 8,
            "height": 6,
            "fov_deg": 75,
        },
        "repair_seconds": 180,
        "deadline_unix": 2_000_000_000,
    }))
    seen = {}

    class Editor:
        def edit(self, _path, _prompt):
            return SimpleNamespace(images=[png.getvalue()], payload={}, error=None)

    class Repair:
        def apply_until(self, _scene, _camera, _rendered, _repaired, **kwargs):
            seen["deadline"] = kwargs.get("deadline")
            return {"n_iters": 20, "l1_before": 0.4, "l1_after": 0.2}

    worker = SceneRunGpuWorker(
        tmp_path,
        scene_loader=lambda _path: object(),
        scene_saver=lambda _scene, path: Path(path).write_bytes(b"ok"),
        image_edit_factory=lambda _config: Editor(),
        repair_factory=lambda params: seen.update(params=params) or Repair(),
        clock=lambda: 1_000.0,
    )
    worker._load_scene_once()
    response = worker.process_request(request_dir)
    assert response["status"] == "ok"
    assert seen["params"]["repair_type"] == "original"
    assert seen["deadline"] is None


def test_worker_can_render_without_qwen_or_browser(tmp_path: Path, monkeypatch):
    (tmp_path / SCENE_NAME).write_bytes(b"scene")
    request_dir = tmp_path / "requests" / "render-00000"
    request_dir.mkdir(parents=True)
    (request_dir / REQUEST_NAME).write_text(json.dumps({
        "request_id": "render-00000",
        "operation": "render",
        "step": 0,
        "camera": {
            "position": [0, 0, 0],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "width": 8,
            "height": 6,
            "fov_deg": 75,
        },
        "repair_seconds": 1,
        "deadline_unix": 2_000_000_000,
    }))
    image_edit_calls = []
    worker = SceneRunGpuWorker(
        tmp_path,
        scene_loader=lambda _path: _Scene(),
        scene_saver=lambda _scene, path: Path(path).write_bytes(b"checkpoint"),
        image_edit_factory=lambda _config: image_edit_calls.append(1),
    )
    worker._load_scene_once()

    class Renderer:
        def render_with_depth(self, camera):
            return (
                np.full((camera.height, camera.width, 3), 127, np.uint8),
                np.full((camera.height, camera.width), 2.5, np.float32),
            )

    monkeypatch.setattr(worker, "_renderer", lambda: Renderer())
    response = worker.process_request(request_dir)

    assert response["status"] == "ok"
    assert (request_dir / RENDERED_NAME).is_file()
    assert np.load(request_dir / DEPTH_NAME).shape == (6, 8)
    assert json.loads((request_dir / RESPONSE_NAME).read_text())["rendered"] == RENDERED_NAME
    assert image_edit_calls == []


def test_pending_requests_ignore_atomic_incoming_directory(tmp_path: Path):
    incoming = tmp_path / "requests" / ".incoming-render-00000"
    incoming.mkdir(parents=True)
    (incoming / REQUEST_NAME).write_text("{}")
    ready = tmp_path / "requests" / "render-00000"
    ready.mkdir()
    (ready / REQUEST_NAME).write_text("{}")

    assert pending_request_dirs(tmp_path) == [ready]
