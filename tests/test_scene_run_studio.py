"""Focused unit and route coverage for the isolated scene-run dashboard."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from splat_explorer.web.scene_run_studio import (
    SCENE_RUN_DEFAULTS,
    SceneRunStudio,
    SceneRunValidationError,
)
from splat_explorer.web.server import DashboardHandler


class _Store:
    def __init__(self, root: Path):
        self.root = root
        self.created = []
        self.stopped = []
        self.runs = [{
            "run_id": "run_20260915_200000",
            "config": dict(SCENE_RUN_DEFAULTS),
            "state": {
                "status": "queued",
                "created_at": "2026-09-15T20:00:00Z",
                "updated_at": "2026-09-15T20:00:00Z",
            },
        }]

    def create_run(self, config):
        self.created.append(config)
        run = dict(self.runs[0])
        run["config"] = dict(config)
        return run

    def list_runs(self):
        return list(self.runs)

    def detail(self, run_id):
        return next((run for run in self.runs if run["run_id"] == run_id), None)

    def request_stop(self, run_id):
        self.stopped.append(run_id)
        return self.root / run_id / "STOP"

    def run_path(self, run_id):
        return self.root / run_id


def _app(tmp_path: Path):
    cfg = SimpleNamespace(
        output=SimpleNamespace(dir=str(tmp_path)),
        agent=SimpleNamespace(model="gpt-5.6-luna"),
    )
    app = SimpleNamespace(cfg=cfg)
    store = _Store(tmp_path / "scene-runs")
    studio = SceneRunStudio(app, store=store)
    studio.scenes = lambda: [
        {"id": "venetian-balcony", "label": "Venetian Balcony"},
        {"id": "pond-shelter", "label": "Pond shelter"},
    ]
    app.scene_runs = studio
    return app, studio, store


def test_defaults_and_start_validation_are_isolated(tmp_path: Path):
    _fake_app, studio, store = _app(tmp_path)
    defaults = studio.defaults()
    assert defaults == {**SCENE_RUN_DEFAULTS, "model": "gpt-5.6-luna"}

    run = studio.create({
        "scene_id": "venetian-balcony",
        "duration_seconds": 7200,
        "send_map": False,
    })
    assert run["run_id"] == "run_20260915_200000"
    assert store.created[0]["duration_seconds"] == 7200
    assert store.created[0]["send_map"] is True
    assert store.created[0]["repair_type"] == "original"
    assert not hasattr(studio, "_thread")


def test_queued_run_pins_capture_visor_to_its_scene(tmp_path: Path):
    app, studio, _store = _app(tmp_path)
    selected = []
    app.select_scene = lambda scene_id: selected.append(scene_id) or (True, "ok")
    assert studio.preferred_visor_scene_id() == "venetian-balcony"
    studio.create({
        "scene_id": "venetian-balcony",
        "duration_seconds": 3600,
    })
    assert selected == ["venetian-balcony"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_id", "does-not-exist"),
        ("width", 0),
        ("height", "wide"),
        ("duration_seconds", 0),
        ("repair_seconds", 50000),
        ("repair_trigger", "sometimes"),
        ("repair_type", "neon"),
        ("backend", "scripted"),
    ],
)
def test_invalid_start_form_is_rejected(tmp_path: Path, field: str, value):
    _fake_app, studio, _store = _app(tmp_path)
    with pytest.raises(SceneRunValidationError):
        studio.validate_config({field: value})


def test_scene_run_image_edit_defaults_to_gpt_and_keeps_qwen(tmp_path: Path):
    _fake_app, studio, _store = _app(tmp_path)
    default = studio.validate_config({})
    assert default["image_edit_backend"] == "gpt-image-2.5-sunburst"
    flare = studio.validate_config({"image_edit_backend": "gpt-image-2.5-flare"})
    assert flare["image_edit_backend"] == "gpt-image-2.5-flare"
    legacy = studio.validate_config({"image_edit_backend": "gpt-image-2"})
    assert legacy["image_edit_backend"] == "gpt-image-2"
    qwen = studio.validate_config({"image_edit_backend": "qwen"})
    assert qwen["image_edit_backend"] == "qwen-image-edit"
    with pytest.raises(SceneRunValidationError):
        studio.validate_config({"image_edit_backend": "midjourney"})


def test_state_recognizes_nested_store_status_and_manager(tmp_path: Path):
    _fake_app, studio, _store = _app(tmp_path)
    state = studio.state()
    assert state["active"] is True
    assert state["active_run"]["run_id"] == "run_20260915_200000"
    assert state["manager"] == {
        "status": "not_detected",
        "active": False,
        "run_id": "run_20260915_200000",
    }


def test_detail_lists_artifacts_and_resolver_blocks_escape(tmp_path: Path):
    _fake_app, studio, store = _app(tmp_path)
    run_dir = store.run_path("run_20260915_200000")
    (run_dir / "images").mkdir(parents=True)
    (run_dir / "images" / "step_001.png").write_bytes(b"png")
    (run_dir / "scene_repaired.ply").write_bytes(b"ply")
    (run_dir / "worker.log").write_text("ok")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")

    detail = studio.detail("run_20260915_200000")
    assert {item["kind"] for item in detail["files"]} == {"image", "ply", "log"}
    assert detail["viser_path"] == "/run_20260915_200000/viser"
    assert detail["ply"] == {"original": False, "repaired": True}
    assert studio.artifact_path(
        "run_20260915_200000", "images/step_001.png",
    ) == run_dir / "images" / "step_001.png"
    assert studio.artifact_path("run_20260915_200000", "../secret.txt") is None
    assert studio.artifact_path("run_20260915_200000", "/etc/passwd") is None


def _request(base: str, path: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            payload = json.loads(raw) if "application/json" in content_type else raw
            return response.status, payload
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_scene_run_routes_and_safe_files(tmp_path: Path):
    app, _studio, store = _app(tmp_path)
    run_dir = store.run_path("run_20260915_200000")
    run_dir.mkdir(parents=True)
    (run_dir / "worker.log").write_text("worker output")
    DashboardHandler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        code, page = _request(base, "/scene-runs")
        assert code == 200
        assert b"Create isolated run" in page
        assert b"original GSFix3D" in page
        assert b"Repair Time" in page

        code, state = _request(base, "/api/scene-runs/state")
        assert code == 200
        assert state["defaults"]["duration_seconds"] == 3600
        assert state["active"] is True

        code, payload = _request(base, "/api/scene-runs/start", {
            "scene_id": "venetian-balcony",
            "duration_seconds": 3600,
            "width": 960,
            "height": 720,
        })
        assert code == 200
        assert payload["run"]["state"]["status"] == "queued"

        code, payload = _request(base, "/api/scene-runs/stop", {
            "id": "run_20260915_200000",
        })
        assert code == 200
        assert payload["ok"] is True
        assert store.stopped == ["run_20260915_200000"]

        code, log = _request(
            base, "/scene-run-files/run_20260915_200000/worker.log",
        )
        assert code == 200
        assert log == b"worker output"
        code, payload = _request(
            base, "/scene-run-files/run_20260915_200000/../secret.txt",
        )
        assert code == 404
        assert payload["error"] == "not found"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class _FakeVisor:
    def __init__(self):
        self.which = None
        self.shown = []
        self.heartbeats = []
        self.released = []

    def snapshot(self, run_id):
        return {
            "ok": True,
            "run_id": run_id,
            "which": self.which,
            "status": "ready" if self.which else "idle",
            "viewer_url": "http://localhost:8082" if self.which else None,
            "original": True,
            "repaired": True,
            "viser_path": f"/{run_id}/viser",
            "message": "",
            "error": None,
        }

    def show(self, run_id, which=None, toggle=False, client=None):
        requested = str(which or "").strip().lower() or None
        if toggle:
            requested = "original" if self.which == "repaired" else "repaired"
        self.which = requested or "repaired"
        self.shown.append((run_id, self.which, toggle, client))
        snap = self.snapshot(run_id)
        snap["ok"] = True
        snap["message"] = f"Showing {self.which}."
        return snap

    def heartbeat(self, run_id, client=None):
        self.heartbeats.append((run_id, client))
        return self.snapshot(run_id)

    def release(self, run_id, client=None):
        self.released.append((run_id, client))
        self.which = None
        snap = self.snapshot(run_id)
        snap["ok"] = True
        return snap


def test_scene_run_visor_page_defaults_to_repaired_and_toggles(tmp_path: Path):
    app, studio, store = _app(tmp_path)
    run_dir = store.run_path("run_20260915_200000")
    run_dir.mkdir(parents=True)
    (run_dir / "scene_original.ply").write_bytes(b"ply")
    (run_dir / "scene_repaired.ply").write_bytes(b"ply")
    visor = _FakeVisor()
    studio._visor = visor
    DashboardHandler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        code, page = _request(base, "/scene-runs")
        assert code == 200
        assert b"Open visor" in page
        assert b"/${encodeURIComponent(selectedId)}/viser" in page

        code, page = _request(base, "/run_20260915_200000/viser")
        assert code == 200
        assert b"Show original" in page
        assert b"id=\"flip\"" in page

        code, payload = _request(base, "/not-a-run/viser")
        assert code == 404

        code, payload = _request(base, "/run_19990101_000000/viser")
        assert code == 404

        code, payload = _request(base, "/api/scene-runs/run_20260915_200000/viser")
        assert code == 200
        assert payload["status"] == "idle"
        assert payload["viser_path"] == "/run_20260915_200000/viser"

        code, payload = _request(base, "/api/scene-runs/run_20260915_200000/viser", {})
        assert code == 200
        assert payload["which"] == "repaired"
        assert payload["viewer_url"] == "http://localhost:8082"
        assert visor.shown == [("run_20260915_200000", "repaired", False, None)]

        code, payload = _request(base, "/api/scene-runs/run_20260915_200000/viser")
        assert visor.shown == [("run_20260915_200000", "repaired", False, None)]

        code, payload = _request(
            base, "/api/scene-runs/run_20260915_200000/viser", {"heartbeat": True, "client": "tab-a"},
        )
        assert code == 200
        assert visor.heartbeats == [("run_20260915_200000", "tab-a")]
        assert visor.shown == [("run_20260915_200000", "repaired", False, None)]

        code, payload = _request(
            base, "/api/scene-runs/run_20260915_200000/viser", {"toggle": True},
        )
        assert code == 200
        assert payload["which"] == "original"
        code, payload = _request(
            base, "/api/scene-runs/run_20260915_200000/viser", {"toggle": True},
        )
        assert payload["which"] == "repaired"

        code, payload = _request(
            base, "/api/scene-runs/run_20260915_200000/viser", {"stop": True, "client": "tab-a"},
        )
        assert code == 200
        assert visor.released == [("run_20260915_200000", "tab-a")]
        assert payload["status"] == "idle"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
