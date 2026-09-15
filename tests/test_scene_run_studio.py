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
    assert not hasattr(studio, "_thread")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_id", "does-not-exist"),
        ("width", 0),
        ("height", "wide"),
        ("duration_seconds", 0),
        ("repair_seconds", 50000),
        ("repair_trigger", "sometimes"),
    ],
)
def test_invalid_start_form_is_rejected(tmp_path: Path, field: str, value):
    _fake_app, studio, _store = _app(tmp_path)
    with pytest.raises(SceneRunValidationError):
        studio.validate_config({field: value})


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
