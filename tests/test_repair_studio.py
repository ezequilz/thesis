"""Repair-studio visor should follow the episode's catalog scene, not the starter room."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from splat_explorer.web.repair_studio import RepairStudio


class _Spec:
    def __init__(self, scene_id: str):
        self.id = scene_id
        self.up_axis = "+y"


class _FakeApp:
    def __init__(self, episode_dir: Path, scene_id: str = "arch-interiors"):
        self.lock = threading.Lock()
        self.run = None
        self._scene_spec = _Spec(scene_id)
        self.scene_status = "ready"
        self.scene = object()
        self.selected: list[str] = []
        self._episodes = {episode_dir.name: episode_dir}
        self.cfg = SimpleNamespace(viewer=SimpleNamespace(port=8080))

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def episode_path(self, episode_id: str):
        return self._episodes.get(episode_id)

    def select_scene(self, scene_id: str):
        self.selected.append(scene_id)
        self._scene_spec = _Spec(scene_id)
        return True, f"Loading {scene_id}"


def _episode(tmp_path: Path, scene: str = "venetian-balcony") -> Path:
    ep = tmp_path / "20260901_190223"
    ep.mkdir()
    (ep / "meta.json").write_text(json.dumps({
        "params": {"scene": scene, "scene_label": "Venetian Balcony"},
    }))
    return ep


def test_ensure_catalog_scene_loads_episode_room(tmp_path: Path):
    ep = _episode(tmp_path)
    app = _FakeApp(ep, scene_id="arch-interiors")
    studio = RepairStudio(app)
    ok, message = studio.ensure_catalog_scene(ep.name)
    assert ok
    assert app.selected == ["venetian-balcony"]
    assert "venetian-balcony" in message or "Loading" in message


def test_ensure_catalog_scene_skips_when_already_on_episode_room(tmp_path: Path):
    ep = _episode(tmp_path)
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.ensure_catalog_scene(ep.name)
    assert ok
    assert app.selected == []
    assert "already" in message.lower()


def test_snapshot_reports_episode_scene(tmp_path: Path):
    ep = _episode(tmp_path)
    app = _FakeApp(ep, scene_id="arch-interiors")
    studio = RepairStudio(app)
    snap = studio.snapshot(ep.name)
    assert snap["episode_scene"] == "venetian-balcony"
    assert snap["episode_scene_label"] == "Venetian Balcony"
    assert snap["scene_id"] == "arch-interiors"
