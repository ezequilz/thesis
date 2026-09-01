"""Repair-studio visor should follow the episode's catalog scene, not the starter room."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

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
        self._scene_generation = 0
        self.scene_status = "ready"
        self.scene = object()
        self.selected: list[str] = []
        self._episodes = {episode_dir.name: episode_dir}
        self.cfg = SimpleNamespace(
            viewer=SimpleNamespace(port=8080),
            camera=SimpleNamespace(up_axis="+y"),
            renderer=SimpleNamespace(fov_deg=75.0),
        )

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
    assert "backends" in snap
    assert snap["backends"]["detected"] in {"gsfix-gsplat", "gsplat-mlx", "cpu-project"}


def test_portable_scene_path_rewrites_host_absolute(tmp_path, monkeypatch):
    from splat_explorer.scene.catalog import openable_scene_path, portable_scene_path

    monkeypatch.chdir(tmp_path)
    ply = tmp_path / "outputs" / "episodes" / "e" / "scene_repaired.ply"
    ply.parent.mkdir(parents=True)
    ply.write_text("x")
    assert portable_scene_path(ply.resolve()) == "outputs/episodes/e/scene_repaired.ply"
    assert openable_scene_path(ply.resolve()).exists()


def test_show_publishes_relative_repaired_ply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ep = tmp_path / "outputs" / "episodes" / "20260901_190223"
    ep.mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({
        "params": {"scene": "venetian-balcony", "scene_label": "Venetian Balcony"},
    }))
    (ep / "scene_repaired.ply").write_bytes(b"ply\n")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.show(ep.name, "repaired")
    assert ok, message
    live = json.loads((tmp_path / "outputs" / "live" / "scene.json").read_text())
    assert live["path"] == "outputs/episodes/20260901_190223/scene_repaired.ply"
    assert Path(live["path"]).is_absolute() is False
    assert live["reload"] is True
    assert live["id"] == "repair-repaired"
    assert live["catalog_id"] == "venetian-balcony"
    assert studio.showing == "repaired"
    assert "scene_repaired.ply" in message


def test_start_replay_focused_requires_that_step(tmp_path: Path):
    ep = _episode(tmp_path)
    Image.new("RGB", (16, 12), (20, 20, 20)).save(ep / "step_004.png")
    Image.new("RGB", (16, 12), (200, 180, 40)).save(ep / "step_004_regen.png")
    (ep / "actions.jsonl").write_text(json.dumps({
        "step": 4,
        "position": [0.0, 0.0, 0.0],
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "frame": "step_004.png",
        "regenerate_frame": "step_004_regen.png",
    }) + "\n")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.start_replay(ep.name, step=2)
    assert ok is False
    assert "step 2" in message.lower()


def _tiny_ply(path: Path, color=(0.2, 0.4, 0.8)) -> None:
    from splat_explorer.scene import GaussianScene, save_ply

    n = 2
    save_ply(
        GaussianScene(
            means=np.zeros((n, 3), np.float32),
            scales=np.full((n, 3), 0.05, np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            opacities=np.full((n,), 0.8, np.float32),
            colors=np.full((n, 3), color, np.float32),
        ),
        path,
    )


def _regen_view(ep: Path, step: int = 4) -> None:
    Image.new("RGB", (16, 12), (20, 20, 20)).save(ep / f"step_{step:03d}.png")
    Image.new("RGB", (16, 12), (200, 180, 40)).save(ep / f"step_{step:03d}_regen.png")
    (ep / "actions.jsonl").write_text(json.dumps({
        "step": step,
        "position": [0.0, 0.0, 0.0],
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "frame": f"step_{step:03d}.png",
        "regenerate_frame": f"step_{step:03d}_regen.png",
    }) + "\n")


def test_start_replay_without_ply_needs_catalog(tmp_path: Path):
    ep = _episode(tmp_path)
    _regen_view(ep)
    app = _FakeApp(ep, scene_id="venetian-balcony")
    app.scene_status = "loading"
    app.scene = None
    studio = RepairStudio(app)
    ok, message = studio.start_replay(ep.name, step=4)
    assert ok is False
    assert "not ready" in message.lower()


def test_start_replay_continues_from_episode_ply_while_catalog_loads(tmp_path: Path):
    ep = _episode(tmp_path)
    _regen_view(ep)
    _tiny_ply(ep / "scene_original.ply", (0.1, 0.1, 0.1))
    _tiny_ply(ep / "scene_repaired.ply", (0.9, 0.2, 0.1))
    app = _FakeApp(ep, scene_id="venetian-balcony")
    app.scene_status = "loading"
    app.scene = None
    studio = RepairStudio(app)
    ok, message = studio.start_replay(ep.name, step=4, resume=True)
    assert ok, message
    studio.stop_replay()


def test_reset_repair_restores_original_ply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ep = tmp_path / "outputs" / "episodes" / "20260901_190223"
    ep.mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({
        "params": {"scene": "venetian-balcony", "scene_label": "Venetian Balcony"},
    }))
    _tiny_ply(ep / "scene_original.ply", (0.2, 0.4, 0.8))
    _tiny_ply(ep / "scene_repaired.ply", (0.9, 0.1, 0.1))
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.reset_repair(ep.name)
    assert ok, message
    from splat_explorer.scene import load_ply
    restored = load_ply(ep / "scene_repaired.ply")
    np.testing.assert_allclose(restored.colors[0], [0.2, 0.4, 0.8], atol=1e-4)
    assert studio.showing == "original"


def test_catalog_id_from_live_ignores_repair_preview():
    from splat_explorer.scene.catalog import catalog_id_from_live

    assert catalog_id_from_live({"id": "repair-repaired"}) is None
    assert catalog_id_from_live({
        "id": "repair-repaired", "catalog_id": "venetian-balcony",
    }) == "venetian-balcony"
    assert catalog_id_from_live({"id": "venetian-balcony"}) == "venetian-balcony"


def test_live_scene_reload_follows_newer_path_even_if_generation_drops():
    from splat_explorer.scene.catalog import live_scene_reload_action

    state = {
        "path": "3dgs_rooms/ArchInteriors_for_UE2_Atlux.sog",
        "generation": 36,
        "updated_at": 1.0,
        "status": "ready",
        "mtime": 0.0,
    }
    req = {
        "path": "3dgs_rooms/Venetian Balcony",
        "generation": 4,
        "updated_at": 2.0,
        "reload": True,
    }
    assert live_scene_reload_action(req, state) == "load"
    stale = dict(req, updated_at=0.5, path="3dgs_rooms/Venetian Balcony")
    assert live_scene_reload_action(stale, state) == "skip"
