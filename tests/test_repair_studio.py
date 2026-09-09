"""Repair-studio visor should follow the episode's catalog scene, not the starter room."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from splat_explorer.web.repair_studio import (
    FOCUSED_REPAIR_MAX_SECONDS,
    RepairStudio,
    focused_cap_label,
    focused_finish_message,
    list_repaired_saves,
    next_repaired_save_index,
    pick_interactive_camera,
    repaired_save_index,
    repaired_save_name,
)


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
        self.published = []

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

    def _publish_pose(self, record, frame_path, params, trajectory, *, snap_camera=False):
        self.published.append({
            "step": record.get("step"),
            "snap_camera": snap_camera,
            "frame": getattr(frame_path, "name", str(frame_path)),
        })


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


def test_ensure_catalog_scene_keeps_repaired_ply_preview(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ep = tmp_path / "outputs" / "episodes" / "20260901_190223"
    ep.mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({
        "params": {"scene": "venetian-balcony", "scene_label": "Venetian Balcony"},
    }))
    _tiny_ply(ep / "scene_repaired.ply")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.show(ep.name, "repaired")
    assert ok, message
    assert studio.showing == "repaired"
    ok, message = studio.ensure_catalog_scene(ep.name)
    assert ok, message
    assert studio.showing == "repaired"
    assert "ply" in message.lower()


def test_snapshot_reports_episode_scene(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    ep = _episode(tmp_path)
    app = _FakeApp(ep, scene_id="arch-interiors")
    studio = RepairStudio(app)
    snap = studio.snapshot(ep.name)
    assert snap["episode_scene"] == "venetian-balcony"
    assert snap["episode_scene_label"] == "Venetian Balcony"
    assert snap["scene_id"] == "arch-interiors"
    assert snap["capture_width"] == 960
    assert snap["capture_height"] == 720
    assert "backends" in snap
    assert snap["gpu_url"] == "/repair/gpu"
    gpu = studio.gpu_snapshot()
    assert "checks" in gpu
    assert gpu["repair"]["status"] == "idle"
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


def test_show_without_force_skips_when_already_previewing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ep = tmp_path / "outputs" / "episodes" / "20260901_190223"
    ep.mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({
        "params": {"scene": "venetian-balcony", "scene_label": "Venetian Balcony"},
    }))
    (ep / "scene_repaired.ply").write_bytes(b"ply\n")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.show(ep.name, "repaired", force=True)
    assert ok, message
    generation = app._scene_generation
    live = json.loads((tmp_path / "outputs" / "live" / "scene.json").read_text())
    ok, message = studio.show(ep.name, "repaired", force=False)
    assert ok, message
    assert "already" in message.lower()
    assert app._scene_generation == generation
    again = json.loads((tmp_path / "outputs" / "live" / "scene.json").read_text())
    assert again["generation"] == live["generation"]
    assert again["updated_at"] == live["updated_at"]


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
    ok, message = studio.start_replay(ep.name, step=4, resume=True, backend="cpu-project")
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
    studio.job = {
        **studio._idle_job(ep.name),
        "status": "error",
        "error": "RuntimeError: command failed (1): ssh … Tensors must have same number of dimensions",
        "message": "RuntimeError: command failed (1): ssh … Tensors must have same number of dimensions",
        "results": [{"step": 4, "status": "error"}],
    }
    ok, message = studio.reset_repair(ep.name)
    assert ok, message
    from splat_explorer.scene import load_ply
    restored = load_ply(ep / "scene_repaired.ply")
    np.testing.assert_allclose(restored.colors[0], [0.2, 0.4, 0.8], atol=1e-4)
    assert studio.showing == "original"
    assert studio.job["status"] == "idle"
    assert studio.job["error"] is None
    assert studio.job["results"] == []
    assert "Restored" in (studio.job["message"] or "")
    assert "RuntimeError" not in (studio.job["message"] or "")


def test_show_highlight_recolors_changed_splats(tmp_path, monkeypatch):
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
    ok, message = studio.show(ep.name, "repaired", highlight=True)
    assert ok, message
    assert studio.showing == "repaired"
    assert studio.showing_highlight is True
    from splat_explorer.repair import HIGHLIGHT_COLOR, HIGHLIGHT_PLY
    from splat_explorer.scene import load_ply
    highlighted = load_ply(ep / HIGHLIGHT_PLY)
    np.testing.assert_allclose(highlighted.colors[0], HIGHLIGHT_COLOR, atol=1e-3)
    live = json.loads((tmp_path / "outputs" / "live" / "scene.json").read_text())
    assert live["id"] == "repair-highlight"
    assert live["path"].endswith(HIGHLIGHT_PLY)
    repaired = load_ply(ep / "scene_repaired.ply")
    np.testing.assert_allclose(repaired.colors[0], [0.9, 0.1, 0.1], atol=1e-3)
    ok, message = studio.show(ep.name, "original", highlight=True)
    assert ok, message
    assert studio.showing == "original"
    assert studio.showing_highlight is False
    original = load_ply(ep / "scene_original.ply")
    np.testing.assert_allclose(original.colors[0], [0.2, 0.4, 0.8], atol=1e-3)


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


def test_metrics_payload_marks_l1_improved(tmp_path: Path):
    ep = _episode(tmp_path)
    (ep / "metrics.json").write_text(json.dumps({
        "step": 4,
        "l1_before": 0.42,
        "l1_after": 0.11,
        "backend": "gsfix-gsplat",
    }))
    (ep / "step_004_repair.json").write_text(json.dumps({
        "step": 4,
        "l1_before": 0.42,
        "l1_after": 0.11,
        "backend": "gsfix-gsplat",
    }))
    app = _FakeApp(ep)
    studio = RepairStudio(app)
    body, name = studio.metrics_payload(ep.name)
    assert name == "metrics.json"
    assert body["l1_improved"] is True
    assert body["l1_after"] == 0.11
    stepped, step_name = studio.metrics_payload(ep.name, step=4)
    assert step_name == "step_004_metrics.json"
    assert stepped["l1_before"] == 0.42
    missing, _ = studio.metrics_payload(ep.name, step=9)
    assert missing is None
    review = studio.episode_review(ep.name)
    assert review["has_metrics"] is True
    assert review["metrics_url"].startswith("/api/repair/metrics")


def test_focused_cap_is_twelve_hours():
    assert FOCUSED_REPAIR_MAX_SECONDS == 12 * 3600
    assert focused_cap_label(FOCUSED_REPAIR_MAX_SECONDS) == "12h"
    assert focused_cap_label(3600) == "1h"
    assert focused_cap_label(90) == "2 min"


def test_focused_finish_message_does_not_claim_cap_on_early_exit():
    early = focused_finish_message(
        stopped=False, hit_deadline=False, step=27, elapsed=11.4, cap=12 * 3600,
    )
    assert "1h cap" not in early
    assert early.startswith("Finished on step 27 after 11s")
    capped = focused_finish_message(
        stopped=False, hit_deadline=True, step=27, elapsed=12 * 3600, cap=12 * 3600,
    )
    assert capped.startswith("Reached 12h cap on step 27")
    stopped = focused_finish_message(
        stopped=True, hit_deadline=False, step=27, elapsed=42, cap=12 * 3600,
    )
    assert stopped.startswith("Stopped on step 27 after 42s")


def test_start_replay_focused_uses_twelve_hour_cap(tmp_path: Path):
    ep = _episode(tmp_path)
    _regen_view(ep)
    _tiny_ply(ep / "scene_original.ply")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message = studio.start_replay(
        ep.name, step=4, backend="cpu-project", reload_code=False,
    )
    assert ok, message
    assert "12h" in message
    assert studio.job["max_seconds"] == FOCUSED_REPAIR_MAX_SECONDS
    studio.stop_replay()


def test_pick_interactive_camera_skips_spectator_and_prefers_recent():
    cameras = [
        {
            "id": 1, "width": 1920, "height": 1080, "usable": False,
            "updated_at": 50.0, "position": [0, 0, 0],
        },
        {
            "id": 2, "width": 960, "height": 720, "usable": True,
            "updated_at": 10.0, "position": [1, 1, 1],
        },
        {
            "id": 3, "width": 640, "height": 800, "usable": False,
            "updated_at": 20.0, "position": [2, 2, 2],
        },
    ]
    picked = pick_interactive_camera(cameras)
    assert picked["id"] == 3
    newer = dict(cameras[1], updated_at=99.0)
    picked = pick_interactive_camera([cameras[0], newer, cameras[2]])
    assert picked["id"] == 2


class _FakeRegen:
    def __init__(self):
        self.submitted = []

    def submit(self, image_path, episode_dir, step, on_done=None):
        from splat_explorer.agent.regenerate import RegenerateResult, write_meta
        self.submitted.append((Path(image_path).name, Path(episode_dir), int(step)))
        write_meta(episode_dir, RegenerateResult(step=int(step), status="queued", model="gpt-image-2"))


def test_add_view_is_dashboard_only_and_queues_image_repair(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    ep = _episode(tmp_path)
    _regen_view(ep, step=4)
    actions = (ep / "actions.jsonl").read_text()
    meta = (ep / "meta.json").read_text()
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    regen = _FakeRegen()
    studio._visor_cameras = lambda: [{
        "id": 7, "width": 640, "height": 800, "usable": False, "updated_at": 1.0,
        "position": [1.0, 1.5, 2.0], "look_at": [2.0, 1.4, 3.0],
    }]
    studio._capture_view_rgb = lambda camera: np.full(
        (camera.height, camera.width, 3), 90, dtype=np.uint8,
    )
    studio._ensure_regenerator = lambda: regen
    ok, message, extra = studio.add_view(ep.name)
    assert ok, message
    assert extra["step"] == 5
    assert extra["custom"] is True
    assert extra["queued"] is True
    assert regen.submitted == [("step_005.png", ep, 5)]
    assert (ep / "step_005.png").is_file()
    assert Image.open(ep / "step_005.png").size == (960, 720)
    assert (ep / "repair_custom_views.jsonl").is_file()
    assert (ep / "actions.jsonl").read_text() == actions
    assert (ep / "meta.json").read_text() == meta
    review = studio.episode_review(ep.name)
    steps = [v["step"] for v in review["views"]]
    assert steps[-1] == 5
    custom = review["views"][-1]
    assert custom["custom"] is True
    assert custom["repaired_url"] is None
    assert custom["regen_status"] == "queued"
    snap = studio.snapshot(ep.name)
    assert snap["pending_regen"] is True
    ok, look_msg = studio.look_at(ep.name, 5)
    assert ok, look_msg
    ok, replay_msg = studio.start_replay(ep.name, step=5, backend="cpu-project")
    assert ok is False
    assert "step 5" in replay_msg.lower()
    Image.new("RGB", (16, 12), (200, 180, 40)).save(ep / "step_005_regen.png")
    ok, replay_all = studio.start_replay(ep.name, backend="cpu-project", reload_code=False)
    assert ok, replay_all
    assert studio.job["n_views"] == 1
    studio.stop_replay()
    ok, replay_msg = studio.start_replay(
        ep.name, step=5, backend="cpu-project", reload_code=False,
    )
    assert ok, replay_msg
    studio.stop_replay()


def test_add_view_blocked_while_episode_runs(tmp_path: Path):
    ep = _episode(tmp_path)
    app = _FakeApp(ep, scene_id="venetian-balcony")
    app.run = {"status": "running"}
    studio = RepairStudio(app)
    ok, message, extra = studio.add_view(ep.name)
    assert ok is False
    assert extra == {}
    assert "episode" in message.lower()


def test_repaired_save_names_skip_highlight():
    assert repaired_save_name(1) == "scene_repaired_1.ply"
    assert repaired_save_index("scene_repaired_1.ply") == 1
    assert repaired_save_index("scene_repaired_12.ply") == 12
    assert repaired_save_index("scene_repaired.ply") is None
    assert repaired_save_index("scene_repaired_highlight.ply") is None
    assert repaired_save_index("scene_repaired_1_highlight.ply") is None


def test_save_repair_numbers_copies_and_survives_reset(tmp_path, monkeypatch):
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    monkeypatch.chdir(tmp_path)
    ep = tmp_path / "outputs" / "episodes" / "20260901_190223"
    ep.mkdir(parents=True)
    (ep / "meta.json").write_text(json.dumps({
        "params": {"scene": "venetian-balcony", "scene_label": "Venetian Balcony"},
    }))
    _tiny_ply(ep / "scene_original.ply", (0.2, 0.4, 0.8))
    _tiny_ply(ep / "scene_repaired.ply", (0.9, 0.1, 0.1))
    (ep / "scene_repaired_highlight.ply").write_bytes(b"ply\n")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)

    ok, message, extra = studio.save_repair(ep.name)
    assert ok, message
    assert extra["save"] == 1
    assert extra["name"] == "scene_repaired_1.ply"
    assert next_repaired_save_index(ep) == 2
    ok, message, extra = studio.save_repair(ep.name)
    assert ok, message
    assert extra["save"] == 2
    assert [n for n, _ in list_repaired_saves(ep)] == [1, 2]

    from splat_explorer.scene import load_ply

    snap1 = load_ply(ep / "scene_repaired_1.ply")
    np.testing.assert_allclose(snap1.colors[0], [0.9, 0.1, 0.1], atol=1e-4)

    ok, message = studio.reset_repair(ep.name)
    assert ok, message
    restored = load_ply(ep / "scene_repaired.ply")
    np.testing.assert_allclose(restored.colors[0], [0.2, 0.4, 0.8], atol=1e-4)
    assert (ep / "scene_repaired_1.ply").is_file()
    assert (ep / "scene_repaired_2.ply").is_file()
    assert "saved snapshot" in message.lower()
    assert studio.showing_save is None

    review = studio.episode_review(ep.name)
    assert [s["id"] for s in review["repaired_saves"]] == [1, 2]
    assert review["repaired_saves"][0]["name"] == "scene_repaired_1.ply"
    snap = studio.snapshot(ep.name)
    assert snap["showing_save"] is None
    assert snap["episode"]["repaired_saves"][0]["id"] == 1

    ok, message = studio.show(ep.name, "repaired", save=1)
    assert ok, message
    assert studio.showing == "repaired"
    assert studio.showing_save == 1
    live = json.loads((tmp_path / "outputs" / "live" / "scene.json").read_text())
    assert live["path"] == "outputs/episodes/20260901_190223/scene_repaired_1.ply"
    assert live["id"] == "repair-save-1"
    shown = load_ply(ep / "scene_repaired_1.ply")
    np.testing.assert_allclose(shown.colors[0], [0.9, 0.1, 0.1], atol=1e-4)

    ok, message = studio.show(ep.name, "repaired")
    assert ok, message
    assert studio.showing_save is None
    live = json.loads((tmp_path / "outputs" / "live" / "scene.json").read_text())
    assert live["path"].endswith("scene_repaired.ply")


def test_save_repair_requires_working_ply(tmp_path: Path):
    ep = _episode(tmp_path)
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    ok, message, extra = studio.save_repair(ep.name)
    assert ok is False
    assert extra == {}
    assert "scene_repaired.ply" in message


def test_save_repair_blocked_while_running(tmp_path: Path):
    ep = _episode(tmp_path)
    _tiny_ply(ep / "scene_repaired.ply")
    app = _FakeApp(ep, scene_id="venetian-balcony")
    studio = RepairStudio(app)
    studio.job = {**studio._idle_job(ep.name), "status": "running"}
    ok, message, extra = studio.save_repair(ep.name)
    assert ok is False
    assert extra == {}
    assert "stop" in message.lower()
    assert not (ep / "scene_repaired_1.ply").is_file()
