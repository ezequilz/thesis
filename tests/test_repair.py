"""View-local 3DGS repair after a gpt-image-2 regenerate comes back."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.agent.actions import Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.loop import run_episode
from splat_explorer.agent.regenerate import RegenerateResult, regenerate_frame_name
from splat_explorer.rendering.base import Camera
from splat_explorer.repair import (
    ORIGINAL_PLY,
    REPAIRED_PLY,
    PhotometricViewRepair,
    SceneRepairer,
    copy_camera,
    scene_from_renderer,
    visible_gaussians,
)
from splat_explorer.scene import GaussianScene, load_ply, save_ply


def _scene(means, colors=None, opacities=None, scales=0.05) -> GaussianScene:
    means = np.asarray(means, dtype=np.float32)
    n = len(means)
    if colors is None:
        colors = np.full((n, 3), 0.5, dtype=np.float32)
    else:
        colors = np.asarray(colors, dtype=np.float32)
    if opacities is None:
        opacities = np.full((n,), 0.8, dtype=np.float32)
    if np.isscalar(scales):
        scales = np.full((n, 3), float(scales), dtype=np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    return GaussianScene(
        means=means, scales=scales, quats=quats, opacities=opacities, colors=colors,
    )


def _front_camera(width=64, height=48) -> Camera:
    return Camera.look_at(
        np.array([0.0, 0.0, -2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        width=width, height=height, fov_deg=75.0,
    )


def test_visible_gaussians_frustum():
    scene = _scene([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -5.0],
        [40.0, 0.0, 0.0],
    ])
    camera = _front_camera()
    idx, uf, vf, z = visible_gaussians(scene, camera)
    assert list(idx) == [0]
    assert z[0] > 0
    assert 0 <= uf[0] < camera.width
    assert 0 <= vf[0] < camera.height


def test_copy_camera_is_independent():
    camera = _front_camera()
    cloned = copy_camera(camera)
    cloned.position[0] = 99.0
    assert camera.position[0] == 0.0


def test_save_ply_roundtrip(tmp_path: Path):
    scene = _scene(
        [[1.0, 2.0, 3.0], [0.5, -0.25, 0.0]],
        colors=[[0.2, 0.4, 0.8], [0.9, 0.1, 0.1]],
    )
    path = tmp_path / "tiny.ply"
    save_ply(scene, path)
    loaded = load_ply(path)
    assert loaded.num_gaussians == 2
    np.testing.assert_allclose(loaded.means, scene.means, atol=1e-5)
    np.testing.assert_allclose(loaded.colors, scene.colors, atol=1e-4)
    np.testing.assert_allclose(loaded.opacities, scene.opacities, atol=1e-4)
    np.testing.assert_allclose(loaded.scales, scene.scales, atol=1e-5)


def test_scene_copy_does_not_alias():
    scene = _scene([[0.0, 0.0, 0.0]])
    cloned = scene.copy()
    cloned.colors[0] = 0.0
    assert scene.colors[0, 0] == 0.5


def test_photometric_updates_only_visible(tmp_path: Path):
    visible = [0.4, 0.4, 0.4]
    hidden = [0.9, 0.1, 0.1]
    scene = _scene(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -6.0]],
        colors=[visible, hidden],
    )
    source = scene.copy()
    camera = _front_camera()
    rendered = np.full((camera.height, camera.width, 3), 80, dtype=np.uint8)
    repaired = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    repaired[..., 1] = 255
    Image.fromarray(rendered).save(tmp_path / "src.png")
    Image.fromarray(repaired).save(tmp_path / "fix.png")

    repairer = SceneRepairer(scene, backend=PhotometricViewRepair(densify=False))
    result = repairer.apply_view(
        step=0, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok"
    assert result.n_visible == 1
    assert result.n_updated == 1
    assert result.n_spawned == 0
    working = repairer._working
    assert working is not None
    assert working.colors[0, 1] > working.colors[0, 0]
    assert working.colors[0, 1] > source.colors[0, 1]
    np.testing.assert_allclose(working.colors[1], hidden, atol=1e-6)
    np.testing.assert_allclose(scene.colors, source.colors)
    assert (tmp_path / ORIGINAL_PLY).is_file()
    assert (tmp_path / REPAIRED_PLY).is_file()
    original = load_ply(tmp_path / ORIGINAL_PLY)
    np.testing.assert_allclose(original.colors[0], visible, atol=1e-4)


def test_densify_fills_empty_pixels(tmp_path: Path):
    scene = _scene([[20.0, 0.0, 0.0]])  # off to the side of a 64x48 75° view
    camera = _front_camera()
    rendered = np.full((camera.height, camera.width, 3), 20, dtype=np.uint8)
    repaired = np.full((camera.height, camera.width, 3), 180, dtype=np.uint8)
    Image.fromarray(rendered).save(tmp_path / "src.png")
    Image.fromarray(repaired).save(tmp_path / "fix.png")
    n0 = scene.num_gaussians
    repairer = SceneRepairer(
        scene,
        backend=PhotometricViewRepair(densify=True, max_new=16, hole_stride=8, hole_l1=0.05),
    )
    result = repairer.apply_view(
        step=1, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok"
    assert result.n_spawned > 0
    assert repairer._working is not None
    assert repairer._working.num_gaussians == n0 + result.n_spawned
    assert scene.num_gaussians == n0


def test_backend_is_swappable(tmp_path: Path):
    scene = _scene([[0.0, 0.0, 0.0]])
    camera = _front_camera()
    Image.fromarray(np.zeros((48, 64, 3), np.uint8)).save(tmp_path / "src.png")
    Image.fromarray(np.full((48, 64, 3), 255, np.uint8)).save(tmp_path / "fix.png")
    calls = []

    class Marker:
        def apply(self, scene, camera, rendered_rgb, repaired_rgb):
            calls.append(scene.num_gaussians)
            scene.colors[:] = 0.1
            return {
                "n_visible": 1, "n_updated": 1, "n_spawned": 0,
                "n_gaussians": 1, "l1_before": 0.5,
            }

    repairer = SceneRepairer(scene, backend=Marker())
    result = repairer.apply_view(
        step=0, camera=camera,
        rendered_path=tmp_path / "src.png",
        repaired_path=tmp_path / "fix.png",
        episode_dir=tmp_path,
    )
    assert result.status == "ok"
    assert calls == [1]
    assert repairer._working.colors[0, 0] == np.float32(0.1)


def test_on_done_runs_after_regenerate(tmp_path: Path):
    scene = _scene([[0.0, 0.0, 0.0]], colors=[[0.5, 0.5, 0.5]])
    camera = _front_camera()
    src = tmp_path / "step_003.png"
    Image.fromarray(np.full((48, 64, 3), 70, np.uint8)).save(src)
    Image.fromarray(np.full((48, 64, 3), (10, 220, 10), np.uint8)).save(
        tmp_path / "step_003_regen.png",
    )
    repairer = SceneRepairer(scene, backend=PhotometricViewRepair(densify=False))
    on_done = repairer.make_callback(3, camera, src, tmp_path)
    on_done(RegenerateResult(step=3, status="ok", image_name="step_003_regen.png"))
    assert repairer.results[0].status == "ok"
    assert repairer.results[0].n_visible == 1
    meta = json.loads((tmp_path / "step_003_repair.json").read_text())
    assert meta["status"] == "ok"
    log = json.loads((tmp_path / "repair_log.json").read_text())
    assert log["repaired_ply"] == REPAIRED_PLY


def test_skips_when_regenerate_has_no_image(tmp_path: Path):
    scene = _scene([[0.0, 0.0, 0.0]])
    repairer = SceneRepairer(scene)
    repairer.make_callback(0, _front_camera(), tmp_path / "missing.png", tmp_path)
    result = repairer.apply_from_regenerate(
        RegenerateResult(step=0, status="no_image"),
    )
    assert result.status == "skipped"
    assert repairer._working is None
    assert not (tmp_path / REPAIRED_PLY).exists()


class _SceneRenderer:
    def __init__(self, scene):
        self.scene = scene

    def render(self, camera):
        return np.full((camera.height, camera.width, 3), 40, dtype=np.uint8)


class _ReportOnce:
    last_debug = None
    allow_done = True

    def decide(self, observation, pose, step, depth_image=None, map_image=None,
               coverage_image=None):
        if step == 0:
            return Action("report_artifact", {
                "description": "hole",
                "image_region": "center",
                "severity": "high",
                "regenerate": "yes",
            })
        return Action("done", {"summary": "ok"})


class _CallingRegen:
    def __init__(self, color=(20, 200, 20)):
        self.color = color

    def submit(self, image_path, episode_dir, step, on_done=None):
        png = episode_dir / regenerate_frame_name(step)
        Image.fromarray(np.full((4, 4, 3), self.color, dtype=np.uint8)).save(png)
        result = RegenerateResult(step=step, status="ok", image_name=png.name, n_images=1)
        if on_done is not None:
            on_done(result)

        class _Fut:
            def result(self, timeout=None):
                return result
        return _Fut()

    def wait(self, timeout=None):
        return []


def test_loop_starts_repair_when_image_regeneration_is_on(tmp_path: Path):
    scene = _scene([[0.0, 1.5, -2.0]], colors=[[0.4, 0.4, 0.4]])
    before = scene.colors.copy()
    episode_dir = run_episode(
        renderer=_SceneRenderer(scene),
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=_ReportOnce(),
        output_dir=tmp_path,
        width=32, height=24, fov_deg=75.0,
        max_steps=2,
        send_map=False,
        image_regeneration=True,
        regenerator=_CallingRegen(),
    )
    assert (episode_dir / ORIGINAL_PLY).is_file()
    assert (episode_dir / REPAIRED_PLY).is_file()
    artifacts = json.loads((episode_dir / "artifacts.json").read_text())
    assert artifacts[0]["repair_status"] == "ok"
    assert artifacts[0]["repair_ply"] == REPAIRED_PLY
    meta = json.loads((episode_dir / "meta.json").read_text())
    assert meta["scene_original"] == ORIGINAL_PLY
    assert meta["scene_repaired"] == REPAIRED_PLY
    np.testing.assert_allclose(scene.colors, before)
    repaired = load_ply(episode_dir / REPAIRED_PLY)
    assert repaired.colors[0, 1] >= before[0, 1]


def test_loop_does_not_repair_when_regeneration_off(tmp_path: Path):
    scene = _scene([[0.0, 1.5, -2.0]])
    episode_dir = run_episode(
        renderer=_SceneRenderer(scene),
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=_ReportOnce(),
        output_dir=tmp_path,
        width=32, height=24, fov_deg=75.0,
        max_steps=2,
        send_map=False,
        image_regeneration=False,
        regenerator=_CallingRegen(),
    )
    assert not (episode_dir / REPAIRED_PLY).exists()
    artifacts = json.loads((episode_dir / "artifacts.json").read_text())
    assert artifacts[0]["regenerate_status"] == "disabled"
    assert "repair_status" not in artifacts[0]


def test_scene_from_renderer_viser_twin():
    scene = _scene([[0.0, 0.0, 0.0]])

    class _Cpu:
        def __init__(self, scene):
            self.scene = scene

    class _Viser:
        def __init__(self, scene):
            self._cpu = _Cpu(scene)

    assert scene_from_renderer(_SceneRenderer(scene)) is scene
    assert scene_from_renderer(_Viser(scene)) is scene
    assert scene_from_renderer(object()) is None
