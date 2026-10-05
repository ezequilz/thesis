"""Scene-run review visor: isolated original/repaired splat dashboard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from splat_explorer.scene.types import GaussianScene
from splat_explorer.web.scene_run_viser import (
    SceneRunViser,
    parse_run_viser_api,
    parse_run_viser_page,
)
from splat_explorer.web.scene_run_studio import SCENE_RUN_DEFAULTS, SceneRunStudio


class _Store:
    def __init__(self, root):
        self.root = root
        self.runs = [{
            "run_id": "run_20260915_200000",
            "config": dict(SCENE_RUN_DEFAULTS),
            "state": {"status": "completed"},
        }]

    def detail(self, run_id):
        return next((run for run in self.runs if run["run_id"] == run_id), None)

    def list_runs(self):
        return list(self.runs)

    def run_path(self, run_id):
        return self.root / run_id


class _FakeSceneApi:
    def __init__(self):
        self.removed = []
        self.splats = []
        self.world_axes = SimpleNamespace(visible=True)

    def remove_by_name(self, name):
        self.removed.append(name)

    def add_gaussian_splats(self, name, **_kwargs):
        self.splats.append(name)
        return SimpleNamespace(name=name, visible=_kwargs.get("visible", True))


class _FakeGui:
    def configure_theme(self, **_kwargs):
        return None

    def set_panel_label(self, *_args):
        return None

    def add_html(self, *_args, **_kwargs):
        return None


class _FakeServer:
    def __init__(self, host="0.0.0.0", port=8082, verbose=False):
        self.host = host
        self._port = port
        self.verbose = verbose
        self.stopped = False
        self.scene = _FakeSceneApi()
        self.gui = _FakeGui()
        self._clients = {}
        self._connect = None

    def get_port(self):
        return self._port

    def get_clients(self):
        return self._clients

    def on_client_connect(self, fn):
        self._connect = fn
        return fn

    def stop(self):
        self.stopped = True


def _tiny_scene():
    n = 4
    return GaussianScene(
        means=np.zeros((n, 3), np.float32),
        scales=np.ones((n, 3), np.float32) * 0.02,
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
        opacities=np.ones((n,), np.float32),
        colors=np.ones((n, 3), np.float32),
    )


def _studio(tmp_path):
    cfg = SimpleNamespace(
        output=SimpleNamespace(dir=str(tmp_path)),
        agent=SimpleNamespace(model="gpt-5.6-luna"),
        viewer=SimpleNamespace(host="127.0.0.1", port=8080, max_splats=0),
        scene=SimpleNamespace(min_opacity=0.0, lod_level=0),
        camera=SimpleNamespace(up_axis="+y", fov_deg=75.0),
    )
    app = SimpleNamespace(cfg=cfg)
    store = _Store(tmp_path / "scene-runs")
    studio = SceneRunStudio(app, store=store)
    studio.scenes = lambda: [{"id": "venetian-balcony", "label": "Venetian Balcony"}]
    run_dir = store.run_path("run_20260915_200000")
    run_dir.mkdir(parents=True)
    (run_dir / "scene_original.ply").write_bytes(b"orig")
    (run_dir / "scene_repaired.ply").write_bytes(b"rep")
    return studio, run_dir


def test_parse_run_viser_paths():
    assert parse_run_viser_page("/run_20260915_200000/viser") == "run_20260915_200000"
    assert parse_run_viser_page("/run_20260915_200000_2/viser/") == "run_20260915_200000_2"
    assert parse_run_viser_page("/scene-runs/viser") is None
    assert parse_run_viser_page("/not-a-run/viser") is None
    assert parse_run_viser_api("/api/scene-runs/run_20260915_200000/viser") == (
        "run_20260915_200000"
    )
    assert parse_run_viser_api("/api/scene-runs/run_20260915_200000") is None


def test_show_defaults_to_repaired_and_toggle_flips(tmp_path):
    studio, run_dir = _studio(tmp_path)
    loaded = []

    def load_scene(path, min_opacity=0.0, lod_level=0):
        loaded.append(str(path))
        return _tiny_scene()

    visor = SceneRunViser(
        studio,
        background=False,
        server_factory=_FakeServer,
        load_scene=load_scene,
        idle_timeout=None,
    )
    first = visor.show("run_20260915_200000")
    assert first["ok"] is True
    assert first["which"] == "repaired"
    assert first["status"] == "ready"
    assert first["viewer_url"] == "http://localhost:8082"
    assert loaded == [str(run_dir / "scene_repaired.ply"), str(run_dir / "scene_original.ply")]
    assert visor._server.scene.splats == ["/review/repaired", "/review/original", "/review/highlight"]
    handles = dict(visor._handles)
    assert handles["repaired"].visible
    assert not handles["original"].visible

    flipped = visor.show("run_20260915_200000", toggle=True)
    assert flipped["which"] == "original"
    assert flipped["status"] == "ready"
    assert handles["original"].visible
    assert not handles["repaired"].visible
    assert loaded[-1] == str(run_dir / "scene_original.ply")

    back = visor.show("run_20260915_200000", toggle=True)
    assert back["which"] == "repaired"
    assert len(loaded) == 2
    assert len(visor._server.scene.splats) == 3
    assert visor._handles == handles
    assert handles["repaired"].visible
    assert not handles["original"].visible
    snap = visor.snapshot("run_20260915_200000")
    assert snap["which"] == "repaired"
    assert snap["original"] is True
    assert snap["repaired"] is True


def test_show_missing_ply_is_an_error(tmp_path):
    studio, run_dir = _studio(tmp_path)
    (run_dir / "scene_repaired.ply").unlink()
    visor = SceneRunViser(
        studio,
        background=False,
        server_factory=_FakeServer,
        load_scene=lambda *_args, **_kwargs: _tiny_scene(),
        idle_timeout=None,
    )
    result = visor.show("run_20260915_200000", which="repaired")
    assert result["ok"] is False
    assert "scene_repaired.ply" in result["message"]

    fallback = visor.show("run_20260915_200000")
    assert fallback["ok"] is True
    assert fallback["which"] == "original"


def test_snapshot_and_heartbeat_do_not_start_server(tmp_path):
    studio, _run_dir = _studio(tmp_path)
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return _FakeServer(**kwargs)

    visor = SceneRunViser(
        studio,
        background=False,
        server_factory=factory,
        load_scene=lambda *_args, **_kwargs: _tiny_scene(),
        idle_timeout=None,
    )
    snap = visor.snapshot("run_20260915_200000")
    assert snap["status"] == "idle"
    assert snap["viewer_url"] is None
    assert created == []

    beat = visor.heartbeat("run_20260915_200000", client="tab-a")
    assert beat["status"] == "idle"
    assert created == []


def test_one_server_is_reused_then_stopped_when_released(tmp_path):
    studio, run_dir = _studio(tmp_path)
    other_id = "run_20260915_200001"
    studio.store.runs.append({
        "run_id": other_id,
        "config": dict(SCENE_RUN_DEFAULTS),
        "state": {"status": "completed"},
    })
    other_dir = studio.store.run_path(other_id)
    other_dir.mkdir(parents=True)
    (other_dir / "scene_repaired.ply").write_bytes(b"rep")
    created = []

    def factory(**kwargs):
        server = _FakeServer(**kwargs)
        created.append(server)
        return server

    visor = SceneRunViser(
        studio,
        background=False,
        server_factory=factory,
        load_scene=lambda *_args, **_kwargs: _tiny_scene(),
        idle_timeout=None,
    )
    first = visor.show("run_20260915_200000", client="tab-a")
    second = visor.show(other_id, client="tab-b")
    assert first["ok"] and second["ok"]
    assert len(created) == 1
    assert created[0] is visor._server
    assert visor._run_id == other_id

    visor.release("run_20260915_200000", client="tab-a")
    assert visor._server is created[0]
    assert not getattr(created[0], "stopped", False)

    visor.release(other_id, client="tab-b")
    assert visor._server is None
    assert created[0].stopped is True
    assert visor._handles == {}
    assert visor.snapshot("run_20260915_200000")["status"] == "idle"


def test_close_during_load_cannot_repopulate_stopped_session(tmp_path):
    studio, _ = _studio(tmp_path)
    loaded = []
    def loader(path, **kwargs):
        loaded.append(path)
        visor.release("run_20260915_200000", client="tab-a")
        return _tiny_scene()
    visor = SceneRunViser(studio, background=False, server_factory=_FakeServer,
                          load_scene=loader, idle_timeout=None)
    visor.show("run_20260915_200000", client="tab-a")
    assert visor._server is None
    assert visor._handles == {}
    assert len(loaded) == 1


def test_highlight_toggle_reuses_payloads(tmp_path):
    studio, _ = _studio(tmp_path)
    visor = SceneRunViser(studio, background=False, server_factory=_FakeServer,
                          load_scene=lambda *a, **kw: _tiny_scene(), idle_timeout=None)
    visor.show("run_20260915_200000")
    highlighted = visor.show("run_20260915_200000", highlight=True)
    assert highlighted["highlight"] and highlighted["which"] == "repaired"
    assert visor._handles["highlight"].visible
    assert not visor._handles["repaired"].visible
    assert not visor.show("run_20260915_200000", toggle=True)["highlight"]
    assert visor._handles["original"].visible
    visor.show("run_20260915_200000", which="repaired")
    assert visor._handles["repaired"].visible
    assert len(visor._server.scene.splats) == 3


def test_difference_strength_and_reordered_pruned_splats():
    from splat_explorer.web.repair_difference import repair_difference
    original = _tiny_scene()
    original.means[:, 0] = np.arange(4)
    original.colors[:] = .4
    repaired = original.filtered(np.array([False, True, True, True]))
    repaired = repaired.filtered(np.array([2, 0, 1]))
    repaired.colors[1] += .01
    repaired.colors[2] += .2
    before = repaired.copy()
    result = repair_difference(original, repaired)
    np.testing.assert_array_equal(result.colors[0], repaired.colors[0])
    small = result.colors[1, 0] - repaired.colors[1, 0]
    large = result.colors[2, 0] - repaired.colors[2, 0]
    assert 0 < small < large
    for field in ("means", "scales", "quats", "opacities"):
        np.testing.assert_array_equal(getattr(result, field), getattr(before, field))
    np.testing.assert_array_equal(repaired.colors, before.colors)
    repaired.quats *= -1
    np.testing.assert_allclose(repair_difference(original, repaired).colors, result.colors)


def test_saved_views_initialize_navigate_and_preserve_free_camera_on_splat_toggle(tmp_path):
    studio, _ = _studio(tmp_path)
    views = [dict(center=np.array([i, 2., 3.]), forward=np.array([0., 0., -1.]),
                  up=np.array([0., 1., 0.]), vfov=0.7, label=str(i)) for i in (1., 4.)]
    studio.review_views = lambda _: views
    viewer = SceneRunViser(studio, background=False, server_factory=_FakeServer,
                           load_scene=lambda *a, **k: _tiny_scene(), idle_timeout=None)
    run = 'run_20260915_200000'
    assert viewer.show(run)['view_index'] == 0
    camera = SimpleNamespace()
    client = SimpleNamespace(camera=camera)
    viewer._server._clients['test'] = client
    viewer._server._connect(client)
    np.testing.assert_array_equal(camera.position, views[0]['center'])
    np.testing.assert_array_equal(camera.look_at, views[0]['center'] + views[0]['forward'])
    assert camera.fov == .7
    assert viewer.move_view(run, 1)['view_index'] == 1
    np.testing.assert_array_equal(camera.position, views[1]['center'])
    camera.position = np.array([8., 9., 10.])  # free orbit/pan after selecting a view
    viewer.show(run, toggle=True)
    np.testing.assert_array_equal(camera.position, [8., 9., 10.])
    assert viewer.move_view(run, 1)['view_index'] == 0  # wrap
    assert viewer.move_view(run, -1)['view_index'] == 1
