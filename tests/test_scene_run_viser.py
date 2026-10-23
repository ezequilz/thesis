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
        self.active = {}
        self.world_axes = SimpleNamespace(visible=True)

    def remove_by_name(self, name):
        self.removed.append(name)
        self.active.pop(name, None)

    def add_gaussian_splats(self, name, **_kwargs):
        self.splats.append(name)
        self.active[name] = _kwargs
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
    assert visor._server.scene.splats == ["/review/repaired"]
    assert set(visor._server.scene.active) == {"/review/repaired"}

    flipped = visor.show("run_20260915_200000", toggle=True)
    assert flipped["which"] == "original"
    assert flipped["status"] == "ready"
    assert set(visor._server.scene.active) == {"/review/original"}
    assert loaded[-1] == str(run_dir / "scene_original.ply")

    back = visor.show("run_20260915_200000", toggle=True)
    assert back["which"] == "repaired"
    assert len(loaded) == 2
    assert len(visor._server.scene.splats) == 3
    assert set(visor._server.scene.active) == {"/review/repaired"}
    visor.show("run_20260915_200000", which="repaired")
    assert len(visor._server.scene.splats) == 3  # same selection is a no-op
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


def test_result_viewer_hidden_outliers_cannot_change_original_depth_range(tmp_path):
    from splat_explorer.web.splatfix_results import ResultViser

    studio, run_dir = _studio(tmp_path)
    original, repaired = _tiny_scene(), _tiny_scene()
    original.means[:, 2] = [-2., -1., 0., 1.]
    repaired.means[:, 2] = [-20000., -1., 0., 40000.]
    paths = {name: run_dir / f"scene_{name}.ply" for name in ("original", "repaired")}
    catalog = SimpleNamespace(cfg=studio.cfg, detail=lambda _, **kw: {"_paths": paths,
        "scene": "room", "run_id": "test", "config": {}}, review_views=lambda _, **kw: [])
    loaded = []

    def loader(path, **kwargs):
        loaded.append(path)
        return original if path == paths["original"] else repaired

    viewer = ResultViser(catalog, background=False, server_factory=_FakeServer,
                         load_scene=loader, idle_timeout=None)
    viewer.show("result", which="original")
    api = viewer._server.scene
    # Viser bins all registered groups, including invisible ones. Merely
    # setting visible=False would make this range 60,000 instead of 3.
    def registered_depth_range():
        return np.ptp(np.concatenate([p["centers"][:, 2] for p in api.active.values()]))

    assert registered_depth_range() == 3
    viewer.show("result", toggle=True)
    assert registered_depth_range() == 60000  # actual reconstruction is unchanged
    viewer.show("result", toggle=True)
    assert registered_depth_range() == 3
    assert len(loaded) == 2
    assert np.max(repaired.means[:, 2]) == 40000
    viewer.release("result")
    assert viewer._scenes == {}


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
    assert visor._scenes == {}
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


def test_highlight_toggle_uses_only_selected_payload(tmp_path):
    studio, _ = _studio(tmp_path)
    visor = SceneRunViser(studio, background=False, server_factory=_FakeServer,
                          load_scene=lambda *a, **kw: _tiny_scene(), idle_timeout=None)
    visor.show("run_20260915_200000")
    highlighted = visor.show("run_20260915_200000", highlight=True)
    assert highlighted["highlight"] and highlighted["which"] == "repaired"
    assert visor._handles["highlight"].visible
    assert set(visor._server.scene.active) == {"/review/highlight"}
    assert not visor.show("run_20260915_200000", toggle=True)["highlight"]
    assert visor._handles["original"].visible
    visor.show("run_20260915_200000", which="repaired")
    assert visor._handles["repaired"].visible
    assert set(visor._server.scene.active) == {"/review/repaired"}
    assert len(visor._server.scene.splats) == 4


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


def test_result_status_is_ready_while_comparison_preloads(tmp_path):
    import threading
    from splat_explorer.web.splatfix_results import ResultViser

    studio, run_dir = _studio(tmp_path)
    paths = {name: run_dir / f"scene_{name}.ply" for name in ("original", "repaired")}
    calls = []
    def detail(key, **kwargs):
        calls.append(key)
        return {"_paths": paths, "scene": "room", "run_id": key, "config": {}}
    catalog = SimpleNamespace(cfg=studio.cfg, detail=detail,
                              review_views=lambda *a, **kw: [])
    preloading, finish, done = threading.Event(), threading.Event(), threading.Event()
    def loader(path, **kwargs):
        if path == paths["original"]:
            preloading.set()
            assert finish.wait(5)
            done.set()
        return _tiny_scene()
    viewer = ResultViser(catalog, server_factory=_FakeServer,
                         load_scene=loader, idle_timeout=None)
    try:
        viewer.show("result", client="tab")
        assert preloading.wait(5)
        assert viewer.snapshot("result")["status"] == "ready"
        assert set(viewer._server.scene.active) == {"/review/repaired"}
        for _ in range(5):
            assert viewer.heartbeat("result", client="tab")["status"] == "ready"
        assert calls == ["result"]
    finally:
        viewer.release("result", client="tab")
        finish.set()
        assert done.wait(5)
    assert calls == ["result"]
    assert viewer._scenes == {}


def test_websocket_keeps_idle_viewer_alive_without_heartbeats(tmp_path):
    studio, _ = _studio(tmp_path)
    viewer = SceneRunViser(studio, background=False, server_factory=_FakeServer,
                          load_scene=lambda *a, **k: _tiny_scene(), idle_timeout=None)
    run = 'run_20260915_200000'
    viewer.show(run, client='tab')
    server = viewer._server
    viewer._idle_timeout = 45
    server._clients['socket'] = object()
    now = __import__('time').monotonic() + 1000
    assert not viewer._stop_if_idle_locked(now)
    # A pagehide beacon must not shut down a still-connected sibling tab.
    viewer.release(run, client='tab')
    assert viewer._server is server
    assert not viewer._stop_if_idle_locked(now + 1000)
    server._clients.clear()
    assert not viewer._stop_if_idle_locked(now + 1001)
    assert not viewer._stop_if_idle_locked(now + 1045)
    assert viewer._stop_if_idle_locked(now + 1046)
    assert server.stopped


def test_viewer_allows_initial_connection_and_reconnection(tmp_path):
    studio, _ = _studio(tmp_path)
    viewer = SceneRunViser(studio, background=False, server_factory=_FakeServer,
                          load_scene=lambda *a, **k: _tiny_scene(), idle_timeout=None)
    viewer.show('run_20260915_200000')
    viewer._idle_timeout = 45
    server = viewer._server
    assert not viewer._stop_if_idle_locked(0)
    assert not viewer._stop_if_idle_locked(44)
    server._clients['socket'] = object()
    assert not viewer._stop_if_idle_locked(45)
    server._clients.clear()
    assert not viewer._stop_if_idle_locked(100)
    server._clients['socket'] = object()
    assert not viewer._stop_if_idle_locked(140)
    server._clients.clear()
    assert not viewer._stop_if_idle_locked(200)
    assert viewer._stop_if_idle_locked(245)


def test_browser_polls_only_while_loading():
    import shutil
    import subprocess
    from pathlib import Path
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for viewer JavaScript test')
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    classList: {toggle(){}, remove(){}}, setAttribute(){}, removeAttribute(name){if(name === "src") this.src = "";},
    firstChild: {}, src: '', disabled: false
  });
  return elements.get(id);
}
let responses = ['loading', 'ready'], requests = [], timers = new Map(), nextTimer = 0;
let visibilityChanged, beacons = 0, lastDelay;
const doc = {getElementById: element, querySelector: element, body: {}, hidden: false,
  addEventListener(name, fn) {if(name === 'visibilitychange') visibilityChanged = fn;}};
const context = vm.createContext({
  document: doc,
  Blob, navigator: {sendBeacon(){beacons++; return true;}},
  location: {pathname: '/splatfix/results/viser', search: '?id=result'},
  crypto: {randomUUID: () => 'tab'}, URLSearchParams,
  window: {addEventListener(){}},
  setTimeout(fn, delay) {lastDelay = delay; timers.set(++nextTimer, fn); return nextTimer;},
  clearTimeout(id) {timers.delete(id);},
  fetch: async (url, options) => {
    requests.push(options?.method || 'GET');
    const status = responses.shift();
    assert(status, 'Unexpected status request');
    return {ok: true, json: async () => ({ok: true, status,
      viewer_url: 'http://localhost:8083', which: 'repaired', original: true,
      repaired: true, view_count: 2})};
  }
});
const flush = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  vm.runInContext(source, context);
  await flush();
  assert.equal(timers.size, 1);
  const [id, poll] = [...timers][0]; timers.delete(id); poll();
  await flush();
  assert.equal(timers.size, 0, 'Ready viewer must not poll');
  assert.deepEqual(requests, ['POST', 'GET']);
  responses.push('ready'); element('flip').onclick(); await flush();
  assert.equal(timers.size, 0, 'Cached splat switch must not poll');
  responses.push('ready'); element('view-next').onclick(); await flush();
  assert.equal(timers.size, 0, 'Camera switch must not poll');
  const before = requests.length;
  doc.hidden = true; visibilityChanged();
  assert.equal(lastDelay, 240000);
  assert.equal(timers.size, 1);
  doc.hidden = false; visibilityChanged();
  assert.equal(timers.size, 0);
  assert.equal(beacons, 0);
  assert.equal(requests.length, before, 'Returning early must reuse the viewer');
  doc.hidden = true; visibilityChanged();
  const [hideId, expire] = [...timers][0]; timers.delete(hideId); expire();
  assert.equal(beacons, 1);
  assert.equal(element('frame').src, '', 'Expired tab must disconnect its WebSocket');
  assert.equal(timers.size, 0);
  responses.push('ready'); doc.hidden = false; visibilityChanged(); await flush();
  assert.equal(requests.length, before + 1, 'Returning after expiry restarts viewer');
  assert.equal(element('frame').src, 'http://localhost:8083');
  assert.equal(timers.size, 0);
  responses.push('loading', 'error'); element('flip').onclick(); await flush();
  assert.equal(timers.size, 1);
  const [lastId, lastPoll] = [...timers][0]; timers.delete(lastId); lastPoll();
  await flush();
  assert.equal(timers.size, 0, 'Failed load must not poll forever');
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    page = Path('src/splat_explorer/web/static/scene_run_viser.html')
    result = subprocess.run([node, '-e', script, str(page)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
