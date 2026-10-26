"""Custom capture owns its scene, client, checkpoint and viewer lifetime."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from splat_explorer.config import Config
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.web import custom_selection_viser as custom


class Handle:
    def __init__(self, content=''):
        self.content = content
        self.removed = False

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def on_click(self, callback): self.callback = callback
    def remove(self): self.removed = True


class Server:
    def __init__(self, host, port):
        self.port = port
        self.stopped = False
        self.clients = {}
        self.scene = SimpleNamespace(world_axes=SimpleNamespace(visible=True))
        self.gui = SimpleNamespace(configure_theme=lambda **kw: None, set_panel_label=lambda _: None,
                                   add_folder=lambda _: Handle(), add_button=lambda _: Handle(), add_markdown=Handle)

    def get_port(self): return self.port
    def get_clients(self): return self.clients
    def on_client_connect(self, callback): self.connect = callback
    def flush(self): pass
    def stop(self): self.stopped = True


@pytest.fixture
def session(tmp_path, monkeypatch):
    source = tmp_path / 'chosen.ply'
    source.touch()
    cp = Checkpoint.create(tmp_path / 'checkpoints', source, target_views=3, metadata={
        'selection': 'manual', 'renderer': {'backend': 'viser'}, 'width': 32, 'height': 16,
        'manual_scene': {'id': 'chosen-room', 'generation': 42}, 'up_axis': '-z',
        'scene_load': {'min_opacity': .02, 'lod_level': 2}})
    loaded, installed, captures = [], [], []
    scene = object()
    def loader(path, **kwargs):
        loaded.append((path, kwargs))
        return scene
    monkeypatch.setattr(custom, '_install_splats', lambda server, value, limit: installed.append((server, value, limit)))
    monkeypatch.setattr(custom, '_view_pose', lambda *a: dict(center=np.array([1., 2., 3.]),
                         forward=np.array([0., 0., 1.]), up=np.array([0., 0., -1.]), vfov=1.))
    from splat_explorer.scene import catalog
    monkeypatch.setattr(catalog, 'publish_live_scene', lambda *a, **kw: pytest.fail('Main scene changed'))
    visor = custom.CustomSelectionViser(Config({'viewer': {'port': 8080}}),
        server_factory=Server, scene_loader=loader, background=False, idle_timeout=45)
    state = visor.start(cp)
    client = SimpleNamespace(client_id=17, atomic=nullcontext, camera=SimpleNamespace(
        wxyz=np.array([1., 0., 0., 0.]), aspect=2., image_width=640, image_height=320, fov=1.))
    def render(**params):
        captures.append(params)
        return np.full((params['height'], params['width'], 3), 80, dtype=np.uint8)
    client.get_render = render
    return SimpleNamespace(visor=visor, state=state, cp=cp, source=source, loaded=loaded,
                           installed=installed, captures=captures, scene=scene, client=client)


def ready(s):
    s.visor._load(s.state['session'])
    server = s.visor._server
    server.clients[17] = s.client
    server.connect(s.client)
    return server


def test_chosen_scene_capture_and_finish_are_isolated(session):
    s = session
    assert s.state['status'] == 'loading'
    assert s.state['viewer_port'] == 8084
    assert not s.loaded
    server = ready(s)
    assert s.loaded == [(s.source, {'min_opacity': .02, 'lod_level': 2})]
    assert s.installed == [(server, s.scene, 0)]
    selection = s.visor._selection
    event = SimpleNamespace(client=s.client)
    selection.save_current(event)
    saved = Checkpoint.load(s.cp.root)
    assert len(saved.views) == 1, selection.status.content
    assert s.captures[0]['width'] == 32 and s.captures[0]['height'] == 16
    np.testing.assert_array_equal(s.captures[0]['position'], s.client.camera.position)
    selection.finish(event)
    state = s.visor.snapshot(s.state['session'])
    assert state['status'] == 'finished' and state['saved_views'] == 1
    assert state['viewer_port'] is None and server.stopped
    assert Checkpoint.load(s.cp.root).complete


def test_close_keeps_partial_checkpoint_and_rejects_stale_tabs(session):
    s = session
    server = ready(s)
    s.visor._selection.save_current(SimpleNamespace(client=s.client))
    with pytest.raises(ValueError, match='current Custom'):
        s.visor.start(s.cp)
    with pytest.raises(ValueError, match='no longer available'):
        s.visor.release('someone-elses-session')
    assert not server.stopped
    result = s.visor.release(s.state['session'])
    assert result['status'] == 'closed' and server.stopped
    saved = Checkpoint.load(s.cp.root)
    assert len(saved.views) == 1 and not saved.complete
    new = s.visor.start(saved)
    with pytest.raises(ValueError):
        s.visor.release(s.state['session'])
    assert s.visor.snapshot(new['session'])['status'] == 'loading'


def test_last_tab_disconnect_stops_after_reconnect_grace(session):
    s = session
    server = ready(s)
    s.visor._maintain_locked(100.)
    server.clients.clear()
    s.visor._maintain_locked(110.)
    s.visor._maintain_locked(150.)
    assert not server.stopped
    server.clients[17] = s.client
    s.visor._maintain_locked(151.)
    server.clients.clear()
    s.visor._maintain_locked(152.)
    s.visor._maintain_locked(198.)
    assert server.stopped


def test_close_during_load_cannot_install_a_late_scene(session):
    s = session
    server = s.visor._server
    def loader(*args, **kwargs):
        s.visor.release(s.state['session'])
        return s.scene
    s.visor._scene_loader = loader
    s.visor._load(s.state['session'])
    assert server.stopped and not s.installed
    assert s.visor.snapshot(s.state['session'])['status'] == 'closed'


def test_failed_load_releases_resources_and_surfaces_error(session):
    s = session
    server = s.visor._server
    def loader(*args, **kwargs):
        raise ValueError('Invalid scene')
    s.visor._scene_loader = loader
    s.visor._load(s.state['session'])
    state = s.visor.snapshot(s.state['session'])
    assert not state['ok'] and state['error'] == 'Invalid scene'
    assert server.stopped and state['viewer_port'] is None


def test_custom_refuses_wrong_runtime_version(session, monkeypatch):
    viser = pytest.importorskip('viser')
    monkeypatch.setattr(viser, '__version__', '1.1.1')
    visor = custom.CustomSelectionViser(Config({}), background=False)
    with pytest.raises(ValueError, match='requires Viser 1.1.0'):
        visor.start(session.cp)
