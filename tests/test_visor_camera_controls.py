"""Camera transitions are batched; RGB capture never drives navigation."""

from contextlib import contextmanager
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from splat_explorer.rendering import viser_viewer as viewer


@pytest.mark.parametrize('action', ['view', 'step', 'restore'])
def test_camera_transition_has_no_partial_pose(action):
    batches, pending = [], []
    in_batch = False

    class Camera:
        def __setattr__(self, name, value):
            assert in_batch, f'{name} would reach the browser as a partial pose'
            pending.append((name, value))

    @contextmanager
    def atomic():
        nonlocal in_batch
        in_batch = True
        try:
            yield
        finally:
            batches.append(dict(pending))
            pending.clear()
            in_batch = False

    client = SimpleNamespace(camera=Camera(), atomic=atomic)
    position, forward, up = np.array([100., 20., 3.]), np.array([0., 0., 1.]), np.array([0., -1., 0.])
    view = dict(center=position, forward=forward, up=up, vfov=1.1)
    if action == 'view':
        viewer._apply_view(SimpleNamespace(get_clients=lambda: {1: client}), view)
    elif action == 'step':
        viewer._set_client_step_camera(client, {'position': position}, forward, view)
    else:
        viewer._restore_client_cameras([(client, position, position + forward, up)])
    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0]['position'], position)
    np.testing.assert_array_equal(batches[0]['look_at'], position + forward)
    np.testing.assert_array_equal(batches[0]['up_direction'], up)
    if action != 'restore':
        assert batches[0]['fov'] == 1.1


@pytest.mark.parametrize('fail', [False, True])
def test_capture_restores_fov_and_leaves_live_camera_and_spectator_alone(monkeypatch, fail):
    live = SimpleNamespace(position=np.array([1., 2., 3.]), look_at=np.array([1., 2., 4.]),
                           up_direction=np.array([0., -1., 0.]), fov=1.2,
                           image_width=640, image_height=360)
    client = SimpleNamespace(camera=live)
    spectator_camera = SimpleNamespace(fov=.8)
    spectator = SimpleNamespace(camera=spectator_camera)
    capture = viewer._VisorCapture(SimpleNamespace(
        get_clients=lambda: {17: client, 18: spectator}, flush=lambda: None), threading.Lock(), [])
    capture.ATTEMPTS = 1
    monkeypatch.setattr(viewer.time, 'sleep', lambda _: None)
    visibility = []
    monkeypatch.setattr(capture, '_set_overlays_visible_on_client',
                        lambda c, names, visible: visibility.append((c, visible)))

    def render(c, **params):
        assert c is client
        assert params['fov'] == live.fov == .6
        np.testing.assert_array_equal(params['position'], [9., 8., 7.])
        np.testing.assert_array_equal(live.position, [1., 2., 3.])
        if fail:
            raise RuntimeError('readback failed')
        return np.zeros((360, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(capture, '_get_render', render)
    params = dict(client_id=17, width=640, height=360, fov=.6,
                  position=[9., 8., 7.], wxyz=[1., 0., 0., 0.])
    if fail:
        with pytest.raises(viewer.CaptureError, match='readback failed'):
            capture.render(params, any_client=True)
    else:
        assert capture.render(params, any_client=True).startswith(b'\xff\xd8')
    np.testing.assert_array_equal(live.position, [1., 2., 3.])
    np.testing.assert_array_equal(live.look_at, [1., 2., 4.])
    np.testing.assert_array_equal(live.up_direction, [0., -1., 0.])
    assert live.fov == 1.2
    assert spectator_camera.fov == .8
    assert visibility == [(client, False), (client, True)]
