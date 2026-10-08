import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.checkpoint import Checkpoint, camera_from_record
from splat_explorer.splatfix.manual_selection import ManualSelection, camera_from_client


class Handle:
    disabled = False
    removed = False
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def on_click(self, callback): self.callback = callback
    def remove(self): self.removed = True


class Gui:
    def add_folder(self, *args): return Handle()
    def add_button(self, *args): return Handle()
    def add_markdown(self, content):
        handle = Handle()
        handle.content = content
        return handle


@pytest.fixture
def selection(tmp_path):
    cp = Checkpoint.create(tmp_path, '/source.ply', target_views=3, metadata={
        'selection': 'manual', 'renderer': {'backend': 'viser'}, 'width': 32, 'height': 16,
        'manual_scene': {'id': 'manual-room', 'generation': 42}})
    state = {'id': 'manual-room', 'generation': 42, 'status': 'ready'}
    captures = []
    def render(params, **kwargs):
        captures.append(params)
        out = io.BytesIO()
        Image.new('RGB', (params['width'], params['height']), (40, 80, 120)).save(out, format='PNG')
        return out.getvalue()
    session = ManualSelection(SimpleNamespace(gui=Gui()), SimpleNamespace(render=render), state)
    session.start({'checkpoint': str(cp.root)})
    client = SimpleNamespace(client_id=17, camera=SimpleNamespace(
        position=np.array([1., 2., 3.]), wxyz=np.array([1., 0., 0., 0.]), fov=np.pi / 3))
    return session, SimpleNamespace(client=client), captures


def test_manual_capture_calibration_and_finish(selection):
    session, event, captures = selection
    expected = camera_from_client(event.client, 32, 16)
    assert expected.vertical_fov_rad() == pytest.approx(np.pi / 3)
    session.save_current(event)
    saved = Checkpoint.load(session.checkpoint.root)
    np.testing.assert_allclose(camera_from_record(saved.views[0]).c2w, expected.c2w)
    assert captures[0]['client_id'] == 17
    assert captures[0]['fov'] == pytest.approx(np.pi / 3)
    saved.require_viser_images()
    assert not saved.complete
    assert np.array(Image.open(saved.image_path(saved.views[0])))[0, 0].tolist() == [40, 80, 120]
    session.finish(event)
    assert session.folder is None
    assert Checkpoint.load(saved.root).complete
    assert Checkpoint.load(saved.root).target_views == 1


def test_wrong_scene_duplicate_and_capture_failure_do_not_add_views(selection):
    session, event, captures = selection
    session.scene_state['generation'] = 43
    session.save_current(event)
    assert not captures and not session.checkpoint.views
    session.scene_state['generation'] = 42
    session.save_current(event)
    session.save_current(event)
    assert len(captures) == len(session.checkpoint.views) == 1
    event.client.camera.position[0] += 1
    def changed_scene(*args, **kwargs):
        session.scene_state['status'] = 'loading'
        return b'not an image'
    session.capture.render = changed_scene
    session.save_current(event)
    assert len(Checkpoint.load(session.checkpoint.root).views) == 1
    assert not session.save_button.disabled


def test_close_retains_partial_and_empty_finish_stays_open(selection):
    session, event, _ = selection
    session.finish(event)
    assert session.folder is not None
    with pytest.raises(ValueError, match='current Custom selection'):
        session.start({'checkpoint': str(session.checkpoint.root)})
    session.save_current(event)
    folder = session.folder
    session.close(event)
    assert folder.removed and session.folder is None
    assert len(Checkpoint.load(session.checkpoint.root).views) == 1


def test_target_limit(selection):
    session, event, captures = selection
    for index in range(3):
        event.client.camera.position[0] = index
        session.save_current(event)
    assert session.save_button.disabled
    assert session.checkpoint.complete


@pytest.mark.parametrize('size', [(832, 480), (1280, 720)])
def test_selected_size_used_for_capture_and_gpt_crop(selection, size):
    from splat_explorer.image_edit import ImageEditResult
    from splat_explorer.splatfix.image_repair import repair_images
    session, event, captures = selection
    session.width, session.height = size
    session.checkpoint.manifest['target_views'] = 1
    session.save_current(event)
    cp = Checkpoint.load(session.checkpoint.root)
    assert (captures[0]['width'], captures[0]['height']) == size
    assert (cp.views[0]['camera']['width'], cp.views[0]['camera']['height']) == size
    with Image.open(cp.image_path(cp.views[0])) as image:
        assert image.size == size
    response = io.BytesIO()
    Image.new('RGB', (1024, 1024), 'red').save(response, format='PNG')
    backend = SimpleNamespace(name='fake-gpt', edit=lambda *_: ImageEditResult(images=[response.getvalue()]))
    repair_images(cp, backend=backend)
    with Image.open(cp.image_path(cp.views[0], repaired=True)) as image:
        assert image.size == size
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((size[0]-1, size[1]-1)) == (255, 0, 0)
    assert cp.views[0]['image_repair']['resize']['method'] == 'cover_center_crop'
