import copy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.config import Config
from splat_explorer.rendering.base import Camera
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.manual_crop import crop_source, save_crop
from splat_explorer.splatfix.resolution import prepare_repair_checkpoint
from splat_explorer.web.splatfix_studio import SplatfixStudio


@pytest.fixture
def crop_checkpoint(tmp_path):
    source = tmp_path / 'scene.ply'; source.write_text('scene')
    cp = Checkpoint.create(tmp_path / 'splatfix', source, target_views=1,
                           metadata={'renderer': {'backend': 'viser'}})
    cp.add_view(np.zeros((720, 1280, 3), np.uint8),
                Camera(np.array([1., 2., 3.]), np.eye(3), width=1280, height=720))
    v = cp.views[0]
    v['repaired_rgb'] = 'views/000/repaired.png'
    raw = Image.new('RGB', (1600, 900), 'red')
    raw.paste('green', (160, 90, 1440, 810))
    raw.save(cp.root / 'views/000/response.image', format='PNG')
    raw.resize((1280, 720)).save(cp.root / v['repaired_rgb'])
    v['image_repair'] = {'response_image': 'views/000/response.image'}
    cp.save()
    return cp


@pytest.mark.parametrize('profile,size', [('720p', (1280, 720)), ('training', (832, 480))])
def test_manual_crop_preserved_and_used_by_step4(crop_checkpoint, tmp_path, monkeypatch, profile, size):
    cp = crop_checkpoint; v = cp.views[0]
    source, token = crop_source(cp, v)
    before = source.read_bytes()
    old_repair = cp.image_path(v, True).read_bytes()
    camera = copy.deepcopy(v['camera'])
    save_crop(cp, v, [160, 90, 1280, 720], token)
    assert source.read_bytes() == before
    assert (cp.root / v['full_repaired_rgb']).read_bytes() == old_repair
    assert v['camera'] == camera
    with Image.open(cp.root / v['manual_resized_rgb']) as crop:
        assert crop.size == (1280, 720)
        assert crop.getpixel((0, 0)) == (0, 128, 0)
    def capture(scene, cameras, destinations, **kwargs):
        for cam, path in zip(cameras, destinations):
            Image.new('RGB', (cam.width, cam.height)).save(path)
    monkeypatch.setattr('splat_explorer.splatfix.viser_capture.capture_rgb', capture)
    prepared = prepare_repair_checkpoint(cp, tmp_path / 'prepared', profile)
    with Image.open(prepared.image_path(prepared.views[0], True)) as repaired:
        assert repaired.size == size
        assert repaired.getpixel((0, 0)) == (0, 128, 0)
    assert prepared.views[0]['image_repair']['reconstruction_resize']['source_kind'] == 'manual_crop'
    # Recropping uses the full original, never the previous green crop.
    save_crop(cp, v, [0, 0, 1600, 900], token)
    with Image.open(cp.root / v['manual_resized_rgb']) as crop:
        assert crop.size == (1600, 900)
        assert crop.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize('box', [[0, 0, 1264, 711], [0, 0, 1280, 721], [-1, 0, 1280, 720],
                                  [400, 0, 1280, 720], [0., 0, 1280, 720]])
def test_invalid_crop_does_not_change_checkpoint(crop_checkpoint, box):
    cp = crop_checkpoint
    before = (cp.root / 'checkpoint.json').read_bytes()
    with pytest.raises(ValueError):
        save_crop(cp, cp.views[0], box, crop_source(cp, cp.views[0])[1])
    assert (cp.root / 'checkpoint.json').read_bytes() == before


def test_crop_api_source_and_stale_guard(crop_checkpoint, tmp_path):
    cp = crop_checkpoint
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})), SceneRunStore(tmp_path / 'scene-runs'))
    body = {'checkpoint': str(cp.root), 'view_id': '000', 'image_set': 'saved'}
    info = studio.crop_checkpoint(body)
    assert (info['width'], info['height']) == (1600, 900)
    assert '.png' in info['url']
    with pytest.raises(ValueError, match='changed'):
        studio.crop_checkpoint({**body, 'box': [0, 0, 1280, 720], 'source_token': 'stale'})
    result = studio.crop_checkpoint({**body, 'box': [0, 0, 1280, 720], 'source_token': info['source_token']})
    assert result['checkpoint']['views'][0]['manual_resized_rgb'].endswith('manual_resized.png')
    with pytest.raises(ValueError, match='existing checkpoint'):
        studio.crop_checkpoint({**body, 'checkpoint': str(tmp_path.parent)})
