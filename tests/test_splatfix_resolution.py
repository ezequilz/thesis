"""Resolution changes must preserve rays and source provenance, across stages."""
import struct

import numpy as np
import pytest
from PIL import Image

from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint, camera_from_record
from splat_explorer.splatfix.resolution import (
    calibrate, prepare_checkpoint, prepare_colmap, profile_size, resize_image, resize_plan,
)


def test_profiles_and_photographic_rays():
    assert profile_size() == (960, 544)
    assert profile_size('720p') == (1280, 720)
    for profile, expected in [('training', [816, 544]), ('720p', [1072, 720])]:
        plan = resize_plan(1237, 822, profile)
        assert plan['output_wh'] == expected
        fx, fy, cx, cy = 1162.376, 1162.376, 618.25, 410.75
        new = calibrate(fx, fy, cx, cy, plan)
        x, y = 901., 611.
        u, v = np.array([x, y])*plan['scale'] - plan['offset_xy']
        np.testing.assert_allclose([(u-new[2])/new[0], (v-new[3])/new[1]],
                                   [(x-cx)/fx, (y-cy)/fy], atol=1e-12)
        assert max(plan['offset_xy']) < 8
        assert resize_image(Image.new('RGB', (1237, 822)), plan).size == tuple(expected)
    with pytest.raises(ValueError):
        resize_plan(0, 720)
    with pytest.raises(ValueError):
        profile_size('1080p')


def test_existing_checkpoint_derivative_preserves_pose_and_intrinsics(tmp_path):
    cp = Checkpoint.create(tmp_path / 'old', tmp_path / 'scene.ply', target_views=1)
    camera = Camera(np.array([1, 2, 3]), np.eye(3), width=1237, height=822)
    cp.add_view(np.zeros((822, 1237, 3), dtype=np.uint8), camera)
    old_manifest = (cp.root / 'checkpoint.json').read_bytes()
    derivative = prepare_checkpoint(cp, tmp_path / 'new')
    changed = camera_from_record(derivative.views[0])
    plan = resize_plan(1237, 822)
    np.testing.assert_allclose(changed.c2w, camera.c2w)
    np.testing.assert_allclose(changed.intrinsics,
        [[camera.fx*plan['scale'], 0, 408], [0, camera.fy*plan['scale'], 272], [0,0,1]], rtol=1e-6)
    assert Image.open(derivative.image_path(derivative.views[0])).size == (816, 544)
    assert (cp.root / 'checkpoint.json').read_bytes() == old_manifest
    assert Image.open(cp.image_path(cp.views[0])).size == (1237, 822)
    assert prepare_checkpoint(derivative, tmp_path / 'unused') is derivative


def test_colmap_copy_calibrates_pixels_observations_and_keeps_world_points(tmp_path):
    source = tmp_path / 'source'; sparse = source / 'sparse/0'
    sparse.mkdir(parents=True); (source / 'images').mkdir()
    (sparse / 'cameras.bin').write_bytes(struct.pack('<QiiQQdddd', 1, 3, 1, 1237, 822, 1162, 1162, 618.25, 410.75))
    pose = struct.pack('<idddddddi', 9, 1, 0, 0, 0, 10, 20, 30, 3)
    (sparse / 'images.bin').write_bytes(struct.pack('<Q', 1)+pose+b'photo.jpg\0'+struct.pack('<Qddq', 1, 901, 611, 42))
    (sparse / 'points3D.bin').write_bytes(b'world points and tracks unchanged')
    Image.new('RGB', (1237, 822), 'red').save(source / 'images/photo.jpg')
    originals = {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    dest = tmp_path / 'prepared'; policy = prepare_colmap(source, dest)
    plan = policy['images']['photo.jpg']
    camera = struct.unpack('<QiiQQdddd', (dest / 'sparse/0/cameras.bin').read_bytes())
    assert camera[3:5] == (816, 544)
    np.testing.assert_allclose(camera[5:], calibrate(1162, 1162, 618.25, 410.75, plan))
    data = (dest / 'sparse/0/images.bin').read_bytes()
    assert data[8:72] == pose
    x, y, point = struct.unpack('<ddq', data[-24:])
    np.testing.assert_allclose([x,y], np.array([901,611])*plan['scale']-plan['offset_xy'])
    assert point == 42
    assert Image.open(dest / 'images/photo.jpg').size == (816, 544)
    for path, data in originals.items():
        assert (source / path).read_bytes() == data
    assert (dest / 'sparse/0/points3D.bin').read_bytes() == originals[next(p for p in originals if p.name=='points3D.bin')]


def test_early_original_preserves_source_bytes_and_uncapped_dimensions(tmp_path):
    assert profile_size('early_original') == (960, 720)
    plan = resize_plan(1237, 822, 'early_original')
    assert plan['output_wh'] == [1237, 822]
    source = tmp_path / 'source'; (source / 'images').mkdir(parents=True)
    Image.new('RGB', (1237, 822), 'red').save(source / 'images/photo.jpg')
    (source / 'sparse/0').mkdir(parents=True)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        (source / 'sparse/0' / name).write_bytes(b'original binary data')
    destination = tmp_path / 'copy'
    policy = prepare_colmap(source, destination, 'early_original')
    assert policy['profile'] == 'early_original' and policy['images'] == {}
    for path in source.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (destination / path.relative_to(source)).read_bytes()
    cp = Checkpoint.create(tmp_path / 'checkpoints', tmp_path / 'scene.ply', target_views=1)
    cp.add_view(np.zeros((822, 1237, 3), dtype=np.uint8),
                Camera(np.zeros(3), np.eye(3), width=1237, height=822))
    assert prepare_checkpoint(cp, tmp_path / 'unused', 'early_original') is cp
