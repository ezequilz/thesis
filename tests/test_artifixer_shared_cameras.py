from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from splat_explorer.splatfix.artifixer.official_worker import shared_distillation_cameras


def test_shared_calibrations_preserve_all_observations_and_exact_intrinsics():
    written = {}
    def write(path, records):
        written[Path(path).name] = records
    official = SimpleNamespace(write_colmap_cameras=write, write_colmap_images=write)
    params = np.array([498., 498., 288., 288., 0., 0., 0., 0.])
    cameras = [SimpleNamespace(id=i + 1, model='OPENCV', width=576, height=576,
                               params=params.copy()) for i in range(2763)]
    # Never merge near-equal calibrations, different sizes, or distortion.
    cameras[1].params[0] = np.nextafter(params[0], np.inf)
    cameras[2].width = 832
    cameras[3].params[4] = .01
    images = [(i + 1, np.array([1., 0., 0., 0.]), np.array([i, 0., 0.]),
               camera.id, f'frame_{i:05d}.png') for i, camera in enumerate(cameras)]
    with shared_distillation_cameras(official):
        official.write_colmap_cameras(Path('/dataset/cameras.bin'), cameras)
        official.write_colmap_images(Path('/dataset/images.bin'), images)
    unique = {c.id: c for c in written['cameras.bin']}
    assert len(unique) == 4
    assert len(written['images.bin']) == len(images)
    for original, remapped, camera in zip(images, written['images.bin'], cameras):
        assert remapped[0] == original[0] and remapped[4] == original[4]
        assert remapped[1] is original[1] and remapped[2] is original[2]
        actual = unique[remapped[3]]
        assert (actual.model, actual.width, actual.height) == (camera.model, camera.width, camera.height)
        np.testing.assert_array_equal(actual.params, camera.params)
    assert official.write_colmap_cameras is write and official.write_colmap_images is write


def test_writer_hooks_restore_after_failure():
    def fail(*args):
        raise RuntimeError('disk full')
    official = SimpleNamespace(write_colmap_cameras=fail, write_colmap_images=fail)
    with pytest.raises(RuntimeError, match='disk full'):
        with shared_distillation_cameras(official):
            official.write_colmap_cameras(Path('/dataset/cameras.bin'), [])
    assert official.write_colmap_cameras is fail and official.write_colmap_images is fail
