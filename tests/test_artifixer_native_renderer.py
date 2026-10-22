import numpy as np
import pytest

from splat_explorer.splatfix.artifixer.native_renderer import expected_camera_z
from splat_explorer.splatfix.artifixer.pipeline_profile import apply_pipeline, rgb_backend, geometry_backend
from splat_explorer.splatfix.jobs import validate_job


def test_default_job_uses_native_renderer_and_fresh_author_recipe():
    opts = validate_job({'stage': 'repair', 'checkpoint': '/saved/cp',
                         'reconstruction_initialization': 'full_splat',
                         'reconstruction_images': 'all', 'regularization_profile': 'base_mcmc'})
    assert opts['artifixer_pipeline'] == 'authors_3dgut'
    assert rgb_backend(opts) == '3dgut'
    assert opts['reconstruction_initialization'] == 'point_cloud'
    assert opts['reconstruction_images'] == 'distilled'
    assert opts['regularization_profile'] == 'artifixer'


def test_legacy_preserves_custom_reconstruction_controls():
    opts = validate_job({'stage': 'repair', 'checkpoint': '/saved/cp',
                         'artifixer_pipeline': 'legacy_gsplat',
                         'reconstruction_initialization': 'full_splat',
                         'reconstruction_images': 'all', 'regularization_profile': 'base_mcmc'})
    assert rgb_backend(opts) == 'viser'
    assert opts['reconstruction_initialization'] == 'full_splat'
    assert opts['reconstruction_images'] == 'all'
    assert opts['regularization_profile'] == 'base_mcmc'
    assert rgb_backend({}) == 'viser'  # Historical worker requests.
    with pytest.raises(ValueError, match='artifixer_pipeline'):
        apply_pipeline({'artifixer_pipeline': 'unknown'})


def test_native_depth_normalizes_accumulation_before_camera_z_conversion():
    alpha = np.array([[.8, 1., 0.]])
    weighted_distance = np.array([[4., 10., 0.]])
    ray_z = np.array([[.6, .8, 1.]])
    depth = expected_camera_z(weighted_distance, alpha, ray_z)
    np.testing.assert_allclose(depth[0, :2], [3., 8.])
    assert np.isinf(depth[0, 2])
    with pytest.raises(ValueError, match='matching shapes'):
        expected_camera_z(weighted_distance, alpha, np.ones((1, 1)))


def test_native_capture_cache_cannot_reuse_viser(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from splat_explorer.rendering.base import Camera
    from splat_explorer.splatfix.artifixer.staging_cache import capture_inputs
    cp = SimpleNamespace(root=tmp_path, manifest={'scene_path': '/same/source.ply', 'metadata': {}})
    camera = Camera(np.zeros(3), np.eye(3), width=16, height=16)
    destination = tmp_path / 'target.png'
    calls = []
    def capture():
        calls.append(1)
        Image.new('RGB', (16, 16)).save(destination)
    legacy = capture_inputs(cp, [camera], [destination], capture, profile='720p')
    native = capture_inputs(cp, [camera], [destination], capture, profile='720p', renderer='3dgut')
    assert legacy != native
    assert len(calls) == 2
    assert capture_inputs(cp, [camera], [destination], capture, profile='720p', renderer='3dgut') == native
    assert len(calls) == 2


def test_hybrid_uses_viser_rgb_native_geometry_and_author_fitting():
    opts = validate_job({'stage': 'repair', 'checkpoint': '/saved/cp',
                         'artifixer_pipeline': 'authors_3dgut_viser_rgb',
                         'reconstruction_initialization': 'full_splat',
                         'reconstruction_images': 'all', 'regularization_profile': 'base_mcmc'})
    assert rgb_backend(opts) == 'viser'
    assert geometry_backend(opts) == '3dgut'
    assert geometry_backend({}) == 'gsplat'
    assert opts['reconstruction_initialization'] == 'point_cloud'
    assert opts['reconstruction_images'] == 'distilled'
    assert opts['regularization_profile'] == 'artifixer'
