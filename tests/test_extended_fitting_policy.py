"""Stable run bounds and source-balanced reconstruction scheduling."""
from types import SimpleNamespace
import numpy as np
import pytest
from splat_explorer.scene_runs_ext.fitting import initial_scale_ceiling, view_schedule, shape_metrics
from splat_explorer.scene_runs_ext.fitting import scale_trust_bounds, constrain_log_scales_


def test_safeguards_default_on_and_explicit_off_survives_validation():
    from splat_explorer.scene_runs_ext.config import validate_options
    assert validate_options()['fitting_safeguards'] is True
    assert validate_options({'fitting_safeguards': False})['fitting_safeguards'] is False
    for invalid in ('false', 0, 1, None):
        with pytest.raises(ValueError, match='boolean'):
            validate_options({'fitting_safeguards': invalid})


@pytest.mark.parametrize('count', [9, 25, 121, 361])
def test_generated_frame_count_cannot_dilute_the_starter(count):
    visits = np.bincount(list(view_schedule(count, 1000, [0])), minlength=count)
    assert visits[0] == 500
    assert visits[1:].sum() == 500
    assert visits[1:].max() - visits[1:].min() <= 1


def test_plain_fitting_remains_uniform_and_bad_ids_fail():
    assert np.bincount(list(view_schedule(4, 12))).tolist() == [3,3,3,3]
    with pytest.raises(ValueError): list(view_schedule(4, 12, [4]))


def test_scale_ceiling_comes_from_initial_asset_even_after_worker_restart(tmp_path, monkeypatch):
    from splat_explorer.scene_runs.gpu_worker import SceneRunGpuWorker, SCENE_NAME, CHECKPOINT_NAME
    from splat_explorer.scene_runs_ext import pipeline
    (tmp_path/SCENE_NAME).touch()
    (tmp_path/CHECKPOINT_NAME).touch()
    original = SimpleNamespace(scales=np.array([[.1, .2, 2.]]))
    repaired = SimpleNamespace(scales=np.array([[.1, .2, 7.]]))
    loaded = []
    def load(path):
        loaded.append(path.name)
        return original if path.name == SCENE_NAME else repaired
    monkeypatch.setattr(pipeline, 'validate_runtime', lambda _: None)
    worker = SceneRunGpuWorker(tmp_path, scene_loader=load)
    worker.config = {'scene_run':{'pipeline':'extended'}}
    worker._load_scene_once()
    assert worker.scene is repaired
    assert worker.extended_scale_ceiling == 4.
    assert loaded == [CHECKPOINT_NAME, SCENE_NAME]
    repaired.scales[:] *= 3
    worker._load_scene_once()
    assert worker.extended_scale_ceiling == 4.
    restarted = SceneRunGpuWorker(tmp_path, scene_loader=load)
    restarted.config = worker.config
    restarted._load_scene_once()
    assert restarted.extended_scale_ceiling == 4.


def test_shape_diagnostics_detect_thinning_even_below_the_global_ceiling():
    original = np.array([[.01, .06, .01], [1., 1., 27.]])
    stretched = original.copy(); stretched[0] = [.004, 4.3, .002]
    assert stretched.max() < initial_scale_ceiling(SimpleNamespace(scales=original))
    assert shape_metrics(original)['axis_ratio_above_100'] == 0
    assert shape_metrics(stretched)['axis_ratio_above_100'] == 1


def test_scale_projection_blocks_needles_and_preserves_source_surfaces():
    torch = pytest.importorskip('torch')
    source = np.array([[.01, .06, .01], [.001, .08, .02], [1., 1., 27.]])
    bounds = tuple(torch.tensor(x) for x in scale_trust_bounds(source, 54.))
    original = torch.tensor(source).log()
    unchanged = original.clone()
    constrain_log_scales_(unchanged, *bounds)
    np.testing.assert_allclose(unchanged.exp(), source, rtol=1e-6)
    damaged = torch.tensor([[.004, 4.3, .00001], [.000001, 2., .01], [1., 1., 100.]]).log()
    constrain_log_scales_(damaged, *bounds)
    result = damaged.exp().numpy()
    assert np.all(result >= source * .5 * (1-1e-6))
    assert np.all(result <= source * 2 * (1+1e-6))
    ratios = result.max(1) / result.min(1)
    assert np.all(ratios <= np.maximum(source.max(1)/source.min(1), 20) * (1+1e-6))
    # A second projection must not progressively change shapes.
    once = damaged.clone()
    constrain_log_scales_(damaged, *bounds)
    torch.testing.assert_close(damaged, once)


@pytest.mark.parametrize('scales', [[], [[0., 1., 1.]], [[np.nan, 1., 1.]]])
def test_scale_bounds_reject_invalid_source(scales):
    with pytest.raises(ValueError):
        scale_trust_bounds(scales, 54.)


@pytest.mark.parametrize('edited', [[], [0], [0, 1], [0, 1, 2, 3]])
def test_per_view_cap_exhausts_views_without_exceeding_total(edited):
    schedule = list(view_schedule(4, 100, edited, per_view_limit=3))
    assert np.bincount(schedule, minlength=4).tolist() == [3, 3, 3, 3]
    limited = list(view_schedule(4, 8, edited, per_view_limit=3))
    assert len(limited) == 8
    assert np.bincount(limited, minlength=4).max() <= 3
    assert limited == list(view_schedule(4, 8, edited, per_view_limit=3))


def test_per_view_limit_defaults_and_validation():
    from splat_explorer.scene_runs_ext.config import validate_options
    assert validate_options()['fit_iterations_per_view'] == 1000
    assert validate_options({'fit_iterations_per_view': 2500})['fit_iterations_per_view'] == 2500
    for invalid in (-1, True, 1.5, '1000', None):
        with pytest.raises(ValueError, match='fit_iterations_per_view'):
            validate_options({'fit_iterations_per_view': invalid})
    assert len(list(view_schedule(1, 50000))) == 1000


def test_unlimited_image_steps_preserves_total_and_source_balance():
    from splat_explorer.scene_runs_ext.config import validate_options
    assert validate_options({'fit_iterations_per_view': 0})['fit_iterations_per_view'] == 0
    visits = np.bincount(list(view_schedule(3, 6000, [0], per_view_limit=0)))
    assert visits.tolist() == [3000, 1500, 1500]
    assert len(list(view_schedule(1, 2500, per_view_limit=0))) == 2500
