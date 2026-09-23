"""Stable run bounds and source-balanced reconstruction scheduling."""
from types import SimpleNamespace
import numpy as np
import pytest
from splat_explorer.scene_runs_ext.fitting import initial_scale_ceiling, view_schedule, shape_metrics


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
