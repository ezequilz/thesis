import numpy as np
import pytest
from splat_explorer.scene_runs_ext.diagnostics import trajectory_diagnostics, image_diagnostics
from splat_explorer.scene_runs_ext.config import validate_options


def test_known_camera_rotation_translation_and_constant_video():
    first, second = np.eye(4), np.eye(4)
    second[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    second[0, 3] = 2
    result = trajectory_diagnostics({'frames': [{'transform_matrix': p} for p in [first, second]]})
    assert result['max_anchor_rotation_degrees'] == pytest.approx(90)
    assert result['max_step_translation_scene_units'] == 2
    assert result['rotation_orthogonality_max_error'] == 0
    stats = image_diagnostics([np.full((4,4,3), 128, dtype=np.uint8)] * 2)
    assert stats['frames'][1]['anchor_rgb_mae'] == 0
    assert stats['frames'][1]['luminance_std'] == pytest.approx(0, abs=1e-6)


def test_ablation_options_are_explicit_and_validated():
    assert validate_options()['source_conditioning'] == 'rendered'
    assert validate_options({'source_conditioning': 'rendered'})['source_conditioning'] == 'rendered'
    assert validate_options({'generated_cache': 'last_denoising'})['generated_cache'] == 'last_denoising'
    for option in ['source_conditioning', 'generated_cache']:
        with pytest.raises(ValueError, match=option):
            validate_options({option: 'invalid'})
