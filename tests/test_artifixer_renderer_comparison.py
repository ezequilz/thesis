import copy
import numpy as np
import pytest

from splat_explorer.splatfix.artifixer.renderer_comparison import (
    arm_trajectory, calibrated_frames, pixel_metrics, opacity_png_values, compare_outputs,
)


def trajectory():
    return {'camera_convention': 'opencv_c2w',
            'transforms': {'w': 32, 'h': 16, 'fl_x': 30., 'fl_y': 29., 'cx': 14., 'cy': 7.,
                           'frames': [{'transform_matrix': np.eye(4).tolist()}]},
            'frames': [{'rgb': 'original.png', 'opacity': 'original.npy'}],
            'anchors': [{'frame_index': 0, 'original_rgb': 'reference.png'}],
            'segments': [{'indices': [0], 'seed_index': 0}]}


@pytest.mark.parametrize('arm', ['current', 'gsplat', '3dgut', 'opacity_only'])
def test_ablation_only_changes_target_inputs(tmp_path, arm):
    original = trajectory()
    saved = copy.deepcopy(original)
    result = arm_trajectory(original, tmp_path, arm)
    assert original == saved
    for key in ('transforms', 'anchors', 'segments', 'camera_convention'):
        assert result[key] == original[key]
    if arm == 'current':
        assert result['frames'] == original['frames']
    elif arm == 'opacity_only':
        assert result['frames'][0]['rgb'] == 'original.png'
        assert result['frames'][0]['opacity'] == str(tmp_path / '3dgut/00000.npy')
    else:
        assert result['frames'][0] == {'rgb': str(tmp_path / arm / '00000.png'),
                                       'opacity': str(tmp_path / arm / '00000.npy')}


def test_calibration_preserves_asymmetric_intrinsics_and_converts_pose():
    data = trajectory()
    data['transforms']['frames'][0]['fl_y'] = 24.
    intr, pose = next(calibrated_frames(data))
    assert (intr['fl_x'], intr['fl_y'], intr['cx'], intr['cy']) == (30., 24., 14., 7.)
    np.testing.assert_array_equal(pose, np.diag([1., -1., -1., 1.]))
    data['transforms']['frames'][0]['k1'] = .1
    with pytest.raises(ValueError, match='undistorted'):
        list(calibrated_frames(data))


def test_metrics_reject_broadcasting_and_nonfinite_inputs():
    a = np.zeros((2, 2))
    assert pixel_metrics(a, a)['psnr_db'] is None
    assert pixel_metrics(a, a + .5)['mae'] == .5
    with pytest.raises(ValueError, match='identically sized'):
        pixel_metrics(a, np.zeros((1, 2)))
    with pytest.raises(ValueError, match='finite'):
        pixel_metrics(a, a + np.nan)


def test_authors_opacity_export_keeps_soft_coverage():
    encoded = opacity_png_values([0., .25, .5, 1.])
    np.testing.assert_array_equal(encoded, [0, 64, 128, 255])
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        opacity_png_values([1.01])


def test_output_comparison_requires_matching_frames(tmp_path):
    from PIL import Image
    relative = 'inference/splatfix/frames/batch_0000/pred/00001.png'
    for arm in ('current', '3dgut'):
        path = tmp_path / f'inference-{arm}' / relative
        path.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
    compare_outputs(tmp_path, ['current', '3dgut'])
    assert (tmp_path / 'output-differences.json').is_file()
    (tmp_path / 'inference-3dgut' / relative).unlink()
    with pytest.raises(ValueError, match='frame sets differ'):
        compare_outputs(tmp_path, ['current', '3dgut'])
