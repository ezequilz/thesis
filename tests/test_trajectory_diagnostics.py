import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from splat_explorer.splatfix.trajectory_diagnostics import summarize_trajectory


def path(points):
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    poses[:, :3, 3] = points
    return poses


def test_zigzag_reversals_ignore_plateaus_and_detect_known_travel():
    poses = path([[0, 0, 0], [1, 0, 1], [2, 0, 1], [3, 0, 0], [4, 0, 2]])
    metrics = summarize_trajectory(poses, up=[0, 0, 1])
    assert metrics['height']['range'] == 2
    assert metrics['height']['total_travel'] == 4
    assert metrics['height']['reversal_count'] == 2
    assert metrics['rotation_total_degrees'] == 0
    assert metrics['translation_total'] == pytest.approx(2 * np.sqrt(2) + 1 + np.sqrt(5))


def test_metrics_respect_global_coordinate_rotation_translation_and_scale():
    poses = path([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
    poses[1, :3, :3] = Rotation.from_euler('y', 15, degrees=True).as_matrix()
    base = summarize_trajectory(poses, up=[0, 1, 0])
    transform = Rotation.from_euler('xyz', [21, 43, 12], degrees=True).as_matrix()
    changed = poses.copy()
    changed[:, :3, :3] = transform @ poses[:, :3, :3]
    changed[:, :3, 3] = 3 * (poses[:, :3, 3] @ transform.T) + [4, 2, 7]
    metrics = summarize_trajectory(changed, up=transform @ [0, 1, 0])
    assert metrics['height']['reversal_count'] == base['height']['reversal_count']
    assert metrics['height']['total_travel'] == pytest.approx(3 * base['height']['total_travel'])
    assert metrics['translation_total'] == pytest.approx(3 * base['translation_total'])
    assert metrics['rotation_total_degrees'] == pytest.approx(30)


def test_single_pose_and_ambiguous_camera_up():
    assert summarize_trajectory(path([[0, 0, 0]]))['translation_total'] == 0
    poses = path([[0, 0, 0], [1, 0, 0]])
    poses[1, :3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    assert summarize_trajectory(poses)['height'] is None
    with pytest.raises(ValueError, match='nonzero'):
        summarize_trajectory(poses, up=[0, 0, 0])


def test_rejects_nonrigid_pose():
    poses = path([[0, 0, 0]])
    poses[0, 0, 0] = 2
    with pytest.raises(ValueError, match='rigid'):
        summarize_trajectory(poses)
