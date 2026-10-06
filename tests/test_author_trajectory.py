import json

import numpy as np
import pytest

from splat_explorer.splatfix import author_trajectory


K = dict(w=100, h=80, fl_x=70, fl_y=70, cx=50, cy=40)


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


class RecordingRenderer:
    def interpolate_orbit_poses(self, poses, **kwargs):
        self.poses, self.kwargs = poses, kwargs
        training = kwargs['training_poses']
        # Known output sequence including an anchor, generated view, supplied
        # nodes and a repeated pose exercises filtering independently of math.
        start = training[0] if training is not None else poses[0]
        middle = start.copy()
        middle[0, 3] = .25
        return np.array([start, middle, *poses, middle]), np.array([-1, -1, *range(len(poses)), -1])


def test_delegates_cv_scale_and_keeps_exact_targets_without_anchor_duplicates():
    renderer = RecordingRenderer()
    result = author_trajectory.build_author_orbit(
        [pose(0), pose(1), pose(2)], K, 2, target_poses=[pose(3), pose(3), pose(1)],
        loop=True, renderer=renderer)
    assert renderer.kwargs['interp_distance'] == .05
    assert renderer.kwargs['loop'] is True
    assert np.array_equal(renderer.kwargs['training_poses'][0], np.diag([1, -1, -1, 1]))
    assert len(renderer.poses) == 3  # unique nodes besides the first reference
    frames = result['trajectory']['frames']
    assert [f['transform_matrix'][0][3] for f in frames] == [.25, 3]
    assert np.array_equal(np.asarray(frames[0]['transform_matrix'])[:3, :3], np.eye(3))
    assert result['provenance']['frames'][1]['requested_target_indices'] == [0, 1]
    assert result['provenance']['reference_frame_indices'] == [0, 2, 3]
    assert result['provenance']['frames'][1]['full_frame_index'] == 4
    assert len(result['full_trajectory']['frames']) == 5
    assert result['provenance']['motion_diagnostics']['generated_targets']['frame_count'] == 2
    assert {f['reason'] for f in result['provenance']['removed_frames']} == {'reference_camera', 'duplicate_target'}


def test_nearby_cameras_are_not_collapsed():
    result = author_trajectory.build_author_orbit(
        [pose(0), pose(1)], K, 1, target_poses=[pose(1 + 1e-10)], renderer=RecordingRenderer())
    assert result['provenance']['unique_input_nodes'] == 3
    assert result['provenance']['frames'][-1]['requested_target_indices'] == [0]


def test_two_nodes_use_original_sort_without_fake_duplicate_training_node():
    renderer = RecordingRenderer()
    result = author_trajectory.build_author_orbit([pose(0), pose(1)], K, 1, renderer=renderer)
    assert renderer.kwargs['training_poses'] is None
    assert not result['provenance']['starts_at_first_reference']


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_rejects_invalid_metric_scale(scale):
    with pytest.raises(ValueError, match='metric_scale'):
        author_trajectory.build_author_orbit([pose(0), pose(1)], K, scale)


def test_requires_distinct_cameras():
    with pytest.raises(ValueError, match='distinct'):
        author_trajectory.build_author_orbit([pose(0), pose(0)], K, 1)


def test_cli_maps_names_and_serializes_both_trajectories(tmp_path, monkeypatch):
    original = author_trajectory.build_author_orbit
    monkeypatch.setattr(author_trajectory, 'build_author_orbit',
                        lambda *args, **kwargs: original(*args, **kwargs, renderer=RecordingRenderer()))
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({**K, 'frames': [
        {'file_path': f'image-{i}.jpg', 'transform_matrix': pose(i).tolist()} for i in range(4)]}))
    selected, names = tmp_path / 'selected.json', tmp_path / 'names.json'
    selected.write_text('[0, 1, 2]')
    names.write_text('["image-3.jpg"]')
    target, provenance, full = [tmp_path / name for name in ('target.json', 'provenance.json', 'full.json')]
    author_trajectory.main(['--repo', str(tmp_path), '--transforms', str(source),
                            '--selected-indices', str(selected), '--target-names', str(names),
                            '--metric-scale', '1', '--output', str(target), '--provenance', str(provenance),
                            '--full-output', str(full)])
    assert json.loads(provenance.read_text())['target_name_to_index'] == {'image-3.jpg': 1}
    assert all('file_path' not in frame for frame in json.loads(target.read_text())['frames'])
    assert json.loads(target.read_text())['camera_model'] == 'OPENCV'
    assert json.loads(full.read_text())['camera_model'] == 'OPENCV'
    assert len(json.loads(full.read_text())['frames']) == 5


def test_target_orbit_keeps_all_references_in_original_training_role():
    class Renderer:
        def interpolate_orbit_poses(self, poses, **kwargs):
            self.poses, self.kwargs = poses, kwargs
            refs = kwargs['training_poses']
            return np.concatenate([refs, poses]), np.array([-1] * len(refs) + list(range(len(poses))))
    renderer = Renderer()
    result = author_trajectory.build_author_orbit(
        [pose(0), pose(1), pose(2)], K, 1, target_poses=[pose(3), pose(4)], renderer=renderer)
    assert len(renderer.kwargs['training_poses']) == 3
    assert len(renderer.poses) == 2
    assert result['provenance']['node_roles'] == 'all_references_as_training'
    assert [m['requested_target_indices'] for m in result['provenance']['frames']] == [[0], [1]]


def test_actual_authors_target_only_trajectory_contract(tmp_path):
    """Optional integration check against an installed, unmodified checkout."""
    import importlib.util
    import os
    from pathlib import Path
    upstream = Path(os.environ.get('ARTIFIXER_REPO', '/private/tmp/splatfix-upstream'))
    source = upstream / 'data_processing/camera_trajectories.py'
    if not source.is_file():
        pytest.skip('Set ARTIFIXER_REPO to run the original authors trajectory contract')
    spec = importlib.util.spec_from_file_location('original_camera_trajectories', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = author_trajectory.build_author_orbit(
        [pose(0), pose(1), pose(2)], K, 1, renderer=RecordingRenderer())
    path = tmp_path / 'trajectory.json'
    path.write_text(json.dumps(result['trajectory']))
    loaded = module.read_camera_trajectory(path)
    module.assert_target_only_trajectory(loaded, str(path))
    assert len(loaded['frames']) == len(result['trajectory']['frames'])
    assert np.array_equal(loaded['frames'][0]['transform_matrix'],
                          result['trajectory']['frames'][0]['transform_matrix'])
    assert loaded['camera_model'] == 'OPENCV'


@pytest.fixture
def original_orbit_renderer():
    """Execute the original CPU helper methods without optional CUDA imports."""
    import ast
    import os
    from pathlib import Path
    from scipy.spatial.transform import Rotation, Slerp
    upstream = Path(os.environ.get('ARTIFIXER_REPO', '/private/tmp/splatfix-upstream'))
    source = upstream / 'thirdparty/3DGRUT-ArtiFixer/threedgrut/render.py'
    if not source.is_file():
        pytest.skip('Set ARTIFIXER_REPO to test unchanged original orbit helpers')
    module = ast.parse(source.read_text())
    renderer = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'Renderer')
    names = {'estimate_center_of_interest', 'compute_pose_distance', 'sort_poses_by_orbit_angle',
             'interpolate_single_pose', 'interpolate_orbit_poses'}
    methods = [node for node in renderer.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    extracted = ast.Module(body=[ast.ClassDef(name='Renderer', bases=[], keywords=[],
                                            body=methods, decorator_list=[])], type_ignores=[])
    namespace = {'np': np, 'Rotation': Rotation, 'Slerp': Slerp}
    exec(compile(ast.fix_missing_locations(extracted), str(source), 'exec'), namespace)
    return namespace['Renderer'].__new__(namespace['Renderer'])


def test_nearby_saved_anchors_refine_original_helper_spacing(original_orbit_renderer):
    anchors = [pose(0), pose(.05)]
    result = author_trajectory.build_author_orbit(anchors, K, 1, renderer=original_orbit_renderer)
    provenance = result['provenance']
    assert provenance['requested_spacing'] == .1
    assert provenance['effective_spacing'] == pytest.approx(.025)
    assert provenance['attempted_normalized_distances'] == [.1, .025]
    assert provenance['spacing_fallback_reason'] == 'requested spacing produced only reference cameras'
    assert len(result['trajectory']['frames']) == 1
    target = np.asarray(result['trajectory']['frames'][0]['transform_matrix'])
    assert np.array_equal(target, pose(.025))
    for source, frame_index in zip(anchors, provenance['reference_frame_indices']):
        assert np.array_equal(result['full_trajectory']['frames'][frame_index]['transform_matrix'], source)
    assert len(result['full_trajectory']['frames']) == 3


def test_usable_original_spacing_is_unchanged(original_orbit_renderer):
    result = author_trajectory.build_author_orbit([pose(0), pose(.5)], K, 1, renderer=original_orbit_renderer)
    provenance = result['provenance']
    assert provenance['attempted_normalized_distances'] == [.1]
    assert provenance['effective_spacing'] == .1
    assert provenance['spacing_fallback_reason'] is None
    assert len(result['trajectory']['frames']) == 4


def test_refinement_is_bounded_for_helper_that_returns_only_anchors():
    class AnchorsOnly:
        attempts = 0
        def interpolate_orbit_poses(self, poses, **kwargs):
            self.attempts += 1
            return poses, np.arange(len(poses))
        def compute_pose_distance(self, first, second):
            return .05, 0
    renderer = AnchorsOnly()
    with pytest.raises(ValueError, match='bounded spacing refinement'):
        author_trajectory.build_author_orbit([pose(0), pose(.05)], K, 1, renderer=renderer)
    assert renderer.attempts == 3
