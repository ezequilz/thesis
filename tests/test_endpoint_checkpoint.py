"""Endpoint-only checkpoints retain all cameras but only two trusted images."""
import numpy as np
import pytest
from test_splatfix_checkpoint import make_checkpoint, Backend
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.image_repair import repair_images
from splat_explorer.splatfix.artifixer.author_trajectory import build_endpoint_path
from splat_explorer.splatfix.artifixer.official_worker import inference_jobs, unique_supervision
from splat_explorer.splatfix.jobs import validate_job


@pytest.mark.parametrize('count', [2, 3, 6, 25])
def test_only_endpoints_are_edited_and_retry_is_free(tmp_path, count):
    cp = make_checkpoint(tmp_path, count)
    cp.manifest['metadata']['repair_policy'] = 'endpoints'
    cp.save()
    originals = [cp.image_path(v).read_bytes() for v in cp.views]
    backend = Backend()
    repair_images(cp, backend=backend)
    cp = Checkpoint.load(cp.root)
    assert [v['id'] for v in cp.views if v.get('repaired_rgb')] == ['000', f'{count-1:03d}']
    assert [cp.image_path(v).read_bytes() for v in cp.views] == originals
    repair_images(cp, backend=backend)
    assert len(backend.calls) == 2
    assert len(cp.views) == count


class PairRenderer:
    def interpolate_orbit_poses(self, poses, **kwargs):
        # Deliberately sort pairs opposite to saved order.
        a, b = poses[::-1]
        if kwargs['interp_distance'] > 1e100:
            return np.array([a, b]), np.array([0, 1])
        middle = a.copy()
        middle[:3, 3] = (a[:3, 3] + b[:3, 3]) / 2
        return np.array([a, middle, b]), np.array([0, -1, 1])


def test_saved_order_unsplit_and_bidirectional_reference_mapping():
    poses = []
    for x in [0, 3, 1, 5, 2, 8]:
        p = np.eye(4); p[0, 3] = x; poses.append(p)
    intrinsics = dict(w=32, h=16, fl_x=20, fl_y=20, cx=16, cy=8)
    result = build_endpoint_path(poses, intrinsics, 1, renderer=PairRenderer())
    full = result['full_trajectory']['frames']
    assert [f['transform_matrix'][0][3] for f in full][::2] == [0, 3, 1, 5, 2, 8]
    assert result['provenance']['segments'][0]['full_frame_indices'] == list(range(1, 10))
    trajectory = {'frames': full, 'anchors': [{'frame_index': 0}, {'frame_index': 10}]}
    request = {'repair_policy': 'endpoints', 'runtime': {}}
    forward, backward = inference_jobs(request, trajectory)
    assert forward[0]['indices'] == backward[0]['indices'][::-1]
    assert forward[1:] == ([0, 1], 'inference')
    assert backward[1:] == ([1, 0], 'inference-back-to-front')
    request['runtime']['endpoint_reference_mode'] = 'initial'
    jobs = inference_jobs(request, trajectory)
    assert [job[1] for job in jobs] == [[0], [1]]
    assert len(inference_jobs(request, trajectory, plus=True)) == 1


def test_endpoint_validation():
    with pytest.raises(ValueError, match='at least two'):
        validate_job({'stage': 'select', 'views': 1, 'repair_policy': 'endpoints'})
    result = validate_job({'stage': 'select', 'views': 6, 'repair_policy': 'endpoints', 'horizontal_fov': 77})
    assert result['horizontal_fov'] == 77
    result = validate_job({'stage': 'repair', 'checkpoint': '/saved', 'split_mode': 'no-split', 'endpoint_reference_mode': 'initial'})
    assert result['endpoint_reference_mode'] == 'initial'


@pytest.mark.parametrize('prior_checkpoint', [False, True])
def test_saved_path_uses_only_endpoint_supervision(tmp_path, monkeypatch, prior_checkpoint):
    from dataclasses import replace
    from test_splatfix_repair import make_checkpoint as make_scene, mock_scene, FakeRenderer
    from splat_explorer.splatfix.checkpoint import camera_from_record
    from splat_explorer.splatfix.artifixer.repair import prepare_saved_path
    cp = make_scene(tmp_path)
    cp.manifest['target_views'] = 6
    cp.manifest['metadata']['repair_policy'] = 'endpoints'
    camera = camera_from_record(cp.views[0])
    for x in range(1, 6):
        cp.add_view(np.full((32, 32, 3), 90, np.uint8), replace(camera, position=camera.position + [x, 0, 0]))
    if prior_checkpoint:
        from splat_explorer.splatfix.artifixer.checkpoint_policy import checkpoint_for_run
        cp.manifest['metadata']['repair_policy'] = 'all'
        repair_images(cp, backend=Backend(size=(32, 32)))
        cp = checkpoint_for_run(cp, 'endpoints')
    mock_scene(monkeypatch)
    _, scale = prepare_saved_path(cp, renderer_factory=FakeRenderer)
    assert [a['frame_index'] for a in scale['anchors']] == [0, 5]
    poses = [camera_from_record(v).c2w for v in cp.views]
    flip = np.diag([1., -1., -1., 1.])
    orbit = build_endpoint_path([p @ flip for p in poses], dict(w=32,h=32,fl_x=camera.fx,fl_y=camera.fx,cx=16,cy=16), 1, renderer=PairRenderer())
    _, path = prepare_saved_path(cp, orbit={'poses': [(np.asarray(f['transform_matrix']) @ flip).tolist() for f in orbit['full_trajectory']['frames']], 'provenance': orbit['provenance']}, renderer_factory=FakeRenderer)
    assert [a['view_id'] for a in path['anchors']] == ['000', '005']
    assert path['segments'] == [{'indices': list(range(1, 10)), 'cache_seed_index': 0}]
    _, supervision = unique_supervision(path, [str(cp.image_path(v)) for v in cp.repair_views])
    assert sum(bool(g['reference']) for g in supervision['groups']) == 2
    assert [g['source_index'] for g in supervision['groups'] if not g['reference']] == list(range(1, 10))


def test_prior_checkpoint_override_is_run_local(tmp_path):
    from splat_explorer.splatfix.artifixer.checkpoint_policy import checkpoint_for_run
    cp = make_checkpoint(tmp_path, 6)
    repair_images(cp, backend=Backend())
    manifest = (cp.root / 'checkpoint.json').read_bytes()
    run = checkpoint_for_run(cp, 'endpoints')
    assert run.endpoint_only
    assert len(run.views) == 6
    assert [v['id'] for v in run.repair_views] == ['000', '005']
    assert all(not v.get('repaired_rgb') for v in run.views[1:-1])
    assert all(v.get('repaired_rgb') for v in cp.views)
    assert not cp.endpoint_only
    assert (cp.root / 'checkpoint.json').read_bytes() == manifest
    assert checkpoint_for_run(cp) is cp
    for original, prepared in zip(cp.views, run.views):
        assert original['camera'] == prepared['camera']
        assert original['original_rgb'] == prepared['original_rgb']


def test_override_options_and_wrong_view_count(tmp_path):
    from splat_explorer.splatfix.artifixer.checkpoint_policy import checkpoint_for_run
    options = validate_job({'stage': 'repair', 'checkpoint': '/saved', 'image_trajectory': 'endpoints'})
    assert options['image_trajectory'] == 'endpoints'
    assert options['split_mode'] == 'no-split'
    assert validate_job({'stage': 'repair', 'checkpoint': '/saved',
                         'image_trajectory': 'endpoints', 'split_mode': 'double-split'})['split_mode'] == 'no-split'
    with pytest.raises(ValueError, match='image_trajectory'):
        validate_job({'stage': 'repair', 'checkpoint': '/saved', 'image_trajectory': 'invalid'})
    with pytest.raises(ValueError, match='at least two'):
        checkpoint_for_run(make_checkpoint(tmp_path, 1), 'endpoints')


@pytest.mark.parametrize('plus', [False, True])
def test_normal_no_split_is_one_sequence_with_all_six_references(plus):
    options = validate_job({'stage': 'repair', 'checkpoint': '/saved', 'split_mode': 'no-split'})
    assert options['image_trajectory'] == 'checkpoint'
    assert options['split_mode'] == 'no-split'
    anchors = [0, 2, 4, 6, 8, 10]
    trajectory = {'frames': [{} for _ in range(11)],
                  'anchors': [{'frame_index': i} for i in anchors],
                  'segments': [{'indices': [1, 3], 'seed_index': 0},
                               {'indices': [9, 7, 5], 'seed_index': 10}]}
    request = {'repair_policy': 'all', 'trajectory_mode': 'authors_orbit',
               'runtime': {'split_mode': options['split_mode'], 'endpoint_reference_mode': 'initial'}}
    jobs = inference_jobs(request, trajectory, plus=plus)
    assert len(jobs) == 1
    segment, reference_order, output = jobs[0]
    assert segment == {'indices': [1, 3, 5, 7, 9], 'cache_seed_index': 0}
    assert reference_order == list(range(6))
    assert output == ('plus' if plus else 'inference')


@pytest.mark.parametrize('count', [2, 3, 7, 25])
def test_variable_endpoint_count_selection_and_path(tmp_path, count):
    from splat_explorer.splatfix.artifixer.checkpoint_policy import checkpoint_for_run
    options = validate_job({'stage': 'select', 'views': count, 'repair_policy': 'endpoints'})
    assert options['views'] == count
    cp = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=count,
                           metadata={'repair_policy': 'endpoints'})
    assert Checkpoint.load(cp.root).target_views == count
    prior = make_checkpoint(tmp_path / 'prior', count)
    run = checkpoint_for_run(prior, 'endpoints')
    assert len(run.views) == count and len(run.repair_views) == 2
    poses = []
    for x in range(count):
        pose = np.eye(4); pose[0, 3] = x; poses.append(pose)
    result = build_endpoint_path(poses, dict(w=32,h=16,fl_x=20,fl_y=20,cx=16,cy=8),
                                 1, renderer=PairRenderer())
    frames = result['full_trajectory']['frames']
    assert len(frames) == 2*count-1
    assert [f['transform_matrix'][0][3] for f in frames][::2] == list(range(count))
    assert result['provenance']['reference_frame_indices'] == [0, len(frames)-1]
    assert len(result['provenance']['segments']) == 1
