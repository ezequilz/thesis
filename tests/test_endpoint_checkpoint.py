"""Endpoint-only checkpoints retain all cameras but only two trusted images."""
import numpy as np
import pytest
from test_splatfix_checkpoint import make_checkpoint, Backend
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.image_repair import repair_images
from splat_explorer.splatfix.artifixer.author_trajectory import build_endpoint_path
from splat_explorer.splatfix.artifixer.official_worker import inference_jobs, unique_supervision
from splat_explorer.splatfix.jobs import validate_job


def test_only_endpoints_are_edited_and_retry_is_free(tmp_path):
    cp = make_checkpoint(tmp_path, 6)
    cp.manifest['metadata']['repair_policy'] = 'endpoints'
    cp.save()
    originals = [cp.image_path(v).read_bytes() for v in cp.views]
    backend = Backend()
    repair_images(cp, backend=backend)
    cp = Checkpoint.load(cp.root)
    assert [v['id'] for v in cp.views if v.get('repaired_rgb')] == ['000', '005']
    assert [cp.image_path(v).read_bytes() for v in cp.views] == originals
    repair_images(cp, backend=backend)
    assert len(backend.calls) == 2
    assert len(cp.views) == 6


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
    with pytest.raises(ValueError, match='six'):
        validate_job({'stage': 'select', 'views': 5, 'repair_policy': 'endpoints'})
    result = validate_job({'stage': 'select', 'views': 6, 'repair_policy': 'endpoints', 'horizontal_fov': 77})
    assert result['horizontal_fov'] == 77
    result = validate_job({'stage': 'repair', 'checkpoint': '/saved', 'split_mode': 'no-split', 'endpoint_reference_mode': 'initial'})
    assert result['endpoint_reference_mode'] == 'initial'


def test_saved_path_uses_only_endpoint_supervision(tmp_path, monkeypatch):
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
