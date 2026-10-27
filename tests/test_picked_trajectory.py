"""Picked waypoint identity, author interpolation, split ownership and options."""
import json
import math

import numpy as np
import pytest

from splat_explorer.splatfix.artifixer import author_trajectory as author, connected_trajectory
from test_author_trajectory import K, pose, original_orbit_renderer


class ReversingPairRenderer:
    """Always reverses a pair; the adapter must restore the requested direction."""
    def interpolate_orbit_poses(self, poses, *, loop, interp_distance, training_poses):
        assert not loop and training_poses is None and len(poses) == 2
        first, last = poses[::-1]
        count = max(1, math.ceil(np.linalg.norm(first[:3, 3] - last[:3, 3]) / interp_distance))
        frames = [first * (1-t) + last * t for t in np.linspace(0, 1, count + 1)]
        frames[0], frames[-1] = first, last
        return np.array(frames), np.full(len(frames), -1)

    def compute_pose_distance(self, first, last):
        return np.linalg.norm(first[:3, 3] - last[:3, 3]), 0


def as_inference_trajectory(result):
    p = result['provenance']
    return {'frames': result['full_trajectory']['frames'],
            'anchors': [{'frame_index': i} for i in p['reference_frame_indices']],
            'recipe': {'orbit': {'provenance': p}},
            'segments': [{'indices': s['full_frame_indices'],
                          'seed_index': s.get('seed_full_frame_index', s.get('cache_seed_full_frame_index'))}
                         for s in p['segments']]}


@pytest.mark.parametrize('split', author.SPLIT_MODES)
@pytest.mark.parametrize('loop', [False, True])
def test_picked_path_survives_reversed_pairs_crossings_and_loops(split, loop):
    references = [pose(x) for x in [0, 2, 1, 3, .5, 4]]
    result = author.build_picked_path(references, K, 2, renderer=ReversingPairRenderer(),
                                     split_mode=split, loop=loop)
    p = result['provenance']
    full = [f['transform_matrix'] for f in result['full_trajectory']['frames']]
    indices = author.picked_reference_indices(full, references, p)
    assert indices == sorted(indices) and len(indices) == 6
    assert all(np.array_equal(full[i], ref) for i, ref in zip(indices, references))
    # View 3 is crossed during 1 -> 2, but retains its later named occurrence.
    assert any(np.array_equal(frame, references[2]) for frame in full[:indices[1]])
    assert indices[2] > indices[1]
    expected = [i for i in range(len(full)) if i not in indices]
    assert sorted(i for s in p['segments'] for i in s['full_frame_indices']) == expected
    assert [f['full_frame_index'] for f in p['frames']] == expected
    assert p['requested_spacing'] == p['effective_spacing'] == .1
    if split == 'double-split':
        plans = connected_trajectory.plan_legs(as_inference_trajectory(result))
        assert len(plans) == (6 if loop else 5)
        pairs = list(zip(indices, indices[1:] + ([0] if loop else [])))
        assert [(s['trajectory_indices'][0], s['trajectory_indices'][-1]) for s in plans] == pairs


@pytest.mark.parametrize('split', author.SPLIT_MODES)
@pytest.mark.parametrize('loop', [False, True])
def test_picked_path_matches_original_authors_pair_samples(original_orbit_renderer, split, loop):
    from scipy.spatial.transform import Rotation
    references = [pose(x) for x in [0, 1, -.2, .7, .05, 1.2]]
    for i, ref in enumerate(references):
        ref[:3, :3] = Rotation.from_euler('y', i * 8, degrees=True).as_matrix()
    result = author.build_picked_path(references, K, 2.6035, renderer=original_orbit_renderer,
                                     split_mode=split, loop=loop)
    indices = result['provenance']['reference_frame_indices']
    full = result['full_trajectory']['frames']
    for i in range(len(references) if loop else len(references)-1):
        first, last = references[i], references[(i+1) % len(references)]
        expected = author.build_author_orbit([first, last], K, 2.6035,
            renderer=original_orbit_renderer, split_mode='single-split')['full_trajectory']['frames']
        if np.array_equal(expected[-1]['transform_matrix'], first):
            expected = expected[::-1]
        actual = full[indices[i]:indices[i+1]+1] if i+1 < len(indices) else [*full[indices[i]:], full[0]]
        assert [f['transform_matrix'] for f in actual] == [f['transform_matrix'] for f in expected]


def test_order_option_defaults_and_invalid_values():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    from splat_explorer.splatfix.artifixer.repair import DEFAULT_RUNTIME
    assert DEFAULT_RUNTIME['trajectory_order'] == 'picked'
    assert validate_options({'stage': 'repair'})['trajectory_order'] == 'picked'
    assert validate_options({'stage': 'repair', 'trajectory_order': 'authors'})['trajectory_order'] == 'authors'
    assert 'trajectory_order' not in validate_options({'stage': 'benchmark'})
    for invalid in [True, None, 'auto', 'PICKED']:
        with pytest.raises(ValueError, match='trajectory_order'):
            validate_options({'stage': 'repair', 'trajectory_order': invalid})


def test_named_waypoints_reject_reordering_and_missing_policy():
    refs = [pose(0), pose(2), pose(1)]
    result = author.build_picked_path(refs, K, 1, renderer=ReversingPairRenderer())
    full = [f['transform_matrix'] for f in result['full_trajectory']['frames']]
    p = result['provenance']
    for wrong in [{**p, 'trajectory_order': 'authors'}, {**p, 'ordering_version': 0},
                  {**p, 'reference_frame_indices': [0, 10, 20]}]:
        with pytest.raises(ValueError, match='selection order'):
            author.picked_reference_indices(full, refs, wrong)


@pytest.mark.parametrize('choice', ['picked', 'authors'])
def test_public_repair_cli_forwards_order(tmp_path, monkeypatch, choice):
    import argparse
    from splat_explorer.config import Config
    from splat_explorer.splatfix.cli import add_parser, repair_splat
    from splat_explorer.splatfix.artifixer import repair
    received = {}
    monkeypatch.setattr(repair, 'run_repair', lambda *a, **kw: received.update(kw) or {})
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(['splatfix', 'repair', str(tmp_path), '--output', str(tmp_path / 'out'),
                              '--trajectory-order', choice])
    repair_splat(Config({}), args)
    assert received['runtime']['trajectory_order'] == choice


def test_author_cli_dispatches_picked_order(tmp_path, monkeypatch):
    original = author.build_picked_path
    monkeypatch.setattr(author, 'build_picked_path', lambda *a, **kw: original(*a, **kw, renderer=ReversingPairRenderer()))
    source = tmp_path / 'transforms.json'
    source.write_text(json.dumps({**K, 'frames': [{'transform_matrix': pose(i).tolist()} for i in range(3)]}))
    selected = tmp_path / 'indices.json'; selected.write_text('[0, 2, 1]')
    provenance = tmp_path / 'provenance.json'
    author.main(['--repo', str(tmp_path), '--transforms', str(source), '--selected-indices', str(selected),
                 '--metric-scale', '1', '--trajectory-order', 'picked', '--output', str(tmp_path / 'out.json'),
                 '--provenance', str(provenance)])
    result = json.loads(provenance.read_text())
    assert result['trajectory_order'] == 'picked' and result['selected_source_indices'] == [0, 2, 1]
    assert result['reference_frame_indices'] == sorted(result['reference_frame_indices'])


def test_trajectory_compatibility_never_crosses_order_policy(tmp_path):
    from copy import deepcopy
    from splat_explorer.splatfix.artifixer.staging_cache import compatible_trajectory
    result = author.build_picked_path([pose(0), pose(1)], K, 1, renderer=ReversingPairRenderer())
    recipe = {'orbit': {'poses': [f['transform_matrix'] for f in result['full_trajectory']['frames']],
                        'provenance': result['provenance']}}
    path = tmp_path / 'saved' / 'trajectory.json'; path.parent.mkdir()
    path.write_text(json.dumps({'recipe': recipe}))
    assert compatible_trajectory(tmp_path, recipe) == path
    other = deepcopy(recipe)
    other['orbit']['provenance']['trajectory_order'] = 'authors'
    assert compatible_trajectory(tmp_path, other) is None


def test_rendered_path_uses_named_waypoint_after_incidental_crossing(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_splatfix_repair import make_checkpoint, mock_scene, FakeRenderer
    from splat_explorer.splatfix.checkpoint import camera_from_record
    from splat_explorer.splatfix.artifixer.repair import prepare_saved_path
    cp = make_checkpoint(tmp_path)
    camera = camera_from_record(cp.views[0])
    cp.manifest['target_views'] = 3
    for x in (2., 1.):
        cp.add_view(np.full((32, 32, 3), 90, np.uint8), replace(camera, position=camera.position + [x, 0, 0]))
    references = [camera_from_record(v).c2w for v in cp.views]
    flip = np.diag([1., -1., -1., 1.])
    result = author.build_picked_path([p @ flip for p in references], K, 1, renderer=ReversingPairRenderer())
    orbit = {'poses': [(np.array(f['transform_matrix']) @ flip).tolist() for f in result['full_trajectory']['frames']],
             'provenance': result['provenance']}
    mock_scene(monkeypatch)
    _, trajectory = prepare_saved_path(cp, orbit=orbit, renderer_factory=FakeRenderer,
                                       runtime={'trajectory_order': 'picked'})
    indices = [a['frame_index'] for a in trajectory['anchors']]
    assert indices == result['provenance']['reference_frame_indices']
    assert indices[2] > indices[1]
    connected_trajectory.plan_legs(trajectory)


def test_rendered_extension_cameras_keep_base_indices_and_separate_context(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_splatfix_repair import make_checkpoint, mock_scene, FakeRenderer
    from splat_explorer.splatfix.checkpoint import camera_from_record
    from splat_explorer.splatfix.artifixer.repair import prepare_saved_path
    from splat_explorer.splatfix.artifixer.teacher_extensions import extend_orbit
    cp=make_checkpoint(tmp_path);camera=camera_from_record(cp.views[0])
    cp.manifest['target_views']=2
    cp.add_view(np.full((32,32,3),90,np.uint8),replace(camera,position=camera.position+[1,0,0]))
    poses=[]
    for t in np.linspace(0,1,5):
        p=camera.c2w.copy();p[:3,3]+=np.array([t,0,0]);poses.append(p.tolist())
    orbit={'poses':poses,'provenance':{'trajectory_order':'picked','ordering_version':1,
        'reference_frame_indices':[0,4],'segments':[{'full_frame_indices':[1,2,3],'seed_full_frame_index':0}]}}
    class Clear:
        def intersects(self,*args):return False
    extended=extend_orbit(orbit,[0,4],4,Clear())
    mock_scene(monkeypatch)
    _,trajectory=prepare_saved_path(cp,orbit=extended,renderer_factory=FakeRenderer,runtime={'trajectory_order':'picked'})
    assert trajectory['base_frame_count']==5
    assert trajectory['context_only_indices']==list(range(5,13))
    assert len(trajectory['frames'])==13
    assert [a['frame_index'] for a in trajectory['anchors']]==[0,4]
    assert trajectory['segments']==[{'indices':[1,2,3],'seed_index':0}]
    assert all((cp.root/f['opacity']).is_file() and (cp.root/f['rgb']).is_file() for f in trajectory['frames'])
