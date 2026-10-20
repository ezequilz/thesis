import ast
import collections
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from test_splatfix_oraclegs import checkpoint
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from splat_explorer.splatfix.oraclegs.scene_memory import rotation_distances, select_context, prepare_memory, world_memory
from splat_explorer.splatfix.oraclegs.repair import seva_request, DEFAULT_RUNTIME, validate_staged_trajectory
from splat_explorer.splatfix.worldexplorer.bundle import digest_file
from splat_explorer.splatfix.jobs import validate_job


def pose(angle):
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    return p.tolist()


def test_rotation_metric_and_mixed_pool_exclude_opposite_origins():
    np.testing.assert_allclose(rotation_distances([pose(0), pose(90), pose(180)], [pose(0)]).ravel(), [0, 90, 180])
    candidates = [{'transform_matrix':pose(i), 'trajectory_pose':pose(0), 'memory_source': 'seva' if i < 2 else 'worldexplorer'} for i in range(7)]
    candidates.insert(0, {'transform_matrix':pose(0), 'trajectory_pose':pose(180), 'memory_source':'excluded'})
    selected = select_context(candidates, [pose(0)], pose(0))
    assert len(selected) == 5
    assert {f['memory_source'] for f in selected} == {'seva', 'worldexplorer'}
    with pytest.raises(ValueError, match='five eligible'):
        select_context(candidates[:4], [pose(0)], pose(0))


def test_rotation_selection_accepts_mixed_seva_and_worldexplorer_pose_shapes():
    full = [pose(angle) for angle in (0, 10, 20, 30, 40, 50, 60)]
    mixed = [p[:3] if i % 2 else p for i, p in enumerate(full)]
    targets = [pose(0), pose(90)[:3]]
    np.testing.assert_allclose(
        rotation_distances(mixed, targets),
        rotation_distances(full, [pose(0), pose(90)]))
    candidates = [{'transform_matrix':p, 'trajectory_pose':pose(0), 'index':i}
                  for i, p in enumerate(mixed)]
    selected = select_context(candidates, targets[:1], pose(0))
    assert {f['index'] for f in selected} == set(range(5))


def stage_fixture(scene, destination):
    destination = Path(destination)
    destination.mkdir(parents=True)
    frames = []
    for i in range(16):
        path = destination/f'{i}.png'
        Image.new('RGB', (576, 576), (0, i, 0)).save(path)
        frames.append({'file_path':path.name, 'w':576, 'h':576, 'fl_x':450, 'fl_y':450, 'cx':288, 'cy':288,
            'transform_matrix':pose(i*45 if i < 8 else 180), 'sha256':digest_file(path),
            'source_role':'scaffold' if i < 8 else 'generated', 'source_frame_index':i if i < 8 else i-8,
            'source_trajectory':'004_in'})
    (destination/'transforms.json').write_text(json.dumps({'frames':frames}))
    return str(destination)


@pytest.mark.parametrize('mode,color', [('edited', (0,0,255)), ('baseline',(255,0,0))])
def test_memory_staging_uses_exactly_one_saved_rgb_set(tmp_path, monkeypatch, mode, color):
    monkeypatch.setattr('splat_explorer.splatfix.worldexplorer.source.stage_source', stage_fixture)
    root, out = tmp_path/'inputs', tmp_path/'out'
    manifest = export_inputs(checkpoint(tmp_path, 6), root, mode=mode,
        runtime={'oracle_reference_mode':'scene_memory', 'oracle_target_frames':4})
    validate_staged_trajectory(root, manifest)
    fixed, candidates = world_memory(root, manifest)
    assert len(fixed) == 8 and len(candidates) == 6
    assert all(f['source_frame_index'] >= 2 for f in candidates)
    for frame in manifest['views']:
        with Image.open(root/frame['image']) as im:
            assert im.getpixel((0, 0)) == color
    request = prepare_memory(root, out, manifest, manifest['seva_segments'][0], out/'first')
    assert request['saved_reference_mode'] == mode
    assert len(request['frames']) == 13
    assert [f['role'] for f in request['frames']] == ['scaffold']*8+['memory']*5
    runtime = seva_request(root, out, manifest, DEFAULT_RUNTIME, manifest['seva_segments'][0])
    assert runtime['num_inputs'] == 6 and runtime['T'] == [27,21]
    # A completed SEVA segment supplies five near-rotation candidates for the next.
    previous = manifest['seva_segments'][0]
    all_frames = json.loads((root/'scene/transforms.json').read_text())['frames']
    folder = out/'seva-segments/000/seva/img2vid/scene'
    folder.mkdir(parents=True)
    exported = []
    for i, global_id in enumerate(previous['frame_order']):
        path = folder/f'{i}.png'
        Image.new('RGB', (1024,576), 'yellow').save(path)
        exported.append({**all_frames[global_id], 'file_path':path.name,
                         'transform_matrix':all_frames[global_id]['transform_matrix'][:3]})
    (folder/'transforms.json').write_text(json.dumps({'frames':exported}))
    later = prepare_memory(root, out, manifest, manifest['seva_segments'][1], out/'second')
    assert any(f['memory_source'] == 'seva' for f in later['frames'][8:])
    assert len(later['frames']) == 13
    for frame in later['frames']:
        with Image.open(frame['file_path']) as image:
            assert image.size == (1024, 576)
        # Square scaffold and wide SEVA images retain centered calibration.
        np.testing.assert_allclose(np.asarray(frame['K'])[:2, 2], [512, 288])
    np.testing.assert_allclose(np.diag(later['frames'][0]['K'])[:2], [800, 800])


def test_defaults_and_missing_scaffold(tmp_path):
    opts = validate_job({'stage':'repair', 'reconstruction_method':'oraclegs', 'checkpoint':'/saved'})
    assert opts['oracle_reference_mode'] == 'scene_memory' and opts['mode'] == 'edited'
    assert opts['oracle_preset'] == 'spiral_expanding' and opts['oracle_preset_scale'] == .25
    assert validate_job({**opts, 'oracle_reference_mode':'all'})['oracle_reference_mode'] == 'all'
    with pytest.raises(ValueError, match='posed WorldExplorer scaffold'):
        export_inputs(checkpoint(tmp_path,6), tmp_path/'inputs', runtime={'oracle_reference_mode':'scene_memory'})


def native_chunking():
    path = Path(__file__).resolve().parents[1]/'src/splat_explorer/splatfix/oraclegs/upstream/seva/seva/eval.py'
    names = {'infer_prior_stats', 'chunk_input_and_test', 'pad_indices'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ns = dict(np=np, math=math, re=re, collections=collections)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes], type_ignores=[])), str(path), 'exec'), ns)
    return ns


def test_native_sparse_slots_reserve_memory_without_exporting_it(monkeypatch):
    from splat_explorer.splatfix.oraclegs.scene_memory_runtime import install_scene_memory
    # This contract exercises real pinned SEVA chunking without model/Torch imports.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    ns = native_chunking()
    evaluation = SimpleNamespace(chunk_input_and_test=ns['chunk_input_and_test'], get_value_dict=lambda *a: a)
    demo = SimpleNamespace(infer_prior_stats=ns['infer_prior_stats'])
    install_scene_memory(evaluation, demo, {'frames':[{}]*13, 'reference_count':19, 'saved_reference_mode':'edited'})
    config = {'T':[27,21], 'options':{'chunk_strategy':'interp-gt'}}
    count = demo.infer_prior_stats([27,21],6,80,config)
    assert count == 8 and config['T'] == [27,21]
    chunks = evaluation.chunk_input_and_test(27, np.zeros((6,3,4)), np.zeros((17,3,4)),
        list(range(6)), list(range(17)), {'sampler_verbose':False}, chunk_strategy='gt', gt_input_inds=list(range(6)))
    assert [len(ids) for ids in chunks[3]] == [8,8,1]
    assert sorted(i for ids in chunks[3] for i in ids) == list(range(17))
    for refs, targets in zip(chunks[2], chunks[4]):
        assert refs == list(range(6)) and max(targets) < 14
        padded_refs, _, _, _ = ns['pad_indices'](refs, targets, 27)
        assert len(set(i for i in padded_refs if i < 14) | set(range(14,27))) == 19
