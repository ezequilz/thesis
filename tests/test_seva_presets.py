"""Preset geometry, independent conditioning and reconstruction boundaries."""
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_splatfix_oraclegs import checkpoint
from splat_explorer.splatfix.oraclegs.trajectory import preset_poses, preset_assignments
from splat_explorer.splatfix.oraclegs.inputs import export_inputs, AXES
from splat_explorer.splatfix.oraclegs.repair import validate_staged_trajectory, seva_request, DEFAULT_RUNTIME
from splat_explorer.splatfix.oraclegs.attention import plan_chunks
from splat_explorer.splatfix.oraclegs.segments import merge_segments
from splat_explorer.splatfix.oraclegs.backend import validate_options
from splat_explorer.splatfix.oraclegs.execution import job_runtime
from PIL import Image


@pytest.mark.parametrize('preset', ['orbit','spiral','lemniscate'])
def test_presets_translate_and_transform_with_saved_camera(preset):
    canonical = preset_poses(np.eye(4),preset,80)
    start = np.eye(4)
    start[:3,:3] = Rotation.from_euler('xyz',[.3,.7,-.4]).as_matrix()
    start[:3,3] = [5,-2,8]
    actual = preset_poses(start,preset,80,.2)
    scaled = canonical.copy()
    scaled[:,:3,3] *= .2
    np.testing.assert_allclose(actual,start[None]@scaled,atol=1e-12)
    np.testing.assert_allclose(actual[0],start)
    assert np.linalg.norm(np.ptp(canonical[:,:3,3],axis=0)) > 1
    np.testing.assert_allclose(np.linalg.det(actual[:,:3,:3]),1,atol=1e-12)
    np.testing.assert_allclose(actual[:,:3,:3].transpose(0,2,1)@actual[:,:3,:3],
                               np.broadcast_to(np.eye(3),(80,3,3)),atol=1e-12)
    if preset == 'orbit':
        np.testing.assert_allclose(np.linalg.norm(canonical[:,:3,3]-[0,0,10],axis=1),10)
        np.testing.assert_allclose(canonical[20,:3,3],[10,0,10],atol=1e-12)
    elif preset == 'lemniscate':
        np.testing.assert_allclose(canonical[:,2,3],0,atol=1e-12)
        np.testing.assert_allclose(canonical[20,:3,3],[-10*np.tan(np.pi/6),0,0],atol=1e-12)


def test_spiral_matches_pinned_author_numpy_implementation():
    source = Path(__file__).resolve().parents[1]/'src/splat_explorer/splatfix/oraclegs/upstream/seva/seva/geometry.py'
    names = {'normalize','viewmatrix','poses_avg','generate_spiral_path'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns = {'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    raw = ns['generate_spiral_path'](AXES[None],np.array([1,5]),n_frames=80,n_rots=2,
                                    zrate=.5,endpoint=False,radii=[1,1,.5]) @ AXES
    poses = np.tile(np.eye(4),(80,1,1));poses[:,:3]=raw
    expected = np.linalg.inv(poses[0])[None]@poses
    np.testing.assert_allclose(preset_poses(np.eye(4),'spiral',80),expected,atol=1e-12)


def test_shuffle_balanced_and_repeatable():
    for n in range(2,9):
        assigned = preset_assignments('shuffle_orbit',n,23)
        counts = Counter(assigned)
        assert max(counts.get(k,0) for k in ['orbit','spiral','lemniscate'])-min(counts.get(k,0) for k in ['orbit','spiral','lemniscate']) <= 1
        assert assigned == preset_assignments('shuffle_orbit',n,23)
    assert Counter(preset_assignments('shuffle_orbit',6,23)) == {'orbit':2,'spiral':2,'lemniscate':2}
    assert preset_assignments('shuffle_orbit',6,23) != preset_assignments('shuffle_orbit',6,24)


@pytest.mark.parametrize('preset',['orbit','spiral','lemniscate','shuffle_orbit'])
def test_six_independent_single_reference_clips(tmp_path,preset):
    cp=checkpoint(tmp_path,6);root=tmp_path/'inputs'
    runtime={'oracle_reference_mode':'single','oracle_adapter':'per_view_preset','oracle_preset':preset,'oracle_preset_scale':.1,'oracle_target_frames':8}
    manifest=export_inputs(cp,root,runtime=runtime)
    validate_staged_trajectory(root,manifest)
    assert len(manifest['seva_segments']) == 6 and manifest['trajectory']['target_count'] == 48
    frames=json.loads((root/'scene/transforms.json').read_text())['frames']
    for segment in manifest['seva_segments']:
        j=segment['index'];native=root/segment['input_subdir']/'scene'
        selected=json.loads((native/'transforms.json').read_text())['frames']
        assert len(selected)==9 and selected[0]==frames[j]
        assert json.loads((native/'train_test_split_1.json').read_text()) == {'train_ids':[0],'test_ids':list(range(1,9))}
        request=seva_request(root/segment['input_subdir'],tmp_path/'out',manifest,DEFAULT_RUNTIME,segment)
        assert request['num_inputs']==1 and request['task']=='img2trajvid'
        assert request['replace_or_include_input'] is True and 'ordered_pair_endpoints' not in request
        assert request['cfg']==[4. if segment['preset']=='orbit' else 3.,2.]
        assert selected[0]['transform_matrix']==selected[1]['transform_matrix']
    for chunk in plan_chunks(manifest):
        assert chunk['segment_indices']==[chunk['source_reference_index']]
    # Corrupt the native camera, which must be caught before launching diffusion.
    native=root/'segments/000/scene/transforms.json'
    data=json.loads(native.read_text());data['frames'][2]['transform_matrix'][0][3]+=1
    native.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='cameras disagree'):
        validate_staged_trajectory(root,manifest)


@pytest.mark.parametrize('reference_mode',['all','single'])
def test_preset_merge_preserves_subset_gt_and_all_targets(tmp_path, reference_mode):
    root=tmp_path/'inputs';out=tmp_path/'output'
    manifest=export_inputs(checkpoint(tmp_path,3),root,runtime={'oracle_adapter':'per_view_preset',
        'oracle_reference_mode':reference_mode,'oracle_preset':'shuffle_orbit','oracle_path_order':[2,0],'oracle_target_frames':4,'seva_resolution':'720p'})
    frames=json.loads((root/'scene/transforms.json').read_text())['frames']
    for segment in manifest['seva_segments']:
        dest=out/'seva-segments'/f"{segment['index']:03d}"/'seva/img2trajvid/scene'
        dest.mkdir(parents=True)
        exported=[]
        for j,global_index in enumerate(segment['frame_order']):
            f=frames[global_index];name=f'{j}.png'
            Image.new('RGB',(1280,768),(segment['index']*50,j,0)).save(dest/name)
            exported.append({**f,'file_path':name,'h':768,'cy':f['cy']+24})
        (dest/'transforms.json').write_text(json.dumps({'frames':exported}))
    merge_segments(root,out,manifest)
    merged=json.loads((out/'seva/img2trajvid/scene/transforms.json').read_text())['frames']
    assert len(merged)==11
    assert merged[1]['file_path']=='input/000001.png' and merged[1]['cy']==frames[1]['cy']+24
    for old,new in zip(frames,merged):
        assert old['transform_matrix']==new['transform_matrix']


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True,'1'])
def test_invalid_preset_scale(value):
    with pytest.raises(ValueError,match='preset_scale'):
        validate_options({'oracle_preset_scale':value})


def test_job_runtime_carries_preset_settings():
    options=validate_options({'oracle_adapter':'per_view_preset','oracle_preset':'shuffle_orbit','oracle_preset_scale':.3})
    runtime=job_runtime({},options)
    assert runtime['oracle_preset']=='shuffle_orbit' and runtime['oracle_preset_scale']==.3


@pytest.mark.parametrize('adapter', ['per_view_preset', 'trajectory'])
@pytest.mark.parametrize('reference_mode,expected_anchors', [('single',20), ('all',15)])
def test_reference_mode_controls_native_sparse_anchor_budget(tmp_path, adapter, reference_mode, expected_anchors):
    import math
    source = Path(__file__).resolve().parents[1]/'src/splat_explorer/splatfix/oraclegs/upstream/seva/seva/eval.py'
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name == 'infer_prior_stats']
    ns = {'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path,6),root,runtime={
        'oracle_adapter':adapter, 'oracle_reference_mode':reference_mode,
        'oracle_split_mode':'double-split', 'oracle_target_frames':80})
    for segment in manifest['seva_segments']:
        native = root/segment['input_subdir']/'scene'
        request = seva_request(native.parent,tmp_path/'out',manifest,DEFAULT_RUNTIME,segment)
        n = request['num_inputs']
        frames = json.loads((native/'transforms.json').read_text())['frames']
        split = json.loads((native/f'train_test_split_{n}.json').read_text())
        assert len(split['train_ids']) == n
        assert len(frames) == n+len(split['test_ids'])
        config = {'T':request.get('T',21), 'options':request.copy()}
        anchors = ns['infer_prior_stats'](config['T'],n,len(split['test_ids']),config)
        assert n == (1 if reference_mode == 'single' else 6)
        assert anchors == expected_anchors
        assert n+anchors == config['T'] == 21
        if reference_mode == 'single':
            assert request['T'] == 21 and request['num_prior_frames'] == 20
            with pytest.raises(ValueError,match='exactly one staged reference'):
                seva_request(native.parent,tmp_path/'out',manifest,DEFAULT_RUNTIME,
                             {**segment,'reference_count':6})


def test_80_frame_presets_preserve_intrinsics_and_accept_legacy_roundoff(tmp_path):
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path,2), root, runtime={
        'oracle_adapter':'per_view_preset', 'oracle_preset':'shuffle_orbit',
        'oracle_reference_mode':'single', 'oracle_target_frames':80})
    path = root/'scene/transforms.json'
    data = json.loads(path.read_text())
    frames = data['frames']
    for segment in manifest['seva_segments']:
        for i in segment['target_indices']:
            for key in ('fl_x','fl_y','cx','cy'):
                assert frames[2+i][key] == frames[segment['from_reference']][key]
    validate_staged_trajectory(root,manifest)

    def change_cy(delta):
        frames[3]['cy'] = frames[0]['cy']+delta
        path.write_text(json.dumps(data))
        segment = manifest['seva_segments'][0]
        native = root/segment['input_subdir']/'scene/transforms.json'
        native.write_text(json.dumps({'frames':[frames[i] for i in segment['frame_order']]}))

    change_cy(5.684341886080802e-14)
    validate_staged_trajectory(root,manifest)
    change_cy(.001)
    with pytest.raises(ValueError,match='path or calibration changed'):
        validate_staged_trajectory(root,manifest)


@pytest.mark.parametrize('adapter,split', [('per_view_preset','single-split'),
                                        ('trajectory','single-split'),
                                        ('trajectory','double-split')])
def test_camera_only_targets_need_no_staged_pngs(tmp_path, adapter, split):
    from splat_explorer.splatfix.oraclegs.adapter import pad_seva_inputs
    from splat_explorer.splatfix.oraclegs.camera_audit import install_camera_only_inputs
    from types import SimpleNamespace
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path,3), root, runtime={
        'oracle_adapter':adapter, 'oracle_split_mode':split,
        'oracle_target_frames':4, 'seva_resolution':'720p'})
    validate_staged_trajectory(root,manifest)
    assert not list(root.rglob('target_*.png'))
    segment = root/manifest['seva_segments'][0]['input_subdir']
    padded = tmp_path/'padded'
    pad_seva_inputs(segment,padded)
    assert not list(padded.rglob('target_*.png'))
    scene = padded/'scene'
    frames = json.loads((scene/'transforms.json').read_text())['frames']
    split_data = json.loads(next(scene.glob('train_test_split_*.json')).read_text())
    def parser(**kwargs):
        return SimpleNamespace(image_paths=[str(scene/f['file_path']) for f in frames],
                               image_names=[Path(f['file_path']).name for f in frames],
                               splits_per_num_input_frames={3:split_data})
    demo = SimpleNamespace(get_parser=parser)
    install_camera_only_inputs(demo)
    parsed = demo.get_parser(data_dir=scene)
    assert all(parsed.image_paths[i] is None for i in split_data['test_ids'])
    assert all(Path(parsed.image_paths[i]).is_file() for i in split_data['train_ids'])
    # Missing real reference images must remain an error, never turn into blanks.
    Path(parsed.image_paths[split_data['train_ids'][0]]).unlink()
    with pytest.raises(FileNotFoundError, match='Missing SEVA image'):
        demo.get_parser(data_dir=scene)


def test_expanding_spiral_local_depth_and_smooth_second_turn():
    from splat_explorer.splatfix.oraclegs.trajectory import preset_poses
    from scipy.spatial.transform import Rotation
    count = 80
    start = np.eye(4)
    base = preset_poses(start, 'spiral', count)
    expanded = preset_poses(start, 'spiral_expanding', count)
    # The first turn changes only saved-camera-local forward translation.
    np.testing.assert_allclose(expanded[:41,:2,3], base[:41,:2,3], atol=1e-12)
    np.testing.assert_allclose(expanded[:41,2,3], 1.5*base[:41,2,3], atol=1e-12)
    # Undo local-Z amplification and the native starting look-at to measure
    # radius about the original spiral axis, not about the saved camera.
    focal = 1/(.25/.9+.75/25)
    forward = np.array([-1.,0.,focal]); forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.,-1.,0.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right,down,forward], axis=-1)
    positions = expanded[:,:3,3].copy(); positions[:,2] /= 1.5
    canonical = positions @ rotation.T + [1.,0.,0.]
    # Across BOTH turns, undoing the 1.5 local-Z multiplier must recover
    # the native canonical depth sinusoid, even as the XY radius expands.
    np.testing.assert_allclose(canonical[:,2], .5*np.sin(np.arange(count)*2*np.pi/count), atol=1e-12)
    radius = np.linalg.norm(canonical[:,:2], axis=1)
    np.testing.assert_allclose(radius[:41], 1, atol=1e-12)
    assert radius[-1] == pytest.approx(1.25)
    assert np.all(np.diff(radius[40:]) > 0)
    assert np.diff(radius)[40] < .002  # no abrupt radius jump at turn boundary
    # The extension remains calibrated, equivariant under saved camera poses,
    # and independent of the user-selected scene-unit scale.
    start[:3,:3] = Rotation.from_euler('xyz', [20,30,40], degrees=True).as_matrix()
    start[:3,3] = [3,4,5]
    moved = preset_poses(start, 'spiral_expanding', count, .3)
    expected = expanded.copy(); expected[:,:3,3] *= .3
    np.testing.assert_allclose(moved, start[None] @ expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(expanded[:,:3,:3]), 1, atol=1e-12)
    np.testing.assert_allclose(expanded[0], np.eye(4), atol=1e-12)


def test_expanding_spiral_stages_and_validates(tmp_path):
    from splat_explorer.splatfix.oraclegs.inputs import export_inputs
    from splat_explorer.splatfix.oraclegs.repair import validate_staged_trajectory
    manifest = export_inputs(checkpoint(tmp_path), tmp_path/'expanded', runtime={
        'oracle_reference_mode':'all', 'oracle_adapter':'per_view_preset',
        'oracle_preset':'spiral_expanding', 'oracle_target_frames':80})
    validate_staged_trajectory(tmp_path/'expanded', manifest)
    assert manifest['trajectory']['closed'] is False
    assert manifest['trajectory']['preset_parameters']['local_forward_multiplier'] == 1.5
