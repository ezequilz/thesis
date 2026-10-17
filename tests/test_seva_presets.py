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
