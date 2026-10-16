import ast
import collections
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from splat_explorer.splatfix.oraclegs.ordered_pair import ordered_chunks, install_ordered_pair
from splat_explorer.splatfix.oraclegs.inputs import export_inputs, AXES
from splat_explorer.splatfix.oraclegs.repair import seva_request, DEFAULT_RUNTIME, validate_staged_trajectory
from test_splatfix_oraclegs import checkpoint


class Array(np.ndarray):
    """Only the CPU tensor conversion interface used by SEVA's task parser."""
    def float(self): return self.astype(np.float32)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


def native():
    root = Path(__file__).resolve().parents[1]/'src/splat_explorer/splatfix/oraclegs/upstream/seva'
    ns = dict(np=np, math=math, re=re, collections=collections,
              torch=SimpleNamespace(tensor=lambda x:np.asarray(x).view(Array)))
    for file,names in [('demo.py',{'parse_task'}),('seva/eval.py',{'infer_prior_stats','infer_prior_inds','chunk_input_and_test'})]:
        nodes = [n for n in ast.parse((root/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),file,'exec'),ns)
    return ns


@pytest.mark.parametrize('order', [[0,1,2,3,4,5],[5,2,0]])
def test_native_ordered_pairs_keep_context_boundaries_and_source_assignments(tmp_path,order):
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path,6),root,runtime={'oracle_path_order':order})
    validate_staged_trajectory(root,manifest)
    ns = native()
    for segment in manifest['seva_segments']:
        data_path = root/segment['input_subdir']
        frames = json.loads((data_path/'scene/transforms.json').read_text())['frames']
        split = json.loads((data_path/'scene/train_test_split_6.json').read_text())
        poses = np.asarray([f['transform_matrix'] for f in frames]) @ AXES
        ns['get_parser'] = lambda **kw:SimpleNamespace(image_paths=[str(data_path/'scene'/f['file_path']) for f in frames],
            camtoworlds=poses,camera_ids=list(range(86)),Ks_dict={i:np.eye(3) for i in range(86)},
            splits_per_num_input_frames={6:split})
        config = {'T':21,'options':{'chunk_strategy':'interp-gt','sampler_verbose':False}}
        def get_value_dict(curr_imgs,curr_input_frame_indices,curr_c2ws,curr_Ks,curr_input_camera_indices,all_c2ws,camera_scale):
            return {}
        saved = []
        evaluation = SimpleNamespace(chunk_input_and_test=ns['chunk_input_and_test'],get_value_dict=get_value_dict,
                                     save_output=lambda samples,path,fps:saved.append(samples))
        demo = SimpleNamespace(parse_task=ns['parse_task'])
        report_path = data_path/'audit.json'
        install_ordered_pair(evaluation,segment['endpoint_frame_indices'],report_path,demo=demo,data_path=data_path)
        parsed = demo.parse_task('img2vid','unused',6,21,config)
        _,n,count,inputs,anchors,c2ws,ks,anchor_poses,_ = parsed
        assert inputs == [0,1,2,3,4,85] and count == 80 and len(anchors)==15
        first = evaluation.chunk_input_and_test(21,c2ws[inputs],anchor_poses,inputs,anchors,
            config['options'],task='img2vid',chunk_strategy='gt-nearest',gt_input_inds=list(range(6)))
        assert len(first[0]) == 1 and first[1][0] == list(range(6))
        ords = sorted(inputs+anchors)
        gt = [ords.index(i) for i in inputs]
        result = evaluation.chunk_input_and_test(21,c2ws[ords],c2ws[split['test_ids']],ords,split['test_ids'],
            config['options'],task='img2vid',chunk_strategy='interp-gt',gt_input_inds=gt)
        assert [j for group in result[3] for j in group] == list(range(80))
        for chunk,ii,ss in zip(result[0],result[1],result[2]):
            assert len(chunk)==21 and len(ii)==len(set(ii)) and len(ss)==len(set(ss))
            assert set(gt).issubset(ii)
        report = json.loads(report_path.read_text())
        for chunk in report['chunks']:
            for b in chunk['bounds']:
                assert 4 <= b['left_order'] <= b['target_order'] <= b['right_order'] <= 85
        a,b = segment['from_reference'],segment['to_reference']
        assert [x['source_reference_index'] for x in report['targets']] == [a]*40+[b]*40
        for anchor in report['anchors']:
            assert anchor['source_reference_index'] == (a if anchor['frame_index']<=44 else b)
        # Actual value hook passes the endpoint assignment to optional reprojection.
        ids = [5,44,45,84]
        value = evaluation.get_value_dict(None,[],c2ws[ids],None,[],None,2)
        assert value['_ordered_pair_sources'] == [a,a,b,b]
        # Images retain canonical export indices; video excludes off-path prefix.
        rgb = np.arange(86)
        evaluation.save_output({'samples-rgb/image':rgb},data_path/'scene',30)
        np.testing.assert_array_equal(saved[0]['samples-rgb/image'],rgb)
        np.testing.assert_array_equal(saved[1]['samples-rgb/video'],rgb[4:86])
        req = seva_request(data_path,tmp_path/'out',manifest,DEFAULT_RUNTIME)
        assert req['task']=='img2vid' and req['ordered_pair_endpoints']==[4,85]


def test_context_budget_midpoint_and_invalid_endpoints():
    # A at 4, B at 10; off-path refs at 0..3 cannot bound any target.
    result,records = ordered_chunks(9,[0,1,2,3,4,7,10],[5,6,7,8,9],[0,1,2,3,4,6],[4,10])
    assert sorted(i for chunk in result[3] for i in chunk)==list(range(5))
    assert all(b['left_order']>=4 for r in records for b in r['bounds'])
    with pytest.raises(ValueError,match='context cannot fit'):
        ordered_chunks(6,[0,1,2,3,4,7,10],[5],[0,1,2,3,4,6],[4,10])
    with pytest.raises(ValueError,match='bound every target'):
        ordered_chunks(21,[0,1,2,3,4,7,10],[3],[0,1,2,3,4,6],[4,10])


def test_v4_inputs_cannot_launch_under_new_worker(tmp_path):
    root=tmp_path/'inputs'
    manifest=export_inputs(checkpoint(tmp_path,2),root)
    manifest['trajectory']['version']=4
    with pytest.raises(ValueError,match='restage'):
        validate_staged_trajectory(root,manifest)


def test_explicit_reprojection_source_overrides_nearest_pose():
    from splat_explorer.splatfix.oraclegs.seva_geometry import ReferenceGuidance
    from test_oraclegs_seva_geometry import scene
    guide=ReferenceGuidance.__new__(ReferenceGuidance)
    first=scene()
    second=list(scene()); second[0]=np.ones_like(second[0]); second[3]=second[3].copy(); second[3][0,3]=.1
    guide.references=[first,second];guide.extent=1
    rgb,inv,valid,pose,k=first
    assert guide.target(pose,k,(32,32))[2]==0
    assert guide.target(pose,k,(32,32),source_reference=1)[2]==1


def test_short_paths_deduplicate_native_anchor_sampling(tmp_path):
    root=tmp_path/'inputs'
    manifest=export_inputs(checkpoint(tmp_path,6),root,runtime={'oracle_target_frames':3})
    data=root/'segments/000'
    frames=json.loads((data/'scene/transforms.json').read_text())['frames']
    split=json.loads((data/'scene/train_test_split_6.json').read_text())
    ns=native()
    ns['get_parser']=lambda **kw:SimpleNamespace(image_paths=[f['file_path'] for f in frames],
        camtoworlds=np.asarray([f['transform_matrix'] for f in frames])@AXES,
        camera_ids=list(range(9)),Ks_dict={i:np.eye(3) for i in range(9)},
        splits_per_num_input_frames={6:split})
    evaluation=SimpleNamespace(chunk_input_and_test=ns['chunk_input_and_test'],
        get_value_dict=lambda **kwargs: {},save_output=lambda *a:None)
    demo=SimpleNamespace(parse_task=ns['parse_task'])
    install_ordered_pair(evaluation,[4,8],data/'audit.json',demo=demo,data_path=data)
    parsed=demo.parse_task('img2vid','unused',6,21,{'T':21,'options':{'chunk_strategy':'interp-gt'}})
    assert parsed[4]==[5,6,7] and len(parsed[7])==3
    report=json.loads((data/'audit.json').read_text())
    assert [a['source_reference_index'] for a in report['anchors']]==[0,0,1]


@pytest.mark.parametrize('resolution',['576p','720p'])
def test_segment_merge_restores_canonical_identity_after_ordering(tmp_path,resolution):
    from PIL import Image
    from splat_explorer.splatfix.oraclegs.segments import merge_segments
    root,out=tmp_path/'inputs',tmp_path/'output'
    manifest=export_inputs(checkpoint(tmp_path,3),root,runtime={
        'oracle_target_frames':3,'oracle_path_order':[2,0,1],'seva_resolution':resolution})
    frames=json.loads((root/'scene/transforms.json').read_text())['frames']
    for segment in manifest['seva_segments']:
        folder=out/'seva-segments'/f"{segment['index']:03d}"/'seva/img2vid/scene'
        folder.mkdir(parents=True)
        exported=[]
        for local,global_index in enumerate(segment['frame_order']):
            f=frames[global_index]
            name=f'{local:03d}.png'
            Image.new('RGB',tuple(manifest['seva_internal_resolution']),(global_index,0,0)).save(folder/name)
            exported.append({**f,'file_path':name,'h':manifest['seva_internal_resolution'][1],
                'cy':f['cy']+(manifest['seva_internal_resolution'][1]-manifest['resolution'][1])/2})
        (folder/'transforms.json').write_text(json.dumps({'frames':exported}))
    merge_segments(root,out,manifest)
    folder=out/'seva/img2vid/scene'
    merged=json.loads((folder/'transforms.json').read_text())['frames']
    for i,f in enumerate(merged):
        assert Image.open(folder/f['file_path']).getpixel((0,0))==(i,0,0)
        assert f['transform_matrix']==frames[i]['transform_matrix']
