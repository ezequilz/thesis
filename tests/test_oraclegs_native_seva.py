"""Run pinned SEVA CPU functions without loading its diffusion weights."""
import ast
import collections
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def native_functions():
    torch = pytest.importorskip('torch')
    root = Path(__file__).resolve().parents[1]/'src/splat_explorer/splatfix/oraclegs/upstream/seva'
    ns = dict(np=np, torch=torch, math=math, re=re, collections=collections)
    for source,names in [(root/'seva/eval.py', {'infer_prior_stats','infer_prior_inds',
            'chunk_input_and_test','pad_indices','assemble','replace_or_include_input_for_dict'}),
            (root/'demo.py', {'parse_task'})]:
        nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),str(source),'exec'),ns)
    return ns


def test_native_parser_and_chunks_use_reference_keyframes_and_keep_all_400_targets():
    ns = native_functions()
    torch = ns['torch']
    # Real native layout: six GT frames, with 80 synthetic queries between each.
    gt = list(range(0,406,81))
    targets = [i for i in range(406) if i not in gt]
    poses = np.repeat(np.eye(4)[None],406,axis=0)
    poses[:,0,3] = np.linspace(-3,9,406)
    parser = SimpleNamespace(image_paths=[f'{i}.png' for i in range(406)],
        camtoworlds=poses,camera_ids=list(range(406)),
        Ks_dict={i:np.eye(3) for i in range(406)},
        splits_per_num_input_frames={6:{'train_ids':gt,'test_ids':targets}})
    ns['get_parser'] = lambda **kwargs: parser
    config = {'T':21,'options':{'chunk_strategy':'interp','sampler_verbose':False}}
    paths,n,count,inputs,anchors,c2ws,ks,anchor_poses,anchor_ks = ns['parse_task'](
        'img2vid','unused',6,21,config)
    assert inputs == gt and count == 400 and paths == parser.image_paths
    assert not set(anchors)&set(gt)
    np.testing.assert_allclose(c2ws[:,:3,3],poses[:,:3,3])
    # This is the native second-pass merge and ordering from run_one_scene.
    prior_order = np.argsort(inputs+anchors)
    prior_indices = np.array(inputs+anchors)[prior_order].tolist()
    gt_inds = [prior_order.tolist().index(i) for i in range(n)]
    result = ns['chunk_input_and_test'](21,c2ws[prior_indices],c2ws[targets],
        prior_indices,targets,config['options'],task='img2vid',chunk_strategy='interp',gt_input_inds=gt_inds)
    _, input_inds, input_sels, test_inds, test_sels = result
    assert sorted(i for chunk in test_inds for i in chunk) == list(range(400))
    used_references = {prior_indices[i] for chunk in input_inds for i in chunk}&set(gt)
    assert used_references == set(gt)
    # Every target is bounded by conditioning cameras in its own keyframe pair.
    for ii,tt in zip(input_inds,test_inds):
        conditioning = [prior_indices[i] for i in ii]
        for t in tt:
            frame = targets[t]
            left = max(i for i in conditioning if i <= frame)
            right = min(i for i in conditioning if i >= frame)
            segment = frame//81
            assert segment*81 <= left <= right <= (segment+1)*81
    # Native final inclusion preserves original RGBs, rather than a generated
    # replacement at a reference pose. Synthetic targets remain separate.
    rgb = torch.zeros(406,3,2,2)
    rgb[gt] = torch.arange(1,7).view(6,1,1,1).float()
    result = ns['replace_or_include_input_for_dict'](
        {'samples-rgb/image':torch.full((400,3,2,2),-1.)},targets,rgb.clone(),c2ws,ks)
    final = result['samples-rgb/image']
    assert torch.equal(final[gt],rgb[gt])
    assert torch.all(final[targets] == -1)
