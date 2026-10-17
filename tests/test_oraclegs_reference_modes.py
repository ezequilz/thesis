"""Reference selection and inward generation without GPU model dependencies."""
import json
import pytest
from test_splatfix_oraclegs import checkpoint
from splat_explorer.splatfix.oraclegs.backend import validate_options
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from splat_explorer.splatfix.oraclegs.repair import validate_staged_trajectory, seva_request, DEFAULT_RUNTIME
from splat_explorer.splatfix.oraclegs.execution import job_runtime


@pytest.mark.parametrize('mode,split', [('all','single-split'),('all','double-split'),('single','single-split'),('single','double-split')])
@pytest.mark.parametrize('count',[4,5])
def test_connected_reference_and_split_combinations(tmp_path,mode,split,count):
    root=tmp_path/'inputs'
    options=validate_options({'oracle_reference_mode':mode,'oracle_split_mode':split,
                              'oracle_target_frames':count,'oracle_path_order':[2,0,1]})
    assert job_runtime({},options)['oracle_reference_mode']==mode
    manifest=export_inputs(checkpoint(tmp_path),root,runtime=options)
    validate_staged_trajectory(root,manifest)
    double=split=='double-split' or mode=='single'
    segments=manifest['seva_segments']
    assert len(segments)==(4 if double else 2)
    assert sorted(i for s in segments for i in s['target_indices'])==list(range(count*2))
    for segment in segments:
        request=seva_request(root/segment['input_subdir'],tmp_path/'out',manifest,DEFAULT_RUNTIME,segment)
        assert request['num_inputs']==(1 if mode=='single' else 3)
        assert ('ordered_pair_endpoints' in request)==(not double)
        refs=[i for i in segment['frame_order'] if i<3]
        if mode=='single':
            assert refs==[segment['from_reference']]
        else:
            assert set(refs)=={0,1,2}
        if double:
            assert segment['frame_order'][len(refs)-1]==segment['from_reference']
    if double:
        for forward,reverse in zip(segments[::2],segments[1::2]):
            assert forward['from_reference']==reverse['to_reference']
            assert forward['to_reference']==reverse['from_reference']
            assert reverse['target_indices']==sorted(reverse['target_indices'],reverse=True)
            assert max(forward['target_indices'])+1==min(reverse['target_indices'])
    native=root/segments[0]['input_subdir']/'scene/transforms.json'
    data=json.loads(native.read_text());data['frames'][-1]['transform_matrix'][0][3]+=1
    native.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='cameras disagree'):
        validate_staged_trajectory(root,manifest)


@pytest.mark.parametrize('mode',['all','single'])
@pytest.mark.parametrize('split',['single-split','double-split'])
def test_presets_never_split(tmp_path,mode,split):
    root=tmp_path/'inputs'
    manifest=export_inputs(checkpoint(tmp_path),root,runtime={'oracle_adapter':'per_view_preset',
        'oracle_reference_mode':mode,'oracle_split_mode':split,'oracle_target_frames':4})
    validate_staged_trajectory(root,manifest)
    assert len(manifest['seva_segments'])==3
    for segment in manifest['seva_segments']:
        assert len(segment['target_indices'])==4
        assert segment['reference_count']==(1 if mode=='single' else 3)


@pytest.mark.parametrize('options',[{'oracle_reference_mode':'invalid'},{'oracle_split_mode':'invalid'}])
def test_invalid_modes(options):
    with pytest.raises(ValueError):
        validate_options(options)


@pytest.mark.parametrize('mode',['all','single'])
def test_inward_merge_preserves_targets_and_unused_references(tmp_path,mode):
    from PIL import Image
    from splat_explorer.splatfix.oraclegs.segments import merge_segments
    root,out=tmp_path/'inputs',tmp_path/'out'
    manifest=export_inputs(checkpoint(tmp_path),root,runtime={'oracle_reference_mode':mode,
        'oracle_split_mode':'double-split','oracle_target_frames':5,'oracle_path_order':[2,0]})
    frames=json.loads((root/'scene/transforms.json').read_text())['frames']
    for segment in manifest['seva_segments']:
        dest=out/'seva-segments'/f"{segment['index']:03d}"/'seva/img2trajvid/scene'
        dest.mkdir(parents=True)
        exported=[]
        for i in segment['frame_order']:
            name=f'{i}.png'
            Image.new('RGB',(1024,576),(i,0,0)).save(dest/name)
            exported.append({**frames[i],'file_path':name})
        (dest/'transforms.json').write_text(json.dumps({'frames':exported}))
    merge_segments(root,out,manifest)
    dest=out/'seva/img2trajvid/scene'
    merged=json.loads((dest/'transforms.json').read_text())['frames']
    assert len(merged)==len(frames)
    for i,(original,actual) in enumerate(zip(frames,merged)):
        assert original['transform_matrix']==actual['transform_matrix']
        if i>=3:
            with Image.open(dest/actual['file_path']) as image:
                assert image.getpixel((0,0))==(i,0,0)
