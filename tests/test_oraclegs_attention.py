"""Reference identity, shuffle scope, and target/source coverage without GPU weights."""
from collections import Counter
import pytest
from splat_explorer.splatfix.oraclegs.attention import plan_chunks
from splat_explorer.splatfix.oraclegs.backend import validate_options
from splat_explorer.splatfix.oraclegs.execution import job_runtime


def manifest(frames=80, order=(0,1,2,3,4,5)):
    return {'views':[{'image':f'scene/images/ref_{i:06d}.png'} for i in range(6)],
        'trajectory':{'segments':[{'index':s,'from_reference':a,'to_reference':b}
            for s,(a,b) in enumerate(zip(order,order[1:]))],
        'frames':[{'target_index':s*frames+i,'segment_index':s,
            'alpha':(i+1)/(frames+1),'reference_index':None,
            'image_name':f'target_{s*frames+i:06d}.png'}
            for s in range(5) for i in range(frames)]}}


@pytest.mark.parametrize('mode',['within_trajectory','global'])
@pytest.mark.parametrize('count',[5,80,201])
def test_complete_unique_targets_all_references_and_nearest_source(mode,count):
    order = (5,2,0,4,1,3)
    m = manifest(count,order)
    chunks = plan_chunks(m,mode=mode,seed=17)
    assert Counter(i for c in chunks for i in c['target_indices']) == Counter(range(5*count))
    refs = {f'ref_{i:06d}.png' for i in range(6)}
    contexts = {}
    for c in chunks:
        assert set(c['reference_images']) == refs
        assert c['image_names'][0] == c['source_image']
        assert c['source_image'] == f"ref_{c['source_reference_index']:06d}.png"
        assert len(c['image_names']) <= 96
        assert set(c['target_images']) <= set(c['context_target_images'])
        contexts.setdefault(c['context_index'],c['context_target_images'])
        assert contexts[c['context_index']] == c['context_target_images']
        for i,alpha in zip(c['target_indices'],c['alphas']):
            assert c['source_reference_index'] == order[i//count+(alpha>.5)]
        if mode == 'within_trajectory':
            assert len(c['segment_indices']) == 1
    assert sum(map(len,contexts.values())) == 5*count
    if mode == 'global':
        assert any(len(c['segment_indices']) > 1 for c in chunks)
    if count == 80 and mode == 'within_trajectory':
        assert len(contexts) == 5 and len(chunks) == 10
        assert all(len(c['target_images']) == 40 for c in chunks)


def test_shuffle_reproducibility_and_midpoint_tie():
    m = manifest(5)
    assert plan_chunks(m,seed=12) == plan_chunks(m,seed=12)
    assert plan_chunks(m,seed=12) != plan_chunks(m,seed=13)
    chunks = plan_chunks(m)
    first = next(c for c in chunks if 2 in c['target_indices'])
    assert first['source_reference_index'] == 0
    with pytest.raises(ValueError,match='batching'):
        plan_chunks(m,mode='unknown')
    with pytest.raises(ValueError,match='room'):
        plan_chunks(m,max_images=6)
    m['trajectory']['frames'].append(m['trajectory']['frames'][0])
    with pytest.raises(ValueError,match='duplicate'):
        plan_chunks(m)


def test_options_reach_remote_runtime():
    assert validate_options({})['oracle_attention_batching'] == 'within_trajectory'
    assert job_runtime({}, {})['oracle_attention_batching'] == 'within_trajectory'
    assert job_runtime({},validate_options({'oracle_attention_batching':'global'}))['oracle_attention_batching'] == 'global'
    with pytest.raises(ValueError,match='batching'):
        validate_options({'oracle_attention_batching':'bad'})


@pytest.mark.parametrize('mode',['within_trajectory','global'])
def test_runner_exports_only_the_assigned_source_map(tmp_path,monkeypatch,mode):
    import sys
    import numpy as np
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace
    from splat_explorer.splatfix.oraclegs.attention import run_attention
    m = manifest(5)
    chunks = plan_chunks(m,mode=mode)
    inputs,output = tmp_path/'combined',tmp_path/'attention'
    inputs.mkdir()
    for c in chunks:
        for name in c['image_names']:
            (inputs/name).touch()
    calls = []
    class Images:
        def __init__(self,names):
            self.names = names
            self.shape = (1,len(names),3,518,518)
        def to(self,*args): return self
        def unsqueeze(self,*args): return self
    class Model:
        aggregator = SimpleNamespace(patch_size=14,patch_start_idx=5)
        def load_state_dict(self,*args): pass
        def to(self,*args): return self
        def eval(self): return self
        def __call__(self,images,**kwargs):
            assert kwargs['vis_source_frame'] == 0
            source = int(images.names[0][4:10])
            layer = kwargs['vis_target_layer']
            calls.append((source,layer))
            return {},np.full((1,len(images.names)*37*37),source+layer/100)
    def save(values,B,S,patch_size,destination,images,source,title,names,**kwargs):
        assert source == 0 and names[0].startswith('ref_')
        assert images.names == names and values.shape == (1,S*37*37)
        Path(destination).mkdir(parents=True)
        for name in names:
            (Path(destination)/name).write_text(str(values[0,0]))
    upstream = SimpleNamespace(VGGT=Model,
        load_and_preprocess_images=lambda paths,mode: Images([Path(p).name for p in paths]),
        normalize_attention_map=lambda raw,*args:raw,save_attention_maps=save)
    torch = SimpleNamespace(manual_seed=lambda seed:None,
        cuda=SimpleNamespace(is_available=lambda:False),float16='fp16',
        hub=SimpleNamespace(load_state_dict_from_url=lambda url:{}),
        no_grad=nullcontext,amp=SimpleNamespace(autocast=lambda **kw:nullcontext()))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'visualize_attn',upstream)
    run_attention({'seed':23,'image_dir':str(inputs),'output_dir':str(output),'chunks':chunks})
    assert len(calls) == 2*len(chunks)
    assert len(list(output.glob('synth_*.png'))) == 25
    for c in chunks:
        for name in c['target_images']:
            assert float((output/name).read_text()) == pytest.approx(c['source_reference_index']+.165)
    # Missing one reference must fail before inference, even if targets exist.
    (inputs/'ref_000005.png').unlink()
    with pytest.raises(ValueError,match='chunk|input'):
        run_attention({'seed':23,'image_dir':str(inputs),'output_dir':str(output),'chunks':chunks})
