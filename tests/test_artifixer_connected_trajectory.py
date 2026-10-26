"""Connected RGB ownership and sampler contracts without CUDA or model weights."""
import contextlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from splat_explorer.splatfix.artifixer import connected_trajectory as connected, official_worker


def trajectory_for(count, *, offset=0):
    middle = (count - 1) // 2
    return {'frames': [{} for _ in range(offset + count)],
            'anchors': [{'frame_index': offset + count - 1}, {'frame_index': offset}],
            'segments': [{'seed_index': offset, 'indices': list(range(offset + 1, offset + middle + 1))},
                         {'seed_index': offset + count - 1,
                          'indices': list(range(offset + count - 2, offset + middle, -1))}]}


def test_exact_proposed_64_pose_plan():
    plan, = connected.plan_legs(trajectory_for(64))
    assert plan['outer_generation_indices'] == [list(range(25)), list(range(63, 38, -1))]
    assert plan['target_indices'] == list(range(19, 44))
    assert plan['keep_indices'] == list(range(21, 42))
    assert plan['overlap_indices'] == [19, 20, 42, 43]
    assert plan['reference_indices'] == [0, 8, 14, 18, 19, 20, 42, 43, 44, 48, 55, 63]


@pytest.mark.parametrize('count', range(2, 65))
def test_all_supported_lengths_have_complete_unique_ownership(count):
    plan, = connected.plan_legs(trajectory_for(count))
    outer = [i for side in plan['outer_generation_indices'] for i in side[1:]]
    final = [i for i in outer if i not in plan['keep_indices']] + plan['keep_indices']
    assert sorted(final) == list(range(1, count - 1))
    assert all(len(side) <= 25 for side in plan['outer_generation_indices'])
    if count <= 2:
        assert not plan['target_indices']
    else:
        assert len(plan['keep_indices']) == max(1, min(count - 2, (count + 1) // 3))
        assert len(plan['target_indices']) + plan['padding_count'] == 25
        assert plan['target_indices'] == sorted(plan['keep_indices'] + plan['overlap_indices'])
        assert set(plan['overlap_indices']) <= {0, count - 1, *outer}
        assert len(plan['reference_indices']) <= 12
        assert len(set(plan['reference_indices'])) == len(plan['reference_indices'])
        assert {0, count - 1} <= set(plan['reference_indices'])
        assert set(plan['reference_indices']).isdisjoint(plan['keep_indices'])
        assert set(plan['reference_indices']) <= {0, count - 1, *outer}
        assert set(plan['overlap_indices']) <= set(plan['reference_indices'])
        assert len(plan['overlap_sides']['left']) == (2 if count >= 6 else 1)
        assert 1 <= len(plan['overlap_sides']['right']) <= 2
        assert plan['keep_indices'][0] >= 1 and plan['keep_indices'][-1] < count - 1
        # Reference provenance must stay on the retained part of its own leg.
        for ids, seed in [(plan['overlap_sides']['left'], 0), (plan['overlap_sides']['right'], count - 1)]:
            own = next((side for side in plan['outer_generation_indices'] if side[0] == seed), [seed])
            assert set(ids) <= set(own)


@pytest.mark.parametrize('count,start,stop,window_start,window_stop', [
    (64,21,42,19,44), (63,21,42,19,44), (48,16,32,14,34),
    (30,10,20,8,22), (12,4,8,2,10), (6,2,4,0,6),
    (5,1,3,0,5), (4,1,2,0,4), (3,1,2,0,3)])
def test_dynamic_middle_third_examples(count, start, stop, window_start, window_stop):
    plan, = connected.plan_legs(trajectory_for(count))
    assert plan['keep_indices'] == list(range(start, stop))
    assert plan['target_indices'] == list(range(window_start, window_stop))
    assert plan['padding_count'] == 25 - (window_stop - window_start)


def test_no_bridge_without_an_interior():
    assert connected.plan_legs({'frames': [{}], 'anchors': [{'frame_index': 0}], 'segments': []}) == []
    plan, = connected.plan_legs(trajectory_for(2))
    assert not plan['target_indices'] and plan['skip_reason']


def test_multiple_spans_use_global_ids_and_unsorted_anchor_slots():
    first, second = trajectory_for(64), trajectory_for(57, offset=63)
    trajectory = {**second, 'anchors': [{'frame_index': i} for i in [63, 119, 0]],
                  'segments': first['segments'] + second['segments']}
    plans = connected.plan_legs(trajectory)
    assert plans[1]['target_indices'] == list(range(80, 103))
    assert plans[1]['reference_indices'][0] == 63
    assert plans[1]['reference_indices'][-1] == 119
    trajectory['segments'][2]['indices'].reverse()
    with pytest.raises(ValueError, match='double-split'):
        connected.plan_legs(trajectory)


def test_unsupported_lengths_and_ownership_fail_closed():
    with pytest.raises(ValueError, match='maximum is 64'):
        connected.plan_legs(trajectory_for(65))
    trajectory = trajectory_for(64)
    trajectory['segments'][1]['indices'].pop()
    with pytest.raises(ValueError, match='double-split'):
        connected.plan_legs(trajectory)
    trajectory = trajectory_for(64)
    trajectory['frames'].append({})
    with pytest.raises(ValueError, match='open tails'):
        connected.plan_legs(trajectory)


def test_options_default_and_explicit_fallbacks():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    base = {'stage': 'repair'}
    default = validate_options(base)
    assert default['connected_trajectory_generation'] and not default['handoff_generation']
    assert default['split_mode'] == 'double-split'
    assert default['connected_bridge_seeding'] == 'seed'
    assert default['artifixer_pipeline'] == 'legacy_gsplat'
    assert connected.enabled({}) is False  # Historical worker requests.
    for raw in ({'handoff_generation': True}, {'image_trajectory': 'endpoints'}, {'split_mode': 'single-split'},
                {'split_mode': 'no-split'}, {'connected_trajectory_generation': False}):
        assert not validate_options({**base, **raw})['connected_trajectory_generation']
    for raw in ({'handoff_generation': True}, {'image_trajectory': 'endpoints'}, {'split_mode': 'single-split'}):
        with pytest.raises(ValueError, match='requires double-split'):
            validate_options({**base, **raw, 'connected_trajectory_generation': True})
    for bad in ('on', 1, None):
        with pytest.raises(ValueError, match='must be a boolean'):
            validate_options({**base, 'connected_trajectory_generation': bad})
    assert 'connected_trajectory_generation' not in validate_options({'stage': 'benchmark'})


@pytest.mark.parametrize('flags,expected', [([], True), (['--no-connected-trajectory-generation'], False),
    (['--handoff-generation', 'on'], False), (['--handoff-generation', 'middle'], False),
    (['--split-mode', 'single-split'], False), (['--image-trajectory', 'endpoints'], False)])
def test_cli_fallbacks_override_config_default(tmp_path, monkeypatch, flags, expected):
    import argparse
    from splat_explorer.config import Config
    from splat_explorer.splatfix.cli import add_parser, repair_splat
    from splat_explorer.splatfix.artifixer import repair
    received = {}
    def run(checkpoint, output, **kwargs):
        received.update(kwargs)
        return {}
    monkeypatch.setattr(repair, 'run_repair', run)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(['splatfix', 'repair', str(tmp_path / 'cp'), '--output', str(tmp_path / 'out'), *flags])
    repair_splat(Config({'splatfix': {'runtime': {'connected_trajectory_generation': True}}}), args)
    assert received['runtime']['connected_trajectory_generation'] is expected


@pytest.fixture
def inference_fixture(tmp_path, monkeypatch, request):
    count = getattr(request, 'param', 64)
    class Tensor(np.ndarray):
        def permute(self, *axes): return self.transpose(axes)
        def float(self): return self.astype(np.float32)
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
    module('torch', bool=np.bool_, from_numpy=lambda x: np.asarray(x).view(Tensor),
           stack=np.stack, tensor=np.asarray, ones=lambda n, dtype: np.ones(n, dtype=dtype),
           inference_mode=contextlib.nullcontext, manual_seed=lambda value: seeds.append(value), device=lambda _: None)
    events, seeds, calls, rays = [], [], [], []
    normal_sampler, starter_sampler = object(), object()
    pipe = SimpleNamespace(frames_per_block=7, generate_samples_from_batch=normal_sampler,
        clear_inference_caches=lambda: events.append('clear'),
        transformer=SimpleNamespace(eval=lambda: SimpleNamespace(requires_grad_=lambda _: None)),
        vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    args = SimpleNamespace(inference_pipeline='kv_cache', num_inference_steps=4)
    def process(pipe, item, args, output, *rest):
        assert events[-1] == 'clear'
        indices = item['frame_indices'].tolist()
        calls.append((indices, item, output, pipe.generate_samples_from_batch))
        events.append('generate')
        assert args.num_inference_steps == 4
        dest = output / 'splatfix/frames/batch_0000/pred'
        dest.mkdir(parents=True, exist_ok=True)
        is_bridge = output.name == 'bridge'
        if is_bridge:
            assert pipe.generate_samples_from_batch is normal_sampler
            assert indices == state['plan']['target_indices']
            for slot in range(len(indices)):
                seeded = state['request']['runtime'].get('connected_bridge_seeding', 'off') != 'off' and indices[slot] in state['plan']['overlap_indices']
                assert np.all(item['opacity'][slot] == (1 if seeded else np.float32(.25)))
                expected = (100 + indices[slot] if indices[slot] in (0, count - 1) else
                            10 if indices[slot] < state['plan']['keep_indices'][0] else 20) if seeded else 1
                assert item['rgb_rendered'][slot, 0, 0, 0] == np.float32(expected / 255)
                green = 0 if seeded and indices[slot] in (0, count - 1) else indices[slot]
                assert item['rgb_rendered'][slot, 1, 0, 0] == np.float32(green / 255)

            assert item['valid_frames_mask'].all()
            refs = (rays[-1][1] if state['request']['runtime'].get('generated_context_references')
                    else state['plan']['reference_indices'])
            assert item['rgb_neighbors'].shape[0] == len(refs)
            assert item['rgb_neighbors'][:, 0, 0, 0].tolist() == pytest.approx([
                (100 + i if i in (0, count - 1) else 10 if i < state['plan']['keep_indices'][0] else 20) / 255 for i in refs])
        else:
            seeded = (state['request']['runtime'].get('initial_image_seeding') is not False
                      or state['request']['runtime'].get('image_cache_insertion', False))
            assert indices in [side if seeded else side[1:] for side in state['plan']['outer_generation_indices']]
            assert bool(item['valid_frames_mask'][0]) is (not seeded)
            assert np.all(item['opacity'][0] == (1 if seeded else .25))
        for i, valid in zip(indices, item['valid_frames_mask']):
            if valid and not (is_bridge and state['missing_frame'] == i):
                color = 200 if is_bridge else 10 if indices[0] < count // 2 else 20
                Image.new('RGB', (8, 8), (color, i, 0)).save(dest / f'{i:05d}.png')
        pipe.starter_inference_audit = {'encoded_starter_preserved': True}
    module('model_eval')
    module('model_eval.run_inference', process_item=process,
           build_parser=lambda: SimpleNamespace(parse_args=lambda _: args), get_eval_pipe=lambda *a: pipe)
    module('model_eval.checkpoint_loading', load_transformer_checkpoint=lambda *a: None)
    module('model_training')
    module('model_training.data')
    def conditioning(cameras, indices, neighbors, **kw):
        rays.append((list(indices), list(neighbors), cameras, kw))
        return {}
    module('model_training.data.utils', compute_camera_rays=conditioning, load_encoded_prompt=lambda _: (None, None))
    monkeypatch.setattr(official_worker, 'validate_caption', lambda *a: None)
    monkeypatch.setattr(official_worker, 'install_image_cache_insertion',
                        lambda p: setattr(p, 'generate_samples_from_batch', starter_sampler))
    trajectory = trajectory_for(count)
    poses = []
    for i, frame in enumerate(trajectory['frames']):
        Image.new('RGB', (8, 8), (1, i, 0)).save(tmp_path / f'{i}.png')
        np.save(tmp_path / f'{i}.npy', np.full((8, 8), .25, dtype=np.float32))
        frame.update(rgb=f'{i}.png', opacity=f'{i}.npy')
        matrix = np.eye(4); matrix[0, 3] = i
        poses.append({'transform_matrix': matrix.tolist()})
    trajectory['transforms'] = {'w': 8, 'h': 8, 'fl_x': 8, 'fl_y': 8, 'cx': 4, 'cy': 4, 'frames': poses}
    references = []
    for i in [count - 1, 0]:
        path = tmp_path / f'ref-{i}.png'
        Image.new('RGB', (8, 8), (100 + i, 0, 0)).save(path)
        references.append(str(path))
    request = {'checkpoint_root': str(tmp_path), 'references': references, 'camera_scale': 2.5,
               'caption_path': 'caption.pt', 'caption_sha256': '', 'seed': 42, 'trajectory_mode': 'authors_orbit',
               'repair_policy': 'all', 'runtime': {'connected_trajectory_generation': True,
                   'checkpoint': 'current-causal.pt', 'model_id': 'test', 'split_mode': 'double-split'}}
    state = dict(plan=connected.plan_legs(trajectory)[0], count=count, root=tmp_path, pipe=pipe, args=args, request=request, trajectory=trajectory,
                 normal_sampler=normal_sampler, starter_sampler=starter_sampler, calls=calls, rays=rays,
                 events=events, seeds=seeds, missing_frame=None)
    return state


@pytest.mark.parametrize('inference_fixture', [3, 5, 12, 64], indirect=True)
@pytest.mark.parametrize('bridge_mode', ['off', 'seed'])
def test_generated_context_pins_tail_and_overlaps_with_fresh_caches(inference_fixture, bridge_mode):
    f = inference_fixture
    f['request']['runtime'].update(generated_context_references=True, connected_bridge_seeding=bridge_mode)
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    rows = json.loads((f['root'] / 'inference-context.json').read_text())['sequences']
    assert not rows[0]['generated_references']
    assert all(row['complete'] for row in rows)
    if len(f['plan']['outer_generation_indices']) == 2:
        tail = f['plan']['outer_generation_indices'][0][1:][-2:]
        assert [r['frame_index'] for r in rows[1]['generated_references'] if r['reason'] == 'boundary'] == tail
    bridge = rows[-1]
    assert bridge['base_reference_indices'][:2] == [f['count'] - 1, 0]
    assert set(f['plan']['reference_indices']) <= set(bridge['base_reference_indices'])
    assert set(f['plan']['overlap_indices']) <= set(bridge['reference_indices'])
    assert set(f['plan']['keep_indices']).isdisjoint(bridge['reference_indices'])
    assert len(bridge['generated_references']) <= 2
    assert f['events'] == ['clear', 'generate', 'clear'] * len(rows)
    assert [r[1] for r in f['rays']] == [r['reference_indices'] for r in rows]
    for row, (_, item, _, _) in zip(rows, f['calls']):
        for record in row['generated_references']:
            index = record['frame_index']
            pixel = np.asarray(Image.open(record['path']))[0, 0] / 255
            assert item['rgb_neighbors'][row['reference_indices'].index(index), :, 0, 0] == pytest.approx(pixel)
    connected.validate_completed(f['root'], f['request'], f['trajectory'])
    f['request']['runtime']['generated_context_references'] = False
    with pytest.raises(ValueError, match='settings mismatch'):
        connected.validate_completed(f['root'], f['request'], f['trajectory'])


def test_generated_context_across_spans_keeps_all_repairs_and_published_middle(inference_fixture, monkeypatch):
    f = inference_fixture
    first, second = trajectory_for(32), trajectory_for(33, offset=31)
    trajectory = f['trajectory']
    trajectory.update(anchors=[{'frame_index': i} for i in [63, 0, 31]],
                      segments=first['segments'] + second['segments'])
    path = f['root'] / 'ref-31.png'
    Image.new('RGB', (8, 8), (131, 0, 0)).save(path)
    f['request']['references'].append(str(path))
    f['request']['runtime']['generated_context_references'] = True
    def process(pipe, item, args, output, *rest):
        assert f['events'][-1] == 'clear'
        f['events'].append('generate')
        folder = output / 'splatfix/frames/batch_0000/pred'
        folder.mkdir(parents=True)
        for index, valid in zip(item['frame_indices'], item['valid_frames_mask']):
            if valid:
                Image.new('RGB', (8, 8), (200 if output.name == 'bridge' else 10, int(index), 0)).save(folder / f'{index:05d}.png')
    monkeypatch.setattr(sys.modules['model_eval.run_inference'], 'process_item', process)
    official_worker.inference(f['root'], f['request'], trajectory)
    manifest = json.loads((f['root'] / 'inference-context.json').read_text())
    rows = manifest['sequences']
    assert manifest['complete'] and len(rows) == 6
    assert all(row['base_reference_indices'][:3] == [63, 0, 31] for row in rows)
    assert rows[3]['generated_references']
    prior = set(range(1, 31))
    for record in rows[3]['generated_references']:
        assert record['frame_index'] in prior
        assert record['source_sequence'] < 3
        if record['frame_index'] in connected.plan_legs(trajectory)[0]['keep_indices']:
            assert record['source_sequence'] == 2
            assert Image.open(record['path']).getpixel((0, 0))[0] == 200
    # A retry starts empty, even with the entire previous result on disk.
    official_worker.inference(f['root'], f['request'], trajectory)
    retry = json.loads((f['root'] / 'inference-context.json').read_text())
    assert retry['sequences'][0]['generated_references'] == []


@pytest.mark.parametrize('endpoint', [False, True])
def test_ordinary_sequences_extend_camera_and_rgb_references_together(inference_fixture, monkeypatch, endpoint):
    f = inference_fixture
    f['request']['runtime'].update(connected_trajectory_generation=False, generated_context_references=True)
    if endpoint:
        f['request']['repair_policy'] = 'endpoints'
        f['trajectory']['anchors'].reverse()
        f['request']['references'].reverse()
    calls = []
    def process(pipe, item, args, output, *rest):
        assert f['events'][-1] == 'clear'
        calls.append(item)
        folder = output / 'splatfix/frames/batch_0000/pred'
        folder.mkdir(parents=True)
        for index, valid in zip(item['frame_indices'], item['valid_frames_mask']):
            if valid:
                Image.new('RGB', (8, 8), (10, int(index), 0)).save(folder / f'{index:05d}.png')
    monkeypatch.setattr(sys.modules['model_eval.run_inference'], 'process_item', process)
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    rows = json.loads((f['root'] / 'inference-context.json').read_text())['sequences']
    assert not rows[0]['generated_references'] and rows[1]['generated_references']
    assert len(calls[0]['rgb_neighbors']) == 2 and len(calls[1]['rgb_neighbors']) == 8
    assert f['rays'][1][1] == rows[1]['reference_indices']
    assert all(r['source_sequence'] == 0 for r in rows[1]['generated_references'])


@pytest.mark.parametrize('bridge_mode', [None, 'off', 'seed'])
@pytest.mark.parametrize('insertion', [False, True])
def test_worker_integrates_fresh_bridge_and_preserves_outer_legs(inference_fixture, insertion, bridge_mode):
    f = inference_fixture
    f['request']['runtime']['image_cache_insertion'] = insertion
    if bridge_mode is not None:
        f['request']['runtime']['connected_bridge_seeding'] = bridge_mode
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    assert len(f['calls']) == 3
    assert f['events'] == ['clear', 'generate', 'clear'] * 3
    assert f['seeds'] == [42, 42, 43, 44]
    assert f['pipe'].generate_samples_from_batch is (f['starter_sampler'] if insertion else f['normal_sampler'])
    indices, neighbors, cameras, kw = f['rays'][-1]
    assert indices == list(range(19, 44))
    assert neighbors == [0, 8, 14, 18, 19, 20, 42, 43, 44, 48, 55, 63]
    assert cameras == official_worker.opengl_transforms(f['trajectory'])
    assert kw['scale'] == 2.5
    prediction = f['root'] / 'inference/splatfix/frames/batch_0000/pred'
    assert sorted(int(p.stem) for p in prediction.glob('*.png')) == list(range(1, 63))
    for i in range(1, 63):
        assert Image.open(prediction / f'{i:05d}.png').getpixel((0, 0))[0] == (10 if i < 21 else 200 if i <= 41 else 20)
    manifest = json.loads((f['root'] / 'inference-connected.json').read_text())
    assert manifest['complete']
    assert manifest['bridge_seeding'] == (bridge_mode or 'off')
    assert manifest['legs'][0]['seeding']['seed_frame_indices'] == ([19, 20, 42, 43] if bridge_mode == 'seed' else [])
    assert len(manifest['legs'][0]['overlaps']) == 4
    assert manifest['legs'][0]['boundary_mae']['left'] > 0
    assert all((f['root'] / row['comparison']).is_file() for row in manifest['legs'][0]['overlaps'])
    transforms, supervision = official_worker.reconstruction_supervision(f['root'], f['request'], f['trajectory'])
    assert supervision['selected_indices'] == [0, 63]


def test_incomplete_bridge_never_publishes_partial_center(inference_fixture):
    f = inference_fixture
    f['missing_frame'] = 31
    with pytest.raises(RuntimeError, match='exactly 25 real'):
        official_worker.inference(f['root'], f['request'], f['trajectory'])
    manifest = json.loads((f['root'] / 'inference-connected.json').read_text())
    assert not manifest['complete']
    prediction = f['root'] / 'inference/splatfix/frames/batch_0000/pred'
    assert not (prediction / '00031.png').exists()
    assert Image.open(prediction / '00021.png').getpixel((0, 0))[0] == 10
    assert f['pipe'].generate_samples_from_batch is f['normal_sampler']
    assert f['events'][-1] == 'clear'


def test_bridge_exception_restores_starter_sampler_and_clears_caches(inference_fixture, monkeypatch):
    f = inference_fixture
    f['request']['runtime']['image_cache_insertion'] = True
    process = sys.modules['model_eval.run_inference'].process_item
    def fail_bridge(pipe, item, args, output, *rest):
        if output.name == 'bridge':
            raise RuntimeError('test sampler failure')
        process(pipe, item, args, output, *rest)
    monkeypatch.setattr(sys.modules['model_eval.run_inference'], 'process_item', fail_bridge)
    with pytest.raises(RuntimeError, match='test sampler failure'):
        official_worker.inference(f['root'], f['request'], f['trajectory'])
    assert f['pipe'].generate_samples_from_batch is f['starter_sampler']
    assert f['events'][-1] == 'clear'
    assert not json.loads((f['root'] / 'inference-connected.json').read_text())['complete']


def test_overlap_metrics_are_finite_json_and_do_not_modify_inputs(tmp_path):
    a, b, comparison = (tmp_path / name for name in ('a.png', 'b.png', 'comparison.png'))
    Image.new('RGB', (8, 8), (0, 0, 0)).save(a)
    Image.new('RGB', (8, 8), (255, 255, 255)).save(b)
    before = [connected.digest(p) for p in (a, b)]
    assert connected.compare_overlap(a, b, comparison) == {'mae': 1, 'rmse': 1, 'psnr_db': 0, 'identical': False}
    metrics = connected.compare_overlap(a, a, comparison)
    assert metrics == {'mae': 0, 'rmse': 0, 'psnr_db': None, 'identical': True}
    json.dumps(metrics, allow_nan=False)
    assert before == [connected.digest(p) for p in (a, b)]
    assert Image.open(comparison).size == (24, 8)


def test_diagnostics_expose_both_sides_and_reject_external_images(tmp_path):
    from splat_explorer.web.connected_details import connected_details
    root = tmp_path / 'result'
    root.mkdir()
    Image.new('RGB', (8, 8)).save(root / 'overlap.png')
    Image.new('RGB', (8, 8)).save(tmp_path / 'outside.png')
    manifest = root / 'inference-preview.json'
    manifest.write_text('{}')
    (root / 'inference-connected.json').write_text(json.dumps({'complete': True, 'legs': [
        {'leg_index': 0, 'status': 'complete', 'boundary_mae': {'left': .1, 'right': .2},
         'overlaps': [{'frame_index': 19, 'side': 'left', 'mae': .1, 'comparison': 'overlap.png'},
                      {'frame_index': 43, 'side': 'right', 'mae': .2, 'comparison': '../outside.png'}]}]}))
    result = connected_details(SimpleNamespace(file_url=lambda p: '/safe/' + p.name), manifest)
    assert result['enabled']
    assert result['passes'][0]['complete']
    leg, = result['passes'][0]['legs']
    assert leg['boundary_mae'] == {'left': .1, 'right': .2}
    assert len(leg['overlaps']) == 1
    assert leg['overlaps'][0]['url'] == '/safe/overlap.png'


@pytest.mark.parametrize('bridge_mode', ['off', 'seed'])
@pytest.mark.parametrize('generated_context', [False, True])
def test_plus_uses_its_own_renders_and_generated_references(inference_fixture, monkeypatch, bridge_mode, generated_context):
    f = inference_fixture
    f['request']['runtime']['connected_bridge_seeding'] = bridge_mode
    f['request']['runtime']['generated_context_references'] = generated_context
    # Use an exact 0.25 float from RGB's channel read for this contract fixture.
    # PNG opacity is quantized, so assert the real expected 64/255 separately.
    root = f['root']
    for folder in ('renders', 'opacity'):
        (root / folder).mkdir()
        for i in range(64):
            color = (1, i, 0) if folder == 'renders' else (64, 64, 64)
            Image.new('RGB', (8, 8), color).save(root / folder / f'{i:05d}.png')
    (root / 'distillation.json').write_text(json.dumps({'render_dir': str(root)}))
    process = sys.modules['model_eval.run_inference'].process_item
    def plus_process(pipe, item, *args):
        mask = item['opacity'] != 1
        assert np.all(item['opacity'][mask] == np.float32(64/255))
        item['opacity'][mask] = .25
        process(pipe, item, *args)
    monkeypatch.setattr(sys.modules['model_eval.run_inference'], 'process_item', plus_process)
    connected.run(root, f['request'], f['trajectory'], pipe=f['pipe'], args=f['args'], device=None,
                  encoded_prompt=None, cameras=official_worker.opengl_transforms(f['trajectory']),
                  normal_sampler=f['normal_sampler'], plus=True)
    assert json.loads((root / 'plus-connected.json').read_text())['complete']
    assert not (root / 'inference').exists()
    if generated_context:
        context = json.loads((root / 'plus-context.json').read_text())
        assert context['complete'] and context['phase'] == 'plus'
        assert not context['sequences'][0]['generated_references']
        assert all('/plus/' in r['source_path'] for row in context['sequences'] for r in row['generated_references'])


def test_explicit_context_off_matches_historical_outputs_and_manifest(inference_fixture):
    f = inference_fixture
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    files = [f['root'] / 'inference-connected.json', f['root'] / 'inference-seeding.json',
             *sorted((f['root'] / 'inference/splatfix/frames/batch_0000/pred').glob('*.png'))]
    before = [p.read_bytes() for p in files]
    f['request']['runtime']['generated_context_references'] = False
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    assert [p.read_bytes() for p in files] == before
    assert not (f['root'] / 'inference-context.json').exists()


@pytest.mark.parametrize('inference_fixture', [3, 4, 5, 6, 12, 19, 25, 30, 48, 57, 63], indirect=True)
@pytest.mark.parametrize('bridge_mode', ['off', 'seed'])
def test_short_bridges_keep_real_ownership_and_matching_padded_cameras(inference_fixture, bridge_mode):
    f = inference_fixture
    f['request']['runtime']['connected_bridge_seeding'] = bridge_mode
    anchors_before = [connected.digest(path) for path in f['request']['references']]
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    plan = f['plan']
    assert len(f['calls']) == len(plan['outer_generation_indices']) + 1
    real = f['calls'][-1][0]
    assert real == plan['target_indices']
    assert f['rays'][-1][0] == real + [real[-1]] * (25 - len(real))
    prediction = f['root'] / 'inference/splatfix/frames/batch_0000/pred'
    assert sorted(int(p.stem) for p in prediction.glob('*.png')) == list(range(1, f['count'] - 1))
    for i in range(1, f['count'] - 1):
        expected = 200 if i in plan['keep_indices'] else 10 if i < plan['keep_indices'][0] else 20
        assert Image.open(prediction / f'{i:05d}.png').getpixel((0, 0))[0] == expected
    manifest = json.loads((f['root'] / 'inference-connected.json').read_text())
    assert manifest['complete']
    leg, = manifest['legs']
    assert [r['frame_index'] for r in leg['overlaps']] == plan['overlap_indices']
    assert [r['side'] for r in leg['overlaps']] == ['left'] * len(plan['overlap_sides']['left']) + ['right'] * len(plan['overlap_sides']['right'])
    bridge_dir = f['root'] / 'inference-connected/leg-000/bridge/splatfix/frames/batch_0000/pred'
    assert sorted(int(p.stem) for p in bridge_dir.glob('*.png')) == real
    assert [connected.digest(path) for path in f['request']['references']] == anchors_before


def test_loop_bridge_and_references_follow_wrapped_trajectory_order():
    trajectory = {'frames': [{}] * 10, 'anchors': [{'frame_index': 2}, {'frame_index': 7}],
                  'recipe': {'orbit': {'provenance': {'loop': True}}},
                  'segments': [{'seed_index': 2, 'indices': [3, 4]}, {'seed_index': 7, 'indices': [6, 5]},
                               {'seed_index': 7, 'indices': [8, 9]}, {'seed_index': 2, 'indices': [1, 0]}]}
    plans = connected.plan_legs(trajectory)
    assert plans[1]['target_indices'] == [7, 8, 9, 0, 1, 2]
    assert plans[1]['keep_indices'] == [9, 0]
    assert plans[1]['overlap_sides'] == {'left': [7, 8], 'right': [1, 2]}
    assert plans[1]['reference_indices'] == [7, 8, 1, 2]
    assert plans[1]['padding_count'] == 19


@pytest.mark.parametrize('inference_fixture', [2], indirect=True)
def test_no_generation_without_interior_and_endpoints_preserved(inference_fixture):
    f = inference_fixture
    before = [connected.digest(p) for p in f['request']['references']]
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    assert f['calls'] == []
    assert [connected.digest(p) for p in f['request']['references']] == before
    manifest = json.loads((f['root'] / 'inference-connected.json').read_text())
    assert manifest['complete'] and manifest['legs'][0]['status'] == 'skipped'


@pytest.mark.parametrize('inference_fixture', [3, 12, 30, 64], indirect=True)
def test_stronger_mode_is_scoped_to_bridge_and_records_dynamic_seed_layout(inference_fixture, monkeypatch):
    f = inference_fixture
    f['request']['runtime'].update(connected_bridge_seeding='seed_preserve', image_cache_insertion=True)
    calls = []
    @contextlib.contextmanager
    def preserve(pipe, normal, *, real_frame_count, overlap_slots):
        assert pipe.generate_samples_from_batch is normal
        calls.append((real_frame_count, overlap_slots))
        # Actual tensor and per-step cache preservation is covered by CPU tests.
        yield {'seed_kv_preserved': True, 'seed_latents_preserved': True}
    monkeypatch.setattr(connected.bridge_seeding, 'preserved_bridge', preserve)
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    plan = f['plan']
    slots = [plan['target_indices'].index(i) for i in plan['overlap_indices']]
    assert calls == [(len(plan['target_indices']), slots)]
    assert f['pipe'].generate_samples_from_batch is f['starter_sampler']
    manifest = json.loads((f['root'] / 'inference-connected.json').read_text())
    seed = manifest['legs'][0]['seeding']
    assert seed['seed_frame_indices'] == plan['overlap_indices']
    assert seed['seed_kv_preserved']
    assert set(seed['constrained_rgb_frame_indices']) <= set(plan['target_indices'])
    assert seed['seed_latent_indices'] == sorted({(i + 3) // 4 for i in slots})


@pytest.mark.parametrize('inference_fixture', [3, 12, 30, 48], indirect=True)
def test_short_outer_cameras_match_final_rgb_padding(inference_fixture):
    f = inference_fixture
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    for (indices, _, _, _), (camera_ids, _, _, _) in zip(f['calls'], f['rays']):
        assert camera_ids == indices + [indices[-1]] * (25 - len(indices))


@pytest.mark.parametrize('value', [None, False, True])
def test_outer_seeding_off_and_historical_none_keep_existing_semantics(inference_fixture, value):
    f = inference_fixture
    f['request']['runtime']['initial_image_seeding'] = value
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    first = f['plan']['outer_generation_indices'][0]
    assert f['calls'][0][0] == (first[1:] if value is False else first)
    audit = json.loads((f['root'] / 'inference-seeding.json').read_text())
    assert audit['series'][0]['enabled'] is (value is not False)


def test_stale_outer_preservation_audit_cannot_pass_next_leg(inference_fixture, monkeypatch):
    f = inference_fixture
    f['request']['runtime']['image_cache_insertion'] = True
    process = sys.modules['model_eval.run_inference'].process_item
    def missing_audit(pipe, *args):
        previous = getattr(pipe, 'starter_inference_audit', None)
        process(pipe, *args)
        if len(f['calls']) == 2:
            pipe.starter_inference_audit = previous
    monkeypatch.setattr(sys.modules['model_eval.run_inference'], 'process_item', missing_audit)
    with pytest.raises(RuntimeError, match='starter preservation'):
        official_worker.inference(f['root'], f['request'], f['trajectory'])
    assert len(f['calls']) == 2  # No bridge on an unverified outer generation.


@pytest.mark.parametrize('inference_fixture', [30], indirect=True)
def test_copy_failure_cannot_publish_half_a_bridge(inference_fixture, monkeypatch):
    f = inference_fixture
    copy = connected.shutil.copyfile
    def fail_copy(source, dest, *args, **kwargs):
        if str(Path(dest).parent.name).startswith('.connected-publish-') and Path(dest).name == '00011.png':
            raise OSError('simulated full disk')
        return copy(source, dest, *args, **kwargs)
    monkeypatch.setattr(connected.shutil, 'copyfile', fail_copy)
    with pytest.raises(OSError, match='full disk'):
        official_worker.inference(f['root'], f['request'], f['trajectory'])
    root = f['root'] / 'inference/splatfix/frames/batch_0000/pred'
    assert Image.open(root / '00010.png').getpixel((0, 0))[0] == 10
    assert Image.open(root / '00011.png').getpixel((0, 0))[0] == 10
    with pytest.raises(ValueError, match='completion'):
        official_worker.distill(f['root'], f['request'], f['trajectory'])


@pytest.mark.parametrize('inference_fixture', [30], indirect=True)
def test_interrupted_publish_with_full_frame_count_cannot_be_reconstructed(inference_fixture, monkeypatch):
    f = inference_fixture
    replace = Path.replace
    def interrupt(source, target):
        if source.parent.name.startswith('.connected-publish-') and source.name == '00011.png':
            raise InterruptedError('simulated interruption')
        return replace(source, target)
    monkeypatch.setattr(Path, 'replace', interrupt)
    with pytest.raises(InterruptedError):
        official_worker.inference(f['root'], f['request'], f['trajectory'])
    root = f['root'] / 'inference/splatfix/frames/batch_0000/pred'
    assert len(list(root.glob('*.png'))) == 28
    assert Image.open(root / '00010.png').getpixel((0, 0))[0] == 200
    with pytest.raises(ValueError, match='completion'):
        official_worker.reconstruction_supervision(f['root'], f['request'], f['trajectory'])


@pytest.mark.parametrize('change', ['missing_manifest', 'unfinished_leg', 'missing_frame', 'extra_frame',
                                   'changed_frame', 'reference', 'camera', 'scale', 'mode', 'seed', 'outer_seeding'])
def test_completed_bridge_is_bound_to_actual_outputs_and_settings(inference_fixture, change):
    f = inference_fixture
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    connected.validate_completed(f['root'], f['request'], f['trajectory'])
    path = f['root'] / 'inference-connected.json'
    prediction = f['root'] / 'inference/splatfix/frames/batch_0000/pred'
    if change == 'missing_manifest': path.unlink()
    elif change == 'unfinished_leg':
        manifest = json.loads(path.read_text()); manifest['legs'][0]['status'] = 'pending'
        path.write_text(json.dumps(manifest))
    elif change == 'missing_frame': (prediction / '00031.png').unlink()
    elif change == 'extra_frame': Image.new('RGB', (8, 8)).save(prediction / '00099.png')
    elif change == 'changed_frame': Image.new('RGB', (8, 8)).save(prediction / '00031.png')
    elif change == 'reference': Image.new('RGB', (8, 8)).save(f['request']['references'][0])
    elif change == 'camera': f['trajectory']['transforms']['frames'][31]['transform_matrix'][0][3] += 1
    elif change == 'scale': f['request']['camera_scale'] *= 2
    elif change == 'mode': f['request']['runtime']['connected_bridge_seeding'] = 'seed_preserve'
    elif change == 'seed': f['request']['seed'] += 1
    elif change == 'outer_seeding': f['request']['runtime']['initial_image_seeding'] = False
    with pytest.raises(ValueError, match='Connected inference'):
        official_worker.reconstruction_supervision(f['root'], f['request'], f['trajectory'])


def test_retry_cannot_fill_a_missing_new_output_from_old_bridge(inference_fixture):
    f = inference_fixture
    official_worker.inference(f['root'], f['request'], f['trajectory'])
    f['missing_frame'] = 31
    with pytest.raises(RuntimeError, match='exactly 25 real'):
        official_worker.inference(f['root'], f['request'], f['trajectory'])
    assert not json.loads((f['root'] / 'inference-connected.json').read_text())['complete']
