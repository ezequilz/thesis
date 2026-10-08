"""Starter conditioning and temporal-cache regressions (small CPU tensors)."""
from types import SimpleNamespace
import pytest
from splat_explorer.scene_runs_ext.starter_inference import (
    generate_from_starter, install_starter_inference, latent_chunks, starter_reference,
)


def test_starter_must_be_edited_and_at_exact_segment_camera():
    manifest = {'references': [{'path': 'anchor.png', 'frame_index': 0, 'kind': 'edited_render'},
                               {'path': 'other.png', 'frame_index': 25, 'kind': 'edited_render'}]}
    assert starter_reference(manifest, {'start': 25})['path'] == 'other.png'
    with pytest.raises(ValueError, match='calibrated'):
        starter_reference(manifest, {'start': 1})
    manifest['references'][1]['kind'] = 'render'
    with pytest.raises(ValueError, match='GPT-edited'):
        starter_reference(manifest, {'start': 25})


@pytest.mark.parametrize('total', [3, 7, 14, 21])
def test_single_starter_then_complete_nonoverlapping_chunks(total):
    chunks = list(latent_chunks(total, 7))
    assert chunks[0] == (0, 1)
    assert [i for a, b in chunks for i in range(a, b)] == list(range(total))


@pytest.mark.parametrize('cache_policy', ['clean', 'last_denoising'])
@pytest.mark.parametrize('source_alpha', [0., .5])
@pytest.mark.parametrize('window', [-1, 21])
@pytest.mark.parametrize('total', [14, 35])
def test_clean_starter_drives_future_frames_and_cache_resets(cache_policy, source_alpha, window, total):
    torch = pytest.importorskip('torch')
    calls = []
    prepared = []
    allocations = []
    fixed_reference = torch.tensor([17.])
    class Transformer:
        patch_size = (1, 1, 1)
        _cp_world_size = 1
        def __call__(self, **kw):
            assert kw['neighbor_hidden_states'] is fixed_reference
            assert kw['ignore_neighbors'] is False
            start = kw['frame_offset']
            value = kw['hidden_states']
            calls.append((start, value.clone(), kw['timestep'].clone(), kw['opacity'].shape[1]))
            assert kw['current_start'] == start * 4
            assert kw['camera_rays'][0, 0, 0] == start
            assert kw['w2cs'][0, 0, 0] == start
            cache = kw['kv_cache']
            if start == 0:
                assert not cache
                assert torch.count_nonzero(kw['timestep']) == 0
                cache['starter'] = value.clone()
            else:
                assert 'starter' in cache, 'Future frames must see clean starter context'
                assert torch.all(kw['opacity'] == source_alpha)
            return (torch.ones_like(value) * cache['starter'].mean(),)
    class Pipe:
        frames_per_block = 7
        local_attn_size = window
        sink_size = 1
        vae = SimpleNamespace(device='cpu', config=SimpleNamespace(scale_factor_temporal=4))
        transformer = Transformer()
        def generate_samples_from_batch(self): return "upstream"
        scheduler = SimpleNamespace(step=lambda noise, *a, **kw: noise,
                                    add_noise=lambda clean, noise, t: clean + noise)
        def _initialize_kv_cache(self, batch, tokens, frames):
            allocations.append((batch, tokens, frames))
            self.kv_cache1 = {}
        def _initialize_crossattn_cache(self, name): setattr(self, name, {})
        def create_denoising_step_list(self, steps): return torch.tensor([1000., 500.])
        def prepare_latents(self, condition, opacity, first):
            assert not first and torch.all(opacity == source_alpha)
            prepared.append(condition.clone())
            return condition * source_alpha + torch.randn_like(condition) * (1-source_alpha)
    pipe = Pipe()
    if cache_policy == 'clean':
        from splat_explorer.splatfix.artifixer.official_worker import install_image_cache_insertion
        install_image_cache_insertion(pipe)
    else:
        install_starter_inference(pipe, generated_cache=cache_policy)
    outputs = []
    with torch.inference_mode():
        for starter in [2., 5.]:
            # Later condition values deliberately differ: they must be ignored.
            condition = torch.full((1, 1, total, 2, 2), 99.)
            condition[:, :, 0] = starter
            opacity = torch.zeros(1, 1 + (total - 1) * 4, 2, 2); opacity[:, 0] = 1
            opacity[:, 1:] = source_alpha
            poses = torch.arange(total).reshape(1, total, 1)
            outputs.append(pipe.generate_samples_from_batch(condition, opacity,
                fixed_reference, poses, poses, torch.ones(1), poses, torch.ones(1),
                torch.ones(1), 2, False))
    assert torch.all(outputs[0] == 2.) and torch.all(outputs[1] == 5.)
    chunks = list(latent_chunks(total, 7))
    assert allocations == [(1, 4, total if window == -1 else min(total, window))] * 2
    assert len(prepared) == 4 * (len(chunks) - 1), 'Initial and intermediate samples must both use opacity mixing'
    assert all(torch.all(value == 99.) for value in prepared)
    repeats = 3 if cache_policy == 'clean' else 2
    expected_offsets = [0] + [start for start, end in chunks[1:] for _ in range(repeats)]
    expected_rgb_counts = [1] + [(end - start) * 4 for start, end in chunks[1:] for _ in range(repeats)]
    assert [c[0] for c in calls] == expected_offsets * 2
    assert [c[3] for c in calls] == expected_rgb_counts * 2
    assert sum(c[2].count_nonzero() == 0 for c in calls) == (2 * len(chunks) if cache_policy == 'clean' else 2)


def test_only_gpt_starter_has_rgb_and_observation_opacity():
    torch = pytest.importorskip('torch')
    from splat_explorer.scene_runs_ext.starter_inference import starter_inputs
    seed = torch.rand(3, 32, 32)
    rgb, opacity = starter_inputs(seed, 25)
    assert torch.equal(rgb[0], seed)
    assert rgb[1:].count_nonzero() == 0
    assert torch.all(opacity[0] == 1)
    assert opacity[1:].count_nonzero() == 0
    rgb[0].zero_()
    assert seed.count_nonzero() > 0, 'Conditioning must not mutate reference images'


@pytest.mark.parametrize('schedule', ['exact_starter', 'periodic_starter', 'upstream'])
@pytest.mark.parametrize('source_mode', ['none', 'rendered'])
def test_bridge_generates_from_each_edit_without_reading_scene_renders(tmp_path, monkeypatch, source_mode, schedule):
    import json
    import numpy as np
    import sys
    from types import ModuleType
    from PIL import Image
    torch = pytest.importorskip('torch')
    from splat_explorer.scene_runs_ext import artifixer_bridge
    count = 45 if schedule == 'periodic_starter' else 9
    for i, value in enumerate([40, 220]):
        Image.new('RGB', (32, 32), (value, value, value)).save(tmp_path / f'edit-{i}.png')
    manifest = {'options': {'seed': 42, 'inference_steps': 4, 'camera_scale': 1., 'source_conditioning': source_mode, 'block_schedule': schedule, 'generated_cache': 'clean'},
                'transforms': {'frames': [{}] * (2*count)},
                'references': [{'path': f'edit-{i}.png', 'frame_index': count*i,
                                'kind': 'edited_render'} for i in range(2)],
                'segments': [{'start': 0, 'count': count}, {'start': count, 'count': count}]}
    (tmp_path / 'bundle.json').write_text(json.dumps(manifest))
    if source_mode == 'rendered':
        (tmp_path / 'inputs').mkdir()
        for i in range(2*count):
            Image.new('RGB', (32,32), (100,100,100)).save(tmp_path / 'inputs' / f'{i:05d}.png')
        np.save(tmp_path / 'opacity.npy', np.full((2*count,32,32), .7, dtype=np.float32))
    # There is intentionally no inputs directory and no opacity.npy.
    parsed = []
    class Parser:
        def parse_args(self, args): parsed.extend(args); return SimpleNamespace()
    class Pipe:
        transformer = SimpleNamespace(eval=lambda: None)
        vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))
        def generate_samples_from_batch(self): return "upstream"
        _initialize_kv_cache = None
        def clear_inference_caches(self): self.cleared = True
    pipe = Pipe()
    seen = []
    prepared = []
    from splat_explorer.scene_runs_ext.periodic_refresh import PeriodicRefresh
    def prepare(refresh):
        refresh.prepared_references = []
        for offset in range(20, len(refresh.indices) - 1, 20):
            i = refresh.indices[offset]
            folder = tmp_path / 'refresh' / f'{i:05d}'
            folder.mkdir(parents=True)
            Image.new('RGB', (32,32), (230,230,230)).save(folder / 'fixed.png')
            refresh.prepared_references.append({'frame_index': i, 'path': f'refresh/{i:05d}/fixed.png', 'kind': 'edited_render'})
            prepared.append(i)
        return refresh.prepared_references
    monkeypatch.setattr(PeriodicRefresh, 'prepare', prepare)
    def get_pipe(*args):
        assert prepared == ([20, 40, 65, 85] if schedule == 'periodic_starter' else [])
        return pipe
    monkeypatch.setattr(artifixer_bridge, 'load_eval_pipe', get_pipe)
    def process(pipe, item, opts, output, rank, device, scale):
        assert pipe.cleared
        pipe.cleared = False
        if schedule != 'upstream':
            assert pipe.generate_samples_from_batch.__func__ is generate_from_starter
        else:
            assert pipe.generate_samples_from_batch.__func__ is Pipe.generate_samples_from_batch
            assert not hasattr(pipe, 'starter_generated_cache')
        assert torch.allclose(item['rgb_neighbors'][0], torch.full((3,32,32), 40/255))
        assert torch.allclose(item['rgb_neighbors'][1], torch.full((3,32,32), 220/255))
        index = len(seen)
        starts = [0, 20, 40, 45, 65, 85] if schedule == 'periodic_starter' else [0, 9]
        values = [40, 230, 230, 220, 230, 230] if schedule == 'periodic_starter' else [40, 220]
        assert torch.allclose(item['rgb_rendered'][0], torch.full((3,32,32), values[index]/255))
        remaining = len(item['frame_indices']) - 1
        if source_mode == 'none':
            assert item['rgb_rendered'][1:].count_nonzero() == 0
            assert item['opacity'][1:].count_nonzero() == 0
        else:
            assert torch.allclose(item['rgb_rendered'][1:], torch.full((remaining,3,32,32), 100/255))
            assert torch.allclose(item['opacity'][1:], torch.full((remaining,32,32), .7))
        assert torch.all(item['opacity'][0] == 1)
        assert item['frame_indices'][0] == starts[index]
        assert item['camera_rays'].flatten().tolist() == item['frame_indices'].tolist(), 'Recompute target cameras for every restart'
        seen.append(item)
        dest = output / 'bundle/frames/batch_0000/pred'
        dest.mkdir(parents=True, exist_ok=True)
        for i in item['frame_indices'].tolist():
            Image.new('RGB', (32,32)).save(dest / f'{i:05d}.png')
        if schedule == 'periodic_starter':
            assert not hasattr(pipe, 'starter_refresh'), 'No mid-video latent splice is allowed'
            assert prepared == [20, 40, 65, 85]
            assert item['neighbor_w2cs'].flatten().tolist() == [0, 45, 20, 40, 65, 85]
            assert item['neighbor_Ks'].flatten().tolist() == [100, 145, 120, 140, 165, 185]
            assert item['rgb_neighbors'].shape[0] == 6
            assert torch.allclose(item['rgb_neighbors'][2:], torch.full((4,3,32,32), 230/255))
    inference = ModuleType('model_eval.run_inference')
    inference.build_parser = Parser
    inference.get_eval_pipe = get_pipe
    inference.process_item = process
    checkpoints = ModuleType('model_eval.checkpoint_loading')
    checkpoints.load_transformer_checkpoint = lambda *a: None
    utils = ModuleType('model_training.data.utils')
    utils.compute_camera_rays = lambda cameras, indices, neighbors, **kw: {
        'camera_rays': torch.tensor(indices),
        'neighbor_w2cs': torch.tensor(neighbors).reshape(-1, 1, 1),
        'neighbor_Ks': (torch.tensor(neighbors) + 100).reshape(-1, 1, 1)}
    utils.load_encoded_prompt = lambda *a: [torch.zeros(1)]
    for name, module in [('model_eval.run_inference', inference),
                         ('model_eval.checkpoint_loading', checkpoints),
                         ('model_training.data.utils', utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'argv', ['bridge', '--request', str(tmp_path), '--repo', str(tmp_path),
                                    '--checkpoint', 'unused', '--model-id', 'test'])
    if schedule == 'periodic_starter' and source_mode == 'none':
        with pytest.raises(ValueError, match='rendered source'):
            artifixer_bridge.main()
        return
    artifixer_bridge.main()
    assert len(seen) == (6 if schedule == 'periodic_starter' else 2)
    assert parsed[parsed.index('--local_attn_size') + 1] == '21'
    assert '--replace_if_exists' in parsed
    metadata = json.loads((tmp_path/'inference.json').read_text())
    assert metadata['starter_frames'] == ([0,20,40,45,65,85] if schedule == 'periodic_starter' else [0,9])
    assert metadata['scene_rgb_conditioning'] is (source_mode == 'rendered')
    assert len(metadata['starter_diagnostics']) == len(seen)
    assert metadata['block_schedule'] == schedule
    assert metadata['exact_starter_preserved'] is (schedule != 'upstream')
    assert metadata['generated_cache'] == ('clean' if schedule != 'upstream' else 'last_denoising')
    diagnostic = 'starter-vae-roundtrip' if schedule != 'upstream' else 'upstream-first-frame'
    assert (tmp_path / f'{diagnostic}-00000.png').exists()
    for i, expected in [(0,40),(count,220)]:
        assert Image.open(tmp_path/f'artifixer-output/bundle/frames/batch_0000/pred/{i:05d}.png').getpixel((0,0)) == ((expected,)*3 if schedule != "upstream" else (0,0,0))
    if schedule == 'periodic_starter':
        assert metadata['initial_reference_views'] == 2 and metadata['reference_views'] == 6
        assert [r['frame_index'] for r in metadata['generated_references']] == [20, 40, 65, 85]
        assert all(r['kind'] == 'edited_render' for r in metadata['generated_references'])
        assert [r['frame_index'] for r in metadata['periodic_refreshes']] == [20, 40, 65, 85]
        for i in [20, 40, 65, 85]:
            assert Image.open(tmp_path/f'artifixer-output/bundle/frames/batch_0000/pred/{i:05d}.png').getpixel((0,0)) == (230,230,230)
            assert Image.open(tmp_path/f'refresh/{i:05d}/vae-roundtrip.png').getpixel((0,0)) == (0,0,0)
            assert (tmp_path/f'refresh/{i:05d}/before.png').is_file()
        assert len(list((tmp_path/'artifixer-output/bundle/frames/batch_0000/pred').glob('*.png'))) == 2*count
        assert all(r['temporal_encoding'] == 'standalone_first_frame' for r in metadata['periodic_refreshes'])
        for i, value in [(count - 1, 40), (2*count - 1, 220)]:
            assert Image.open(tmp_path/f'artifixer-output/bundle/frames/batch_0000/pred/{i:05d}.png').getpixel((0,0)) == (value,)*3
