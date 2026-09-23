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


def test_clean_starter_drives_future_frames_and_cache_resets():
    torch = pytest.importorskip('torch')
    calls = []
    class Transformer:
        patch_size = (1, 1, 1)
        _cp_world_size = 1
        def __call__(self, **kw):
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
                assert kw['opacity'].count_nonzero() == 0
            return (torch.ones_like(value) * cache['starter'].mean(),)
    class Pipe:
        frames_per_block = 7
        local_attn_size = -1
        vae = SimpleNamespace(device='cpu', config=SimpleNamespace(scale_factor_temporal=4))
        transformer = Transformer()
        generate_samples_from_batch = None
        scheduler = SimpleNamespace(step=lambda noise, *a, **kw: noise,
                                    add_noise=lambda clean, noise, t: clean + noise)
        def _initialize_kv_cache(self, *args): self.kv_cache1 = {}
        def _initialize_crossattn_cache(self, name): setattr(self, name, {})
        def create_denoising_step_list(self, steps): return torch.tensor([1000., 500.])
    pipe = Pipe()
    install_starter_inference(pipe)
    outputs = []
    with torch.inference_mode():
        for starter in [2., 5.]:
            # Later condition values deliberately differ: they must be ignored.
            condition = torch.full((1, 1, 14, 2, 2), 99.)
            condition[:, :, 0] = starter
            opacity = torch.zeros(1, 53, 2, 2); opacity[:, 0] = 1
            poses = torch.arange(14).reshape(1, 14, 1)
            outputs.append(pipe.generate_samples_from_batch(condition, opacity,
                torch.ones(1), poses, poses, torch.ones(1), poses, torch.ones(1),
                torch.ones(1), 2, False))
    assert torch.all(outputs[0] == 2.) and torch.all(outputs[1] == 5.)
    assert [c[0] for c in calls[:7]] == [0, 1, 1, 1, 8, 8, 8]
    assert [c[3] for c in calls[:7]] == [1, 28, 28, 28, 24, 24, 24]
    assert all(calls[i][2].count_nonzero() == 0 for i in [0, 3, 6, 7, 10, 13])


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


def test_bridge_generates_from_each_edit_without_reading_scene_renders(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType
    from PIL import Image
    torch = pytest.importorskip('torch')
    from splat_explorer.scene_runs_ext import artifixer_bridge
    for i, value in enumerate([40, 220]):
        Image.new('RGB', (32, 32), (value, value, value)).save(tmp_path / f'edit-{i}.png')
    manifest = {'options': {'seed': 42, 'inference_steps': 4, 'camera_scale': 1.},
                'transforms': {'frames': [{}] * 18},
                'references': [{'path': f'edit-{i}.png', 'frame_index': 9*i,
                                'kind': 'edited_render'} for i in range(2)],
                'segments': [{'start': 0, 'count': 9}, {'start': 9, 'count': 9}]}
    (tmp_path / 'bundle.json').write_text(json.dumps(manifest))
    # There is intentionally no inputs directory and no opacity.npy.
    parsed = []
    class Parser:
        def parse_args(self, args): parsed.extend(args); return SimpleNamespace()
    class Pipe:
        transformer = SimpleNamespace(eval=lambda: None)
        vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))
        generate_samples_from_batch = None
        _initialize_kv_cache = None
        def clear_inference_caches(self): self.cleared = True
    pipe = Pipe()
    seen = []
    def process(pipe, item, opts, output, rank, device, scale):
        assert pipe.cleared
        pipe.cleared = False
        assert pipe.generate_samples_from_batch.__func__ is generate_from_starter
        index = len(seen)
        assert torch.allclose(item['rgb_rendered'][0], torch.full((3,32,32), [40,220][index]/255))
        assert item['rgb_rendered'][1:].count_nonzero() == 0
        assert item['opacity'][1:].count_nonzero() == 0
        assert item['frame_indices'][0] == index*9
        seen.append(item)
        dest = output / 'bundle/frames/batch_0000/pred'
        dest.mkdir(parents=True, exist_ok=True)
        for i in item['frame_indices'].tolist():
            Image.new('RGB', (32,32)).save(dest / f'{i:05d}.png')
    inference = ModuleType('model_eval.run_inference')
    inference.build_parser = Parser
    inference.get_eval_pipe = lambda *a: pipe
    inference.process_item = process
    checkpoints = ModuleType('model_eval.checkpoint_loading')
    checkpoints.load_transformer_checkpoint = lambda *a: None
    utils = ModuleType('model_training.data.utils')
    utils.compute_camera_rays = lambda cameras, indices, neighbors, **kw: {}
    utils.load_encoded_prompt = lambda *a: [torch.zeros(1)]
    for name, module in [('model_eval.run_inference', inference),
                         ('model_eval.checkpoint_loading', checkpoints),
                         ('model_training.data.utils', utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'argv', ['bridge', '--request', str(tmp_path), '--repo', str(tmp_path),
                                    '--checkpoint', 'unused', '--model-id', 'test'])
    artifixer_bridge.main()
    assert len(seen) == 2
    assert parsed[parsed.index('--local_attn_size') + 1] == '-1'
    assert '--replace_if_exists' in parsed
    metadata = json.loads((tmp_path/'inference.json').read_text())
    assert metadata['starter_frames'] == [0,9]
    assert metadata['scene_rgb_conditioning'] is False
    for i, expected in [(0,40),(9,220)]:
        assert Image.open(tmp_path/f'artifixer-output/bundle/frames/batch_0000/pred/{i:05d}.png').getpixel((0,0)) == (expected,)*3
