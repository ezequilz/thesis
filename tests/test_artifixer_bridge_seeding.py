"""Bridge modes, clean boundary memory and failure cleanup on small CPU tensors."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from splat_explorer.splatfix.artifixer import bridge_seeding as bridge


@pytest.mark.parametrize('value', [None, True, False, 1, 'on', [], {}])
def test_invalid_bridge_modes_rejected(value):
    from splat_explorer.splatfix.artifixer.backend import validate_options
    from splat_explorer.splatfix.artifixer.repair import validate_runtime
    for validate in (lambda x: validate_options({'stage': 'repair', **x}), validate_runtime):
        with pytest.raises(ValueError, match='connected_bridge_seeding'):
            validate({'connected_bridge_seeding': value})


def test_new_defaults_and_saved_request_baseline():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    from splat_explorer.splatfix.artifixer.repair import DEFAULT_RUNTIME
    assert bridge.mode({}) == 'off'
    assert bridge.mode(DEFAULT_RUNTIME) == 'seed'
    assert validate_options({'stage': 'repair'})['connected_bridge_seeding'] == 'seed'
    for mode in bridge.MODES:
        assert validate_options({'stage': 'repair', 'connected_bridge_seeding': mode})['connected_bridge_seeding'] == mode


@pytest.mark.parametrize('mode', bridge.MODES)
def test_cli_passes_bridge_mode(tmp_path, monkeypatch, mode):
    import argparse
    from splat_explorer.config import Config
    from splat_explorer.splatfix.cli import add_parser, repair_splat
    from splat_explorer.splatfix.artifixer import repair
    received = {}
    def run(*args, **kwargs):
        received.update(kwargs)
        return {}
    monkeypatch.setattr(repair, 'run_repair', run)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(['splatfix', 'repair', str(tmp_path), '--output', str(tmp_path / 'out'),
                             '--connected-bridge-seeding', mode])
    repair_splat(Config({}), args)
    assert received['runtime']['connected_bridge_seeding'] == mode


@pytest.fixture
def cpu_pipe(monkeypatch):
    torch = pytest.importorskip('torch')
    class OriginalProcessor:
        _attention_backend = None
        _cp_mesh = None
    helpers = ModuleType('model_training.net.transformer')
    helpers.KvCacheWanSelfAttnProcessor = OriginalProcessor
    helpers._get_qkv_projections = lambda attn, x, _: (x.clone(), x.clone(), x.clone())
    helpers._apply_rotary_emb = lambda x, cos, sin: x * cos + torch.stack((-x[..., 1], x[..., 0]), -1) * sin
    def attention(q, k, v, **kwargs):
        assert kwargs == dict(attn_mask=None, dropout_p=0.0, is_causal=False, backend=None)
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    helpers.dispatch_attention_fn = attention
    for name in ('model_training', 'model_training.net'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['model_training.net'].transformer = helpers
    monkeypatch.setitem(sys.modules, 'model_training.net.transformer', helpers)
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = 1
            self.processor = OriginalProcessor()
            self.norm_q = self.norm_k = torch.nn.Identity()
            self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        def set_processor(self, processor): self.processor = processor
    class Transformer(torch.nn.Module):
        patch_size = (1, 1, 1)
        _cp_world_size = 1
        def __init__(self):
            super().__init__()
            self.blocks = [SimpleNamespace(attn1=Attention()) for _ in range(2)]
            self.calls, self.cache_history = [], []
        def forward(self, *, hidden_states, timestep, **kwargs):
            self.calls.append((hidden_states.clone(), timestep.clone(), kwargs))
            assert kwargs['current_start'] == kwargs['frame_offset'] == 0
            # Timestep-sensitive projections ensure a clean seed K/V must be
            # actively preserved; fixing only the input latent would not suffice.
            hidden = hidden_states.permute(0, 2, 3, 4, 1).reshape(1, 14, 2) + timestep[0] / 1000
            angle = torch.arange(14).reshape(1, 14, 1, 1) / 14
            for block, cache in zip(self.blocks, kwargs['kv_cache']):
                hidden = block.attn1.processor(block.attn1, hidden, None, None,
                    (angle.cos(), angle.sin()), cache, 0, 2)
            self.cache_history.append([(c['k'].clone(), c['v'].clone()) for c in kwargs['kv_cache']])
            return (hidden.reshape(1, 7, 1, 2, 2).permute(0, 4, 1, 2, 3),)
    class Scheduler:
        def step(self, noise, timestep, latents, *, to_final):
            if pipe.fail_step and len(pipe.transformer.calls) == 3:
                raise RuntimeError('scheduler failure')
            return noise + 3
        def add_noise(self, clean, noise, timestep): return clean * .7 + noise
    class Pipe:
        frames_per_block, local_attn_size = 7, 21
        vae = SimpleNamespace(device='cpu', config=SimpleNamespace(scale_factor_temporal=4))
        def __init__(self):
            self.transformer, self.scheduler = Transformer(), Scheduler()
            self.samplings, self.fail_step = 0, False
        def generate_samples_from_batch(self, condition, opacity, neighbors, rays, w2cs,
                neighbor_w2cs, Ks, neighbor_Ks, prompt, steps, use_exit_flag, **flags):
            # Exercise the upstream contract: allocate first; transformer and
            # scheduler methods are looked up for each of the four updates.
            self.samplings += 1
            assert steps == 4 and use_exit_flag is False
            assert flags['ignore_neighbors'] is False
            caches = [dict(k=torch.zeros(1, 42, 1, 2), v=torch.zeros(1, 42, 1, 2),
                           global_end_index=torch.zeros(1, dtype=torch.long),
                           local_end_index=torch.zeros(1, dtype=torch.long)) for _ in self.transformer.blocks]
            self.kv_cache1 = caches
            kwargs = dict(current_start=0, frame_offset=0, kv_cache=caches, opacity=opacity,
                neighbor_hidden_states=neighbors, camera_rays=rays, w2cs=w2cs,
                neighbor_w2cs=neighbor_w2cs, Ks=Ks, neighbor_Ks=neighbor_Ks, encoder_hidden_states=prompt)
            latents = condition.clone()
            for index, t in enumerate([1000., 750., 500., 250.]):
                timestep = torch.tensor([t])
                noise = self.transformer(hidden_states=latents, timestep=timestep, **kwargs)[0]
                latents = self.scheduler.step(noise, timestep, latents, to_final=True)
                if index < 3:
                    latents = self.scheduler.add_noise(latents, condition.clone(), timestep - 250)
            return latents
    pipe = Pipe()
    return torch, pipe


def inputs(torch, right=9.):
    condition = torch.full((1, 2, 7, 1, 2), .5)
    condition[:, :, 0] = 2
    condition[:, :, 1] = 3
    condition[:, :, 6] = right
    opacity = torch.full((1, 25, 8, 16), .25)
    opacity[:, [0, 1, 23, 24]] = 1
    poses = torch.arange(7).reshape(1, 7, 1)
    refs = torch.arange(12).reshape(1, 12, 1)
    return [condition, opacity, refs, poses, poses, refs, poses, refs, torch.ones(1), 4, False]


def test_four_steps_preserve_both_boundaries_in_every_layer(cpu_pipe):
    torch, pipe = cpu_pipe
    original_sampler = pipe.generate_samples_from_batch
    original_processors = [b.attn1.processor for b in pipe.transformer.blocks]
    values = inputs(torch)
    before = values[0].clone()
    with torch.inference_mode(), bridge.preserved_bridge(pipe, original_sampler, real_frame_count=25, overlap_slots=[0, 1, 23, 24]) as audit:
        output = pipe.generate_samples_from_batch(*values)
    assert pipe.samplings == 1, 'Keep the normal joint sampler'
    assert torch.equal(before, values[0]), 'Do not mutate VAE conditioning'
    assert torch.equal(output[:, :, [0, 1, 6]], before[:, :, [0, 1, 6]])
    assert not torch.equal(output[:, :, 2:6], before[:, :, 2:6])
    assert audit['seed_latents_preserved'] and audit['seed_kv_preserved']
    assert audit['verified_denoising_steps'] == 4 and audit['verified_attention_layers'] == 2
    assert [t.item() for _, t, _ in pipe.transformer.calls] == [0, 1000, 750, 500, 250]
    token_indices = [0, 1, 2, 3, 12, 13]
    for states, _, kw in pipe.transformer.calls:
        assert torch.equal(states[:, :, [0, 1, 6]], before[:, :, [0, 1, 6]])
        assert kw['neighbor_hidden_states'] is values[2]
        assert kw['camera_rays'] is values[3] and kw['w2cs'] is values[4]
        assert kw['neighbor_w2cs'] is values[5] and kw['Ks'] is values[6] and kw['neighbor_Ks'] is values[7]
    for history in pipe.transformer.cache_history[1:]:
        for (keys, vals), (first_k, first_v) in zip(history, pipe.transformer.cache_history[0]):
            assert torch.equal(keys[:, token_indices], first_k[:, token_indices])
            assert torch.equal(vals[:, token_indices], first_v[:, token_indices])
    assert_restored(pipe, original_sampler, original_processors)


def assert_restored(pipe, sampler, processors):
    assert pipe.generate_samples_from_batch == sampler
    assert 'generate_samples_from_batch' not in vars(pipe)
    assert 'forward' not in vars(pipe.transformer)
    assert 'step' not in vars(pipe.scheduler) and 'add_noise' not in vars(pipe.scheduler)
    assert [b.attn1.processor for b in pipe.transformer.blocks] == processors


def test_right_seed_influences_middle_and_fresh_calls_do_not_share_memory(cpu_pipe):
    torch, pipe = cpu_pipe
    outputs = []
    with torch.inference_mode():
        for value in [9., -3.]:
            with bridge.preserved_bridge(pipe, pipe.generate_samples_from_batch, real_frame_count=25, overlap_slots=[0, 1, 23, 24]):
                outputs.append(pipe.generate_samples_from_batch(*inputs(torch, right=value)))
    assert not torch.equal(outputs[0][:, :, 2:6], outputs[1][:, :, 2:6])
    assert pipe.samplings == 2


def test_stronger_mode_restores_all_hooks_after_sampler_failure(cpu_pipe):
    torch, pipe = cpu_pipe
    sampler = pipe.generate_samples_from_batch
    processors = [b.attn1.processor for b in pipe.transformer.blocks]
    pipe.fail_step = True
    with torch.inference_mode(), pytest.raises(RuntimeError, match='scheduler failure'):
        with bridge.preserved_bridge(pipe, sampler, real_frame_count=25, overlap_slots=[0, 1, 23, 24]) as audit:
            pipe.generate_samples_from_batch(*inputs(torch))
    assert not audit['seed_latents_preserved'] and not audit['seed_kv_preserved']
    assert_restored(pipe, sampler, processors)


def test_preservation_requires_actual_sampler_call(cpu_pipe):
    torch, pipe = cpu_pipe
    sampler = pipe.generate_samples_from_batch
    with torch.inference_mode(), pytest.raises(RuntimeError, match='sampler was not called'):
        with bridge.preserved_bridge(pipe, sampler, real_frame_count=25, overlap_slots=[0, 1, 23, 24]):
            pass
    assert pipe.generate_samples_from_batch == sampler


@pytest.mark.parametrize('real,overlaps,groups', [
    (25, [0,1,23,24], [0,1,6]), (20,[0,1,18,19],[0,1,5]),
    (14,[0,1,12,13],[0,1,3,4]), (8,[0,1,6,7],[0,1,2]),
    (3,[0,2],[0,1])])
def test_preserved_groups_follow_real_overlaps_not_padding(cpu_pipe, real, overlaps, groups):
    torch, pipe = cpu_pipe
    layout = bridge.seed_layout(real, overlaps)
    assert layout['seed_latent_indices'] == groups
    assert layout['seed_rgb_local_indices'] == overlaps
    assert all(i < real for i in layout['constrained_rgb_local_indices'])
    values = inputs(torch)
    with torch.inference_mode(), bridge.preserved_bridge(pipe, pipe.generate_samples_from_batch,
            real_frame_count=real, overlap_slots=overlaps) as audit:
        result = pipe.generate_samples_from_batch(*values)
    assert torch.equal(result[:, :, groups], values[0][:, :, groups])
    assert audit['seed_latent_indices'] == groups
    assert audit['all_real_latent_groups_preserved'] is (real <= 8)


@pytest.mark.parametrize('fail', [False, True])
def test_joint_cache_window_is_bounded_and_restored(cpu_pipe, fail):
    _, pipe = cpu_pipe
    for block in pipe.transformer.blocks:
        block.attn1.processor.local_attn_size = 21
        block.attn1.processor.sink_size = 3
    try:
        with bridge.single_block_cache(pipe):
            assert pipe.local_attn_size == 7
            assert all(block.attn1.processor.local_attn_size == 7 for block in pipe.transformer.blocks)
            if fail:
                raise RuntimeError('sampling failed')
    except RuntimeError:
        assert fail
    assert pipe.local_attn_size == 21
    assert all(block.attn1.processor.local_attn_size == 21 for block in pipe.transformer.blocks)


def test_preservation_has_no_free_middle_below_the_twenty_pose_boundary():
    # N=19 has a 6-frame middle + four overlaps; N=20 has seven + four.
    assert bridge.seed_layout(10, [0, 1, 8, 9])['free_middle_latent_indices'] == []
    assert bridge.seed_layout(11, [0, 1, 9, 10])['free_middle_latent_indices'] == [2]
