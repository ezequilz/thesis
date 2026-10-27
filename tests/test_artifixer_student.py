"""Student routing and lossless residency must preserve the small-model contract."""
from copy import deepcopy

import pytest

from splat_explorer.splatfix.artifixer.inference_policy import MODELS, model_runtime
from splat_explorer.splatfix.artifixer.student_memory import GIB, plan_residency, staged_caches
from splat_explorer.splatfix.jobs import validate_job


def test_student_job_has_identical_generation_options_to_small_model():
    raw = {'stage': 'repair', 'checkpoint': '/saved', 'reconstruction_method': 'artifixer'}
    small = validate_job({**raw, 'inference_model': '1.3b'})
    large = validate_job(raw)
    assert large == {**small, 'inference_model': '14b'}
    cfg = model_runtime({**large, 'checkpoint': '/models/artifixer-1.3b.pt',
                         'model_id': MODELS['1.3b'][1]})
    assert cfg['checkpoint'] == '/models/artifixer-14b.pt'
    assert cfg['model_id'] == MODELS['14b'][1]
    assert not cfg['teacher_inference']
    assert not cfg['connected_trajectory_generation']
    assert cfg['generated_context_references']
    assert cfg['initial_image_seeding']


def test_saved_teacher_opt_out_without_model_stays_small():
    cfg = model_runtime({'checkpoint': '/models/artifixer-1.3b.pt',
                         'model_id': MODELS['1.3b'][1], 'teacher_inference': False})
    assert cfg['inference_model'] == '1.3b'


def test_student_keeps_causal_segments_even_with_stale_teacher_layout():
    from splat_explorer.splatfix.artifixer.official_worker import inference_jobs
    trajectory = {'frames': [{}]*7, 'anchors': [{'frame_index': 0}, {'frame_index': 6}],
                  'segments': [{'indices': [1, 2, 3], 'seed_index': 0},
                               {'indices': [5, 4], 'seed_index': 6}]}
    request = {'trajectory_mode': 'authors_orbit', 'runtime': {
        'inference_model': '14b', 'teacher_clip_layout': 'forward', 'split_mode': 'double-split'}}
    assert [job[0] for job in inference_jobs(request, trajectory)] == trajectory['segments']


def test_native_720p_six_reference_cache_fits_80gb_with_host_offload():
    # 40 layers, full 21-frame temporal window, six references, 512 text tokens.
    size = (21 * 3600 + 6 * 3600 + 512) * 5120 * 2 * 2
    sizes = [size] * 40
    cpu = plan_residency(sizes, 47 * GIB, 55 * GIB)
    assert cpu and len(cpu) < 40
    assert sum(sizes[i] for i in cpu) + 4 * GIB <= 55 * GIB
    assert sum(sizes[i] for i in range(40) if i not in cpu) + size + 12 * GIB <= 47 * GIB
    with pytest.raises(RuntimeError, match='will not be reduced'):
        plan_residency(sizes, 47 * GIB, 8 * GIB)
    assert plan_residency([GIB] * 3, 20 * GIB, 8 * GIB) == set()


def test_host_budget_reclaims_unmapped_checkpoint_pages_but_not_tensor_memory():
    from splat_explorer.splatfix.artifixer.student_memory import reclaimable_file_bytes
    assert reclaimable_file_bytes({'file': 50*GIB, 'file_mapped': 2*GIB, 'shmem': GIB,
                                   'file_dirty': GIB, 'anon': 8*GIB, 'active_file': 30*GIB}) == 46*GIB
    assert reclaimable_file_bytes({'total_cache': 50*GIB, 'total_mapped_file': 2*GIB,
                                   'total_writeback': GIB, 'total_rss': 8*GIB}, v1=True) == 47*GIB


def test_repaired_plus_generated_references_keep_the_complete_cache():
    from splat_explorer.splatfix.artifixer.student_memory import choose_residency
    size = (21*3600 + 12*3600 + 512)*5120*2*2
    layers, gpu_margin, host_margin = choose_residency([size]*40, 46.4*GIB, 59.8*GIB)
    assert (gpu_margin, host_margin) == (8, 2)
    assert len(layers)*size + host_margin*GIB <= 59.8*GIB
    assert (40-len(layers)+1)*size + gpu_margin*GIB <= 46.4*GIB
    with pytest.raises(RuntimeError):
        choose_residency([size]*40, 20*GIB, 20*GIB)


@pytest.mark.parametrize('failure', [False, True])
def test_cache_staging_preserves_mutations_and_new_reference_caches(failure):
    torch = pytest.importorskip('torch')
    original = torch.arange(12, dtype=torch.bfloat16).reshape(1, 6, 2)
    temporal = {'k': original.clone(), 'v': original.clone(), 'local_end_index': torch.tensor([3])}
    reference = {'is_init': False}
    backing = temporal['k']
    try:
        with staged_caches([temporal, None, reference], 'cpu'):
            assert temporal['k'].data_ptr() != backing.data_ptr()
            temporal['k'][:, :3] = temporal['k'][:, 3:].clone()
            temporal['v'] = temporal['v'] + 7  # Upstream may replace a tensor.
            temporal['local_end_index'].fill_(6)
            reference.update(is_init=True, k=original + 2, v=original + 4)
            if failure:
                raise RuntimeError('test forward failure')
    except RuntimeError:
        assert failure
    assert temporal['k'] is backing
    assert torch.equal(temporal['k'][:, :3], original[:, 3:])
    assert torch.equal(temporal['v'], original + 7)
    assert temporal['local_end_index'].item() == 6
    assert reference['is_init'] and torch.equal(reference['k'], original + 2)
    before = deepcopy(reference)
    with staged_caches([reference], 'cpu'):
        assert reference['is_init'] and torch.equal(reference['v'], before['v'])
    assert torch.equal(reference['k'], before['k'])


def test_installed_offload_matches_resident_rolling_attention(monkeypatch, tmp_path):
    """Compare outputs over cache eviction, reference reuse, and a fresh sequence."""
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from splat_explorer.splatfix.artifixer import student_memory as memory

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = SimpleNamespace(heads=1, inner_dim=4)

        def forward(self, x, kv_cache, crossattn_cache, neighbor_crossattn_cache):
            end = int(kv_cache['local_end_index'])
            if end == 6:
                for key in ('k', 'v'):
                    kv_cache[key][:, :4] = kv_cache[key][:, 2:].clone()
                end = 4
            kv_cache['k'][:, end:end+2] = x
            kv_cache['v'][:, end:end+2] = x * 2
            kv_cache['local_end_index'].fill_(end+2)
            kv_cache['global_end_index'].add_(2)
            for cache in (crossattn_cache, neighbor_crossattn_cache):
                if not cache['is_init']:
                    cache.update(is_init=True, k=x.clone(), v=x.clone())
            q = x.flatten(2)
            keys = kv_cache['k'][:, :end+2].flatten(2)
            values = kv_cache['v'][:, :end+2].flatten(2)
            return ((q @ keys.transpose(-1, -2)).softmax(-1) @ values
                    + neighbor_crossattn_cache['v'].flatten(2)).unflatten(-1, (1, 4))

    class Pipe:
        def __init__(self):
            self.vae = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float32,
                                       config=SimpleNamespace(scale_factor_spatial=8))
            self.transformer = SimpleNamespace(blocks=[Block(), Block()], _cp_world_size=1,
                                               patch_size=(1, 2, 2))
            self.frames_per_block = 7

        def clear_inference_caches(self):
            self.kv_cache1 = None

        def _initialize_kv_cache(self, batch, spatial, frames):
            self.kv_cache1 = [{'k': torch.zeros(batch, spatial*frames, 1, 4),
                              'v': torch.zeros(batch, spatial*frames, 1, 4),
                              'local_end_index': torch.zeros(1, dtype=torch.long),
                              'global_end_index': torch.zeros(1, dtype=torch.long)} for _ in range(2)]

        def denoise_to_latents(self, neighbors, prompt):
            self._initialize_kv_cache(1, 2, 3)
            texts = [{'is_init': False} for _ in range(2)]
            refs = [{'is_init': False} for _ in range(2)]
            outputs = []
            for i in range(6):
                x = torch.arange(8).reshape(1, 2, 1, 4).float() / 10 + i / 100
                for layer, block in enumerate(self.transformer.blocks):
                    x = block(x, self.kv_cache1[layer], texts[layer], refs[layer])
                outputs.append(x)
            return torch.stack(outputs)

    monkeypatch.setattr(memory.StudentMemory, 'record', lambda *args, **kwargs: None)
    monkeypatch.setattr(memory, 'available_host_bytes', lambda: 60*GIB)
    monkeypatch.setattr(memory, 'plan_residency', lambda *args, **kwargs: {0})
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *args: (40*GIB, 80*GIB))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    resident, offloaded = Pipe(), Pipe()
    memory.install(offloaded, tmp_path/'memory.jsonl')
    with torch.no_grad():
        for _ in range(2):
            args = {'neighbors': torch.ones(1, 6, 3, 64, 64), 'prompt': torch.ones(1, 3, 4)}
            assert torch.equal(resident.denoise_to_latents(**args), offloaded.denoise_to_latents(**args))
            for a, b in zip(resident.kv_cache1, offloaded.kv_cache1):
                assert all(torch.equal(a[key], b[key]) for key in a)
