"""Periodic repair regressions: real CPU tensor math and a fake file transport."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from splat_explorer.scene_runs_ext.config import validate_options
from splat_explorer.scene_runs_ext.starter_inference import latent_chunks, install_starter_inference
from splat_explorer.scene_runs_ext.periodic_refresh import PeriodicRefresh, service_refresh


def test_periodic_policy_requires_scene_rgb_and_opacity():
    result = validate_options({'block_schedule': 'periodic_starter'})
    assert result['source_conditioning'] == 'rendered'
    with pytest.raises(ValueError, match='opacity'):
        validate_options({'block_schedule': 'periodic_starter', 'source_conditioning': 'none'})


def test_split_chunks_preserve_original_boundaries_and_cover_each_latent_once():
    chunks = list(latent_chunks(21, 7, 5))
    assert chunks == [(0, 1), (1, 6), (6, 8), (8, 11), (11, 15), (15, 16), (16, 21)]
    assert [i for start, end in chunks for i in range(start, end)] == list(range(21))
    assert [4 * (end - 1) for _, end in chunks if end > 1 and (end - 1) % 5 == 0] == [20, 40, 60, 80]


@pytest.mark.parametrize('cache_policy', ['clean', 'last_denoising'])
def test_repaired_latent_drives_next_block_without_losing_opacity_or_history(cache_policy):
    torch = pytest.importorskip('torch')
    calls, refreshes = [], []
    class Transformer:
        patch_size = (1, 1, 1)
        def __call__(self, **kw):
            start = kw['frame_offset']
            cache = kw['kv_cache']
            value = kw['hidden_states']
            calls.append((start, float(kw['timestep'][0]), value.clone()))
            assert torch.all(kw['opacity'] == (1 if start == 0 else .6))
            assert kw['camera_rays'][0, 0, 0] == start
            if start:
                assert cache[0] == 2
            if start == 6:
                assert cache[5] == 77, 'Following block must see corrected KV memory'
            corrected = start >= 6 or (start == 1 and kw['timestep'][0] == 0)
            count = 2 if corrected else 1
            assert kw['neighbor_hidden_states'].shape[2] == count
            assert kw['neighbor_w2cs'].shape[1] == count
            assert kw['neighbor_Ks'].shape[1] == count
            ref_cache = kw['neighbor_crossattn_cache']
            if 'count' in ref_cache:
                assert ref_cache['count'] == count, 'Reference KV must be invalidated after append'
            ref_cache['count'] = count
            if corrected:
                assert kw['neighbor_hidden_states'][0, 0, -1, 0, 0] == 88
                assert kw['neighbor_w2cs'][0, -1, 0, 0] == 20
                assert kw['neighbor_Ks'][0, -1, 0, 0] == 200
            kw['crossattn_cache'].setdefault('original', True)
            previous = cache.get(start - 1, 2)
            for i in range(value.shape[2]):
                cache[start + i] = float(value[:, :, i].mean())
            return (torch.full_like(value, previous + (88 if start >= 6 else 0)),)
    class Pipe:
        frames_per_block = 7
        local_attn_size = -1
        vae = SimpleNamespace(device='cpu', config=SimpleNamespace(scale_factor_temporal=4))
        transformer = Transformer()
        scheduler = SimpleNamespace(step=lambda noise, *a, **kw: noise)
        def generate_samples_from_batch(self): pass
        def _initialize_kv_cache(self, *args): self.kv_cache1 = {}
        def _initialize_crossattn_cache(self, name): setattr(self, name, {})
        def create_denoising_step_list(self, steps): return torch.tensor([1000.])
        def prepare_latents(self, condition, opacity, first):
            assert not first and torch.all(opacity == .6)
            return torch.zeros_like(condition)
    pipe = Pipe()
    install_starter_inference(pipe, generated_cache=cache_policy)
    def refresh(pipe, prefix, frame):
        assert frame == 20 and prefix.shape[2] == 6
        assert pipe.kv_cache1[0] == 2
        assert any(start == 1 and t == 1000 for start, t, _ in calls)
        refreshes.append(frame)
        return torch.full_like(prefix[:, :, -1:], 77)
    pipe.starter_refresh = refresh
    def update_references(pipe, frame, neighbors, w2cs, Ks):
        assert frame == 20
        assert pipe.crossattn_cache['original'], 'Text KV memory must remain intact'
        assert pipe.neighbor_crossattn_cache['count'] == 1
        return (torch.cat([neighbors, torch.full_like(neighbors, 88)], dim=2),
                torch.cat([w2cs, torch.full_like(w2cs, 20)], dim=1),
                torch.cat([Ks, torch.full_like(Ks, 200)], dim=1))
    refresh.update_reference_conditioning = update_references
    pipe.starter_rgb_count = 25  # Upstream pads to 49 RGB frames / 13 latents here.
    condition = torch.zeros(1, 1, 14, 2, 2)
    condition[:, :, 0] = 2
    opacity = torch.full((1, 53, 2, 2), .6); opacity[:, 0] = 1
    poses = torch.arange(14).reshape(1, 14, 1)
    with torch.inference_mode():
        output = pipe.generate_samples_from_batch(condition, opacity, torch.ones(1, 1, 1, 2, 2),
            poses, poses, torch.ones(1, 1, 4, 4), poses, torch.ones(1, 1, 3, 3), torch.ones(1), 1, False)
    assert refreshes == [20], 'Padded views must never trigger a GPT call'
    assert torch.all(output[:, :, 5] == 77)
    assert torch.all(output[:, :, 6:8] == 165), 'Next block must receive both corrected temporal and reference context'
    assert any(start == 1 and t == 0 and value[0, 0, -1, 0, 0] == 77
               for start, t, value in calls), 'Even last-denoising mode must refresh corrected cache'


def test_refresh_reencodes_generated_context_without_clearing_cache(tmp_path):
    torch = pytest.importorskip('torch')
    (tmp_path / 'inputs').mkdir()
    Image.new('RGB', (16, 16), (30, 30, 30)).save(tmp_path / 'inputs/00020.png')
    Image.new('RGB', (16, 16), (200, 200, 200)).save(tmp_path / 'anchor.png')
    folder = tmp_path / 'refresh/00020'; folder.mkdir(parents=True)
    Image.new('RGB', (32, 32), (240, 240, 240)).save(folder / 'regenerated.png')
    (folder / 'response.json').write_text('{"status":"ok"}')
    caches = object()
    seen = []
    def encode(video):
        assert video.shape == (1, 21, 3, 16, 16)
        assert torch.allclose(video[:, :-1], torch.full_like(video[:, :-1], .25))
        assert torch.allclose(video[:, -1], torch.full_like(video[:, -1], 240 / 255))
        seen.append(video)
        return torch.full((1, 1, 6, 2, 2), 9.)
    pipe = SimpleNamespace(vae=SimpleNamespace(device='cpu'), kv_cache1=caches,
        latents_to_rgb=lambda prefix: torch.full((1, 21, 3, 16, 16), .25),
        video_processor=SimpleNamespace(postprocess_video=lambda value, **kw: value),
        encode_video_frames=encode,
        decode_latents_to_video=lambda value: pytest.fail('Must not clear cache'))
    refresh = PeriodicRefresh(tmp_path, list(range(25)))
    result = refresh(pipe, torch.zeros(1, 1, 6, 2, 2), 20)
    assert result.shape == (1, 1, 1, 2, 2) and torch.all(result == 9)
    assert pipe.kv_cache1 is caches and len(seen) == 1
    assert Image.open(folder / 'rendered.png').getpixel((0, 0)) == (30, 30, 30)
    assert refresh.records[0]['frame_index'] == 20
    assert Image.open(folder / 'fixed.png').size == (16, 16)


@pytest.mark.parametrize('failure', [False, True])
def test_local_relay_uses_original_prompt_and_anchor_and_publishes_response_last(tmp_path, monkeypatch, failure):
    from splat_explorer import repair_lrz
    request = tmp_path / 'requests/repair-00001'; request.mkdir(parents=True)
    (request / 'request.json').write_text('{"prompt":"Repair the damaged door."}')
    transfers, edits = [], []
    def transfer(args):
        transfers.append(args[-2:])
        if args[-2].startswith('user@host:'):
            folder = Path(args[-1])
            Image.new('RGB', (16, 16)).save(folder / 'rendered.png')
            Image.new('RGB', (16, 16)).save(folder / 'anchor.png')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(repair_lrz, '_mux_run', transfer)
    monkeypatch.setattr(repair_lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    def edit(source, target, prompt, references):
        edits.append(prompt)
        assert prompt.startswith('Repair the damaged door.')
        assert 'FIRST image' in prompt and 'SECOND image' in prompt
        assert references == [source.parent / 'anchor.png']
        if failure:
            raise RuntimeError('Relay failed')
        Image.new('RGB', (16, 16)).save(target)
        return {'model': 'test'}
    transport = SimpleNamespace(cfg={'user': 'user', 'host': 'host'}, remote_dir='/remote',
        run_dir=tmp_path, _run_config={}, _remote_json=lambda name: {'frame_index': 20},
        _edit_with_clirelay=edit, _progress=lambda *a: None)
    for _ in range(2):
        service_refresh(transport, 'repair-00001', deadline=float('inf'), should_stop=lambda: False)
    assert len(edits) == 1
    assert transfers[-1][0].endswith('response.json')
    response = json.loads((request / 'extended/refresh/00020/response.json').read_text())
    assert response['status'] == ('error' if failure else 'ok')
    if not failure:
        assert transfers[-2][0].endswith('regenerated.png')


def test_cancelled_refresh_does_not_call_relay():
    transport = SimpleNamespace(_remote_json=lambda name: pytest.fail('Cancelled request must not be fetched'))
    service_refresh(transport, 'repair-00001', deadline=float('inf'), should_stop=lambda: True)


def test_periodic_edits_are_direct_fitting_targets(tmp_path):
    from splat_explorer.scene_runs_ext.pipeline import repair
    # Use the same tiny scene/renderer fixtures as the existing pipeline tests.
    from test_scene_runs_ext import scene, camera, Renderer, propagate
    Image.new('RGB', (32, 32), (210, 210, 210)).save(tmp_path / 'anchor.png')
    def generate(root, runtime, stop):
        result = propagate(root, runtime, stop)
        Image.new('RGB', (32, 32), (240, 240, 240)).save(root / 'fixed-20.png')
        (root / 'inference.json').write_text(json.dumps({
            'periodic_refreshes': [{'frame_index': 20}],
            'generated_references': [{'frame_index': 20, 'path': 'fixed-20.png', 'kind': 'edited_render'}]}))
        return result
    def fit(candidate, cameras, targets, **kwargs):
        assert kwargs['edited_indices'] == [0, 20]
        assert len(cameras) == 24
        assert int(targets[20][0, 0, 0]) == 240
        return {}
    _, metrics = repair(scene(), camera(), tmp_path / 'anchor.png', tmp_path,
        options={'frames': 25, 'fit_iterations': 1, 'block_schedule': 'periodic_starter'},
        runtime={}, proposal={}, should_stop=lambda: False, on_progress=lambda _: None,
        renderer_factory=Renderer, propagator=generate, fitter=fit)
    manifest = json.loads((tmp_path / 'extended/bundle.json').read_text())
    assert len(manifest['references']) == 1, 'Replay must start with only initial references'
    assert manifest['generated_references'][0]['frame_index'] == 20
    assert metrics['reference_views'] == manifest['reference_views'] == 2


def test_periodic_option_persists_and_rejects_non_gpt_backend(tmp_path):
    from splat_explorer.scene_runs.models import SceneRunConfig
    from splat_explorer.scene_runs.store import SceneRunStore
    from splat_explorer.web.scene_run_studio import SceneRunStudio
    app = SimpleNamespace(cfg=SimpleNamespace(output=SimpleNamespace(dir=str(tmp_path))))
    store = SceneRunStore(tmp_path / 'runs')
    studio = SceneRunStudio(app, store, pipeline='extended')
    created = studio.create({'extended': {'block_schedule': 'periodic_starter'}})
    assert store.get_run(created['run_id']).config.extended['block_schedule'] == 'periodic_starter'
    with pytest.raises(ValueError, match='GPT-image'):
        SceneRunConfig(pipeline='extended', extended={'block_schedule': 'periodic_starter'},
                       image_edit_backend='qwen-image-edit')


def test_wait_response_services_periodic_requests(tmp_path, monkeypatch):
    from splat_explorer import repair_lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.scene_runs_ext import periodic_refresh
    calls = []
    monkeypatch.setattr(periodic_refresh, 'service_refresh', lambda *a, **kw: calls.append(a[1]))
    monkeypatch.setattr(repair_lrz, '_ssh_run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='{"status":"ok"}'))
    transport = object.__new__(LrzSceneRunTransport)
    transport._run_config = {'extended': {'block_schedule': 'periodic_starter'}}
    transport.remote_dir = '/remote'
    transport.cfg = {}
    assert transport._wait_response('repair-00001', deadline=float('inf'), should_stop=lambda: False) == {'status': 'ok'}
    assert calls == ['repair-00001']


def test_repairs_are_independent_calibrated_references_with_bounded_active_bank(tmp_path):
    torch = pytest.importorskip('torch')
    # Nonzero segment start catches confusion between local RGB offsets and
    # global frame IDs. Poses are marker tensors, not temporal averages.
    refresh = PeriodicRefresh(tmp_path, list(range(100, 381)))
    refresh.reference_w2cs = torch.arange(281).float()[:, None, None].expand(-1, 4, 4)
    refresh.reference_Ks = (1000 + torch.arange(281)).float()[:, None, None].expand(-1, 3, 3)
    encoded_pixels = []
    def encode_neighbors(rgb, max_neighbors_per_encode):
        assert rgb.shape == (1, 1, 3, 16, 16)
        assert max_neighbors_per_encode == 1
        value = round(float(rgb[0, 0, 0, 0, 0]) * 255)
        encoded_pixels.append(value)
        return torch.full((1, 1, 1, 2, 2), value, dtype=torch.float32)
    pipe = SimpleNamespace(vae=SimpleNamespace(device='cpu'), encode_neighbors=encode_neighbors)
    neighbors = torch.full((1, 1, 1, 2, 2), 17.)
    poses = torch.zeros(1, 1, 4, 4)
    Ks = torch.full((1, 1, 3, 3), 1000.)
    for n, frame in enumerate(range(20, 281, 20), 1):
        path = f'repair-{frame}.png'
        Image.new('RGB', (16, 16), (n, n, n)).save(tmp_path / path)
        refresh.records.append({'frame_index': frame + 100, 'path': path})
        neighbors, poses, Ks = refresh.update_reference_conditioning(pipe, frame, neighbors, poses, Ks)
        assert neighbors[0, 0, 0, 0, 0] == 17, 'Original anchor must remain active'
        assert neighbors[0, 0, -1, 0, 0] == n, 'Use the repaired image, not the video latent'
        assert poses[0, -1, 0, 0] == frame
        assert Ks[0, -1, 0, 0] == frame + 1000
        assert neighbors.shape[2] == poses.shape[1] == Ks.shape[1] == min(n + 1, 12)
        assert refresh.records[-1]['active_reference_indices'] == refresh.active_reference_indices
        assert refresh.records[-1]['reference_camera'] == 'exact_rgb_pose'
    assert encoded_pixels == list(range(1, 15))
    assert len(refresh.records) == 14, 'All repairs stay in the saved bank and fitting targets'
    assert refresh.active_reference_indices == [100, *range(180, 381, 20)]
    assert neighbors[0, 0, :, 0, 0].tolist() == [17, *range(4, 15)]
    assert poses[0, :, 0, 0].tolist() == [0, *range(80, 281, 20)]
    assert Ks[0, :, 0, 0].tolist() == [1000, *range(1080, 1281, 20)]
