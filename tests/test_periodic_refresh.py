"""Periodic repair regressions: real CPU tensor math and a fake file transport."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from splat_explorer.scene_runs_ext.config import validate_options
from splat_explorer.scene_runs_ext.periodic_refresh import PeriodicRefresh, service_refresh, loop_closure_reference


def test_periodic_policy_requires_scene_rgb_and_opacity():
    result = validate_options({'block_schedule': 'periodic_starter'})
    assert result['source_conditioning'] == 'rendered'
    with pytest.raises(ValueError, match='opacity'):
        validate_options({'block_schedule': 'periodic_starter', 'source_conditioning': 'none'})


@pytest.mark.parametrize('start,count', [(0, 21), (0, 25), (0, 121), (9, 45)])
def test_periodic_sequences_restart_at_each_edit_and_cover_all_frames(start, count, tmp_path):
    refresh = PeriodicRefresh(tmp_path, list(range(start, start + count)),
                              closure_reference={'path': 'anchor.png'})
    starter = {'frame_index': start, 'path': 'anchor.png'}
    refresh.prepared_references = [{'frame_index': i, 'path': f'{i}.png'}
        for i in range(start + 20, start + count - 1, 20)]
    sequences = list(refresh.sequences(starter))
    assert [s['start'] for s, _ in sequences] == [start, *range(start + 20, start + count - 1, 20)]
    frames = []
    for i, (sequence, reference) in enumerate(sequences):
        assert sequence['start'] == reference['frame_index']
        indices = list(range(sequence['start'], sequence['start'] + sequence['count']))
        if i:
            assert frames[-1] == indices[0]
            indices = indices[1:]
        frames.extend(indices)
    assert frames == list(range(start, start + count))


def test_prepared_restart_records_actual_roundtrip_before_exact_export(tmp_path):
    (tmp_path / 'inputs').mkdir()
    Image.new('RGB', (16, 16), (30, 30, 30)).save(tmp_path / 'inputs/00020.png')
    folder = tmp_path / 'refresh/00020'; folder.mkdir(parents=True)
    Image.new('RGB', (32, 32), (240, 240, 240)).save(folder / 'regenerated.png')
    (folder / 'response.json').write_text('{"status":"ok"}')
    refresh = PeriodicRefresh(tmp_path, list(range(25)))
    reference, = refresh.prepare()
    (folder / 'response.json').unlink()
    roundtrip = tmp_path / 'roundtrip.png'
    Image.new('RGB', (16, 16), (230, 230, 230)).save(roundtrip)
    refresh.record_restart(reference, roundtrip, 10 / 255)
    assert Image.open(folder / 'vae-roundtrip.png').getpixel((0, 0)) == (230,)*3
    assert Image.open(folder / 'fixed.png').size == (16, 16)
    assert refresh.records[0]['temporal_encoding'] == 'standalone_first_frame'
    assert refresh.records[0]['vae_roundtrip_mae_0_1'] == 10 / 255


@pytest.mark.parametrize('failure', [False, True])
def test_local_relay_uses_original_prompt_and_anchor_and_publishes_response_last(tmp_path, monkeypatch, failure):
    from splat_explorer import repair_lrz
    request = tmp_path / 'requests/repair-00001'; request.mkdir(parents=True)
    (request / 'request.json').write_text('{"prompt":"Repair the damaged door."}')
    transfers, edits = [], []
    def transfer(args):
        transfers.append(args[-2:])
        if args[-2].startswith('user@host:'):
            Image.new('RGB', (16, 16)).save(Path(args[-1]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(repair_lrz, '_mux_run', transfer)
    monkeypatch.setattr(repair_lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    def edit(source, target, prompt, references):
        edits.append(prompt)
        assert prompt.startswith('Repair the damaged door.')
        assert 'FIRST image' in prompt and 'SECOND image' in prompt
        assert source == request / 'extended/inputs/00020.png'
        assert references == [request / 'extended/anchor.png']
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


@pytest.mark.parametrize('count', [9, 20, 21, 25, 41, 281])
def test_all_planned_references_are_prepared_without_eviction(tmp_path, monkeypatch, count):
    refresh = PeriodicRefresh(tmp_path, list(range(100, 100 + count)))
    calls = []
    def prepare(frame):
        calls.append(frame)
        return {'frame_index': refresh.indices[frame], 'path': f'{frame}.png', 'kind': 'edited_render'}
    monkeypatch.setattr(refresh, 'prepare_frame', prepare)
    prepared = refresh.prepare()
    assert calls == list(range(20, count, 20))
    assert [r['frame_index'] for r in prepared] == list(range(120, 100 + count, 20))
    assert refresh.records == [], 'Preparation must not perform temporal replacements'


def test_failed_preparation_aborts_before_any_replacements(tmp_path, monkeypatch):
    refresh = PeriodicRefresh(tmp_path, list(range(45)))
    def fail(frame):
        raise RuntimeError('GPT repair failed')
    monkeypatch.setattr(refresh, 'prepare_frame', fail)
    with pytest.raises(RuntimeError, match='GPT repair failed'):
        refresh.prepare()
    assert not refresh.records and not refresh.prepared_references


def test_refreshes_share_anchor_and_retry_without_downloading_inputs(tmp_path, monkeypatch):
    from splat_explorer import repair_lrz
    request = tmp_path / 'requests/repair-00001'
    request.mkdir(parents=True)
    (request / 'request.json').write_text('{}')
    downloads, edits = [], []
    pending = {'frame_index': 20}
    fail_upload = [True]

    def transfer(args):
        source, destination = args[-2:]
        if source.startswith('user@host:'):
            downloads.append(source)
            color = 200 if source.endswith('/anchor.png') else int(Path(source).stem)
            Image.new('RGB', (16, 16), (color, color, color)).save(destination)
        elif source.endswith('response.json') and fail_upload[0]:
            fail_upload[0] = False
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    def edit(source, target, prompt, references):
        edits.append(source.name)
        assert Image.open(source).getpixel((0, 0))[0] == pending['frame_index']
        assert references == [request / 'extended/anchor.png']
        assert Image.open(references[0]).getpixel((0, 0))[0] == 200
        Image.new('RGB', (16, 16)).save(target)
        return {}

    monkeypatch.setattr(repair_lrz, '_mux_run', transfer)
    monkeypatch.setattr(repair_lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    transport = SimpleNamespace(cfg={'user': 'user', 'host': 'host'}, remote_dir='/remote',
        run_dir=tmp_path, _run_config={}, _remote_json=lambda name: pending,
        _edit_with_clirelay=edit, _progress=lambda *a: None)
    with pytest.raises(RuntimeError, match='transfer failed'):
        service_refresh(transport, 'repair-00001', deadline=float('inf'), should_stop=lambda: False)
    service_refresh(transport, 'repair-00001', deadline=float('inf'), should_stop=lambda: False)
    pending['frame_index'] = 40
    service_refresh(transport, 'repair-00001', deadline=float('inf'), should_stop=lambda: False)
    assert edits == ['00020.png', '00040.png']
    assert len(downloads) == 3
    assert sum(path.endswith('/anchor.png') for path in downloads) == 1
    assert not list((request / 'extended/refresh').glob('*/anchor.png'))
    assert not list((request / 'extended/refresh').glob('*/rendered.png'))


@pytest.mark.parametrize('count', [21, 25, 121])
def test_closed_loop_skips_final_gpt_call_and_exports_exact_starter(tmp_path, monkeypatch, count):
    # Cover both a scheduled refresh at closure and a non-boundary closing view.
    first = {'transform_matrix': [[1, 0], [0, 1]]}
    manifest = {'transforms': {'frames': [first.copy() for _ in range(count)]}}
    closure = loop_closure_reference(manifest, {'start': 0, 'count': count},
                                     {'path': 'anchor.png'})
    Image.new('RGB', (16, 16), (213, 57, 109)).save(tmp_path / 'anchor.png')
    refresh = PeriodicRefresh(tmp_path, list(range(count)), closure_reference=closure)
    calls = []
    def prepare(frame):
        assert frame != count - 1, 'Closure must never request a GPT repair'
        calls.append(frame)
        return {'frame_index': frame, 'path': f'{frame}.png', 'kind': 'edited_render'}
    monkeypatch.setattr(refresh, 'prepare_frame', prepare)
    bank = refresh.prepare()
    assert calls == list(range(20, count - 1, 20))
    assert all(r['frame_index'] != count - 1 for r in bank)
    output = tmp_path / 'output'
    output.mkdir()
    Image.new('RGB', (16, 16)).save(output / f'{count - 1:05d}.png')
    refresh.export_repairs(output)
    assert (output / f'{count - 1:05d}.png').read_bytes() == (tmp_path / 'anchor.png').read_bytes()
    assert not (tmp_path / 'refresh-request.json').exists()
    manifest['transforms']['frames'][-1] = {'transform_matrix': [[1, 1], [0, 1]]}
    with pytest.raises(ValueError, match='identical'):
        loop_closure_reference(manifest, {'start': 0, 'count': count}, {'path': 'anchor.png'})
