"""Shared requests must never assign a wrong or partial image set to cameras."""
import base64
import copy
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from splat_explorer.image_edit_multiview import request_repairs, validate_response
from splat_explorer.splatfix.checkpoint import Checkpoint, atomic_json
from splat_explorer.splatfix.image_repair import repair_images
from splat_explorer.splatfix.jobs import validate_job
from test_splatfix_checkpoint import make_checkpoint


def response(count=6):
    outputs = []
    for i in range(count):
        buf = io.BytesIO()
        Image.new('RGB', (16, 12), (20 * i, 80, 120)).save(buf, format='PNG')
        outputs.append(dict(type='image_generation_call', id=f'ig_{i}', status='completed',
                            revised_prompt=f'TARGET_VIEW={i:03d}\nRepair this camera.',
                            result=base64.b64encode(buf.getvalue()).decode()))
    return dict(status='completed', output=outputs, tools=[{'type': 'image_generation', 'model': 'gpt-image-2.5-sunburst'}])


class Backend:
    name = model = 'gpt-image-2.5-sunburst'

    def __init__(self, payload=None, mutate=None):
        self.payload = payload or response()
        self.calls = 0
        self.mutate = mutate

    def edit_views(self, targets, prompt, response_path):
        self.calls += 1
        atomic_json(response_path, self.payload)
        if self.mutate:
            self.mutate()
        return validate_response(self.payload, [t['id'] for t in targets])


def test_single_reasoning_request_with_six_labelled_inputs(tmp_path):
    cp = make_checkpoint(tmp_path, count=6)
    calls = []
    client = SimpleNamespace(with_options=lambda **kwargs: calls.append(('options', kwargs)) or client,
                             responses=SimpleNamespace(create=lambda **kwargs:
                                 calls.append(('request', kwargs)) or SimpleNamespace(model_dump=lambda **kw: response())))
    targets = [dict(id=v['id'], path=cp.image_path(v), width=16, height=12) for v in cp.views]
    output = tmp_path / 'response.json'
    result = request_repairs(client, Backend.model, targets, 'repair', output)
    assert set(result) == {f'{i:03d}' for i in range(6)}
    assert len(calls) == 3
    assert calls[0][1]['max_retries'] == 0
    assert calls[1][1]['tool_choice'] == 'none'
    request = calls[2][1]
    assert request['reasoning'] == {'effort': 'high'}
    assert request['tools'][0]['model'] == Backend.model
    assert request['tools'][0]['action'] == 'edit'
    assert 'max_tool_calls' not in request  # rejected by live CliRelay transport
    assert sum(p['type'] == 'input_image' for p in request['input'][0]['content']) == 6
    assert json.loads(output.read_text()) == response()


@pytest.mark.parametrize('problem', ['missing', 'duplicate_label', 'no_label', 'failed', 'duplicate_pixels', 'incomplete', 'bad_base64'])
def test_response_mapping_fails_closed(problem):
    body = response()
    if problem == 'missing': body['output'].pop()
    elif problem == 'duplicate_label': body['output'][1]['revised_prompt'] = body['output'][0]['revised_prompt']
    elif problem == 'no_label': body['output'][0]['revised_prompt'] = 'Repair all views 000 001 002 003 004 005'
    elif problem == 'failed': body['output'][0]['status'] = 'failed'
    elif problem == 'duplicate_pixels': body['output'][1]['result'] = body['output'][0]['result']
    elif problem == 'incomplete': body['status'] = 'incomplete'
    elif problem == 'bad_base64': body['output'][0]['result'] = 'not base64!'
    with pytest.raises(ValueError):
        validate_response(body, [f'{i:03d}' for i in range(6)])


def test_outputs_map_by_labels_not_order():
    body = response()
    body['output'].reverse()
    results = validate_response(body, [f'{i:03d}' for i in range(6)])
    for i in range(6):
        assert results[f'{i:03d}']['call_id'] == f'ig_{i}'


def test_all_six_atomic_switch_preserves_previous_and_is_idempotent(tmp_path):
    cp = make_checkpoint(tmp_path, count=6)
    # Existing files can be reused downstream while the full new set is generated.
    for view in cp.views:
        view['repaired_rgb'] = view['original_rgb']
    cp.save()
    old_paths = [cp.image_path(v, repaired=True) for v in cp.views]
    snapshots = []
    backend = Backend(mutate=lambda: snapshots.append(Checkpoint.load(cp.root).manifest))
    before = copy.deepcopy(cp.manifest)
    repair_images(cp, backend=backend, repair_mode='shared_six', repair_request_id='test')
    assert snapshots == [before]
    assert backend.calls == 1
    restored = Checkpoint.load(cp.root)
    assert all(v['image_repair']['request_id'] == 'test' for v in restored.views)
    assert all(p.exists() for p in old_paths)
    assert all(cp.image_path(v, repaired=True) not in old_paths for v in restored.views)
    repair_images(cp, backend=backend, repair_mode='shared_six', repair_request_id='test')
    assert backend.calls == 1


@pytest.mark.parametrize('problem', ['partial', 'aspect', 'concurrent'])
def test_bad_set_never_replaces_existing_repairs(tmp_path, problem):
    cp = make_checkpoint(tmp_path, count=6)
    for view in cp.views: view['repaired_rgb'] = view['original_rgb']
    cp.save()
    before = (cp.root / 'checkpoint.json').read_bytes()
    payload = response(5 if problem == 'partial' else 6)
    mutate = None
    if problem == 'aspect':
        buf = io.BytesIO(); Image.new('RGB', (12, 12)).save(buf, format='PNG')
        payload['output'][-1]['result'] = base64.b64encode(buf.getvalue()).decode()
    if problem == 'concurrent':
        def mutate():
            other = Checkpoint.load(cp.root)
            other.manifest['metadata']['concurrent_change'] = True
            other.save()
    with pytest.raises(ValueError):
        repair_images(cp, backend=Backend(payload, mutate), repair_mode='shared_six', repair_request_id='test')
    restored = Checkpoint.load(cp.root)
    assert all(v['repaired_rgb'] == v['original_rgb'] for v in restored.views)
    if problem == 'concurrent': assert restored.manifest['metadata']['concurrent_change'] is True
    else: assert (cp.root / 'checkpoint.json').read_bytes() == before
    assert (cp.root / 'image-repairs/shared-test/response.json').exists()


def test_saved_response_recovers_without_paid_call(tmp_path):
    cp = make_checkpoint(tmp_path, count=6)
    backend = Backend()
    original_save = cp.save
    cp.save = lambda: (_ for _ in ()).throw(OSError('disk error'))
    with pytest.raises(OSError):
        repair_images(cp, backend=backend, repair_mode='shared_six', repair_request_id='recover')
    cp.save = original_save
    repair_images(cp, backend=backend, repair_mode='shared_six', repair_request_id='recover')
    assert backend.calls == 1
    assert all(v.get('repaired_rgb') for v in cp.views)


def test_only_six_and_valid_mode_accepted(tmp_path):
    cp = make_checkpoint(tmp_path, count=2)
    backend = Backend()
    with pytest.raises(ValueError, match='exactly six'):
        repair_images(cp, backend=backend, repair_mode='shared_six')
    assert backend.calls == 0
    raw = dict(stage='edit', checkpoint=str(cp.root))
    assert validate_job(raw)['repair_mode'] == 'auto'
    assert validate_job({**raw, 'repair_mode': 'shared_six'})['repair_mode'] == 'shared_six'
    with pytest.raises(ValueError, match='repair_mode'):
        validate_job({**raw, 'repair_mode': 'anything'})


def test_new_set_clears_stale_manual_derivatives_and_archives_manifest(tmp_path):
    cp = make_checkpoint(tmp_path, count=6)
    for view in cp.views:
        view.update(repaired_rgb=view['original_rgb'], manual_resized_rgb=view['original_rgb'],
                    full_repaired_rgb=view['original_rgb'], manual_crop={'old': True})
    cp.save()
    repair_images(cp, backend=Backend(), repair_mode='shared_six', repair_request_id='new')
    assert all('manual_resized_rgb' not in v and 'manual_crop' not in v for v in cp.views)
    archived = json.loads((cp.root / 'image-repairs/shared-new/checkpoint-before.json').read_text())
    assert all(v.get('manual_crop') == {'old': True} for v in archived['views'])


def test_shared_mode_reaches_cli_and_gpt_backend(tmp_path, monkeypatch):
    from splat_explorer.config import Config, load_config
    from splat_explorer.splatfix import cli, executor
    from splat_explorer.image_edit_gpt import GptImageEditBackend
    cp = make_checkpoint(tmp_path, count=6)
    cfg = Config({'agent': {'vlm_backend': 'cli_relay'}})
    run = SimpleNamespace(run_id='job', config=SimpleNamespace(image_edit_backend=Backend.model,
        splatfix={'stage': 'edit', 'checkpoint': str(cp.root), 'resolution_profile': 'training',
                  'repair_mode': 'shared_six'}))
    backend = Backend()
    monkeypatch.setattr('splat_explorer.image_edit_gpt._cli_relay_client', lambda cfg: object())
    monkeypatch.setattr(GptImageEditBackend, 'edit_views', lambda self, *args: backend.edit_views(*args))
    def process(argv, *unused):
        get = lambda flag: argv[argv.index(flag) + 1]
        assert get('--repair-mode') == 'shared_six'
        assert get('--repair-request-id') == 'job'
        cli.edit_views(load_config(get('--config')), SimpleNamespace(checkpoint=cp.root,
            image_set=get('--image-set'), repair_mode=get('--repair-mode'), repair_request_id=get('--repair-request-id')))
    monkeypatch.setattr(executor, 'run_process', process)
    root = tmp_path / 'job'; root.mkdir()
    store = SimpleNamespace(get_run=lambda _: SimpleNamespace(state=SimpleNamespace(details={})))
    executor.SplatfixExecutor(cfg, store)._local(run, root, lambda: False, lambda **kw: None)
    assert backend.calls == 1
    assert all(v['image_repair']['request_id'] == 'job' for v in Checkpoint.load(cp.root).views)


def test_shared_repairs_update_nested_77_capture_set_only(tmp_path):
    cp = make_checkpoint(tmp_path, count=6)
    for view in cp.views:
        view['capture_77'] = {'original_rgb': view['original_rgb'],
                              'camera': copy.deepcopy(view['camera']), 'rgb_renderer': 'viser'}
    cp.save()
    repair_images(cp, backend=Backend(), image_set='77', repair_mode='shared_six', repair_request_id='wide')
    restored = Checkpoint.load(cp.root)
    assert all(not v.get('repaired_rgb') for v in restored.views)
    assert all(v['capture_77']['image_repair']['request_id'] == 'wide' for v in restored.views)
    repair_images(cp, backend=Backend(), image_set='77', repair_mode='shared_six', repair_request_id='wide')


def test_model_substitution_rejected_before_image_generation(tmp_path):
    cp = make_checkpoint(tmp_path, count=6)
    calls = []
    payload = response()
    payload['output'] = []
    payload['tools'][0]['model'] = 'gpt-image-2-codex'
    client = SimpleNamespace(with_options=lambda **kw: client,
        responses=SimpleNamespace(create=lambda **kwargs:
            calls.append(kwargs) or SimpleNamespace(model_dump=lambda **kw: payload)))
    targets = [dict(id=v['id'], path=cp.image_path(v), width=16, height=12) for v in cp.views]
    with pytest.raises(ValueError, match='silent model substitution'):
        request_repairs(client, Backend.model, targets, 'repair', tmp_path / 'response.json')
    assert len(calls) == 1 and calls[0]['tool_choice'] == 'none'
    assert not (tmp_path / 'response.json').exists()
    assert (tmp_path / 'preflight.json').exists()


def test_model_substitution_rejected_in_saved_response():
    payload = response()
    payload['tools'][0]['model'] = 'gpt-image-2-codex'
    with pytest.raises(ValueError, match='silent model substitution'):
        validate_response(payload, [f'{i:03d}' for i in range(6)], expected_model=Backend.model)


def test_native_mode_is_explicit_and_records_actual_model(tmp_path, monkeypatch):
    from splat_explorer.image_edit_gpt import GptImageEditBackend
    from splat_explorer.image_edit_multiview import NATIVE_IMAGE_MODEL
    cp = make_checkpoint(tmp_path, count=6)
    configured = GptImageEditBackend(client=object(), model=Backend.model)
    seen = []
    def edit(self, targets, prompt, response_path):
        seen.append(self.model)
        body = response(); body['tools'][0]['model'] = NATIVE_IMAGE_MODEL
        atomic_json(response_path, body)
        return validate_response(body, [t['id'] for t in targets], expected_model=self.model)
    monkeypatch.setattr(GptImageEditBackend, 'edit_views', edit)
    repair_images(cp, backend=configured, repair_mode='shared_six_native', repair_request_id='native')
    assert seen == [NATIVE_IMAGE_MODEL]
    assert configured.model == Backend.model  # never change sequential Sunburst configuration
    assert all(v['image_repair']['model'] == NATIVE_IMAGE_MODEL for v in cp.views)
    assert all(v['image_repair']['repair_mode'] == 'shared_six_native' for v in cp.views)
    repair_images(cp, backend=configured, repair_mode='shared_six_native', repair_request_id='native')
    assert len(seen) == 1
