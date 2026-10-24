"""Offline sequencing/transport regressions; no GPU or paid image requests."""
import contextlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from splat_explorer.splatfix.artifixer import handoff, official_worker


def test_options_opt_in_and_scoped():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    base = {'stage': 'repair', 'mode': 'edited'}
    assert validate_options(base)['handoff_generation'] is False
    assert validate_options(base)['split_mode'] == 'double-split'
    assert validate_options({**base, 'handoff_generation': True})['split_mode'] == 'single-split'
    for bad in ('on', 1, None):
        with pytest.raises(ValueError, match='boolean'):
            validate_options({**base, 'handoff_generation': bad})
    for bad in ({'mode': 'baseline'}, {'image_trajectory': 'endpoints'}):
        with pytest.raises(ValueError, match='edited checkpoint'):
            validate_options({**base, **bad, 'handoff_generation': True})
    assert 'handoff_generation' not in validate_options({'stage': 'benchmark'})


@pytest.mark.parametrize('plus', [False, True])
@pytest.mark.parametrize('insertion', [False, True])
def test_generation_waits_repairs_then_seeds_exact_boundary(tmp_path, monkeypatch, plus, insertion):
    class Tensor(np.ndarray):
        def permute(self, *axes): return self.transpose(axes)
        def float(self): return self.astype(np.float32)
    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
    module('torch', manual_seed=lambda _: None, device=lambda _: None, bool=np.bool_,
           from_numpy=lambda x: np.asarray(x).view(Tensor), stack=np.stack, tensor=np.asarray,
           ones=lambda count, dtype: np.ones(count, dtype=dtype), inference_mode=contextlib.nullcontext)
    events, items, rays = [], [], []
    pipe = SimpleNamespace(transformer=SimpleNamespace(eval=lambda: SimpleNamespace(requires_grad_=lambda _: None)),
        clear_inference_caches=lambda: events.append('clear'),
        vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    def process(pipe, item, args, output, *rest):
        events.append('generate')
        items.append(item)
        pred = output / 'splatfix/frames/batch_0000/pred'
        pred.mkdir(parents=True, exist_ok=True)
        for index, valid in zip(item['frame_indices'], item['valid_frames_mask']):
            if valid:
                Image.new('RGB', (8, 8), (int(index),) * 3).save(pred / f'{index:05d}.png')
        pipe.starter_inference_audit = {'encoded_starter_preserved': True}
    module('model_eval')
    module('model_eval.run_inference', build_parser=lambda: SimpleNamespace(parse_args=lambda _: None),
           get_eval_pipe=lambda *a: pipe, process_item=process)
    module('model_eval.checkpoint_loading', load_transformer_checkpoint=lambda *a: None)
    module('model_training')
    module('model_training.data')
    def conditioning(cameras, indices, neighbors, **kw):
        rays.append((list(indices), list(neighbors)))
        return {}
    module('model_training.data.utils', compute_camera_rays=conditioning,
           load_encoded_prompt=lambda _: (None, None))
    monkeypatch.setattr(official_worker, 'validate_caption', lambda *a: None)
    monkeypatch.setattr(official_worker, 'install_image_cache_insertion', lambda *a: None)
    frames = []
    for i in range(16):
        Image.new('RGB', (8, 8)).save(tmp_path / f'{i}.png')
        np.save(tmp_path / f'{i}.npy', np.zeros((8, 8)))
        pose = np.eye(4); pose[0, 3] = i
        frames.append({'rgb': f'{i}.png', 'opacity': f'{i}.npy'})
    trajectory = {'frames': frames, 'anchors': [{'frame_index': i} for i in (3, 0, 6, 9, 12, 15)],
        'segments': [{'indices': [1, 2], 'cache_seed_index': 0},
                     {'indices': [4, 5], 'cache_seed_index': 3},
                     {'indices': [7, 8], 'cache_seed_index': 6},
                     {'indices': [10, 11], 'cache_seed_index': 9},
                     {'indices': [13, 14], 'cache_seed_index': 12}]}
    cameras = {'w': 8, 'h': 8, 'frames': [{'transform_matrix': [[i]]} for i in range(16)]}
    monkeypatch.setattr(official_worker, 'opengl_transforms', lambda _: cameras)
    refs = []
    for i in (3, 0, 6, 9, 12, 15):
        path = tmp_path / f'ref{i}.png'
        Image.new('RGB', (8, 8), (100 + i,) * 3).save(path)
        refs.append(str(path))
    repaired = []
    def repair(root, source, index, camera, *, phase):
        events.append('repair')
        assert source.is_file() and camera['transform_matrix'] == [[index]]
        assert index == [2, 5, 8, 11][len(repaired)]
        archive = root / f'boundary-{index}'
        archive.mkdir()
        (archive / 'before.png').write_bytes(source.read_bytes())
        Image.new('RGB', (8, 8), (200 + index,) * 3).save(source)
        fixed = archive / 'repaired.png'; fixed.write_bytes(source.read_bytes())
        repaired.append(index)
        return fixed
    monkeypatch.setattr(handoff, 'repair_boundary', repair)
    if plus:
        for kind in ('renders', 'opacity'):
            (tmp_path / kind).mkdir()
            for i in range(16): Image.new('RGB', (8, 8)).save(tmp_path / kind / f'{i:05d}.png')
        (tmp_path / 'distillation.json').write_text(json.dumps({'render_dir': str(tmp_path)}))
    request = {'runtime': {'checkpoint': '', 'model_id': '', 'handoff_generation': True,
                           'split_mode': 'single-split', 'initial_image_seeding': False,
                           'image_cache_insertion': insertion},
               'references': refs, 'checkpoint_root': str(tmp_path), 'seed': 1,
               'camera_scale': 1, 'caption_path': '', 'caption_sha256': '',
               'trajectory_mode': 'authors_orbit', 'mode': 'edited', 'upstream_revision': '',
               'initialization': '', 'input_adaptation': ''}
    if plus:
        prior_paths = list(refs)
        for slot, index in ((0, 2), (2, 5), (3, 8), (4, 11)):
            path = tmp_path / f'prior-{index}.png'
            Image.new('RGB', (8, 8), (200 + index,) * 3).save(path)
            prior_paths[slot] = str(path)
        (tmp_path / 'inference-references.json').write_text(json.dumps({
            'original_frame_indices': [3, 0, 6, 9, 12, 15],
            'frame_indices': [2, 0, 5, 8, 11, 15], 'paths': prior_paths,
            'sha256': [handoff.digest(path) for path in prior_paths]}))
    official_worker.inference(tmp_path, request, trajectory, plus=plus)
    assert events == ['clear', 'generate'] + ['repair', 'clear', 'generate'] * 4
    expected_rays = [([0, 1, 2], [3, 0, 6, 9, 12, 15]),
                    ([2, 4, 5], [2, 0, 6, 9, 12, 15]),
                    ([5, 7, 8], [2, 0, 5, 9, 12, 15]),
                    ([8, 10, 11], [2, 0, 5, 8, 12, 15]),
                    ([11, 13, 14], [2, 0, 5, 8, 11, 15])]
    if plus:
        expected_rays = [(indices, [2, 0, 5, 8, 11, 15]) for indices, _ in expected_rays]
    assert rays == expected_rays
    for item, colors in zip(items, [(103, 100, 106, 109, 112, 115),
                                  (202, 100, 106, 109, 112, 115),
                                  (202, 100, 205, 109, 112, 115),
                                  (202, 100, 205, 208, 112, 115),
                                  (202, 100, 205, 208, 211, 115)]):
        if plus:
            colors = (202, 100, 205, 208, 211, 115)
        assert len(item['rgb_neighbors']) == 6
        for reference, color in zip(item['rgb_neighbors'], colors):
            np.testing.assert_allclose(reference, color / 255)
    for item, color in zip(items, (100, 202, 205, 208, 211)):
        assert item['valid_frames_mask'].tolist() == [False, True, True]
        np.testing.assert_allclose(item['rgb_rendered'][0], color / 255)
        assert (item['opacity'][0] == 1).all()
    pred = tmp_path / ('plus' if plus else 'inference') / 'splatfix/frames/batch_0000/pred'
    assert [p.name for p in sorted(pred.glob('*.png'))] == [f'{i:05d}.png' for i in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)]
    assert Image.open(pred / '00002.png').getpixel((0, 0)) == (202,) * 3
    assert Image.open(pred / '00005.png').getpixel((0, 0)) == (205,) * 3

    records = json.loads((tmp_path / ('plus-handoffs.json' if plus else 'inference-handoffs.json')).read_text())
    assert [r['replaced_reference_frame_index'] for r in records] == [3, 6, 9, 12]
    assert records[-1]['reference_frame_indices'] == [2, 0, 5, 8, 11, 15]
    assert [a['frame_index'] for a in trajectory['anchors']] == [3, 0, 6, 9, 12, 15]
    for path, index in zip(request['references'], (3, 0, 6, 9, 12, 15)):
        assert Image.open(path).getpixel((0, 0)) == (100 + index,) * 3
    for index in (2, 5, 8, 11):
        assert Image.open(tmp_path / f'boundary-{index}/before.png').getpixel((0, 0)) == (index,) * 3
        assert Image.open(tmp_path / f'boundary-{index}/repaired.png').getpixel((0, 0)) == (200 + index,) * 3


def test_repair_archives_full_response_and_fits_changed_aspect(tmp_path):
    Image.new('RGB', (16, 8)).save(tmp_path / 'before.png')
    request = {'id': 'a' * 32, 'input_sha256': handoff.digest(tmp_path / 'before.png')}
    def answer(size):
        data = io.BytesIO(); Image.new('RGB', size).save(data, format='PNG')
        return SimpleNamespace(error=None, images=[data.getvalue()])
    backend = SimpleNamespace(edit=lambda *args: answer((32, 16)))
    handoff.edit_request(tmp_path, request, backend=backend)
    result = json.loads((tmp_path / 'response.json').read_text())
    assert 'error' not in result
    assert result['model'] == 'gpt-image-2.5-sunburst'
    assert Image.open(tmp_path / 'repaired.png').size == (16, 8)
    backend.edit = lambda *args: pytest.fail('Raw response must be reused without paying again')
    handoff.edit_request(tmp_path, request, backend=backend)
    (tmp_path / 'response.image').write_bytes(answer((32, 32)).images[0])
    handoff.edit_request(tmp_path, request, backend=backend)
    result = json.loads((tmp_path / 'response.json').read_text())
    assert 'error' not in result
    assert Image.open(tmp_path / 'response-original.png').size == (32, 32)
    assert Image.open(tmp_path / 'repaired.png').size == (16, 8)
    assert result['reconstruction_resize']['source_crop_box'] == [0, 8, 32, 24]
    assert result['reconstruction_resize']['method'] == 'cover_center_crop'


def test_boundary_handshake_replaces_only_same_index(tmp_path, monkeypatch):
    mailbox = tmp_path / 'mailbox'; mailbox.mkdir()
    monkeypatch.setenv('SPLATFIX_HANDOFF_MAILBOX', str(mailbox))
    source = tmp_path / '00007.png'; Image.new('RGB', (8, 8)).save(source)
    original = source.read_bytes()
    def respond(_):
        request = json.loads((mailbox / 'request.json').read_text())
        folder = mailbox / request['id']
        Image.new('RGB', (8, 8), 'red').save(folder / 'repaired.png')
        (folder / 'response.image').write_bytes((folder / 'repaired.png').read_bytes())
        handoff.write_json(folder / 'response.json', {**request, 'sha256': handoff.digest(folder / 'repaired.png')})
    monkeypatch.setattr(handoff.time, 'sleep', respond)
    fixed = handoff.repair_boundary(tmp_path, source, 7, {'pose': 'unchanged'}, phase='inference')
    assert source.read_bytes() == fixed.read_bytes() != original
    assert fixed.with_name('before.png').read_bytes() == original
    assert json.loads(fixed.with_name('camera.json').read_text())['camera'] == {'pose': 'unchanged'}
    assert list(tmp_path.glob('0*.png')) == [source]


def test_service_keeps_polling_and_publishes_pixels_before_response(tmp_path, monkeypatch):
    import threading
    gate = threading.Event()
    started = threading.Event()
    ident = 'b' * 32
    uploaded = []
    source = tmp_path / 'remote-before.png'; Image.new('RGB', (8, 8)).save(source)
    request = {'id': ident, 'input_sha256': handoff.digest(source)}
    def ssh(command):
        return json.dumps(request) if '/request.json' in command else ''
    def sync(src, dst):
        if src.startswith('host:'):
            Path(dst).write_bytes(source.read_bytes())
        else:
            uploaded.append(Path(src).name)
    def edit(folder, request, cfg):
        started.set()
        assert gate.wait(5)
        (folder / 'response.image').write_bytes(source.read_bytes())
        (folder / 'repaired.png').write_bytes(source.read_bytes())
        handoff.write_json(folder / 'response.json', request)
    monkeypatch.setattr(handoff, 'edit_request', edit)
    service = handoff.HandoffService(tmp_path / 'local', remote='/remote/handoff',
                                    endpoint='host', ssh=ssh, sync=sync)
    try:
        service.poll()
        assert started.wait(5)
        service.poll()  # Must return while repair is blocked.
        assert uploaded == []
        gate.set()
        service.pending[1].result(timeout=5)
        service.poll()
        assert uploaded == ['response.image', 'repaired.png', 'response.json']
        service.poll()
        assert len(uploaded) == 3
    finally:
        gate.set()
        service.close()


@pytest.mark.parametrize('case', ['stop', 'error', 'checksum'])
def test_boundary_failures_never_replace_input(tmp_path, monkeypatch, case):
    mailbox = tmp_path / 'mailbox'; mailbox.mkdir()
    monkeypatch.setenv('SPLATFIX_HANDOFF_MAILBOX', str(mailbox))
    source = tmp_path / '00001.png'; Image.new('RGB', (8, 8)).save(source)
    original = source.read_bytes()
    if case == 'stop':
        (tmp_path / 'STOP').touch()
    def respond(_):
        request = json.loads((mailbox / 'request.json').read_text())
        folder = mailbox / request['id']
        Image.new('RGB', (8, 8), 'red').save(folder / 'repaired.png')
        handoff.write_json(folder / 'response.json', {
            **request, **({'error': 'relay failed'} if case == 'error' else {'sha256': 'invalid'})})
    monkeypatch.setattr(handoff.time, 'sleep', respond)
    with pytest.raises((InterruptedError, RuntimeError, ValueError)):
        handoff.repair_boundary(tmp_path, source, 1, {}, phase='inference')
    assert source.read_bytes() == original


def test_worker_sets_and_restores_private_mailbox(tmp_path, monkeypatch):
    import os
    from splat_explorer.splatfix.artifixer import worker
    request = {'checkpoint': 'cp', 'output': str(tmp_path / 'result'), 'mode': 'edited',
               'frames': 25, 'span_fraction': .04, 'runtime': {'handoff_generation': True}}
    (tmp_path / 'worker-request.json').write_text(json.dumps(request))
    monkeypatch.setenv('SPLATFIX_HANDOFF_MAILBOX', '/previous/mailbox')
    def run(*args, **kwargs):
        assert os.environ['SPLATFIX_HANDOFF_MAILBOX'] == str(tmp_path / 'handoff')
        raise RuntimeError('test stop')
    monkeypatch.setattr(worker, 'run_repair', run)
    worker.execute(tmp_path)
    assert os.environ['SPLATFIX_HANDOFF_MAILBOX'] == '/previous/mailbox'
    assert json.loads((tmp_path / 'worker-status.json').read_text())['status'] == 'error'


def test_reconstruction_uses_handoffs_and_first_reference_excluding_last_endpoint(tmp_path, monkeypatch):
    original = [0, 3, 6, 9, 12, 15]
    final = [0, 2, 5, 8, 11, 15]
    frames = []
    for i in range(16):
        pose = np.eye(4); pose[0, 3] = i
        frames.append({'transform_matrix': pose.tolist()})
    cameras = {'frames': frames, 'w': 8, 'h': 8, 'fl_x': 8, 'fl_y': 8, 'cx': 4, 'cy': 4}
    monkeypatch.setattr(official_worker, 'opengl_transforms', lambda _: cameras)
    trajectory = {'frames': frames, 'anchors': [{'frame_index': i} for i in original]}
    paths = []
    for i in original:
        path = tmp_path / f'original-{i}.png'; Image.new('RGB', (8, 8)).save(path)
        paths.append(str(path))
    replacements = list(paths)
    for slot, i in enumerate(final[1:-1], 1):
        path = tmp_path / f'repaired-{i}.png'; Image.new('RGB', (8, 8), 'red').save(path)
        replacements[slot] = str(path)
    request = {'runtime': {'handoff_generation': True}, 'references': paths}
    state = {'original_frame_indices': original, 'frame_indices': final, 'paths': replacements,
             'sha256': [handoff.digest(path) for path in replacements]}
    (tmp_path / 'inference-references.json').write_text(json.dumps(state))
    transforms, supervision = official_worker.reconstruction_supervision(tmp_path, request, trajectory)
    kept = [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
    assert len(transforms['frames']) == len(kept)
    assert supervision['distillation_to_original'] == kept
    assert [supervision['original_to_distillation'][i] for i in original[1:]] == [None] * 5
    selected = supervision['selected_indices']
    assert len(selected) == 5
    assert [supervision['groups'][i]['reference'] for i in selected] == replacements[:-1]
    assert [transforms['frames'][i]['transform_matrix'][0][3] for i in selected] == final[:-1]
    assert [g['source_index'] for g in supervision['groups'] if g['reference'] is None] == [1, 4, 7, 10, 13, 14]
    assert supervision['excluded_endpoint_frame_index'] == 15
    assert supervision['reconstruction_reference_slots'] == [0, 1, 2, 3, 4]
    assert json.loads((tmp_path / 'inference-references.json').read_text()) == state
    assert Path(paths[-1]).is_file()
    assert request['references'] == paths
    assert [a['frame_index'] for a in trajectory['anchors']] == original
    Path(replacements[1]).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        official_worker.reconstruction_supervision(tmp_path, request, trajectory)
    # Off never reads the replacement manifest and retains all saved anchors.
    transforms, supervision = official_worker.reconstruction_supervision(tmp_path,
        {**request, 'runtime': {'handoff_generation': False}}, trajectory)
    assert len(transforms['frames']) == 16
    assert supervision['selected_indices'] == original
