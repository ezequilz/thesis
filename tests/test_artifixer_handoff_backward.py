import contextlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.artifixer import official_worker as worker


def test_backward_halves_use_completed_handoffs_and_existing_final_repair():
    trajectory = {'anchors': [{'frame_index': i} for i in [6, 0, 12]],
                  'segments': [{'indices': [1, 2, 3, 4, 5]}, {'start': 7, 'count': 5}]}
    jobs = worker.handoff_backward_jobs(trajectory, [5, 0, 12])
    assert [j['generation_global_indices'] for j in jobs] == [[5, 4, 3], [12, 11, 10]]
    assert [j['export_global_indices'] for j in jobs] == [[4, 3], [11, 10]]
    assert len({i for j in jobs for i in j['export_global_indices']}) == 4


@pytest.mark.parametrize('plus', [False, True])
@pytest.mark.parametrize('insertion', [False, True])
@pytest.mark.parametrize('generated_context', [False, True])
def test_backward_replaces_only_far_halves_and_clears_cache(tmp_path, monkeypatch, plus, insertion, generated_context):
    class Tensor(np.ndarray):
        def permute(self, *axes): return self.transpose(axes)
        def float(self): return self.astype(np.float32)
    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
    module('torch', bool=np.bool_, from_numpy=lambda x: np.asarray(x).view(Tensor),
           stack=np.stack, tensor=np.asarray, ones=lambda n, dtype: np.ones(n, dtype=dtype),
           inference_mode=contextlib.nullcontext)
    events = []
    phase = 'plus' if plus else 'inference'
    prediction = tmp_path / phase / 'splatfix/frames/batch_0000/pred'
    prediction.mkdir(parents=True)
    frames = []
    for i in range(13):
        Image.new('RGB', (8, 8), (i, 0, 0)).save(tmp_path / f'{i}.png')
        np.save(tmp_path / f'{i}.npy', np.zeros((8, 8)))
        Image.new('RGB', (8, 8), (i, 0, 0)).save(prediction / f'{i:05d}.png')
        frames.append({'rgb': f'{i}.png', 'opacity': f'{i}.npy'})
    references = []
    for index in [5, 0, 12]:
        path = tmp_path / f'ref-{index}.png'
        Image.new('RGB', (8, 8), (100 + index, 0, 0)).save(path)
        references.append(str(path))
    if plus:
        for kind in ('renders', 'opacity'):
            (tmp_path / kind).mkdir()
            for i in range(13): Image.new('RGB', (8, 8)).save(tmp_path / kind / f'{i:05d}.png')
        (tmp_path / 'distillation.json').write_text(json.dumps({'render_dir': str(tmp_path)}))
    pipe = SimpleNamespace(clear_inference_caches=lambda: events.append('clear'),
                           vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    def process(pipe, item, args, output, *rest):
        assert events[-1] == 'clear'
        events.append('generate')
        indices = item['frame_indices'].tolist()
        assert indices == ([[5, 4, 3], [12, 11, 10]][events.count('generate') - 1])
        assert not item['valid_frames_mask'][0]
        assert np.all(item['opacity'][0] == 1)
        assert item['rgb_rendered'][0, 0, 0, 0] == pytest.approx((100 + indices[0]) / 255)
        assert item['rgb_neighbors'][:3, 0, 0, 0].tolist() == pytest.approx([105/255, 100/255, 112/255])
        assert len(item['rgb_neighbors']) == (5 if generated_context and events.count('generate') == 2 else 3)
        dest = output / 'splatfix/frames/batch_0000/pred'
        dest.mkdir(parents=True, exist_ok=True)
        for index, valid in zip(indices, item['valid_frames_mask']):
            if valid: Image.new('RGB', (8, 8), (200, index, 0)).save(dest / f'{index:05d}.png')
        pipe.starter_inference_audit = {'encoded_starter_preserved': True}
    module('model_eval')
    module('model_eval.run_inference', process_item=process)
    module('model_training')
    module('model_training.data')
    def rays(cameras, indices, neighbors, **kw):
        assert neighbors[:3] == [5, 0, 12]
        return {}
    module('model_training.data.utils', compute_camera_rays=rays)
    trajectory = {'frames': frames, 'anchors': [{'frame_index': i} for i in [6, 0, 12]],
                  'segments': [{'indices': [1, 2, 3, 4, 5]}, {'indices': [7, 8, 9, 10, 11]}]}
    state = {'original_frame_indices': [6, 0, 12], 'frame_indices': [5, 0, 12], 'paths': references}
    cameras = {'w': 8, 'h': 8, 'frames': []}
    for i in range(13):
        pose = np.eye(4); pose[0, 3] = i
        cameras['frames'].append({'transform_matrix': pose.tolist()})
    worker.run_handoff_backward(tmp_path, {'runtime': {'image_cache_insertion': insertion,
        'generated_context_references': generated_context},
        'checkpoint_root': str(tmp_path), 'camera_scale': 1}, trajectory, state,
        pipe=pipe, args=SimpleNamespace(), device=None, encoded_prompt=None, cameras=cameras, plus=plus)
    assert events == ['clear', 'generate', 'clear', 'generate']
    for i in range(13):
        assert Image.open(prediction / f'{i:05d}.png').getpixel((0, 0))[0] == (200 if i in [3, 4, 10, 11] else i)
    archive = tmp_path / (phase + '-forward-original') / 'splatfix/frames/batch_0000/pred'
    assert sorted(p.stem for p in archive.glob('*.png')) == ['00003', '00004', '00010', '00011']
    assert Image.open(archive / '00003.png').getpixel((0, 0))[0] == 3
    manifest = json.loads((tmp_path / (phase + '-backward.json')).read_text())
    assert manifest['complete'] is True
    assert [s['seed_global_index'] for s in manifest['series']] == [5, 12]


def test_new_handoff_default_and_historical_runtime():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    from splat_explorer.splatfix.artifixer.handoff import backward_enabled
    assert validate_options({'stage': 'repair', 'handoff_generation': True})['handoff_backward_pass'] is True
    assert validate_options({'stage': 'repair', 'handoff_generation': True, 'handoff_backward_pass': False})['handoff_backward_pass'] is False
    assert backward_enabled({}) is False
    with pytest.raises(ValueError, match='boolean'):
        backward_enabled({'handoff_backward_pass': 'on'})
