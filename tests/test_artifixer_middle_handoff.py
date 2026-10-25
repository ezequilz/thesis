"""Offline midpoint inference, pose ownership, and additive reference regressions."""
import contextlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.artifixer import middle_handoff, official_worker
from splat_explorer.splatfix.artifixer.handoff import digest


def trajectory_for(count):
    return {'frames': [{} for _ in range(count + 2)],
            'anchors': [{'frame_index': count + 1}, {'frame_index': 0}],
            'segments': [{'indices': list(range(1, count + 1)), 'cache_seed_index': 0}]}


@pytest.mark.parametrize('count', [1, 2, 3, 4, 5, 61, 62, 63, 64])
def test_middle_plan_preserves_indices_and_shared_camera(count):
    job, = middle_handoff.plan_legs(trajectory_for(count))
    middle = job['midpoint']
    assert job['forward'][-1] == job['backward'][-1] == middle
    assert job['fuse_backward'][0] == job['fuse_forward'][0] == middle
    assert len(job['forward'][1:]) == count // 2 + 1
    assert set(job['forward'][1:]) | set(job['backward'][1:]) == set(range(1, count + 1))
    assert set(job['forward'][1:]) & set(job['backward'][1:]) == {middle}
    replaced = set(job['fuse_backward']) | set(job['fuse_forward'])
    assert len(replaced) == max(1, (count + 2) // 3)
    assert sorted(replaced) == list(range(min(replaced), max(replaced) + 1))


def test_middle_options_are_opt_in_and_validated():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    assert validate_options({'stage': 'repair'})['handoff_type'] == 'boundary'
    options = validate_options({'stage': 'repair', 'handoff_generation': True, 'handoff_type': 'middle'})
    assert options['handoff_type'] == 'middle'
    assert options['split_mode'] == 'single-split'
    with pytest.raises(ValueError, match='handoff_type'):
        validate_options({'stage': 'repair', 'handoff_type': 'unknown'})


@pytest.mark.parametrize('insertion', [False, True])
def test_middle_passes_fit_and_export_keep_originals(tmp_path, monkeypatch, insertion):
    class Tensor(np.ndarray):
        def permute(self, *axes): return self.transpose(axes)
        def float(self): return self.astype(np.float32)
    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    module('torch', bool=np.bool_, from_numpy=lambda x: np.asarray(x).view(Tensor),
           stack=np.stack, tensor=np.asarray, ones=lambda n, dtype: np.ones(n, dtype=dtype),
           inference_mode=contextlib.nullcontext)
    events = []
    pipe = SimpleNamespace(clear_inference_caches=lambda: events.append('clear'),
                           vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    def process(pipe, item, args, output, *rest):
        assert events[-1] == 'clear'
        events.append(output.name)
        seed = item['frame_indices'][0]
        assert not item['valid_frames_mask'][0]
        assert np.all(item['opacity'][0] == 1)
        assert item['rgb_rendered'][0, 0, 0, 0] == pytest.approx((200 if seed == 5 else 100 + seed) / 255)
        dest = output / 'splatfix/frames/batch_0000/pred'
        dest.mkdir(parents=True)
        color = {'forward': 10, 'backward': 20, 'fuse_backward': 30, 'fuse_forward': 40}[output.name]
        for index in item['frame_indices'][1:]:
            Image.new('RGB', (8, 8), (color, int(index), 0)).save(dest / f'{index:05d}.png')
        pipe.starter_inference_audit = {'encoded_starter_preserved': True}
    module('model_eval')
    module('model_eval.run_inference', process_item=process)
    module('model_training')
    module('model_training.data')
    module('model_training.data.utils', compute_camera_rays=lambda *a, **kw: {})
    trajectory = trajectory_for(9)
    camera_frames = []
    for i, frame in enumerate(trajectory['frames']):
        Image.new('RGB', (8, 8)).save(tmp_path / f'{i}.png')
        np.save(tmp_path / f'{i}.npy', np.zeros((8, 8)))
        frame.update(rgb=f'{i}.png', opacity=f'{i}.npy')
        pose = np.eye(4); pose[0, 3] = i
        camera_frames.append({'transform_matrix': pose.tolist()})
    cameras = {'frames': camera_frames, 'w': 8, 'h': 8, 'fl_x': 8, 'fl_y': 8, 'cx': 4, 'cy': 4}
    trajectory['transforms'] = cameras
    refs = []
    for i in (10, 0):
        path = tmp_path / f'ref-{i}.png'
        Image.new('RGB', (8, 8), (100 + i, 0, 0)).save(path)
        refs.append(str(path))
    original_hashes = [digest(p) for p in refs]
    def repair(root, source, index, camera, **kw):
        events.append('repair')
        assert index == kw['reference_frame_index'] == 5
        assert camera['transform_matrix'] == camera_frames[5]['transform_matrix']
        assert Image.open(source).getpixel((0, 0)) == (10, 5, 0)
        assert Image.open(kw['reference']).getpixel((0, 0)) == (20, 5, 0)
        assert kw['noise']['handoff_noise_vignette']
        assert kw['noise']['handoff_noise_mode'] == 'repaired_pixel_sprinkle'
        fixed = root / 'midpoint.png'
        Image.new('RGB', (8, 8), (200, 0, 0)).save(fixed)
        source.write_bytes(fixed.read_bytes())
        return fixed
    request = {'checkpoint_root': str(tmp_path), 'references': refs, 'camera_scale': 1,
               'runtime': {'handoff_generation': True, 'handoff_type': 'middle', 'image_cache_insertion': insertion}}
    kwargs = dict(pipe=pipe, args=None, device=None, encoded_prompt=None, cameras=cameras,
                  repair_boundary=repair, digest=digest)
    middle_handoff.run(tmp_path, request, trajectory, **kwargs)
    assert events == ['clear', 'forward', 'clear', 'backward', 'repair', 'clear', 'fuse_backward', 'clear', 'fuse_forward']
    state = official_worker.load_handoff_references(tmp_path, request, trajectory)
    assert state['frame_indices'] == [10, 0, 5]
    assert state['paths'][:2] == refs
    assert [digest(p) for p in refs] == original_hashes
    prediction = tmp_path / 'inference/splatfix/frames/batch_0000/pred'
    assert sorted(int(p.stem) for p in prediction.glob('*.png')) == list(range(1, 10))
    assert [Image.open(prediction / f'{i:05d}.png').getpixel((0, 0))[0] for i in range(1, 10)] == [10, 10, 10, 30, 200, 40, 20, 20, 20]
    monkeypatch.setattr(official_worker, 'opengl_transforms', lambda _: cameras)
    _, supervision = official_worker.reconstruction_supervision(tmp_path, request, trajectory)
    assert supervision['selected_indices'] == [0, 5, 10]
    assert supervision['excluded_original_reference_indices'] == []

    from splat_explorer.splatfix.oraclegs.artifixer_inputs import middle_handoff_assets, trajectory_references
    (tmp_path / 'request.json').write_text(json.dumps(request))
    data = lambda name: json.loads((tmp_path / name).read_text())
    studio = SimpleNamespace(result_artifact=lambda manifest, raw: tmp_path / raw)
    row = {'_manifest': tmp_path / 'result.json'}
    exported, references, predictions = middle_handoff_assets(studio, row, trajectory, list(map(Path, refs)),
        {int(p.stem): p for p in prediction.glob('*.png')}, data)
    assert [a['frame_index'] for a in exported['anchors']] == [10, 0, 5]
    assert len(references) == 3
    assignments = trajectory_references(studio, row, exported, predictions)
    assert {i: slot for i, (slot, _) in assignments.items()} == {1: 1, 2: 1, 3: 1, 4: 2, 6: 2, 7: 0, 8: 0, 9: 0}

    # The plus pass reuses the repaired midpoint and never replaces a reference.
    for kind in ('renders', 'opacity'):
        (tmp_path / kind).mkdir()
        for i in range(11): Image.new('RGB', (8, 8)).save(tmp_path / kind / f'{i:05d}.png')
    (tmp_path / 'distillation.json').write_text(json.dumps({'render_dir': str(tmp_path)}))
    middle_handoff.run(tmp_path, request, trajectory, previous_state=state, plus=True, **kwargs)
    assert events.count('repair') == 1
    assert data('plus-references.json') == state

    state['paths'][0] = state['paths'][-1]
    (tmp_path / 'inference-references.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='retain every original'):
        official_worker.load_handoff_references(tmp_path, request, trajectory)


def test_middle_plan_handles_unsorted_anchor_slots_and_closed_leg():
    trajectory = {'frames': [{} for _ in range(12)],
                  'anchors': [{'frame_index': i} for i in (4, 0, 8)],
                  'segments': [{'indices': [1, 2, 3], 'cache_seed_index': 0},
                               {'indices': [5, 6, 7], 'cache_seed_index': 4},
                               {'indices': [9, 10, 11], 'cache_seed_index': 8}]}
    plans = middle_handoff.plan_legs(trajectory)
    assert [job['backward'][0] for job in plans] == [4, 8, 0]
    assert [job['midpoint'] for job in plans] == [2, 6, 10]
    trajectory['segments'][1]['indices'].append(3)
    with pytest.raises(ValueError, match='distinct target'):
        middle_handoff.plan_legs(trajectory)


def test_middle_plan_rejects_missing_camera_coverage():
    trajectory = trajectory_for(9)
    trajectory['segments'][0]['indices'].pop()
    with pytest.raises(ValueError, match='every non-reference'):
        middle_handoff.plan_legs(trajectory)
