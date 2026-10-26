"""Teacher settings must reach the matching checkpoint and retain camera calibration."""
from copy import deepcopy
from pathlib import Path

import pytest

from splat_explorer.splatfix.artifixer.inference_policy import (
    MODELS, TEACHER_DISABLED, inference_options, inference_size, model_runtime,
)
from splat_explorer.splatfix.artifixer.teacher_inference import clip_jobs, resized_cameras


def test_teacher_runtime_pairs_checkpoint_and_sampler_and_disables_causal_adapters():
    source = {'checkpoint': '/cache/artifixer-1.3b.pt', 'model_id': MODELS['1.3b'][1],
              'inference_model': '14b-teacher', **{key: True for key in TEACHER_DISABLED}}
    runtime = model_runtime(source)
    assert runtime['checkpoint'] == '/cache/artifixer-s1-14b.pt'
    assert runtime['model_id'] == MODELS['14b-teacher'][1]
    assert runtime['teacher_steps'] == 50
    assert runtime['teacher_max_frames'] == 17
    assert runtime['teacher_resolution'] == '480p'
    assert all(runtime[key] is False for key in TEACHER_DISABLED)
    assert source['checkpoint'].endswith('1.3b.pt')


def test_historical_custom_checkpoint_path_is_preserved():
    cfg = {'checkpoint': '/custom/experiment.pt', 'model_id': MODELS['1.3b'][1]}
    assert model_runtime(cfg)['checkpoint'] == cfg['checkpoint']
    teacher = {'checkpoint': '/cache/artifixer-s1-14b.pt', 'model_id': MODELS['14b'][1]}
    assert model_runtime(teacher)['inference_model'] == '14b-teacher'


@pytest.mark.parametrize('change', [
    {'teacher_max_frames': 18}, {'teacher_max_frames': True}, {'teacher_max_frames': 85},
    {'teacher_steps': 0}, {'teacher_steps': 101}, {'teacher_steps': True},
    {'teacher_resolution': 'typo'}, {'inference_model': 'teacher'},
])
def test_invalid_teacher_settings_rejected(change):
    with pytest.raises(ValueError):
        inference_options({'inference_model': '14b-teacher', **change})


def test_teacher_job_snapshot_contains_effective_options():
    from splat_explorer.splatfix.jobs import validate_job
    job = validate_job({'stage': 'repair', 'checkpoint': '/checkpoint',
                        'reconstruction_method': 'artifixer', 'inference_model': '14b-teacher',
                        'teacher_steps': 30, 'teacher_resolution': 'native',
                        'handoff_generation': True, 'connected_trajectory_generation': True})
    assert job['inference_model'] == '14b-teacher'
    assert job['teacher_steps'] == 30
    assert job['teacher_resolution'] == 'native'
    assert all(job[key] is False for key in TEACHER_DISABLED)


def test_clip_partition_keeps_order_and_output_ownership():
    indices = list(range(3, 54))
    jobs = [({'indices': indices}, [0, 1], 'inference'),
            ({'indices': indices[::-1]}, [1, 0], 'inference-back-to-front')]
    clips = list(clip_jobs(jobs, 17))
    assert all(len(indices) <= 17 for indices, _, _ in clips)
    assert [i for ids, _, path in clips if path == 'inference' for i in ids] == indices
    assert [i for ids, _, path in clips if path == 'inference-back-to-front' for i in ids] == indices[::-1]
    assert all(refs == ([0, 1] if path == 'inference' else [1, 0]) for _, refs, path in clips)


def test_inference_resize_preserves_normalized_intrinsics_and_poses():
    camera = {'w': 1280, 'h': 720, 'fl_x': 1000., 'fl_y': 950., 'cx': 601., 'cy': 367.,
              'k1': .02, 'frames': [{'transform_matrix': [[1, 0], [0, 1]], 'fl_x': 1050.}]}
    before = deepcopy(camera)
    size = inference_size(1280, 720, '480p')
    assert size == (832, 464)
    result = resized_cameras(camera, size)
    assert camera == before
    for old, new in zip([camera, *camera['frames']], [result, *result['frames']]):
        for key, dimension in [('fl_x', 'w'), ('cx', 'w'), ('fl_y', 'h'), ('cy', 'h')]:
            assert new[key] / new[dimension] == pytest.approx(old.get(key, camera[key]) / camera[dimension])
    assert result['frames'][0]['transform_matrix'] == camera['frames'][0]['transform_matrix']
    assert result['k1'] == camera['k1']
    assert inference_size(1280, 720, 'native') == (1280, 720)
    assert inference_size(320, 240, '480p') == (320, 240)


def test_cli_accepts_teacher_inference_only_controls():
    import argparse
    from splat_explorer.splatfix.cli import _add_stages
    parser = argparse.ArgumentParser()
    _add_stages(parser)
    args = parser.parse_args(['repair', '/checkpoint', '--output', '/output',
                             '--inference-model', '14b-teacher', '--teacher-resolution', '480p',
                             '--teacher-max-frames', '17', '--teacher-steps', '50'])
    assert args.inference_model == '14b-teacher'
    assert args.teacher_resolution == '480p'


def test_teacher_runs_bounded_cpu_items_and_exports_only_targets(tmp_path, monkeypatch):
    import argparse
    import json
    import sys
    from types import ModuleType, SimpleNamespace
    import numpy as np
    from PIL import Image
    torch = pytest.importorskip('torch')
    from splat_explorer.splatfix.artifixer import official_worker, teacher_inference

    def module(name, **members):
        value = ModuleType(name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, value)
    checkpoint = tmp_path / 'artifixer-s1-14b.pt'
    checkpoint.touch()
    for index in range(12):
        Image.new('RGB', (160, 96), (index, 10, 30)).save(tmp_path / f'{index}.png')
        np.save(tmp_path / f'{index}.npy', np.full((96, 160), .5, dtype=np.float32))
    frames = [{'rgb': f'{i}.png', 'opacity': f'{i}.npy'} for i in range(12)]
    transforms = {'w': 160, 'h': 96, 'fl_x': 100., 'fl_y': 100., 'cx': 80., 'cy': 48.,
                  'frames': [{'transform_matrix': np.eye(4).tolist()} for _ in frames]}
    trajectory = {'frames': frames, 'transforms': transforms, 'anchors': [{'frame_index': 0}, {'frame_index': 10}],
                  'segments': [{'indices': list(range(12))}]}
    request = {'runtime': {'checkpoint': str(checkpoint), 'model_id': MODELS['14b'][1],
                          'inference_model': '14b-teacher', 'teacher_max_frames': 5},
               'checkpoint_root': str(tmp_path), 'references': [str(tmp_path / '0.png'), str(tmp_path / '10.png')],
               'camera_scale': 1., 'caption_path': 'unused', 'caption_sha256': 'unused', 'seed': 42}
    monkeypatch.setattr(official_worker, 'validate_caption', lambda *a: None)
    monkeypatch.setattr(teacher_inference, 'load_pipe', lambda *a: SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))))
    monkeypatch.setattr(teacher_inference, 'memory_record', lambda *a, **kw: {'phase': a[2], **kw})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    parsed, observed, camera_indices = [], [], []
    def parse(arguments):
        parsed.extend(arguments)
        return SimpleNamespace(inference_pipeline='bidirectional')
    def process(pipe, item, args, output, rank, device, scale):
        assert args.inference_pipeline == 'bidirectional'
        assert item['rgb_rendered'].device.type == 'cpu'
        assert item['rgb_neighbors'].device.type == 'cpu'
        assert item['rgb_neighbors'].shape[0] == 2
        assert len(item['frame_indices']) <= 5
        observed.extend(item['frame_indices'].tolist())
        folder = output / 'splatfix/frames/batch_0000/pred'
        folder.mkdir(parents=True, exist_ok=True)
        for index in item['frame_indices'].tolist():
            Image.new('RGB', (item['target_w'], item['target_h'])).save(folder / f'{index:05d}.png')
    def rays(cameras, indices, neighbors, **kwargs):
        camera_indices.extend(indices)
        assert neighbors == [0, 10]
        assert kwargs['image_shape'] == (96, 160)
        return {}
    module('model_eval')
    module('model_eval.run_inference', build_parser=lambda: SimpleNamespace(parse_args=parse), process_item=process)
    module('model_training')
    module('model_training.data')
    module('model_training.data.utils', compute_camera_rays=rays, load_encoded_prompt=lambda *a: (torch.zeros(2), None))
    teacher_inference.run(tmp_path / 'output', request, trajectory)
    assert observed == [i for i in range(12) if i not in (0, 10)] == camera_indices
    assert parsed[parsed.index('--num_inference_steps') + 1] == '50'
    assert parsed[parsed.index('--max_neighbors_per_encode') + 1] == '1'
    assert parsed[parsed.index('--inference_pipeline') + 1] == 'bidirectional'
    report = json.loads((tmp_path / 'output/inference-teacher.json').read_text())
    assert report['complete'] and len(report['clips']) == 3
