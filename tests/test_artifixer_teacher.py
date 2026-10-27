"""Teacher settings must reach the matching checkpoint and retain camera calibration."""
from copy import deepcopy
from pathlib import Path

import pytest

from splat_explorer.splatfix.artifixer.inference_policy import (
    MODELS, TEACHER_DISABLED, inference_options, inference_size, model_runtime,
)
from splat_explorer.splatfix.artifixer.teacher_inference import clip_jobs, planned_clips, resized_cameras


def test_teacher_runtime_pairs_checkpoint_and_sampler_and_disables_causal_adapters():
    source = {'checkpoint': '/cache/artifixer-1.3b.pt', 'model_id': MODELS['1.3b'][1],
              'inference_model': '14b-teacher', **{key: True for key in TEACHER_DISABLED}}
    runtime = model_runtime(source)
    assert runtime['checkpoint'] == '/cache/artifixer-s1-14b.pt'
    assert runtime['model_id'] == MODELS['14b-teacher'][1]
    assert runtime['teacher_steps'] == 50
    assert runtime['teacher_max_frames'] == 37
    assert runtime['teacher_resolution'] == 'native'
    assert all(runtime[key] is False for key in TEACHER_DISABLED)
    assert source['checkpoint'].endswith('1.3b.pt')


def test_historical_custom_checkpoint_path_is_preserved():
    cfg = {'checkpoint': '/custom/experiment.pt', 'model_id': MODELS['1.3b'][1]}
    assert model_runtime(cfg)['checkpoint'] == cfg['checkpoint']
    teacher = {'checkpoint': '/cache/artifixer-s1-14b.pt', 'model_id': MODELS['14b'][1]}
    assert model_runtime(teacher)['inference_model'] == '14b-teacher'


def test_new_jobs_default_to_teacher_but_legacy_opt_out_and_history_survive():
    from splat_explorer.splatfix.jobs import validate_job
    raw = {'stage': 'repair', 'checkpoint': '/checkpoint', 'reconstruction_method': 'artifixer'}
    teacher = validate_job(raw)
    assert teacher['teacher_inference'] and teacher['inference_model'] == '14b-teacher'
    assert teacher['teacher_max_frames'] == 25 and teacher['teacher_steps'] == 50
    assert teacher['teacher_clip_layout'] == 'full-leg'
    assert teacher['initial_image_seeding'] and teacher['image_cache_insertion']
    assert not inference_options({'inference_model': '14b-teacher'})['image_cache_insertion']
    legacy = validate_job({**raw, 'teacher_inference': False, 'initial_image_seeding': True})
    assert legacy['inference_model'] == '1.3b' and legacy['initial_image_seeding']
    explicit = validate_job({**raw, 'inference_model': '14b'})
    assert not explicit['teacher_inference'] and explicit['inference_model'] == '14b'
    historical = model_runtime({'checkpoint': '/cache/artifixer-1.3b.pt', 'model_id': MODELS['1.3b'][1]})
    assert historical['inference_model'] == '1.3b'
    with pytest.raises(ValueError, match='boolean'):
        validate_job({**raw, 'teacher_inference': 'false'})


@pytest.mark.parametrize('change', [
    {'teacher_max_frames': 18}, {'teacher_max_frames': True}, {'teacher_max_frames': 85},
    {'teacher_steps': 0}, {'teacher_steps': 151}, {'teacher_steps': True},
    {'teacher_resolution': 'typo'}, {'inference_model': 'teacher'}, {'teacher_clip_layout': 'typo'},
    {'initial_image_seeding': 'false'}, {'image_cache_insertion': 1},
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


def test_teacher_starters_use_correct_endpoint_and_fit_inside_clip_budget():
    anchors = [0, 13]
    jobs = [({'indices': list(range(14)), 'seed_index': 0}, [0, 1], 'inference'),
            ({'indices': list(range(13, -1, -1)), 'cache_seed_index': 13}, [1, 0], 'inference-back-to-front')]
    clips = list(planned_clips(jobs, 5, anchors, seeding=True))
    assert len(clips) == 6
    assert all(len(c['indices']) + 1 <= 5 for c in clips)
    for directory, seed, order in [('inference', 0, list(range(1, 13))),
                                   ('inference-back-to-front', 13, list(range(12, 0, -1)))]:
        selected = [c for c in clips if c['output'] == directory]
        assert all(c['seed_index'] == seed for c in selected)
        assert [i for c in selected for i in c['indices']] == order
    with pytest.raises(ValueError, match='trusted reference'):
        list(planned_clips([({'indices': [1, 2], 'seed_index': 8}, [0, 1], 'inference')], 5, anchors, seeding=True))
    with pytest.raises(ValueError, match='trusted endpoint'):
        list(planned_clips([({'indices': [1, 2]}, [0, 1], 'inference')], 5, anchors, seeding=True))


def test_preserved_teacher_starter_is_restored_each_step_without_changing_other_frames():
    torch = pytest.importorskip('torch')
    from splat_explorer.splatfix.artifixer.teacher_seeding import preserve_starting_latent
    class Scheduler:
        def step(self, prediction, timestep, latents, *, return_dict):
            return (latents + prediction,)
    class Pipe:
        scheduler = Scheduler()
        def prepare_latents(self, condition, opacity, first):
            return condition.clone()
    pipe = Pipe()
    condition = torch.arange(6.).reshape(1, 1, 3, 1, 2)
    original = condition.clone()
    with preserve_starting_latent(pipe) as audit:
        current = pipe.prepare_latents(condition, torch.ones(1, 9, 1, 2), True)
        for index in range(3):
            assert torch.equal(current[:, :, :1], original[:, :, :1])
            current = pipe.scheduler.step(torch.ones_like(current), index, current, return_dict=False)[0]
            assert torch.equal(current[:, :, 1:], original[:, :, 1:] + index + 1)
        assert torch.equal(current[:, :, :1], original[:, :, :1])
    assert audit['scheduler_steps'] == 3 and audit['encoded_starter_preserved']
    assert torch.equal(condition, original)
    assert 'prepare_latents' not in pipe.__dict__ and 'step' not in pipe.scheduler.__dict__
    with pytest.raises(RuntimeError, match='test failure'):
        with preserve_starting_latent(pipe):
            raise RuntimeError('test failure')
    assert 'prepare_latents' not in pipe.__dict__ and 'step' not in pipe.scheduler.__dict__


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


@pytest.mark.parametrize('layout,seeding,extension', [('segment',False,0),('segment',True,0),
    ('full-leg',False,0),('full-leg',True,0),('full-leg',True,4)])
@pytest.mark.parametrize('opacity_scale,teacher_steps', [(.75, 50), (.5, 150)])
def test_teacher_runs_bounded_cpu_items_and_exports_only_targets(tmp_path, monkeypatch, layout, seeding, extension, opacity_scale, teacher_steps):
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
    limit = 9 if extension else 5
    reference_opacity = .8 if extension else 1.
    frame_count = 20 if extension else 12
    for index in range(frame_count):
        Image.new('RGB', (160, 96), (index, 10, 30)).save(tmp_path / f'{index}.png')
        np.save(tmp_path / f'{index}.npy', np.broadcast_to(np.linspace(0, 1, 160, dtype=np.float32), (96, 160)))
    frames = [{'rgb': f'{i}.png', 'opacity': f'{i}.npy'} for i in range(frame_count)]
    transforms = {'w': 160, 'h': 96, 'fl_x': 100., 'fl_y': 100., 'cx': 80., 'cy': 48.,
                  'frames': [{'transform_matrix': np.eye(4).tolist()} for _ in frames]}
    trajectory = {'frames': frames, 'transforms': transforms, 'anchors': [{'frame_index': 0}, {'frame_index': 10}],
                  'segments': [{'indices': list(range(12)), 'cache_seed_index': 0}]}
    request = {'runtime': {'checkpoint': str(checkpoint), 'model_id': MODELS['14b'][1],
                          'inference_model': '14b-teacher', 'teacher_max_frames': limit,
                          'teacher_steps': teacher_steps,
                          'teacher_clip_layout': layout, 'initial_image_seeding': seeding,
                          'teacher_endpoint_extension':extension, 'teacher_opacity_scale':opacity_scale,
                          'teacher_reference_opacity':reference_opacity},
               'checkpoint_root': str(tmp_path), 'references': [str(tmp_path / '0.png'), str(tmp_path / '10.png')],
               'camera_scale': 1., 'caption_path': 'unused', 'caption_sha256': 'unused', 'seed': 42}
    if extension:
        trajectory.update(base_frame_count=12,context_only_indices=list(range(12,20)),teacher_extensions=[
            {'seed_frame_index':0,'interior_frame_index':1,'frame_indices':list(range(12,16))},
            {'seed_frame_index':10,'interior_frame_index':9,'frame_indices':list(range(16,20))}])
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
        assert len(item['frame_indices']) <= limit
        exported = item['frame_indices'][item['valid_frames_mask']].tolist()
        observed.extend(exported)
        for position,index in enumerate(item['frame_indices'].tolist()):
            if index in (0,10):
                assert not item['valid_frames_mask'][position]
                assert torch.all(item['opacity'][position] == reference_opacity)
                assert position in (0,4)
            else:
                assert torch.allclose(item['opacity'][position], torch.from_numpy(np.load(tmp_path / f'{index}.npy')) * opacity_scale)
                assert torch.allclose(item['rgb_rendered'][position, 0], torch.full((96,160), index / 255))
        folder = output / 'splatfix/frames/batch_0000/pred'
        folder.mkdir(parents=True, exist_ok=True)
        for index in exported:
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
    expected = [i for i in range(12) if i not in (0, 10)] + (list(range(12,20)) if extension else [])
    assert sorted(observed) == expected
    assert set(camera_indices) <= set(range(frame_count))
    predicted = tmp_path / 'output/inference/splatfix/frames/batch_0000/pred'
    assert sorted(int(p.stem) for p in predicted.glob('*.png')) == expected
    assert parsed[parsed.index('--num_inference_steps') + 1] == str(teacher_steps)
    assert parsed[parsed.index('--max_neighbors_per_encode') + 1] == '1'
    assert parsed[parsed.index('--inference_pipeline') + 1] == 'bidirectional'
    report = json.loads((tmp_path / 'output/inference-teacher.json').read_text())
    assert report['complete'] and len(report['clips']) == (4 if layout == 'full-leg' else 3)
    assert report['teacher_clip_layout'] == layout
    assert report['opacity_conditioning']['interpolated_view_multiplier'] == opacity_scale
    assert all(row['opacity_conditioning']['target_min'] == 0 and row['opacity_conditioning']['target_max'] == opacity_scale for row in report['clips'])
    assert all(row['opacity_conditioning']['seeded_reference_opacity'] is None or
               row['opacity_conditioning']['seeded_reference_opacity'] == pytest.approx(reference_opacity) for row in report['clips'])
    assert all(len(row['input_frame_indices']) == row['input_frame_count'] <= limit for row in report['clips'])
    if extension:
        assert report['clips'][0]['seeding']['reference_input_position']==4
        assert not report['image_cache_insertion']
        assert set(range(12,20)) <= set(observed)
        assert len(observed)==len(set(observed))
    if layout == 'full-leg' and not extension:
        assert report['clips'][2]['context_frame_indices']
        assert len(report['clips'][2]['input_frame_indices']) == 4 + int(seeding)


def test_progress_preview_from_isolated_filename_worker(tmp_path):
    import json
    import os
    import site
    import subprocess
    import sys
    from splat_explorer.splatfix.artifixer import teacher_inference
    script = r'''
import json, sys
from pathlib import Path
# Reproduce the filename worker without editable installs or project PYTHONPATH.
sys.path = [p for p in sys.path if 'splat_explorer' not in p and p not in ('', sys.argv[3])]
sys.path.extend(json.loads(sys.argv[2]))
sys.path.insert(0, sys.argv[1])
import teacher_inference
assert not teacher_inference.__package__
before = list(sys.path)
root = Path(sys.argv[4])
teacher_inference.preview_writer()(root, root/'pred', [1, 2], 1,
    inference_model='14b-teacher', inference_complete=False, expected_frame_count=4)
assert sys.path == before
assert json.loads((root/'inference-preview.json').read_text())['reconstruction_inputs'] == [1, 2]
'''
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-S', '-c', script, str(Path(teacher_inference.__file__).parent),
                    json.dumps(site.getsitepackages()), str(Path(teacher_inference.__file__).parents[3]),
                    str(tmp_path)], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)


def test_full_leg_clips_fill_budget_instead_of_inheriting_double_split():
    anchors = [0, 32]
    jobs = [({'indices': list(range(1, 17)), 'seed_index': 0}, [0, 1], 'inference'),
            ({'indices': list(range(31, 16, -1)), 'seed_index': 32}, [0, 1], 'inference')]
    before = deepcopy(jobs)
    clips = list(planned_clips(jobs, 25, anchors, seeding=True, layout='full-leg'))
    assert jobs == before
    assert len(clips) == 2
    assert clips[0]['indices'] == list(range(1, 25))
    assert clips[1]['indices'] == list(range(31, 24, -1))
    assert clips[1]['input_indices'] == list(range(31, 7, -1))
    assert [c['seed_index'] for c in clips] == [0, 32]
    assert all(len(c['input_indices']) + 1 == 25 for c in clips)
    exported = [i for c in clips for i in c['indices']]
    assert sorted(exported) == list(range(1, 32))
    assert len(set(exported)) == len(exported)
    assert set(clips[1]['context_frame_indices']).isdisjoint(clips[1]['indices'])
    # Saved historical requests retain their original ownership and short inputs.
    old = list(planned_clips(jobs, 37, anchors, seeding=True))
    assert [len(c['indices']) for c in old] == [16, 15]


@pytest.mark.parametrize('seeding', [False, True])
def test_full_leg_windows_do_not_cross_anchors_or_missing_captures(seeding):
    anchors = [0, 5, 40]
    targets = [1, 2, 3, 4, *range(6, 30), *range(32, 40)]
    jobs = [({'indices': targets}, [0, 1, 2], 'inference')]
    clips = list(planned_clips(jobs, 25, anchors, seeding=seeding, layout='full-leg'))
    assert sorted(i for c in clips for i in c['indices']) == targets
    assert [len(c['input_indices']) for c in clips] == [4, 24, 8]
    for c in clips:
        assert not set(c['input_indices']).intersection(anchors)
        assert len(c['input_indices']) == len(set(c['input_indices']))
        assert all(abs(a - b) == 1 for a, b in zip(c['input_indices'], c['input_indices'][1:]))


def test_full_leg_endpoint_reverse_outputs_and_single_reference_policy():
    from splat_explorer.splatfix.artifixer.official_worker import inference_jobs
    trajectory = {'anchors': [{'frame_index': 0}, {'frame_index': 32}], 'frames': [{}] * 33}
    request = {'repair_policy': 'endpoints', 'runtime': {'endpoint_reference_mode': 'initial'}}
    clips = list(planned_clips(inference_jobs(request, trajectory), 25, [0, 32],
                              seeding=True, layout='full-leg'))
    for output, seed, expected in [('inference', 0, list(range(1, 32))),
                                   ('inference-back-to-front', 32, list(range(31, 0, -1)))]:
        selected = [c for c in clips if c['output'] == output]
        assert [i for c in selected for i in c['indices']] == expected
        assert all(c['seed_index'] == seed and len(c['references']) == 1 for c in selected)
        assert all(len(c['input_indices']) == 24 for c in selected)
    assert {c['output'] for c in planned_clips(inference_jobs(request, trajectory, plus=True),
        25, [0, 32], seeding=True, layout='full-leg')} == {'plus'}


def test_saved_teacher_request_keeps_37_and_segment_layout():
    source = {'checkpoint': '/cache/artifixer-s1-14b.pt', 'model_id': MODELS['14b'][1],
              'teacher_inference': True, 'inference_model': '14b-teacher', 'initial_image_seeding': True}
    old = model_runtime(source)
    assert old['teacher_max_frames'] == 37
    assert old['teacher_clip_layout'] == 'segment'
    new = model_runtime({**source, 'teacher_max_frames': 25, 'teacher_clip_layout': 'full-leg'})
    assert new['teacher_max_frames'] == 25 and new['teacher_clip_layout'] == 'full-leg'
