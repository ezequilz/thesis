import json
from pathlib import Path

import pytest
from splat_explorer.splatfix import benchmark as b
from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.jobs import validate_job
from splat_explorer.splatfix.repair import UPSTREAM_REVISION
from splat_explorer.splatfix.repeat_benchmark import prepare_repeat


def test_repeat_rebases_segment_provenance_for_both_passes(tmp_path):
    from splat_explorer.splatfix.repeat_benchmark import inherited_inference_command
    old = {'stages': [{'phase': phase, 'status': 'complete', 'command': [
        'python', '/old/segmented_inference.py', '--repo', '/authors',
        '--trajectory-provenance', '/old/provenance.json', '--render_trajectory', 'trajectory',
        '--num_views', '3', '--split_path', '/old/split.json', '--save_dir', '/old/output']}
        for phase in ('inference', 'plus')]}
    for phase in ('inference', 'plus'):
        command = inherited_inference_command(old, phase, tmp_path / 'split.json', tmp_path / phase)
        assert command[command.index('--trajectory-provenance') + 1] == str(tmp_path / 'author-orbit-provenance.json')
        assert command[command.index('--save_dir') + 1] == str(tmp_path / phase)


def fixture(tmp_path):
    prior = tmp_path / 'prior'
    p = prior / 'prepared/bicycle'
    (p / 'artifixer3d/renders').mkdir(parents=True)
    (p / 'artifixer3d/opacity').mkdir()
    (p / 'artifixer3d/repaired.pt').write_bytes(b'repaired splat')
    (p / 'base.pt').write_bytes(b'simple splat')
    (p / 'caption.h5').write_bytes(b'caption')
    atomic_json(p / 'selected.json', [0, 1, 2])
    atomic_json(p / 'artifixer3d/selected.json', [0, 1, 2])
    atomic_json(p / 'targets.json', [3])
    atomic_json(p / 'transforms.json', {'w': 1237, 'h': 822, 'frames': [{'file_path': x} for x in ['a', 'b', 'c', 'd']]})
    original = dict(prompt_path='caption.h5', transforms_path='transforms.json', selected_indices_path='selected.json',
                    target_indices_path='targets.json', reconstruction_checkpoint='base.pt', metric_scale=2, camera_scale=.02)
    repaired = {**original, 'reconstruction_checkpoint': 'artifixer3d/repaired.pt',
                'render_dir': 'artifixer3d/renders', 'opacity_dir': 'artifixer3d/opacity',
                'selected_indices_path': 'artifixer3d/selected.json'}
    atomic_json(p / 'split_trajectory.json', {'test': {'bicycle': original}})
    atomic_json(p / 'split_artifixer3d_plus.json', {'test': {'bicycle': repaired}})
    atomic_json(prior / 'benchmark-run.json', dict(status='complete', upstream_revision=UPSTREAM_REVISION,
        trajectory='author_orbit', selected_images=['a','b','c'],
        stages=[dict(phase=phase, status='complete', command=['python', '-m', 'model_eval.run_inference', '--render_trajectory', 'trajectory', '--num_views', '3', '--split_path', 'old-split', '--save_dir', phase]) for phase in ('inference', 'plus')],
        result={'inference_split': 'prepared/bicycle/split_trajectory.json'}))
    (prior / 'artifixer3d.ply').write_bytes(b'exported splat')
    for name in ('author-orbit.json', 'author-orbit-provenance.json', 'orbit-target-names.json'):
        atomic_json(prior / name, {})
    return prior


def test_repeat_preserves_historical_cameras_and_uses_repaired_checkpoint(tmp_path):
    prior = fixture(tmp_path)
    before = {str(f.relative_to(prior)): f.read_bytes() for f in prior.rglob('*') if f.is_file()}
    _, proof, split = prepare_repeat(prior, tmp_path / 'second')
    _, entry = b._split_entry(split)
    assert b._metadata_path(split, entry, 'reconstruction_checkpoint').read_bytes() == b'repaired splat'
    assert json.loads(b._metadata_path(split, entry, 'transforms_path').read_text())['w'] == 1237
    assert not (split.parent / 'artifixer3d').exists()  # fresh training cannot reuse pass one
    assert not (split.parent / 'split_artifixer3d_plus.json').exists()
    assert proof['repair_pass'] == 2
    assert before == {str(f.relative_to(prior)): f.read_bytes() for f in prior.rglob('*') if f.is_file()}


def test_repeat_rejects_changed_reference_indices_before_copy(tmp_path):
    prior = fixture(tmp_path)
    atomic_json(prior / 'prepared/bicycle/artifixer3d/selected.json', [0, 2, 1])
    with pytest.raises(ValueError, match='selected_indices_path'):
        prepare_repeat(prior, tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


def test_repeat_option_survives_queue_validation():
    options = dict(stage='benchmark', source='/source', repeat_from='run_20261005_142842')
    assert validate_job(options)['repeat_from'] == options['repeat_from']
    with pytest.raises(ValueError, match='cannot be combined'):
        validate_job({**options, 'preparation_from': 'run_20261005_142842'})


def test_historical_inference_without_settings_replays_exact_command(tmp_path):
    from splat_explorer.splatfix.repeat_benchmark import inherited_inference_command
    prior = fixture(tmp_path)
    old = json.loads((prior / 'benchmark-run.json').read_text())
    assert 'inference_settings' not in old
    old['stages'][0]['command'] += ['--sink_size', '11']
    command = inherited_inference_command(old, 'inference', '/new/split', '/new/inference')
    expected = list(old['stages'][0]['command'])
    expected[expected.index('--split_path') + 1] = '/new/split'
    expected[expected.index('--save_dir') + 1] = '/new/inference'
    assert command == expected
    assert old['stages'][0]['command'][old['stages'][0]['command'].index('--split_path') + 1] == 'old-split'


def test_training_alignment_preserves_camera_poses(tmp_path, monkeypatch):
    from splat_explorer.splatfix.repeat_benchmark import align_repeat_inputs
    from splat_explorer.splatfix import resolution
    root = tmp_path / 'prepared'; root.mkdir()
    pose = [[1,0,0,2],[0,1,0,3],[0,0,1,4],[0,0,0,1]]
    calibration = dict(w=1237,h=822,fl_x=1162.,fl_y=1162.,cx=618.25,cy=410.75)
    atomic_json(root / 'transforms.json', {**calibration,'frames':[{'transform_matrix':pose}, {**calibration,'transform_matrix':pose,'file_path':'images/a.jpg'}]})
    split = root / 'split.json'
    atomic_json(split, {'test':{'bicycle':dict(image_root='input',transforms_path='transforms.json')}})
    monkeypatch.setattr(resolution, 'prepare_colmap', lambda *args: {'profile':'training'})
    assert align_repeat_inputs(split,'training')['profile']=='training'
    _,entry=b._split_entry(split)
    result=json.loads((root/entry['transforms_path']).read_text())
    assert (result['w'], result['h'])==(720,480)
    assert all(f['transform_matrix']==pose for f in result['frames'])
    assert all((f['w'],f['h'])==(720,480) for f in result['frames'])
    assert result['frames'][1]['file_path']=='images/a.jpg'


@pytest.mark.parametrize('value', [0, -1, 1.5, '2', True, None])
def test_chain_count_rejects_non_positive_integers(value):
    with pytest.raises(ValueError, match='positive integer'):
        validate_job(dict(stage='benchmark', source='/source', benchmark_runs=value))


def test_chain_count_defaults_and_schedule():
    assert validate_job(dict(stage='benchmark', source='/source'))['benchmark_runs'] == 1
    options = validate_job(dict(stage='benchmark', source='/source', benchmark_runs=3,
                                scheduled_at='2026-10-10T12:00:00+02:00'))
    assert options['benchmark_runs'] == 3
    assert options['scheduled_at'] == '2026-10-10T10:00:00+00:00'


@pytest.mark.parametrize('count', [1, 2, 3])
def test_worker_chains_previous_output_and_evaluates_each_pass(tmp_path, monkeypatch, count):
    from splat_explorer.splatfix.artifixer import worker, benchmark, repeat_benchmark, run_evaluation
    calls, evaluations, events = [], [], []
    runtime = {'resolution_profile': 'training', 'regularization_profile': 'artifixer'}
    atomic_json(tmp_path / 'worker-request.json', dict(stage='benchmark', source='/original',
                output=str(tmp_path / 'results'), runtime=runtime, benchmark_runs=count))
    def run(source, output, **kwargs):
        calls.append(source)
        kwargs['on_progress']({'phase': 'inference'})
        events.append(json.loads((tmp_path / 'worker-status.json').read_text()))
        return {'output_dir': str(Path(output) / f'pass-{len(calls)}')}
    monkeypatch.setattr(benchmark, 'run_benchmark', run)
    monkeypatch.setattr(repeat_benchmark, 'run_repeat_benchmark', run)
    monkeypatch.setattr(run_evaluation, 'finalize_run_evaluation', lambda root, *a, **kw: evaluations.append(root))
    worker.execute(tmp_path)
    status = json.loads((tmp_path / 'worker-status.json').read_text())
    assert status['status'] == 'completed'
    assert calls == ['/original'] + [str(tmp_path / 'results' / f'pass-{i}') for i in range(1, count)]
    assert evaluations == [str(tmp_path / 'results' / f'pass-{i}') for i in range(1, count + 1)]
    assert [e['benchmark_pass'] for e in events] == list(range(1, count + 1))
    assert all(e['benchmark_runs'] == count for e in events)
    assert status['result']['output_dir'] == evaluations[-1]


@pytest.mark.parametrize('cancel', [False, True])
def test_chain_stops_before_next_pass_on_failure_or_cancel(tmp_path, monkeypatch, cancel):
    from splat_explorer.splatfix.artifixer import worker, benchmark, repeat_benchmark, run_evaluation
    atomic_json(tmp_path / 'worker-request.json', dict(stage='benchmark', source='/original',
                output=str(tmp_path / 'results'), runtime={}, benchmark_runs=3))
    def first(*args, **kwargs):
        if not cancel:
            raise RuntimeError('inference failed')
        (tmp_path / 'STOP').touch()
        return {'output_dir': str(tmp_path / 'results/pass-1')}
    monkeypatch.setattr(benchmark, 'run_benchmark', first)
    monkeypatch.setattr(repeat_benchmark, 'run_repeat_benchmark', lambda *a, **kw: pytest.fail('Must not start another pass'))
    monkeypatch.setattr(run_evaluation, 'finalize_run_evaluation', lambda *a, **kw: None)
    worker.execute(tmp_path)
    status = json.loads((tmp_path / 'worker-status.json').read_text())
    assert status['status'] == ('stopped' if cancel else 'error')


def test_third_pass_uses_second_checkpoint_and_original_conditioning(tmp_path):
    import shutil
    prior = fixture(tmp_path)
    old, _, split = prepare_repeat(prior, tmp_path / 'second')
    second = tmp_path / 'second'
    shutil.copytree(prior / 'prepared/bicycle/artifixer3d', split.parent / 'artifixer3d')
    (split.parent / 'artifixer3d/repaired.pt').write_bytes(b'second repaired splat')
    _, entry = b._split_entry(split)
    entry.update(reconstruction_checkpoint='artifixer3d/repaired.pt', render_dir='artifixer3d/renders',
                 opacity_dir='artifixer3d/opacity', selected_indices_path='artifixer3d/selected.json')
    atomic_json(split.parent / 'split_artifixer3d_plus.json', {'test': {'bicycle': entry}})
    atomic_json(second / 'benchmark-run.json', {**old, 'repeat_repair': {'repair_pass': 2}})
    (second / 'artifixer3d.ply').write_bytes(b'second export')
    _, proof, third_split = prepare_repeat(second, tmp_path / 'third')
    _, third = b._split_entry(third_split)
    assert b._metadata_path(third_split, third, 'reconstruction_checkpoint').read_bytes() == b'second repaired splat'
    assert proof['repair_pass'] == 3
    for key in ('transforms_path', 'selected_indices_path', 'target_indices_path', 'prompt_path'):
        _, original = b._split_entry(prior / 'prepared/bicycle/split_trajectory.json')
        assert b._metadata_path(third_split, third, key).read_bytes() == b._metadata_path(
            prior / 'prepared/bicycle/split_trajectory.json', original, key).read_bytes()


def test_repeat_reuses_already_aligned_references_and_rerenders_repaired_splat(tmp_path, monkeypatch):
    from splat_explorer.splatfix.artifixer import repeat_benchmark as repeat
    prior = fixture(tmp_path)
    old = json.loads((prior / 'benchmark-run.json').read_text())
    old['runtime'] = {'resolution_profile': 'training', 'model_variant': '1.3b', 'python': 'python', 'repo': '/authors'}
    old['resolution_policy'] = {'profile': 'training'}
    atomic_json(prior / 'benchmark-run.json', old)
    monkeypatch.setattr(b, 'benchmark_runtime', lambda runtime: runtime)
    monkeypatch.setattr(repeat, 'runtime_environment', lambda cfg: {})
    monkeypatch.setattr(repeat, 'align_repeat_inputs', lambda *a: pytest.fail('References must stay unchanged'))
    commands = []
    def stop_after_render(command, **kwargs):
        commands.append(command)
        raise InterruptedError('Test stopped at repaired input rendering')
    monkeypatch.setattr(repeat, 'run_worker', stop_after_render)
    with pytest.raises(InterruptedError, match='Test stopped'):
        repeat.run_repeat_benchmark(prior, tmp_path / 'results', resolution_profile='training')
    assert len(commands) == 1
    assert Path(commands[0][1]).name == 'repeat_render.py'
    split = Path(commands[0][commands[0].index('--split') + 1])
    _, entry = b._split_entry(split)
    assert b._metadata_path(split, entry, 'reconstruction_checkpoint').read_bytes() == b'repaired splat'
    assert b._metadata_path(split, entry, 'transforms_path').read_bytes() == (prior / 'prepared/bicycle/transforms.json').read_bytes()
