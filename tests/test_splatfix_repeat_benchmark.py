import json
from pathlib import Path

import pytest
from splat_explorer.splatfix import benchmark as b
from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.jobs import validate_job
from splat_explorer.splatfix.repair import UPSTREAM_REVISION
from splat_explorer.splatfix.repeat_benchmark import prepare_repeat


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
    assert (result['w'], result['h'])==(816,544)
    assert all(f['transform_matrix']==pose for f in result['frames'])
    assert all((f['w'],f['h'])==(816,544) for f in result['frames'])
    assert result['frames'][1]['file_path']=='images/a.jpg'
