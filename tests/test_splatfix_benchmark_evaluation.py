import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from splat_explorer.splatfix import benchmark_evaluation as evaluation
from splat_explorer.splatfix.repair import digest_file


def fixture_run(tmp_path):
    root = tmp_path / 'run'
    source = tmp_path / 'source'
    source.mkdir()
    names = [f'photo_{i:03d}.png' for i in range(30)]
    test_names = names[3:28]
    frames = [{'file_path': f'images/{name}', 'transform_matrix': np.eye(4).tolist()} for name in reversed(names)]
    selected = [29, 28, 27]
    prepared = root / 'prepared/bicycle'
    prepared.mkdir(parents=True)
    (prepared / 'transforms.json').write_text(json.dumps({'frames': frames}))
    (prepared / 'selected.json').write_text(json.dumps(selected))
    hashes = {}
    for name in names:
        target = source / 'colmap/images' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((16, 24, 3), np.uint8)).save(target)
        hashes[str(target.relative_to(source))] = digest_file(target)
    metadata = {'name': 'bicycle', 'test_images': test_names, 'selected_images': names[:3], 'sha256': hashes}
    (source / 'benchmark.json').write_text(json.dumps(metadata))
    directories = {method: root / method for method in evaluation.METHODS}
    for offset, directory in enumerate(directories.values()):
        directory.mkdir()
        for index in range(30):
            # Extra nonpublished predictions must never enter aggregates.
            color = 250 if index in (0, 1, 27, 28, 29) else index + offset
            Image.fromarray(np.full((16, 24, 3), color, np.uint8)).save(directory / f'{index:05d}.png')
    for name, directory in [('split.json', directories['baseline']), ('split_artifixer3d_plus.json', directories['artifixer3d'])]:
        (prepared / name).write_text(json.dumps({'test': {'bicycle': {
            'transforms_path': 'transforms.json', 'selected_indices_path': 'selected.json',
            'render_dir': str(directory)}}}))
    (root / 'benchmark-run.json').write_text(json.dumps({'status': 'complete', 'source_dir': str(source),
        'source_metadata': metadata, 'runtime': {'repo': 'unused-test-repo'}, 'limitation': 'Test fixture'}))
    (root / 'result.json').write_text(json.dumps({'prediction_frames': str(directories['artifixer']),
                                                'plus_frames': str(directories['artifixer3d_plus'])}))
    return root, source


def test_scores_exactly_published_names_remapped_not_all_nonreferences(tmp_path, monkeypatch):
    root, _ = fixture_run(tmp_path)
    calls = []
    def compute(prediction, target):
        calls.append((prediction.shape, target.shape))
        value = float(prediction.mean())
        return {'psnr': value, 'ssim': value, 'lpips': value}
    monkeypatch.setattr(evaluation, 'load_official_metrics', lambda *args: (compute, {'implementation': 'test stub'}))
    result = evaluation.evaluate_benchmark(root)
    assert len(calls) == 100
    assert result['published_test_count'] == 25
    assert [row['prepared_index'] for row in result['per_image']] == list(range(26, 1, -1))
    assert result['aggregate']['baseline']['psnr'] == 14
    assert result['aggregate']['artifixer3d_plus']['psnr'] == 17
    assert result['metrics']['implementation'] == 'test stub'
    assert (root / 'published-test-metrics.json').is_file()
    assert all(shape == ((16, 24, 3), (16, 24, 3)) for shape in calls)


def test_shape_mismatch_fails_without_resampling(tmp_path):
    root, _ = fixture_run(tmp_path)
    Image.new('RGB', (16, 16)).save(root / 'artifixer/00002.png')
    with pytest.raises(ValueError, match='no implicit resize'):
        evaluation.evaluation_plan(root)


def test_authored_orbit_uses_its_own_split_not_source_camera_order(tmp_path):
    root, _ = fixture_run(tmp_path)
    prepared = root / 'prepared/bicycle'
    (prepared / 'split.json').rename(prepared / 'split_trajectory.json')
    (prepared / 'split.json').write_text('{}')  # Source split is not the inference catalogue.
    result = json.loads((root / 'result.json').read_text())
    result['inference_split'] = 'prepared/bicycle/split_trajectory.json'
    (root / 'result.json').write_text(json.dumps(result))
    plan = evaluation.evaluation_plan(root)
    assert len(plan['pairs']) == 25
    assert [row['prepared_index'] for row in plan['pairs']] == list(range(26, 1, -1))


def test_inference_split_cannot_point_to_a_different_run(tmp_path):
    root, _ = fixture_run(tmp_path)
    result = json.loads((root / 'result.json').read_text())
    result['inference_split'] = '../another-run/split.json'
    (root / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='must belong'):
        evaluation.evaluation_plan(root)


@pytest.mark.parametrize('wrong_pose', [False, True])
def test_target_only_orbit_uses_verified_pose_mapping(tmp_path, wrong_pose):
    root, _ = fixture_run(tmp_path)
    prepared = root / 'prepared/bicycle'
    source = json.loads((prepared / 'transforms.json').read_text())
    mapping = {}
    for index, frame in enumerate(source['frames']):
        if 2 <= index <= 26:
            mapping[Path(frame.pop('file_path')).name] = index
    if wrong_pose:
        source['frames'][2]['transform_matrix'][0][3] = 1
    (prepared / 'orbit.json').write_text(json.dumps(source))
    split = json.loads((prepared / 'split.json').read_text())
    split['test']['bicycle']['transforms_path'] = 'orbit.json'
    (prepared / 'split_trajectory.json').write_text(json.dumps(split))
    plus = json.loads((prepared / 'split_artifixer3d_plus.json').read_text())
    plus['test']['bicycle']['transforms_path'] = 'orbit.json'
    (prepared / 'split_artifixer3d_plus.json').write_text(json.dumps(plus))
    result = json.loads((root / 'result.json').read_text())
    result.update(inference_split='prepared/bicycle/split_trajectory.json', trajectory_mode='author_orbit')
    (root / 'result.json').write_text(json.dumps(result))
    manifest = json.loads((root / 'benchmark-run.json').read_text())
    manifest['orbit_provenance'] = {'target_name_to_index': mapping}
    (root / 'benchmark-run.json').write_text(json.dumps(manifest))
    if wrong_pose:
        with pytest.raises(ValueError, match='original published test pose'):
            evaluation.evaluation_plan(root)
    else:
        assert len(evaluation.evaluation_plan(root)['pairs']) == 25


def test_published_ground_truth_digest_is_checked(tmp_path):
    root, source = fixture_run(tmp_path)
    Image.new('RGB', (24, 16), color='red').save(source / 'colmap/images/photo_003.png')
    with pytest.raises(ValueError, match='ground truth changed'):
        evaluation.evaluation_plan(root)


def test_missing_test_prediction_is_not_silently_dropped(tmp_path):
    root, _ = fixture_run(tmp_path)
    (root / 'artifixer3d_plus/00002.png').unlink()
    with pytest.raises(FileNotFoundError, match='missing published test prediction'):
        evaluation.evaluation_plan(root)


def test_reference_leakage_is_rejected(tmp_path):
    root, source = fixture_run(tmp_path)
    metadata = json.loads((source / 'benchmark.json').read_text())
    metadata['test_images'][0] = metadata['selected_images'][0]
    (source / 'benchmark.json').write_text(json.dumps(metadata))
    manifest = json.loads((root / 'benchmark-run.json').read_text())
    manifest['source_metadata'] = metadata
    (root / 'benchmark-run.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='exclude all three reference'):
        evaluation.evaluation_plan(root)


def test_plus_camera_order_mismatch_is_rejected(tmp_path):
    root, _ = fixture_run(tmp_path)
    prepared = root / 'prepared/bicycle'
    transforms = json.loads((prepared / 'transforms.json').read_text())
    transforms['frames'] = list(reversed(transforms['frames']))
    (prepared / 'plus-transforms.json').write_text(json.dumps(transforms))
    split = json.loads((prepared / 'split_artifixer3d_plus.json').read_text())
    split['test']['bicycle']['transforms_path'] = 'plus-transforms.json'
    (prepared / 'split_artifixer3d_plus.json').write_text(json.dumps(split))
    with pytest.raises(ValueError, match='frame order differs'):
        evaluation.evaluation_plan(root)


def metric_repo(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / 'upstream'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init', '-q')
    git('config', 'core.fileMode', 'true')
    metric = repo / 'metric.py'
    metric.write_text('def score(): return 1\n')
    metric.chmod(0o644)
    git('add', 'metric.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-q', '-m', 'Pinned metric fixture')
    revision = git('rev-parse', 'HEAD')
    monkeypatch.setattr(evaluation, 'UPSTREAM_REVISION', revision)
    return repo, metric, git, revision


def test_metric_checkout_tolerates_mount_mode_changes(tmp_path, monkeypatch):
    repo, metric, git, revision = metric_repo(tmp_path, monkeypatch)
    metric.chmod(0o755)
    assert git('status', '--porcelain')  # The ordinary check really would reject it.
    assert evaluation._validate_metric_source(repo) == revision


@pytest.mark.parametrize('staged', [False, True])
def test_metric_checkout_rejects_actual_content_changes_despite_mode_change(tmp_path, monkeypatch, staged):
    repo, metric, git, _ = metric_repo(tmp_path, monkeypatch)
    metric.chmod(0o755)
    metric.write_text('def score(): return 999\n')
    if staged:
        git('add', 'metric.py')
    with pytest.raises(ValueError, match='modified tracked files'):
        evaluation._validate_metric_source(repo)


def test_metric_checkout_still_requires_pinned_revision(tmp_path, monkeypatch):
    repo, _, _, _ = metric_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, 'UPSTREAM_REVISION', '0' * 40)
    with pytest.raises(ValueError, match='must be pinned'):
        evaluation._validate_metric_source(repo)


def test_declared_calibrated_resolution_applies_to_ground_truth_only(tmp_path, monkeypatch):
    from splat_explorer.splatfix.resolution import resize_plan
    root, source = fixture_run(tmp_path)
    manifest_path = root / 'benchmark-run.json'
    manifest = json.loads(manifest_path.read_text())
    policy = resize_plan(24, 16)
    manifest['resolution_policy'] = {'version': 1, 'profile': 'training',
                                    'images': {p.name: policy for p in (source / 'colmap/images').iterdir()}}
    manifest_path.write_text(json.dumps(manifest))
    for method in evaluation.METHODS:
        for p in (root / method).glob('*.png'):
            with Image.open(p) as image:
                image.crop((4, 0, 20, 16)).save(p)
    calls = []
    def compute(pred, gt):
        assert pred.shape == gt.shape == (16, 16, 3)
        calls.append(1)
        return {'psnr': 20., 'ssim': .9, 'lpips': .1}
    monkeypatch.setattr(evaluation, 'load_official_metrics', lambda *args: (compute, {}))
    result = evaluation.evaluate_benchmark(root)
    assert len(calls) == 100
    assert 'calibrated' in result['image_policy']
    assert Image.open(next((source / 'colmap/images').iterdir())).size == (24, 16)
    manifest['resolution_policy']['images'][manifest['source_metadata']['test_images'][0]] = {**policy, 'scale': .5}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='resolution plan differs'):
        evaluation.evaluation_plan(root)
