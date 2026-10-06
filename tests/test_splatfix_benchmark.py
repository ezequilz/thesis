import json
import struct
from PIL import Image
from pathlib import Path
import pytest
from splat_explorer.splatfix import benchmark, repair


CALIBRATION = {'w': 640, 'h': 480, 'fl_x': 500, 'fl_y': 500, 'cx': 320, 'cy': 240}
POSE = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def source_fixture(tmp_path):
    root = tmp_path / 'source'
    (root / 'colmap/images').mkdir(parents=True)
    (root / 'colmap/sparse/0').mkdir(parents=True)
    names = ['a.jpg', 'b.jpg', 'c.jpg']
    for name in names + ['d.jpg']:
        Image.new('RGB', (640, 480)).save(root / 'colmap/images' / name)
    sparse = root / 'colmap/sparse/0'
    (sparse / 'cameras.bin').write_bytes(struct.pack('<QiiQQdddd', 1, 1, 1, 640, 480, 500, 500, 320, 240))
    with (sparse / 'images.bin').open('wb') as f:
        f.write(struct.pack('<Q', 4))
        for index, name in enumerate(names + ['d.jpg']):
            f.write(struct.pack('<idddddddi', index+1, 1, 0, 0, 0, 0, 0, 0, 1))
            f.write(name.encode() + b'\0' + struct.pack('<Q', 0))
    (sparse / 'points3D.bin').write_bytes(struct.pack('<Q', 0))
    (root / 'selected_images.txt').write_text('\n'.join(names))
    hashes = {str(p.relative_to(root)): repair.digest_file(p) for p in root.rglob('*') if p.is_file()}
    (root / 'benchmark.json').write_text(json.dumps({'name': 'bicycle', 'selected_images': names,
                                                   'test_images': ['d.jpg'], 'sha256': hashes}))
    return root


def install_worker(monkeypatch, missing_scale=False, fail_phase=None, trajectory_mode="source_cameras"):
    calls = []
    cfg = {**repair.DEFAULT_RUNTIME, 'model_variant': '1.3b'}
    if trajectory_mode is not None:
        cfg['trajectory_mode'] = trajectory_mode
    monkeypatch.setattr(benchmark, 'benchmark_runtime', lambda runtime: {**cfg, **(runtime or {})})
    def worker(command, *, log_path, **kwargs):
        def value(flag):
            return command[command.index(flag) + 1]
        phase = Path(log_path).stem
        calls.append((phase, command, kwargs))
        Path(log_path).write_text('official stage')
        if phase == fail_phase:
            if phase == 'orbit_render':
                (Path(value('--output_root')) / 'split.json').write_text('partially replaced split')
            raise RuntimeError('fixture stage failure')
        if phase == 'caption':
            prepared = Path(value('--output_root'))
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / 'caption.h5').write_bytes(b'real caption fixture')
            (prepared / 'base.pt').write_bytes(b'base reconstruction')
            (prepared / 'selected.json').write_text('[0,1,2]')
            (prepared / 'transforms.json').write_text(json.dumps({**CALIBRATION, 'frames': [{'file_path': n, 'transform_matrix': POSE} for n in ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']]}))
            entry = {'prompt_path': 'caption.h5', 'transforms_path': 'transforms.json', 'selected_indices_path': 'selected.json',
                     'reconstruction_checkpoint': 'base.pt', 'camera_scale': .025, 'metric_scale': 2.5}
            if missing_scale:
                del entry['metric_scale']
            (prepared / 'split.json').write_text(json.dumps({'test': {'bicycle': entry}}))
        elif phase == 'orbit_path':
            Path(value('--output')).write_text(json.dumps({**CALIBRATION, 'frames': [{'transform_matrix': POSE}, {'transform_matrix': POSE}]}))
            Path(value('--provenance')).write_text(json.dumps({'target_name_to_index': {'d.jpg': 0}, 'method': 'authors'}))
        elif phase == 'orbit_render':
            prepared = Path(value('--output_root'))
            entry = json.loads((prepared / 'split.json').read_text())['test']['bicycle']
            (prepared / 'orbit-selected.json').write_text('[2,3,4]')
            (prepared / 'orbit-targets.json').write_text('[0,1]')
            (prepared / 'orbit-transforms.json').write_text(json.dumps({**CALIBRATION, 'frames': [{'transform_matrix': POSE}, {'transform_matrix': POSE}] + [{'file_path': n, 'transform_matrix': POSE} for n in ['a.jpg', 'b.jpg', 'c.jpg']]}))
            entry.update(transforms_path='orbit-transforms.json', selected_indices_path='orbit-selected.json', target_indices_path='orbit-targets.json')
            (prepared / 'split.json').write_text(json.dumps({'test': {'bicycle': entry}}))
        elif phase in ('inference', 'plus'):
            directory = Path(value('--save_dir')) / 'checkpoint/official_run/bicycle/frames/batch_0000/pred'
            directory.mkdir(parents=True)
            for index in range(4):
                (directory / f'{index:05d}.png').write_bytes(b'official prediction')
        elif phase == 'artifixer3d':
            prepared = Path(value('--scene_root'))
            checkpoint = prepared / 'fresh.pt'
            checkpoint.write_bytes(b'fresh authors reconstruction')
            (prepared / 'split_artifixer3d_plus.json').write_text(json.dumps({'test': {'bicycle': {'reconstruction_checkpoint': 'fresh.pt'}}}))
        elif phase == 'export':
            Path(value('--output')).write_bytes(b'fresh PLY only')
    monkeypatch.setattr(repair, 'run_worker', worker)
    return calls


def test_original_photographic_recipe_and_final_export(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    calls = install_worker(monkeypatch)
    result = benchmark.run_benchmark(source, tmp_path / 'runs')
    assert [p for p, _, _ in calls] == ['prepare', 'reconstruct', 'render', 'scale', 'caption', 'inference', 'artifixer3d', 'export', 'plus']
    for phase, command, kwargs in calls:
        assert '--metric_scale' not in command
        assert '--base_checkpoint' not in command
        assert '--reconstruction_checkpoint' not in command
        assert kwargs['env']['HF_HUB_OFFLINE'] == '0'
        if phase in ('inference', 'plus'):
            assert command[command.index('--evalset') + 1] == 'reconstructed_colmap'
            assert command[command.index('--render_trajectory') + 1] == 'all_frames'
            assert command[command.index('--num_views') + 1] == '3'
        if phase == 'artifixer3d':
            assert command[2:4] == ['data_processing.run_artifixer3d', '--scene_root']
    assert Path(result['splat_path']).read_bytes() == b'fresh PLY only'
    assert result['metric_scale'] == 2.5 and result['merged'] is False
    assert 'orbit' in result['limitation']
    manifest = json.loads((Path(result['output_dir']) / 'benchmark-run.json').read_text())
    assert manifest['status'] == 'complete'
    assert all(stage['status'] == 'complete' for stage in manifest['stages'])
    assert manifest['source_metadata']['name'] == 'bicycle'
    evaluation = manifest['evaluation']
    assert evaluation['published_test_images'] == ['d.jpg']
    assert evaluation['published_test_prepared_indices'] == [3]
    assert evaluation['source_image_count'] == 4
    assert evaluation['generated_supervision_count'] == 1
    assert evaluation['metrics_computed'] is False
    assert 'all source photographs' in evaluation['caption_input']


def test_missing_metric_scale_never_falls_back_to_synthetic_default(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    calls = install_worker(monkeypatch, missing_scale=True)
    with pytest.raises(ValueError, match='measured metric scale'):
        benchmark.run_benchmark(source, tmp_path / 'runs')
    assert calls[-1][0] == 'caption'
    manifest_path = next((tmp_path / 'runs').glob('*/benchmark-run.json'))
    assert json.loads(manifest_path.read_text())['status'] == 'failed'


def test_import_hashes_reject_changed_photographs_before_gpu(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    (source / 'colmap/images/a.jpg').write_bytes(b'changed')
    calls = install_worker(monkeypatch)
    with pytest.raises(ValueError, match='changed since import'):
        benchmark.run_benchmark(source, tmp_path / 'runs')
    assert not calls


def test_cancel_between_stages_records_cancelled_state(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    calls = install_worker(monkeypatch)
    with pytest.raises(InterruptedError):
        benchmark.run_benchmark(source, tmp_path / 'runs', should_stop=lambda: len(calls) >= 2)
    assert [row[0] for row in calls] == ['prepare', 'reconstruct']
    path = next((tmp_path / 'runs').glob('*/benchmark-run.json'))
    assert json.loads(path.read_text())['status'] == 'cancelled'


def test_release_model_pairing(monkeypatch):
    monkeypatch.setattr(repair, 'validate_runtime', lambda config: config)
    cfg = benchmark.benchmark_runtime({'model_variant': '14b'})
    assert cfg['model_id'] == benchmark.MODEL_IDS['14b']
    assert cfg['checkpoint'].endswith('artifixer-14b.pt')
    with pytest.raises(ValueError, match='paired release'):
        benchmark.benchmark_runtime({'model_variant': '14b', 'checkpoint': '/models/artifixer-1.3b.pt'})
    with pytest.raises(ValueError, match='do not match'):
        benchmark.benchmark_runtime({'model_variant': '14b', 'model_id': benchmark.MODEL_IDS['1.3b']})


def failed_prepared_run(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    install_worker(monkeypatch, fail_phase='caption')
    with pytest.raises(RuntimeError, match='fixture stage failure'):
        benchmark.run_benchmark(source, tmp_path / 'old-runs')
    prior = next((tmp_path / 'old-runs').iterdir())
    prepared = prior / 'prepared/bicycle'
    prepared.mkdir(parents=True)
    (prepared / 'base.pt').write_bytes(b'expensive completed reconstruction')
    (prepared / 'scale_info.txt').write_text('Scale factor: 2.5\n')
    (prepared / 'photo.jpg').symlink_to(source / 'colmap/images/a.jpg')
    (prior / 'inference').mkdir()
    (prior / 'inference/old.png').write_bytes(b'old generated image')
    # Downstream reconstruction must not survive a newly generated inference.
    (prepared / 'artifixer3d').mkdir()
    (prepared / 'artifixer3d/old.pt').write_bytes(b'stale downstream reconstruction')
    return source, prior


def test_resume_copies_only_preparation_materializes_links_and_runs_real_phases(tmp_path, monkeypatch):
    source, prior = failed_prepared_run(tmp_path, monkeypatch)
    old_manifest = (prior / 'benchmark-run.json').read_bytes()
    old_checkpoint = (prior / 'prepared/bicycle/base.pt').read_bytes()
    calls = install_worker(monkeypatch)
    delegate = repair.run_worker
    prefixes = {'prepare': 'Skipping prepare;', 'reconstruct': 'Skipping reconstruction;',
                'render': 'Skipping render;', 'scale': 'Skipping metric alignment;'}
    def worker(command, *, log_path, **kwargs):
        phase = Path(log_path).stem
        root = Path(log_path).parent
        if phase in prefixes:
            prepared = root / 'prepared/bicycle'
            assert (prepared / 'base.pt').read_bytes() == old_checkpoint
            assert not (prepared / 'photo.jpg').is_symlink()
            assert (prepared / 'photo.jpg').read_bytes() == (source / 'colmap/images/a.jpg').read_bytes()
            assert not (root / 'inference').exists()
            assert not (prepared / 'artifixer3d').exists()
            assert not (root / 'caption.log').exists()
        delegate(command, log_path=log_path, **kwargs)
        if phase in prefixes:
            Path(log_path).write_text(prefixes[phase] + ' complete copied artifact\n')
    monkeypatch.setattr(repair, 'run_worker', worker)
    result = benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime={'resume_from': str(prior)})
    root = Path(result['output_dir'])
    manifest = json.loads((root / 'benchmark-run.json').read_text())
    resume = manifest['resume_source']
    assert resume['root'] == str(prior)
    assert resume['manifest_sha256'] == repair.digest_file(prior / 'benchmark-run.json')
    assert resume['copied_file_count'] == 3
    assert 'bicycle/photo.jpg' in resume['copied_prepared_sha256']
    assert [phase for phase, _, _ in calls][:5] == ['prepare', 'reconstruct', 'render', 'scale', 'caption']
    assert all(stage['reuse_evidence'] for stage in manifest['stages'][:4])
    assert (prior / 'benchmark-run.json').read_bytes() == old_manifest
    assert (prior / 'prepared/bicycle/base.pt').read_bytes() == old_checkpoint
    assert (prior / 'prepared/bicycle/photo.jpg').is_symlink()
    assert manifest['resume_supported'] is False  # Complete runs are not failed resumptions.


@pytest.mark.parametrize('field,value', [
    ('status', 'running'), ('input_sha256', {}), ('selected_images', ['wrong.jpg']),
    ('upstream_revision', 'wrong-revision'), ('runtime', {'model_id': 'wrong-model'}),
])
def test_resume_rejects_incompatible_prior_before_new_worker(tmp_path, monkeypatch, field, value):
    source, prior = failed_prepared_run(tmp_path, monkeypatch)
    path = prior / 'benchmark-run.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    calls = install_worker(monkeypatch)
    with pytest.raises(ValueError):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime={'resume_from': str(prior)})
    assert calls == []
    assert not (tmp_path / 'new-runs').exists()


def test_failure_after_resumed_preparation_is_marked_resumable(tmp_path, monkeypatch):
    source, prior = failed_prepared_run(tmp_path, monkeypatch)
    install_worker(monkeypatch, fail_phase='caption')
    with pytest.raises(RuntimeError, match='fixture stage failure'):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime={'resume_from': str(prior)})
    manifest_path = next((tmp_path / 'new-runs').glob('*/benchmark-run.json'))
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'failed'
    assert manifest['resume_supported'] is True
    assert manifest['stages'][-1]['phase'] == 'caption'
    assert manifest['stages'][-1]['status'] == 'failed'


def test_resume_cannot_write_inside_preserved_prior_run(tmp_path, monkeypatch):
    source, prior = failed_prepared_run(tmp_path, monkeypatch)
    calls = install_worker(monkeypatch)
    with pytest.raises(ValueError, match='outside the preserved failed run'):
        benchmark.run_benchmark(source, prior, runtime={'resume_from': str(prior)})
    assert not calls


def test_cancel_materialized_copy_stops_without_mutating_prior(tmp_path):
    prior = tmp_path / 'prior'
    (prior / 'prepared').mkdir(parents=True)
    (prior / 'prepared/data.bin').write_bytes(b'original')
    with pytest.raises(InterruptedError, match='copy stopped'):
        benchmark._copy_preparation({'root': str(prior)}, tmp_path / 'new', lambda: True)
    assert (prior / 'prepared/data.bin').read_bytes() == b'original'


def conditioning_cache(tmp_path, monkeypatch, *, mismatch=False, regular_target=False, blob_layout='repository'):
    import hashlib
    import sys
    from types import SimpleNamespace
    hf_home = tmp_path / 'hf'
    snapshots = {}
    content = {'tokenizer/tokenizer_config.json': b'{}', 'tokenizer/tokenizer.json': b'{"version":1}',
               'text_encoder/config.json': b'{}', 'text_encoder/model.safetensors': b'encoder-fixture'}
    for variant, model_id in benchmark.MODEL_IDS.items():
        cache = hf_home / 'hub' / ('models--' + model_id.replace('/', '--'))
        snapshot = cache / 'snapshots' / (('a' if variant == '1.3b' else 'b') * 40)
        snapshots[model_id] = snapshot
        for name, data in content.items():
            if mismatch and variant == '14b' and name.endswith('.safetensors'):
                data = b'changed-fixture'
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if regular_target and variant == '14b':
                path.write_bytes(data)
            else:
                blob = cache / 'blobs' / hashlib.sha256(data).hexdigest()
                if blob_layout in ('global_sharded', 'outside'):
                    hub = hf_home / 'hub' if blob_layout == 'global_sharded' else tmp_path / 'outside'
                    blob = hub / 'blobs' / blob.name[:2] / blob.name
                blob.parent.mkdir(parents=True, exist_ok=True)
                blob.write_bytes(data)
                path.symlink_to(blob)
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        assert kwargs['local_files_only'] is True
        assert Path(kwargs['cache_dir']) == hf_home / 'hub'
        assert kwargs['allow_patterns'] == ['tokenizer/*', 'text_encoder/*']
        return str(snapshots[kwargs['repo_id']])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot_download))
    return hf_home, calls


def test_global_sharded_hf_blobs_use_identity_without_reading_weights(tmp_path, monkeypatch):
    hf_home, calls = conditioning_cache(tmp_path, monkeypatch, blob_layout='global_sharded')
    def unexpected_hash(path):
        pytest.fail(f'Cached immutable blob must not be rehashed: {path}')
    monkeypatch.setattr(repair, 'digest_file', unexpected_hash)
    identities = [benchmark._conditioning_identity(model, hf_home) for model in benchmark.MODEL_IDS.values()]
    assert identities[0]['files'] == identities[1]['files']
    assert all(record['kind'] == 'huggingface_blob' for record in identities[0]['files'].values())
    assert len(calls) == 2


def test_external_sharded_blob_targets_are_hashed(tmp_path, monkeypatch):
    hf_home, _ = conditioning_cache(tmp_path, monkeypatch, blob_layout='outside')
    original_hash = repair.digest_file
    hashed = []
    def tracked_hash(path):
        hashed.append(path)
        return original_hash(path)
    monkeypatch.setattr(repair, 'digest_file', tracked_hash)
    result = benchmark._conditioning_identity(benchmark.MODEL_IDS['1.3b'], hf_home)
    assert len(hashed) == 4
    assert all(record['kind'] == 'sha256' for record in result['files'].values())


def completed_preparation_run(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    install_worker(monkeypatch)
    result = benchmark.run_benchmark(source, tmp_path / 'old-runs')
    return source, Path(result['output_dir'])


def comparison_runtime(prior, hf_home):
    return {'preparation_from': str(prior), 'hf_home': str(hf_home), 'model_variant': '14b',
            'model_id': benchmark.MODEL_IDS['14b'], 'checkpoint': '/models/artifixer-14b.pt'}


@pytest.mark.parametrize('regular_target', [False, True])
def test_cross_variant_preparation_requires_equal_cached_text_conditioning(tmp_path, monkeypatch, regular_target):
    source, prior = completed_preparation_run(tmp_path, monkeypatch)
    old_manifest = (prior / 'benchmark-run.json').read_bytes()
    old_caption = (prior / 'prepared/bicycle/caption.h5').read_bytes()
    hf_home, cache_calls = conditioning_cache(tmp_path, monkeypatch, regular_target=regular_target)
    calls = install_worker(monkeypatch)
    result = benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime=comparison_runtime(prior, hf_home))
    manifest = json.loads((Path(result['output_dir']) / 'benchmark-run.json').read_text())
    reuse = manifest['resume_source']
    assert reuse['kind'] == 'preparation_from'
    assert reuse['prior_status'] == 'complete'
    assert reuse['shared_conditioning']['verified_equal'] is True
    assert reuse['shared_conditioning']['file_count'] == 4
    assert reuse['shared_conditioning']['caption_sha256'] == repair.digest_file(prior / 'prepared/bicycle/caption.h5')
    assert reuse['shared_conditioning']['initial_checkpoint_sha256'] == repair.digest_file(prior / 'prepared/bicycle/base.pt')
    assert reuse['shared_conditioning']['post_preparation_verified_sha256'] == {
        key: reuse['shared_conditioning'][key] for key in ('caption_sha256', 'initial_checkpoint_sha256')}
    assert len(cache_calls) == 2
    assert [phase for phase, _, _ in calls][:5] == ['prepare', 'reconstruct', 'render', 'scale', 'caption']
    assert manifest['preparation_supported'] is True
    assert 'not a deterministic' in manifest['comparison_limitations']
    assert (prior / 'benchmark-run.json').read_bytes() == old_manifest
    assert (prior / 'prepared/bicycle/caption.h5').read_bytes() == old_caption


@pytest.mark.parametrize('changed_file', ['caption.h5', 'base.pt'])
def test_comparison_rejects_unexpected_preparation_regeneration(tmp_path, monkeypatch, changed_file):
    source, prior = completed_preparation_run(tmp_path, monkeypatch)
    prior_bytes = (prior / 'prepared/bicycle' / changed_file).read_bytes()
    hf_home, _ = conditioning_cache(tmp_path, monkeypatch)
    calls = install_worker(monkeypatch)
    original_worker = repair.run_worker
    def changed_worker(command, **kwargs):
        original_worker(command, **kwargs)
        if Path(kwargs['log_path']).stem == 'caption':
            prepared = Path(command[command.index('--output_root') + 1])
            (prepared / changed_file).write_bytes(b'unexpected regeneration')
    monkeypatch.setattr(repair, 'run_worker', changed_worker)
    with pytest.raises(ValueError, match='Shared preparation changed'):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime=comparison_runtime(prior, hf_home))
    assert calls[-1][0] == 'caption'
    assert (prior / 'prepared/bicycle' / changed_file).read_bytes() == prior_bytes
    manifest = json.loads(next((tmp_path / 'new-runs').glob('*/benchmark-run.json')).read_text())
    assert manifest['status'] == 'failed'
    assert 'post_preparation_verified_sha256' not in manifest['resume_source']['shared_conditioning']


def test_cross_variant_preparation_rejects_different_cached_encoder(tmp_path, monkeypatch):
    source, prior = completed_preparation_run(tmp_path, monkeypatch)
    hf_home, _ = conditioning_cache(tmp_path, monkeypatch, mismatch=True)
    calls = install_worker(monkeypatch)
    with pytest.raises(ValueError, match='text conditioning differs'):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime=comparison_runtime(prior, hf_home))
    assert not calls
    assert not (tmp_path / 'new-runs').exists()


@pytest.mark.parametrize('problem', ['running', 'missing_caption', 'incomplete_caption'])
def test_comparison_requires_terminal_source_and_completed_caption(tmp_path, monkeypatch, problem):
    source, prior = completed_preparation_run(tmp_path, monkeypatch)
    path = prior / 'benchmark-run.json'
    manifest = json.loads(path.read_text())
    if problem == 'running':
        manifest['status'] = 'running'
    elif problem == 'incomplete_caption':
        next(stage for stage in manifest['stages'] if stage['phase'] == 'caption')['status'] = 'failed'
    else:
        (prior / 'prepared/bicycle/caption.h5').unlink()
    path.write_text(json.dumps(manifest))
    calls = install_worker(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime=comparison_runtime(prior, tmp_path / 'unused'))
    assert not calls


def test_resume_remains_same_model_and_options_are_exclusive(tmp_path, monkeypatch):
    source, prior = failed_prepared_run(tmp_path, monkeypatch)
    calls = install_worker(monkeypatch)
    runtime = comparison_runtime(prior, tmp_path / 'unused')
    runtime['resume_from'] = runtime.pop('preparation_from')
    with pytest.raises(ValueError, match='model runtime'):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime=runtime)
    runtime['preparation_from'] = str(prior)
    with pytest.raises(ValueError, match='not both'):
        benchmark.run_benchmark(source, tmp_path / 'new-runs', runtime=runtime)
    assert not calls


def test_default_uses_author_orbit_and_preserves_source_preparation(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    calls = install_worker(monkeypatch, trajectory_mode=None)
    result = benchmark.run_benchmark(source, tmp_path / 'runs')
    root = Path(result['output_dir'])
    assert result['trajectory_mode'] == 'author_orbit'
    assert result['inference_split'] == 'prepared/bicycle/split_trajectory.json'
    source_entry = json.loads((root / 'prepared/bicycle/split.json').read_text())['test']['bicycle']
    assert source_entry['transforms_path'] == 'transforms.json'
    manifest = json.loads((root / 'benchmark-run.json').read_text())
    assert manifest['evaluation']['published_test_prepared_indices'] == [0]
    assert manifest['evaluation']['generated_supervision_count'] == 2
    assert manifest['orbit_provenance']['target_name_to_index'] == {'d.jpg': 0}
    for phase, command, _ in calls:
        if phase in ('inference', 'plus'):
            assert command[command.index('--render_trajectory') + 1] == 'trajectory'
        if phase == 'artifixer3d':
            assert command[command.index('--split_path') + 1].endswith('/split_trajectory.json')
        if phase == 'orbit_path':
            assert command[command.index('--metric-scale') + 1] == '2.5'
            assert command[command.index('--interp-distance') + 1] == '0.1'


def test_failed_orbit_render_restores_source_split(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    install_worker(monkeypatch, trajectory_mode=None, fail_phase='orbit_render')
    with pytest.raises(RuntimeError, match='fixture stage failure'):
        benchmark.run_benchmark(source, tmp_path / 'runs')
    root = next((tmp_path / 'runs').glob('benchmark_*'))
    source_entry = json.loads((root / 'prepared/bicycle/split.json').read_text())['test']['bicycle']
    assert source_entry['transforms_path'] == 'transforms.json'


def test_invalid_trajectory_mode_fails_before_author_commands(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    calls = install_worker(monkeypatch)
    with pytest.raises(ValueError, match='trajectory_mode'):
        benchmark.run_benchmark(source, tmp_path / 'runs', runtime={'trajectory_mode': 'unknown'})
    assert not calls


@pytest.mark.parametrize('mutation, message', [
    ('missing', 'every published'), ('reference', 'distinct target'),
    ('pose', 'exact published'), ('calibration', 'calibration'),
])
def test_orbit_mapping_verifies_original_target_cameras(mutation, message):
    import copy
    source = {**CALIBRATION, 'frames': [{'file_path': 'heldout.jpg', 'transform_matrix': POSE}]}
    prepared = {**CALIBRATION, 'frames': [{'transform_matrix': copy.deepcopy(POSE)}, {'file_path': 'ref.jpg', 'transform_matrix': POSE}]}
    provenance = {'target_name_to_index': {'heldout.jpg': 0}}
    if mutation == 'missing':
        provenance['target_name_to_index'] = {}
    elif mutation == 'reference':
        provenance['target_name_to_index']['heldout.jpg'] = 1
    elif mutation == 'pose':
        prepared['frames'][0]['transform_matrix'][0][3] += .001
    elif mutation == 'calibration':
        prepared['fl_x'] += .25
    with pytest.raises(ValueError, match=message):
        benchmark._orbit_test_indices(source, prepared, provenance, ['heldout.jpg'], [1])


def test_orbit_mapping_rejects_two_test_names_sharing_camera_index():
    source = {**CALIBRATION, 'frames': [{'file_path': name, 'transform_matrix': POSE} for name in ['a.jpg', 'b.jpg']]}
    prepared = {**CALIBRATION, 'frames': [{'transform_matrix': POSE}]}
    with pytest.raises(ValueError, match='distinct target'):
        benchmark._orbit_test_indices(source, prepared, {'target_name_to_index': {'a.jpg': 0, 'b.jpg': 0}}, ['a.jpg', 'b.jpg'], [])


def test_old_or_different_resolution_preparation_cannot_be_reused(tmp_path, monkeypatch):
    source, prior = failed_prepared_run(tmp_path, monkeypatch)
    install_worker(monkeypatch)
    with pytest.raises(ValueError, match='resolution policy differs'):
        benchmark.run_benchmark(source, tmp_path / 'runs', runtime={'resume_from': str(prior), 'resolution_profile': '720p'})
    path = prior / 'benchmark-run.json'
    old = json.loads(path.read_text()); del old['resolution_policy']; path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='resolution policy differs'):
        benchmark.run_benchmark(source, tmp_path / 'runs', runtime={'resume_from': str(prior)})
