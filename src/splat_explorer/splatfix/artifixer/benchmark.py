"""Photographic COLMAP benchmark through the pinned authors' command line tools.

This path does not use the splatfix rendered-anchor adapter. Authors orbit
interpolation provides a reproducible smooth camera trajectory; the unpublished project-page orbit is not claimed.
"""
from __future__ import annotations
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

MODEL_IDS = {'1.3b': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
             '14b': 'Wan-AI/Wan2.1-T2V-14B-Diffusers'}
# Explicit defaults from the pinned release's add_inference_args. Record these
# in the run and pass them to both diffusion stages so the recipe is auditable.
# These are release defaults, not a claim about the unpublished website run.
INFERENCE_DEFAULTS = {
    'inference_pipeline': 'kv_cache',
    'num_inference_steps': 4,
    'frames_per_block': 7,
    'local_attn_size': 21,
    'sink_size': 7,
    'context_parallel_size': 1,
}
LIMITATION = ('Uses the authors orbit interpolation with published reference and test cameras. The exact project-page Bicycle '
              'orbit and original random state are unavailable; this is an authors-code '
              'reproduction with the published reference photographs, not pixel-identical video reproduction. '
              'Generic authors preparation receives all source photographs for metric alignment and '
              'caption generation (194 for the released Bicycle input, including its 25 published test views). '
              'Interpolated targets include the exact published test cameras; reference cameras are appended once as photographic context. '
              'This is not a verified reproduction of the paper evaluation protocol.')
SOURCE_CAMERA_LIMITATION = LIMITATION.replace(
    'Uses the authors orbit interpolation with published reference and test cameras.',
    'Diagnostic mode uses all source COLMAP cameras in source order.').replace(
    'Interpolated targets include the exact published test cameras; reference cameras are appended once as photographic context.',
    'All nonreference source cameras are generated supervision (191 for Bicycle).')


def benchmark_runtime(runtime=None):
    from .repair import DEFAULT_RUNTIME, validate_runtime
    supplied = dict(runtime or {})
    variant = supplied.get('model_variant')
    if variant is None:
        model_id = supplied.get('model_id', DEFAULT_RUNTIME['model_id'])
        variant = next((key for key, value in MODEL_IDS.items() if value == model_id), None)
    if variant not in MODEL_IDS:
        raise ValueError('Benchmark requires the authors 1.3b or 14b release model')
    expected = MODEL_IDS[variant]
    if supplied.get('model_id', expected) != expected:
        raise ValueError('model_variant and model_id do not match')
    supplied['model_id'] = expected
    supplied['model_variant'] = variant
    supplied.setdefault('checkpoint', str(Path(DEFAULT_RUNTIME['checkpoint']).with_name(f'artifixer-{variant}.pt')))
    if Path(supplied['checkpoint']).name != f'artifixer-{variant}.pt':
        raise ValueError(f'Use the paired release checkpoint artifixer-{variant}.pt')
    return validate_runtime(supplied)


def _split_entry(path):
    body = json.loads(path.read_text())
    scenes = body['test']
    if len(scenes) != 1:
        raise ValueError('Benchmark expects exactly one prepared COLMAP scene')
    scene_id, entry = next(iter(scenes.items()))
    return scene_id, entry


def _metadata_path(split, entry, key):
    path = Path(entry[key])
    return path if path.is_absolute() else split.parent / path


def _verify_preparation(split, selected_names):
    scene_id, entry = _split_entry(split)
    for key in ('prompt_path', 'transforms_path', 'selected_indices_path', 'reconstruction_checkpoint'):
        if not _metadata_path(split, entry, key).is_file():
            raise FileNotFoundError(f'Official preparation did not produce {key}')
    metric = entry.get('metric_scale')
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric) or metric <= 0:
        raise ValueError('Official preparation did not produce a positive measured metric scale')
    if not math.isclose(entry['camera_scale'], metric * .01):
        raise ValueError('Official camera conditioning scale differs from metric_scale * 0.01')
    selected = json.loads(_metadata_path(split, entry, 'selected_indices_path').read_text())
    frames = json.loads(_metadata_path(split, entry, 'transforms_path').read_text())['frames']
    actual = [Path(frames[index]['file_path']).name for index in selected]
    if actual != selected_names:
        raise ValueError('Prepared references differ from the published selected photographs')
    return scene_id, entry, len(frames), selected


def _orbit_test_indices(source_data, prepared_data, provenance, names, selected):
    """Resolve target identities without adding forbidden file_path to targets."""
    import numpy as np
    mapping = provenance.get('target_name_to_index')
    if not isinstance(mapping, dict) or set(mapping) != set(names):
        raise ValueError('Orbit provenance must map every published test photograph exactly once')
    indices = [mapping[name] for name in names]
    frames = prepared_data['frames']
    if (any(type(index) is not int or not 0 <= index < len(frames) for index in indices)
            or len(set(indices)) != len(indices) or set(indices) & set(selected)):
        raise ValueError('Orbit published test indices must identify distinct target cameras')
    source_by_name = {}
    for frame in source_data['frames']:
        name = Path(frame['file_path']).name
        if name in source_by_name:
            raise ValueError('Source transforms contain duplicate image basenames')
        source_by_name[name] = frame
    for name, index in zip(names, indices):
        if name not in source_by_name:
            raise ValueError('Source transforms omitted a published test photograph')
        original, target = source_by_name[name], frames[index]
        left, right = np.asarray(original['transform_matrix']), np.asarray(target['transform_matrix'])
        if (left.shape != (4, 4) or right.shape != (4, 4)
                or not np.allclose(left, right, rtol=0, atol=1e-12)):
            raise ValueError(f'Orbit changed the exact published test pose: {name}')
        for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'):
            a, b = original.get(key, source_data.get(key)), target.get(key, prepared_data.get(key))
            if a is None or b is None or not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'Orbit changed published test calibration {key}: {name}')
    return indices


def _prediction_frames(save_dir, scene_id, required):
    matches = list(save_dir.glob(f'*/*/{scene_id}/frames/batch_0000/pred'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one official prediction directory below {save_dir}; found {len(matches)}')
    predicted = matches[0]
    missing = [i for i in required if not (predicted / f'{i:05d}.png').is_file()]
    if missing:
        raise RuntimeError(f'Official inference is missing prediction frames: {missing[:10]}')
    return predicted


def _conditioning_identity(model_id, hf_home):
    """Inspect only cached tokenizer/text-encoder files; never contact the Hub."""
    from huggingface_hub import snapshot_download
    from .repair import digest_file
    cache = Path(hf_home).expanduser().resolve() / 'hub'
    snapshot = Path(snapshot_download(repo_id=model_id, cache_dir=str(cache), local_files_only=True,
                                    allow_patterns=['tokenizer/*', 'text_encoder/*']))
    files = {}
    for folder in ('tokenizer', 'text_encoder'):
        for path in sorted((snapshot / folder).rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(snapshot))
            target = path.resolve()
            # HF caches use either repository-local blobs or shared, sharded
            # blobs. Trust content-addressed names only inside this cache;
            # external symlink targets must still be hashed from their bytes.
            blob = (path.is_symlink() and target.is_relative_to(cache)
                    and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', target.name)
                    and target.parent in (snapshot.parent.parent / 'blobs',
                                          cache / 'blobs' / target.name[:2]))
            files[relative] = {'kind': 'huggingface_blob' if blob else 'sha256',
                               'identity': target.name if blob else digest_file(path),
                               'size': path.stat().st_size}
    required = {'tokenizer/tokenizer_config.json', 'text_encoder/config.json'}
    if (not required <= files.keys() or not any(name.endswith('.safetensors') for name in files)
            or not any(name in files for name in ('tokenizer/spiece.model', 'tokenizer/tokenizer.json'))):
        raise ValueError(f'Incomplete cached tokenizer/text encoder for {model_id}')
    for name in files:
        if name.startswith('text_encoder/') and name.endswith('.index.json'):
            index = json.loads((snapshot / name).read_text())
            missing = {f'text_encoder/{part}' for part in index['weight_map'].values()} - files.keys()
            if missing:
                raise ValueError(f'Missing cached text encoder shards for {model_id}: {sorted(missing)}')
    return {'model_id': model_id, 'snapshot': str(snapshot), 'revision': snapshot.name, 'files': files}


def _shared_conditioning_proof(old_runtime, cfg):
    from .repair import digest_file
    models = [old_runtime.get('model_id'), cfg.get('model_id')]
    if any(model not in MODEL_IDS.values() for model in models):
        raise ValueError('Preparation reuse requires paired release model IDs')
    for runtime in (old_runtime, cfg):
        variant = runtime.get('model_variant')
        if (MODEL_IDS.get(variant) != runtime.get('model_id')
                or Path(runtime.get('checkpoint', '')).name != f'artifixer-{variant}.pt'):
            raise ValueError('Preparation reuse requires paired release model variants and checkpoints')
    source = _conditioning_identity(models[0], cfg['hf_home'])
    target = _conditioning_identity(models[1], cfg['hf_home'])
    if source['files'].keys() != target['files'].keys():
        raise ValueError('Preparation tokenizer/text-encoder cached file sets differ')
    for name, left in source['files'].items():
        right = target['files'][name]
        if left['size'] != right['size']:
            raise ValueError(f'Preparation text conditioning differs: {name}')
        if left['kind'] == right['kind']:
            equal = left['identity'] == right['identity']
        else:
            # One cache may materialize a previously symlinked HF blob.
            equal = digest_file(Path(source['snapshot']) / name) == digest_file(Path(target['snapshot']) / name)
        if not equal:
            raise ValueError(f'Preparation text conditioning differs: {name}')
    return {'verified_equal': True, 'lookup': 'local-only targeted tokenizer/text_encoder snapshots',
            'identity_policy': 'immutable Hugging Face blob ID and size; SHA256 for materialized files',
            'source': source, 'target': target, 'file_count': len(source['files'])}


def _resume_preparation(cfg, input_hashes, names, metadata, upstream_revision):
    """Validate failed-run resume or shared-preparation comparison before copying."""
    from .repair import digest_file
    if cfg.get('resume_from') and cfg.get('preparation_from'):
        raise ValueError('Choose resume_from or preparation_from, not both')
    comparison = bool(cfg.get('preparation_from'))
    selected_source = cfg.get('preparation_from') or cfg.get('resume_from')
    if not selected_source:
        return None
    prior = Path(selected_source).expanduser().resolve()
    path = prior / 'benchmark-run.json'
    old = json.loads(path.read_text())
    if comparison and old.get('status') not in ('complete', 'failed'):
        raise ValueError('preparation_from must identify a completed or failed terminal benchmark')
    if not comparison and old.get('status') != 'failed':
        raise ValueError('resume_from must identify a failed benchmark run')
    checks = {'input_sha256': input_hashes, 'selected_images': names,
              'source_metadata': metadata, 'upstream_revision': upstream_revision,
              'base_reconstruction_steps': 10000, 'artifixer3d_steps': 30000}
    for key, expected in checks.items():
        if old.get(key) != expected:
            raise ValueError(f'Resume source differs in {key}')
    old_runtime = old.get('runtime', {})
    from ..resolution import POLICY_VERSION
    policy = old.get('resolution_policy', {})
    if policy.get('version') != POLICY_VERSION or policy.get('profile') != cfg.get('resolution_profile', 'training'):
        raise ValueError('Preparation resolution policy differs; start a fresh benchmark')
    if not comparison:
        for key in ('model_variant', 'model_id', 'checkpoint'):
            if old_runtime.get(key) != cfg.get(key):
                raise ValueError(f'Resume source differs in model runtime {key}')
    phases = {}
    for phase in ('prepare', 'reconstruct', 'render', 'scale', *(('caption',) if comparison else ())):
        matches = [stage for stage in old.get('stages', []) if stage.get('phase') == phase]
        if len(matches) != 1 or matches[0].get('status') != 'complete':
            raise ValueError(f'Resume requires a completed prior {phase} phase')
        log = prior / f'{phase}.log'
        phases[phase] = {'prior_status': 'complete', 'prior_command': matches[0].get('command'),
                         'prior_log_sha256': digest_file(log) if log.is_file() else None}
    if not (prior / 'prepared/bicycle').is_dir():
        raise FileNotFoundError('Failed run has no prepared Bicycle tree to resume')
    proof = None
    if comparison:
        split = prior / 'prepared/bicycle/split.json'
        _verify_preparation(split, names)
        _, entry = _split_entry(split)
        caption = _metadata_path(split, entry, 'prompt_path')
        if not caption.stat().st_size:
            raise ValueError('Preparation source caption is empty')
        proof = _shared_conditioning_proof(old_runtime, cfg)
        proof['caption_sha256'] = digest_file(caption)
        proof['initial_checkpoint_sha256'] = digest_file(_metadata_path(split, entry, 'reconstruction_checkpoint'))
    return {'root': str(prior), 'manifest_sha256': digest_file(path), 'prior_status': old['status'],
            'kind': 'preparation_from' if comparison else 'resume_from',
            'shared_conditioning': proof,
            'prior_completed_phases': phases,
            'policy': 'Materialized copy of prepared tree only; every official phase is invoked again and decides reuse.'}


def _copy_preparation(resume, destination, should_stop):
    """Materialize symlinks and hash bytes during the copy; never mutate prior data."""
    source = Path(resume['root']) / 'prepared'
    hashes = {}
    class CopyStopped(RuntimeError):
        pass
    def copy_file(src, dst):
        if should_stop():
            raise CopyStopped('Benchmark preparation copy stopped')
        digest = hashlib.sha256()
        with Path(src).open('rb') as incoming, Path(dst).open('wb') as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b''):
                if should_stop():
                    raise CopyStopped('Benchmark preparation copy stopped')
                outgoing.write(block)
                digest.update(block)
        shutil.copystat(src, dst)
        hashes[str(Path(dst).relative_to(destination))] = digest.hexdigest()
        return dst
    # Top-level phase logs and inference output directories are never copied.
    # Logs nested inside training outputs are unnecessary for artifact reuse.
    try:
        shutil.copytree(source, destination, symlinks=False, copy_function=copy_file,
                        ignore=shutil.ignore_patterns('*.log', 'artifixer3d', 'split_artifixer3d_plus.json'))
    except CopyStopped as exc:
        # copytree aggregates OSError (including InterruptedError); use a
        # separate exception internally so cancellation stops copying promptly.
        raise InterruptedError(str(exc)) from exc
    resume['copied_prepared_sha256'] = hashes
    resume['copied_file_count'] = len(hashes)


def run_benchmark(source_dir, output_dir, *, runtime=None, should_stop=lambda: False,
                  on_progress=lambda event: None):
    """Run fresh official prep, inference, reconstruction/export and plus pass."""
    from ..checkpoint import atomic_json, utc_now
    from .repair import UPSTREAM_REVISION, RUNTIME_COMPATIBILITY, digest_file, run_worker, runtime_environment
    from .official_worker import ARTIFIXER3D_STEPS, reconstruction_recipe, reconstruction_cli_args, reconstruction_command
    source = Path(source_dir).expanduser().resolve()
    metadata = json.loads((source / 'benchmark.json').read_text())
    if not isinstance(metadata, dict):
        raise ValueError('benchmark.json must contain source provenance')
    selection = source / 'selected_images.txt'
    names = [line.strip() for line in selection.read_text().splitlines() if line.strip()]
    if len(names) != 3 or len(set(names)) != 3 or any(Path(name).name != name for name in names):
        raise ValueError('Bicycle benchmark requires three distinct selected image basenames')
    colmap = source / 'colmap'
    files = [source / 'benchmark.json', selection]
    files.extend(colmap / 'sparse/0' / name for name in ('cameras.bin', 'images.bin', 'points3D.bin'))
    files.extend(sorted((colmap / 'images').glob('*')))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in names:
        if not (colmap / 'images' / name).is_file():
            raise FileNotFoundError(f'Missing selected photograph: {name}')
    if metadata.get('selected_images') != names:
        raise ValueError('Selected photographs differ from benchmark provenance')
    published_test_names = metadata.get('test_images', [])
    if (not isinstance(published_test_names, list)
            or any(not isinstance(name, str) or Path(name).name != name for name in published_test_names)
            or len(set(published_test_names)) != len(published_test_names)
            or set(published_test_names) & set(names)):
        raise ValueError('Published test photographs must be distinct basenames, separate from references')
    if any(not (colmap / 'images' / name).is_file() for name in published_test_names):
        raise FileNotFoundError('Published test photograph is missing from the source')
    expected_hashes = metadata.get('sha256')
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError('Benchmark source must include importer SHA256 provenance')
    for relative, expected in expected_hashes.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'Invalid benchmark source artifact: {relative}')
        if digest_file(path) != expected:
            raise ValueError(f'Benchmark source changed since import: {relative}')
    if should_stop():
        raise InterruptedError('Benchmark stopped')
    cfg = benchmark_runtime(runtime)
    trajectory_mode = cfg.get('trajectory_mode', 'author_orbit')
    from .author_trajectory import validate_split_mode
    cfg['split_mode'] = validate_split_mode(cfg.get('split_mode', 'double-split'))
    if trajectory_mode not in ('author_orbit', 'source_cameras'):
        raise ValueError('trajectory_mode must be author_orbit or source_cameras')
    if trajectory_mode == 'author_orbit' and not published_test_names:
        raise ValueError('Author orbit benchmark requires published target camera names')
    cfg['trajectory_mode'] = trajectory_mode
    limitation = LIMITATION if trajectory_mode == 'author_orbit' else SOURCE_CAMERA_LIMITATION
    # Match authors normal Hugging Face metadata lookups. Cached tokenizer
    # weights alone do not satisfy every optional-config lookup in offline mode.
    cfg['model_hub_offline'] = (runtime or {}).get('model_hub_offline', False)
    input_hashes = {str(path.relative_to(source)): digest_file(path) for path in files}
    resume = _resume_preparation(cfg, input_hashes, names, metadata, UPSTREAM_REVISION)
    root = Path(output_dir).expanduser().resolve() / f'benchmark_{cfg["model_variant"]}_{uuid.uuid4().hex[:12]}'
    if root.is_relative_to(source):
        raise ValueError('Benchmark outputs must be outside the immutable source directory')
    if resume and root.is_relative_to(Path(resume['root'])):
        raise ValueError('Resumed output must be outside the preserved failed run')
    root.mkdir(parents=True)
    prepared = root / 'prepared/bicycle'
    env = runtime_environment(cfg)
    if cfg.get('moge_model_path'):
        env['MOGE_MODEL_PATH'] = str(cfg['moge_model_path'])
    manifest = {'reconstruction_method': 'artifixer', 'status': 'running', 'created_at': utc_now(), 'source_dir': str(source),
                'source_metadata': metadata, 'selected_images': names, 'runtime': cfg,
                'upstream_revision': UPSTREAM_REVISION, 'runtime_compatibility': RUNTIME_COMPATIBILITY,
                'native_library_path_active': bool(cfg.get('native_library_dir') and Path(cfg['native_library_dir']).is_dir()), 'trajectory': trajectory_mode,
                'limitation': limitation, 'stages': [],
                'input_sha256': input_hashes,
                'base_reconstruction_steps': 10000, 'artifixer3d_steps': ARTIFIXER3D_STEPS,
                'reconstruction_recipe': reconstruction_recipe(cfg.get('regularization_profile', 'artifixer')),
                'inference_settings': dict(INFERENCE_DEFAULTS),
                'reference_kind': 'original photographs', 'initialization': 'original COLMAP sparse points',
                'text_conditioning': 'official generated Qwen caption encoded by Wan UMT5',
                'metric_scale': 'official MoGe alignment (no override)',
                'evaluation': {
                    'protocol': 'generic authors COLMAP pipeline; paper protocol equivalence unverified',
                    'source_image_count': len(list((colmap / 'images').glob('*'))),
                    'published_test_images': published_test_names,
                    'published_test_count': len(published_test_names),
                    'metric_alignment_input': 'all source photographs and COLMAP observations after the recorded calibrated resolution transform',
                    'caption_input': 'all source photographs passed to the official video processor; processor controls sampling',
                    'generated_supervision': ('authors interpolated orbit targets' if trajectory_mode == 'author_orbit' else 'all nonreference source cameras'),
                    'metrics_computed': False,
                    'metric_policy': ('Any later published-split score must match filenames to published_test_images, '
                                      'exclude all reference images, use photographic ground truth only for scoring, '
                                      'and disclose all-source caption/scale and full-scene sparse initialization. '
                                      'Do not report a paper-equivalent score without establishing its preprocessing protocol.'),
                }}
    if resume:
        manifest['resume_source'] = resume
        if resume['kind'] == 'preparation_from':
            manifest['comparison_limitations'] = (
                'Initial reconstruction, caption, cameras and cached text conditioning are shared. '
                'Unchanged authors inference samples random noise without an exposed deterministic seed; '
                'fresh MCMC reconstruction randomness also remains. This is not a deterministic or '
                'strict one-variable paired trial; authors RNG behavior is preserved.')
    atomic_json(root / 'benchmark-run.json', manifest)

    def stage(name, command):
        if should_stop():
            raise InterruptedError('Benchmark stopped')
        record = {'phase': name, 'command': command, 'log': f'{name}.log', 'status': 'running', 'started_at': utc_now()}
        manifest['stages'].append(record)
        atomic_json(root / 'benchmark-run.json', manifest)
        on_progress({'phase': name, 'output_dir': str(root), 'log': str(root / record['log'])})
        started = time.monotonic()
        try:
            run_worker(command, cwd=cfg['repo'], env=env, log_path=root / record['log'], should_stop=should_stop)
        except Exception as exc:
            record.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
            raise
        else:
            record['status'] = 'complete'
            if resume and name in resume['prior_completed_phases']:
                # This is evidence from the newly executed command, not an
                # assumed completed stage copied from the previous manifest.
                prefix = {'prepare': 'Skipping prepare;', 'reconstruct': 'Skipping reconstruction;',
                          'render': 'Skipping render;', 'scale': 'Skipping metric alignment;',
                          'caption': 'Skipping captioning;'}[name]
                record['reuse_evidence'] = [line for line in (root / record['log']).read_text(errors='replace').splitlines()
                                            if line.startswith(prefix)]
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(root / 'benchmark-run.json', manifest)

    from ..resolution import prepare_colmap
    colmap = root / 'conditioning-colmap'
    prep = [cfg['python'], '-m', 'data_processing.prepare_colmap_artifixer_inputs',
            '--colmap_dir', str(colmap), '--output_root', str(prepared),
            '--selected_image_names_file', str(selection), '--text_encoder_model_id', cfg['model_id']]
    def inference(split, destination):
        entrypoint = (['-m', 'model_eval.run_inference'] if trajectory_mode != 'author_orbit' else
                      [str(Path(__file__).with_name('segmented_inference.py').resolve()),
                       '--repo', cfg['repo'], '--trajectory-provenance', str(provenance_path)])
        return [cfg['python'], *entrypoint, '--evalset', 'reconstructed_colmap',
                '--checkpoint_pt', cfg['checkpoint'], '--model_id', cfg['model_id'],
                '--save_dir', str(destination), '--split_path', str(split),
                '--num_views', '3', '--render_trajectory',
                'trajectory' if trajectory_mode == 'author_orbit' else 'all_frames', '--save_frame_outputs_only',
                *[part for key, value in INFERENCE_DEFAULTS.items()
                  for part in ('--' + key, str(value))]]
    try:
        manifest['resolution_policy'] = prepare_colmap(source / 'colmap', colmap, cfg.get('resolution_profile', 'training'))
        atomic_json(root / 'benchmark-run.json', manifest)
        if resume:
            on_progress({'phase': 'copy_preparation', 'output_dir': str(root), 'resume_from': resume['root']})
            _copy_preparation(resume, root / 'prepared', should_stop)
            atomic_json(root / 'benchmark-run.json', manifest)
        # Keep the authors defaults; separate subprocesses release model memory.
        for phase in ('prepare', 'reconstruct', 'render', 'scale', 'caption'):
            stage(phase, prep + ['--phases', phase])
        split = prepared / 'split.json'
        scene_id, entry, frame_count, selected = _verify_preparation(split, names)
        if resume and resume['kind'] == 'preparation_from':
            proof = resume['shared_conditioning']
            verified = {}
            for field, key in (('prompt_path', 'caption_sha256'),
                               ('reconstruction_checkpoint', 'initial_checkpoint_sha256')):
                actual = digest_file(_metadata_path(split, entry, field))
                if actual != proof[key]:
                    raise ValueError(f'Shared preparation changed after official phases: {field}')
                verified[key] = actual
            proof['post_preparation_verified_sha256'] = verified
            atomic_json(root / 'benchmark-run.json', manifest)
        if trajectory_mode == 'author_orbit':
            source_transforms = json.loads(_metadata_path(split, entry, 'transforms_path').read_text())
            target_names_path = root / 'orbit-target-names.json'
            atomic_json(target_names_path, published_test_names)
            trajectory_path = root / 'author-orbit.json'
            provenance_path = root / 'author-orbit-provenance.json'
            stage('orbit_path', [cfg['python'], str(Path(__file__).with_name('author_trajectory.py').resolve()),
                '--split-mode', cfg['split_mode'],
                '--repo', cfg['repo'], '--transforms', str(_metadata_path(split, entry, 'transforms_path')),
                '--selected-indices', str(_metadata_path(split, entry, 'selected_indices_path')),
                '--target-names', str(target_names_path), '--metric-scale', str(entry['metric_scale']),
                '--interp-distance', '0.1', '--output', str(trajectory_path), '--provenance', str(provenance_path)])
            # The upstream CLI always writes split.json. Preserve its source-camera
            # split for preparation reuse; retain the separate trajectory split.
            source_split_bytes = split.read_bytes()
            trajectory_split = prepared / 'split_trajectory.json'
            try:
                stage('orbit_render', prep + ['--phases', 'render', '--trajectory_path', str(trajectory_path)])
                trajectory_split.write_bytes(split.read_bytes())
            finally:
                split.write_bytes(source_split_bytes)
            split = trajectory_split
            scene_id, entry, frame_count, selected = _verify_preparation(split, names)
            targets = json.loads(_metadata_path(split, entry, 'target_indices_path').read_text())
            if targets != [i for i in range(frame_count) if i not in selected]:
                raise ValueError('Official orbit target indices overlap or omit photographic contexts')
            manifest['orbit_provenance'] = json.loads(provenance_path.read_text())
            manifest['inference_series_policy'] = manifest['orbit_provenance'].get('series_policy')
            manifest['inference_split'] = str(split.relative_to(root))
        required = [i for i in range(frame_count) if i not in selected]
        if not required:
            raise ValueError('Benchmark needs held-out source views in addition to the three references')
        prepared_transforms = json.loads(_metadata_path(split, entry, 'transforms_path').read_text())
        frames = prepared_transforms['frames']
        if trajectory_mode == 'author_orbit':
            test_indices = _orbit_test_indices(source_transforms, prepared_transforms,
                manifest['orbit_provenance'], published_test_names, selected)
        else:
            prepared_names = [Path(frame['file_path']).name for frame in frames]
            if len(set(prepared_names)) != len(prepared_names):
                raise ValueError('Prepared source trajectory contains duplicate image basenames')
            name_to_index = {name: index for index, name in enumerate(prepared_names)}
            if any(name not in name_to_index for name in published_test_names):
                raise ValueError('Prepared source trajectory omitted a published test photograph')
            test_indices = [name_to_index[name] for name in published_test_names]
        if set(test_indices) & set(selected):
            raise ValueError('Prepared reference cameras overlap published test photographs')
        manifest['evaluation'].update(prepared_frame_count=frame_count,
                                      generated_supervision_count=len(required),
                                      published_test_prepared_indices=test_indices)
        atomic_json(root / 'benchmark-run.json', manifest)
        stage('inference', inference(split, root / 'inference'))
        predictions = _prediction_frames(root / 'inference', scene_id, required)
        from .inference_preview import publish_preview
        publish_preview(root, predictions, required, len(selected), frame_count=frame_count,
                        model_variant=cfg['model_variant'], trajectory_mode=trajectory_mode,
                        resolution_policy=manifest['resolution_policy'])
        stage('artifixer3d', [*reconstruction_command(cfg['python'], cfg['repo'], cfg.get('regularization_profile', 'artifixer')),
                             '--scene_root', str(prepared), '--artifixer_frames_dir', str(predictions),
                             '--split_path', str(split), *reconstruction_cli_args()])
        plus_split = prepared / 'split_artifixer3d_plus.json'
        plus_scene, plus_entry = _split_entry(plus_split)
        if plus_scene != scene_id:
            raise ValueError('Official plus split changed scene identity')
        checkpoint = _metadata_path(plus_split, plus_entry, 'reconstruction_checkpoint')
        if not checkpoint.is_file():
            raise FileNotFoundError('Official ArtiFixer3D did not produce its fresh checkpoint')
        dataset = prepared / 'artifixer3d/distillation_input' / scene_id
        splat = root / 'artifixer3d.ply'
        stage('export', [cfg['python'], str(Path(__file__).resolve()), '--repo', cfg['repo'],
                         '--export', str(checkpoint), '--dataset', str(dataset), '--output', str(splat)])
        if not splat.is_file():
            raise FileNotFoundError('Official PLYExporter did not produce the reconstructed splat')
        stage('plus', inference(plus_split, root / 'plus'))
        plus_frames = _prediction_frames(root / 'plus', scene_id, required)
        result = {'reconstruction_method': 'artifixer', 'resolution_policy': manifest['resolution_policy'], 'output_dir': str(root), 'splat_path': str(splat), 'plus_frames': str(plus_frames),
                  'prediction_frames': str(predictions), 'reconstruction_checkpoint': str(checkpoint),
                  'source_dir': str(source), 'model_variant': cfg['model_variant'],
                  'selected_images': names, 'metric_scale': entry['metric_scale'],
                  'camera_scale': entry['camera_scale'], 'frame_count': frame_count,
                  'upstream_revision': UPSTREAM_REVISION, 'limitation': limitation,
                  'trajectory_mode': trajectory_mode, 'inference_split': str(split.relative_to(root)),
                  'merged': False, 'ply_stage': 'ArtiFixer3D; + produces postprocessed images'}
        if trajectory_mode == 'author_orbit':
            result['orbit_provenance_path'] = str(provenance_path.relative_to(root))
            result['target_name_to_index'] = dict(manifest['orbit_provenance']['target_name_to_index'])
        atomic_json(root / 'result.json', result)
        manifest.update(status='complete', result=result)
        return result
    except Exception as exc:
        manifest.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        atomic_json(root / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
    finally:
        completed = {stage['phase'] for stage in manifest['stages'] if stage['status'] == 'complete'}
        manifest['resume_supported'] = (manifest['status'] == 'failed'
            and {'prepare', 'reconstruct', 'render', 'scale'} <= completed and prepared.is_dir())
        manifest['preparation_supported'] = (manifest['status'] in ('complete', 'failed')
            and {'prepare', 'reconstruct', 'render', 'scale', 'caption'} <= completed and prepared.is_dir())
        atomic_json(root / 'benchmark-run.json', manifest)


def _export(checkpoint, dataset, output):
    """Only serialize the fresh official model; no fitting or merging here."""
    from threedgrut.render import Renderer
    from threedgrut.export.ply_exporter import PLYExporter
    renderer = Renderer.from_checkpoint(checkpoint_path=checkpoint, path=str(dataset),
        out_dir=str(output.parent / 'export'), save_gt=False, computes_extra_metrics=False)
    PLYExporter().export(renderer.model, output)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    _export(args.export, args.dataset, args.output)
