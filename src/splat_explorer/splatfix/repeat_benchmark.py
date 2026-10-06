"""Repeat a completed benchmark repair, preserving its historical preparation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time
import uuid

from .checkpoint import atomic_json, utc_now
from . import benchmark as b
from .repair import UPSTREAM_REVISION, digest_file, run_worker, runtime_environment


def prepare_repeat(prior, root, should_stop=lambda: False):
    prior, root = Path(prior), Path(root)
    old = json.loads((prior / 'benchmark-run.json').read_text())
    if old.get('status') != 'complete' or old.get('upstream_revision') != UPSTREAM_REVISION:
        raise ValueError('Repeat repair requires a completed benchmark at the pinned revision')
    if old.get('inference_settings') != b.INFERENCE_DEFAULTS or old.get('trajectory') != 'author_orbit':
        raise ValueError('Repeat repair requires the matching inference recipe and author orbit')
    original_split = prior / old['result']['inference_split']
    scene, original = b._split_entry(original_split)
    plus_split = prior / 'prepared/bicycle/split_artifixer3d_plus.json'
    plus_scene, repaired = b._split_entry(plus_split)
    if scene != plus_scene:
        raise ValueError('Repaired scene identity differs')
    for key in ('transforms_path', 'selected_indices_path', 'target_indices_path', 'prompt_path'):
        if digest_file(b._metadata_path(original_split, original, key)) != digest_file(b._metadata_path(plus_split, repaired, key)):
            raise ValueError(f'Repaired conditioning differs: {key}')
    if original['camera_scale'] != repaired['camera_scale']:
        raise ValueError('Repaired camera scale differs')
    checkpoint = b._metadata_path(plus_split, repaired, 'reconstruction_checkpoint')
    # Both are outputs of the same upstream training/export stages. Use the native
    # checkpoint and its saved RGB/alpha to avoid a lossy PLY import/render roundtrip.
    proof = {'source_result': str(prior), 'manifest_sha256': digest_file(prior / 'benchmark-run.json'),
             'input_checkpoint': str(checkpoint), 'input_checkpoint_sha256': digest_file(checkpoint),
             'input_ply': str(prior / 'artifixer3d.ply'), 'input_ply_sha256': digest_file(prior / 'artifixer3d.ply'),
             'repair_pass': old.get('repeat_repair', {}).get('repair_pass', 1) + 1,
             'policy': 'Use previous repaired checkpoint RGB/opacity; preserve original photographic references, cameras, caption, scale and COLMAP initialization. Fresh inference and fresh 30000-step training.'}
    destination = root / 'prepared/bicycle'
    def copy_file(src, dst):
        if should_stop():
            raise InterruptedError('Repeat preparation stopped')
        return shutil.copy2(src, dst)
    shutil.copytree(prior / 'prepared/bicycle', destination, copy_function=copy_file,
                    ignore=shutil.ignore_patterns('depth', '*.log', 'previous_repair'))
    (destination / 'artifixer3d').rename(destination / 'previous_repair')
    entry = copy.deepcopy(repaired)
    for key in ('render_dir', 'opacity_dir', 'reconstruction_checkpoint', 'selected_indices_path'):
        value = Path(entry[key])
        if value.is_absolute() or value.parts[0] != 'artifixer3d' or '..' in value.parts:
            raise ValueError(f'Unexpected repaired artifact location: {key}')
        entry[key] = str(Path('previous_repair', *value.parts[1:]))
    entry['metric_scale'] = original['metric_scale']
    split = destination / 'split_trajectory.json'
    atomic_json(split, {'test': {scene: entry}})
    (destination / 'split_artifixer3d_plus.json').unlink()
    b._verify_preparation(split, old['selected_images'])
    if digest_file(b._metadata_path(split, entry, 'reconstruction_checkpoint')) != proof['input_checkpoint_sha256']:
        raise ValueError('Copied repaired checkpoint differs')
    for name in ('author-orbit.json', 'author-orbit-provenance.json', 'orbit-target-names.json'):
        shutil.copy2(prior / name, root / name)
    return old, proof, split


def run_repeat_benchmark(prior, output_dir, *, should_stop=lambda: False, on_progress=lambda event: None):
    prior = Path(prior).resolve()
    old = json.loads((prior / 'benchmark-run.json').read_text())
    # Preserve historical dimensions by copying preparation, never re-preprocessing.
    runtime = {k: v for k, v in old['runtime'].items() if k not in ('resume_from', 'preparation_from', 'repeat_from')}
    cfg = b.benchmark_runtime(runtime)
    root = Path(output_dir).resolve() / f'benchmark_{cfg["model_variant"]}_{uuid.uuid4().hex[:12]}'
    if root.is_relative_to(prior):
        raise ValueError('Repeat output must be outside the prior result')
    root.mkdir(parents=True)
    manifest = copy.deepcopy(old)
    for key in ('result', 'resume_source', 'comparison_limitations', 'error', 'resolution_policy'):
        manifest.pop(key, None)
    manifest.update(status='running', created_at=utc_now(), runtime=runtime, stages=[],
                    resume_supported=False, preparation_supported=False,
                    historical_resolution_preserved=True)
    if 'resolution_policy' in old:
        manifest['resolution_policy'] = old['resolution_policy']
    atomic_json(root / 'benchmark-run.json', manifest)
    env = runtime_environment(cfg)
    def stage(name, command):
        if should_stop():
            raise InterruptedError('Repeat repair stopped')
        record = dict(phase=name, command=command, log=f'{name}.log', status='running', started_at=utc_now())
        manifest['stages'].append(record)
        atomic_json(root / 'benchmark-run.json', manifest)
        on_progress(dict(phase=name, output_dir=str(root), log=str(root / record['log'])))
        start = time.monotonic()
        try:
            run_worker(command, cwd=cfg['repo'], env=env, log_path=root / record['log'], should_stop=should_stop)
            record['status'] = 'complete'
        except Exception:
            record['status'] = 'failed'
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - start
            atomic_json(root / 'benchmark-run.json', manifest)
    def inference(split, destination):
        return [cfg['python'], '-m', 'model_eval.run_inference', '--evalset', 'reconstructed_colmap',
                '--checkpoint_pt', cfg['checkpoint'], '--model_id', cfg['model_id'], '--save_dir', str(destination),
                '--split_path', str(split), '--num_views', '3', '--render_trajectory', 'trajectory',
                '--save_frame_outputs_only', *[p for k, v in old['inference_settings'].items() for p in ('--' + k, str(v))]]
    try:
        on_progress(dict(phase='copy_repaired_input', output_dir=str(root)))
        _, proof, split = prepare_repeat(prior, root, should_stop)
        manifest['repeat_repair'] = proof
        atomic_json(root / 'benchmark-run.json', manifest)
        scene, entry, count, selected = b._verify_preparation(split, old['selected_images'])
        required = [i for i in range(count) if i not in selected]
        # Validate complete saved renders before launching costly inference.
        for key in ('render_dir', 'opacity_dir'):
            directory = b._metadata_path(split, entry, key)
            if len(list(directory.glob('*.png'))) != count:
                raise ValueError(f'Incomplete repaired {key}: expected {count} PNGs')
        stage('inference', inference(split, root / 'inference'))
        predictions = b._prediction_frames(root / 'inference', scene, required)
        prepared = split.parent
        stage('artifixer3d', [cfg['python'], '-m', 'data_processing.run_artifixer3d', '--scene_root', str(prepared),
                            '--artifixer_frames_dir', str(predictions), '--split_path', str(split)])
        plus = prepared / 'split_artifixer3d_plus.json'
        plus_scene, plus_entry = b._split_entry(plus)
        if plus_scene != scene:
            raise ValueError('Second repair changed scene identity')
        checkpoint = b._metadata_path(plus, plus_entry, 'reconstruction_checkpoint')
        splat = root / 'artifixer3d.ply'
        stage('export', [cfg['python'], str(Path(b.__file__).resolve()), '--repo', cfg['repo'], '--export', str(checkpoint),
                         '--dataset', str(prepared / 'artifixer3d/distillation_input' / scene), '--output', str(splat)])
        if not splat.is_file():
            raise FileNotFoundError('Missing second repair PLY')
        stage('plus', inference(plus, root / 'plus'))
        result = copy.deepcopy(old['result'])
        result.update(output_dir=str(root), splat_path=str(splat), reconstruction_checkpoint=str(checkpoint),
                      prediction_frames=str(predictions), plus_frames=str(b._prediction_frames(root / 'plus', scene, required)),
                      inference_split=str(split.relative_to(root)), repeat_repair=proof)
        atomic_json(root / 'result.json', result)
        manifest.update(status='complete', result=result)
        return result
    except Exception as exc:
        manifest.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        atomic_json(root / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
    finally:
        atomic_json(root / 'benchmark-run.json', manifest)
