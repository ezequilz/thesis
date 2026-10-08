"""Repeat a completed benchmark repair, preserving its historical preparation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time
import uuid

from ..checkpoint import atomic_json, utc_now
from . import benchmark as b
from .repair import UPSTREAM_REVISION, digest_file, run_worker, runtime_environment
from .official_worker import ARTIFIXER3D_STEPS, reconstruction_recipe, reconstruction_cli_args, reconstruction_command


def inherited_inference_command(old, phase, split, destination):
    """Replay the recorded command, including implicit pinned-release defaults."""
    records = [s for s in old.get('stages', []) if s.get('phase') == phase and s.get('status') == 'complete']
    if len(records) != 1:
        raise ValueError(f'Repeat requires one completed original {phase} command')
    command = list(records[0]['command'])
    segmented = Path(command[1]).name == 'segmented_inference.py'
    if command[1:3] != ['-m', 'model_eval.run_inference'] and not segmented:
        raise ValueError('Unexpected original inference entry point')
    if segmented:
        command[1] = str(Path(__file__).with_name('segmented_inference.py').resolve())
        command[command.index('--trajectory-provenance') + 1] = str(
            Path(destination).parent / 'author-orbit-provenance.json')
    for flag, value in (('--render_trajectory', 'trajectory'), ('--num_views', '3')):
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError(f'Original inference requires {flag} {value}')
    for flag, value in (('--split_path', split), ('--save_dir', destination)):
        if command.count(flag) != 1:
            raise ValueError(f'Original inference requires one {flag}')
        command[command.index(flag) + 1] = str(value)
    return command


def prepare_repeat(prior, root, should_stop=lambda: False):
    prior, root = Path(prior), Path(root)
    old = json.loads((prior / 'benchmark-run.json').read_text())
    if old.get('status') != 'complete' or old.get('upstream_revision') != UPSTREAM_REVISION:
        raise ValueError('Repeat repair requires a completed benchmark at the pinned revision')
    trajectory = old.get('trajectory') or old.get('runtime', {}).get('trajectory_mode') or old.get('result', {}).get('trajectory_mode')
    if trajectory != 'author_orbit':
        raise ValueError('Repeat repair requires the original author orbit')
    for phase in ('inference', 'plus'):
        inherited_inference_command(old, phase, 'validation-split', 'validation-output')
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
             'conditioning_sha256': {key: digest_file(b._metadata_path(original_split, original, key))
                                    for key in ('transforms_path', 'selected_indices_path', 'target_indices_path', 'prompt_path')},
             'inference_recipe_source': 'Original completed inference and plus commands; omitted options retain defaults of the same pinned upstream revision',
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


def align_repeat_inputs(split, profile):
    from ..resolution import prepare_colmap, resize_plan, calibrate
    scene, entry = b._split_entry(split)
    source = b._metadata_path(split, entry, 'image_root')
    destination = split.parent / 'aligned-input' / scene
    policy = prepare_colmap(source, destination, profile)
    original = json.loads(b._metadata_path(split, entry, 'transforms_path').read_text())
    aligned = copy.deepcopy(original)
    def adjust(camera):
        plan = resize_plan(camera['w'], camera['h'], profile)
        camera['fl_x'], camera['fl_y'], camera['cx'], camera['cy'] = calibrate(
            camera['fl_x'], camera['fl_y'], camera['cx'], camera['cy'], plan)
        camera['w'], camera['h'] = plan['output_wh']
    adjust(aligned)
    for old_frame, frame in zip(original['frames'], aligned['frames']):
        calibration = {**{k: v for k, v in original.items() if k != 'frames'}, **old_frame}
        adjust(calibration)
        for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'):
            frame[key] = calibration[key]
        assert frame['transform_matrix'] == old_frame['transform_matrix']
    transforms = split.parent / 'trajectory/transforms_aligned.json'
    transforms.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(transforms, aligned)
    entry.update(image_root=str(destination.relative_to(split.parent)),
                 transforms_path=str(transforms.relative_to(split.parent)))
    atomic_json(split, {'test': {scene: entry}})
    return policy


def run_repeat_benchmark(prior, output_dir, *, resolution_profile='training', regularization_profile='artifixer', should_stop=lambda: False, on_progress=lambda event: None):
    prior = Path(prior).resolve()
    old = json.loads((prior / 'benchmark-run.json').read_text())
    # Preserve historical dimensions by copying preparation, never re-preprocessing.
    runtime = {k: v for k, v in old['runtime'].items() if k not in ('resume_from', 'preparation_from', 'repeat_from')}
    runtime['resolution_profile'] = resolution_profile
    runtime['regularization_profile'] = regularization_profile
    cfg = b.benchmark_runtime(runtime)
    root = Path(output_dir).resolve() / f'benchmark_{cfg["model_variant"]}_{uuid.uuid4().hex[:12]}'
    if root.is_relative_to(prior):
        raise ValueError('Repeat output must be outside the prior result')
    root.mkdir(parents=True)
    manifest = copy.deepcopy(old)
    manifest['reconstruction_method'] = 'artifixer'
    for key in ('result', 'resume_source', 'comparison_limitations', 'error', 'resolution_policy'):
        manifest.pop(key, None)
    manifest.update(status='running', created_at=utc_now(), runtime=runtime, stages=[],
                    artifixer3d_steps=ARTIFIXER3D_STEPS, reconstruction_recipe=reconstruction_recipe(regularization_profile),
                    resume_supported=False, preparation_supported=False,
                    historical_resolution_preserved=resolution_profile == 'early_original')
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
        return inherited_inference_command(old, destination.name, split, destination)
    try:
        on_progress(dict(phase='copy_repaired_input', output_dir=str(root)))
        _, proof, split = prepare_repeat(prior, root, should_stop)
        manifest['repeat_repair'] = proof
        if resolution_profile != 'early_original':
            manifest['resolution_policy'] = align_repeat_inputs(split, resolution_profile)
            stage('rerender_repaired_input', [cfg['python'], str(Path(__file__).with_name('repeat_render.py')),
                '--repo', cfg['repo'], '--split', str(split)])
            proof['resolution_adaptation'] = 'Original camera poses preserved; calibrated v1 references and newly rendered repaired-splat RGB/opacity'
        atomic_json(root / 'benchmark-run.json', manifest)
        scene, entry, count, selected = b._verify_preparation(split, old['selected_images'])
        required = [i for i in range(count) if i not in selected]
        # Validate complete saved renders before launching costly inference.
        for key in ('render_dir', 'opacity_dir'):
            directory = b._metadata_path(split, entry, key)
            if len(list(directory.glob('*.png'))) != count:
                raise ValueError(f'Incomplete repaired {key}: expected {count} PNGs')
        from .inference_preview import publish_inputs
        publish_inputs(root, split=split)
        stage('inference', inference(split, root / 'inference'))
        predictions = b._prediction_frames(root / 'inference', scene, required)
        from .inference_preview import publish_preview
        publish_preview(root, predictions, required, len(selected), frame_count=count,
                        model_variant=old['result'].get('model_variant'),
                        trajectory_mode=old['result'].get('trajectory_mode'),
                        inference_split=str(split.relative_to(root)), output_dir=str(root),
                        resolution_policy=manifest.get('resolution_policy', {}))
        prepared = split.parent
        stage('artifixer3d', [*reconstruction_command(cfg['python'], cfg['repo'], regularization_profile), '--scene_root', str(prepared),
                            '--artifixer_frames_dir', str(predictions), '--split_path', str(split),
                            *reconstruction_cli_args()])
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
        result['reconstruction_method'] = 'artifixer'
        result.update(output_dir=str(root), splat_path=str(splat), reconstruction_checkpoint=str(checkpoint),
                      prediction_frames=str(predictions), plus_frames=str(b._prediction_frames(root / 'plus', scene, required)),
                      inference_split=str(split.relative_to(root)), repeat_repair=proof)
        if 'resolution_policy' in manifest:
            result['resolution_policy'] = manifest['resolution_policy']
        atomic_json(root / 'result.json', result)
        manifest.update(status='complete', result=result)
        return result
    except Exception as exc:
        manifest.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        atomic_json(root / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
    finally:
        atomic_json(root / 'benchmark-run.json', manifest)
