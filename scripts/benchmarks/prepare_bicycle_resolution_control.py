"""Prepare calibrated, reduced-resolution inference inputs; never launch a job.

Keeps the complete historical target sequence and original fitted scene. This
tests diffusion input resolution, not retraining at another resolution, and is
not claimed to recover the unpublished website recipe.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.splatfix.repair import digest_file


def prepare(result_root: Path, output: Path, width=1088, height=720):
    result_root, output = result_root.resolve(), output.resolve()
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError('Use positive dimensions divisible by 16')
    if output.is_relative_to(result_root):
        raise ValueError('Preserve historical results; use a separate output directory')
    result = json.loads((result_root / 'result.json').read_text())
    if result['model_variant'] != '1.3b':
        raise ValueError('This control requires the 1.3B run')
    source = result_root / result['inference_split']
    split = json.loads(source.read_text())
    scene = split['test']['bicycle']
    original_scene = copy.deepcopy(scene)
    def local(key):
        path = Path(original_scene[key])
        if path.is_absolute():
            path = path.relative_to(Path(result['output_dir']))
            return result_root / path
        return source.parent / path
    transforms = json.loads(local('transforms_path').read_text())
    selected = json.loads(local('selected_indices_path').read_text())
    targets = json.loads(local('target_indices_path').read_text())
    if len(selected) != 3 or set(selected) & set(targets) or len(set(targets)) != len(targets):
        raise ValueError('Expected three references and unique disjoint targets')
    remote = Path(result['output_dir']).parent.parent / 'controls' / output.name
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    def resize(source_path, relative):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            if image.size != (transforms['w'], transforms['h']):
                raise ValueError(f'Unexpected source image size: {source_path}')
            image.resize((width, height), Image.Resampling.BILINEAR).save(destination)
        hashes[str(relative)] = {'source_sha256': digest_file(source_path),
                                 'output_sha256': digest_file(destination)}
    for index in targets:
        for key, folder in [('render_dir', 'renders'), ('opacity_dir', 'opacity')]:
            resize(local(key) / f'{index:05d}.png', Path(folder) / f'{index:05d}.png')
    updated = copy.deepcopy(transforms)
    for index in selected:
        old = transforms['frames'][index]['file_path']
        relative = Path('images') / (Path(old).stem + '.png')
        resize(local('image_root') / old, relative)
        updated['frames'][index]['file_path'] = str(relative)
    # Scale intrinsics wherever supplied. Normalized camera rays and C2W must
    # remain unchanged, including frame-level overrides used by the loader.
    for before, after in [(transforms, updated), *zip(transforms['frames'], updated['frames'])]:
        old_w, old_h = before.get('w', transforms['w']), before.get('h', transforms['h'])
        for key in ('fl_x', 'cx'):
            if key in before:
                after[key] = before[key] * width / old_w
        for key in ('fl_y', 'cy'):
            if key in before:
                after[key] = before[key] * height / old_h
        if 'w' in before:
            after['w'] = width
        if 'h' in before:
            after['h'] = height
        for key, divisor in [('fl_x', 'w'), ('cx', 'w'), ('fl_y', 'h'), ('cy', 'h')]:
            a = before.get(key, transforms[key]) / before.get(divisor, transforms[divisor])
            b = after.get(key, updated[key]) / after.get(divisor, updated[divisor])
            if not np.isclose(a, b, rtol=0, atol=1e-12):
                raise ValueError('Resize changed normalized calibration')
        if 'transform_matrix' in before and before['transform_matrix'] != after['transform_matrix']:
            raise ValueError('Resize changed a camera pose')
    for key in ('prompt_path', 'reconstruction_checkpoint'):
        old = Path(original_scene[key])
        scene[key] = str(old if old.is_absolute() else Path(result['output_dir']) / Path(result['inference_split']).parent / old)
    for key, relative in [('image_root', '.'), ('render_dir', 'renders'), ('opacity_dir', 'opacity'),
                          ('transforms_path', 'transforms.json'), ('selected_indices_path', 'selected_indices.json'),
                          ('target_indices_path', 'target_indices.json')]:
        scene[key] = str(remote / relative)
    manifest = {'status': 'prepared_not_launched', 'model_variant': '1.3b',
                'source_result': str(result_root), 'remote_control': str(remote),
                'source_split_sha256': digest_file(source),
                'source_wh': [transforms['w'], transforms['h']], 'output_wh': [width, height],
                'target_count': len(targets), 'reference_indices': selected,
                'normalized_calibration_verified': True, 'poses_unchanged': True,
                'interpolation': 'Pillow BILINEAR; lossless PNG output',
                'comparison': 'Full historical sequence; only spatial resolution/preprocessing changes. Original checkpoint, metric scale, caption and target ordering retained.',
                'limitations': ['Not the verified website resolution.',
                               'Original reconstruction is resampled, not retrained.',
                               'Historical inference noise is unavailable; not a noise-paired trial.'],
                'files': hashes}
    for name, data in [('split.json', split), ('transforms.json', updated),
                       ('selected_indices.json', selected), ('target_indices.json', targets), ('control.json', manifest)]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n')
    return {key: value for key, value in manifest.items() if key != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1088)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    print(json.dumps(prepare(args.result_root, args.output, args.width, args.height), indent=2))
