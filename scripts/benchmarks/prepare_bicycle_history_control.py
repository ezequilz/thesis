"""Prepare an isolated short-history inference split without launching GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(result_root: Path, output: Path, start: int = 280, stop: int = 343):
    result = json.loads((result_root / 'result.json').read_text())
    if result.get('model_variant') != '1.3b':
        raise ValueError('This development control requires the 1.3B model')
    source = result_root / result['inference_split']
    split = json.loads(source.read_text())
    scene = split['test']['bicycle']
    transforms = json.loads((source.parent / scene['transforms_path']).read_text())
    selected = json.loads((source.parent / scene['selected_indices_path']).read_text())
    original_targets = json.loads((source.parent / scene['target_indices_path']).read_text())
    targets = list(range(start, stop))
    # A 28-frame offset aligns subsequent block boundaries and four-frame
    # sampling phase, not the causal VAE boundary state. The first block covers
    # 25 RGB frames, followed by 28-frame blocks; a fresh run reinitializes it.
    if start % 28 or not targets or not set(targets).issubset(original_targets):
        raise ValueError('Use a nonempty original target interval with a start offset divisible by 28')
    if set(targets) & set(selected) or 321 not in targets:
        raise ValueError('Control must contain failing camera321 and exclude references')
    remote_result = Path(result['output_dir'])
    remote_prepared = remote_result / Path(result['inference_split']).parent
    remote_control = remote_result.parent.parent / 'controls' / output.name
    for key, value in list(scene.items()):
        if key.endswith('_path') or key.endswith('_dir') or key == 'image_root':
            path = Path(value)
            scene[key] = str(path if path.is_absolute() else remote_prepared / path)
    scene['target_indices_path'] = str(remote_control / 'target_indices.json')
    manifest = {
        'source_result': str(result_root.resolve()), 'remote_control': str(remote_control),
        'model_variant': '1.3b', 'status': 'prepared_not_launched',
        'source_split_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_target_count': len(original_targets), 'target_indices': targets,
        'reference_indices': selected, 'failing_target_index': 321,
        'failing_target_position': targets.index(321), 'rgb_block_alignment': 28,
        'temporal_alignment': {
            'first_rgb_block_length': 25, 'later_rgb_block_length': 28,
            'offset_multiple': 28, 'identical_vae_context': False,
            'note': 'A 28-frame offset aligns subsequent block boundaries, but restart changes '
                    'causal VAE boundary context, sink content, and the first block.',
        },
        'camera_scale': scene['camera_scale'],
        'failing_target_pose': transforms['frames'][321]['transform_matrix'],
        'unchanged': ['model', 'reference images', 'caption', 'calibration',
                      'render and opacity files', 'frame identities', 'authors inference defaults'],
        'changed': ['preceding autoregressive target history', 'VAE boundary context',
                    'sink content', 'first block', 'random sampling realization'],
        'limits': ['Authors CLI has no seed option; comparison with historical output is not noise-paired.',
                   'Restart also changes VAE boundary context, sink content, and the first block; '
                   'this does not isolate KV-history length alone.',
                   'A changed outcome would motivate repeated controlled trials, not prove a logic bug.',
                   'This is a diagnostic, not a proposed replacement baseline.'],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('split.json', split), ('target_indices.json', targets), ('control.json', manifest)]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.result_root, args.output), indent=2))
