"""Score the released Bicycle 25-image test split using pinned author metrics.

Run in the ArtiFixer environment after a benchmark completes. This utility does
not train, generate images, resize inputs, or evaluate the 191-view complement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from .checkpoint import atomic_json
from .repair import UPSTREAM_REVISION, digest_file

METHODS = ('baseline', 'artifixer', 'artifixer3d', 'artifixer3d_plus')
METRICS = ('psnr', 'ssim', 'lpips')


def _read(path):
    return json.loads(Path(path).read_text())


def _split(path):
    scenes = _read(path)['test']
    if len(scenes) != 1:
        raise ValueError('Expected exactly one Bicycle scene in the prepared split')
    return next(iter(scenes.items()))


def _resolve(split, entry, key):
    path = Path(entry[key])
    return path.resolve() if path.is_absolute() else (split.parent / path).resolve()


def _names(frames):
    names = [Path(frame['file_path']).name for frame in frames]
    if len(set(names)) != len(names):
        raise ValueError('Ambiguous duplicate source filenames in prepared transforms')
    return names


def _rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def evaluation_plan(run_dir):
    """Resolve every score by published filename, validating all 100 image pairs."""
    root = Path(run_dir).expanduser().resolve()
    run = _read(root / 'benchmark-run.json')
    result = _read(root / 'result.json')
    if run.get('status') != 'complete':
        raise ValueError('Complete all benchmark stages before evaluating all four methods')
    source = Path(run['source_dir']).resolve()
    metadata = _read(source / 'benchmark.json')
    if metadata != run['source_metadata']:
        raise ValueError('Source provenance differs from the benchmark run snapshot')
    names = metadata.get('test_images')
    references = metadata.get('selected_images')
    if (not isinstance(names, list) or len(names) != 25 or len(set(names)) != 25
            or any(not isinstance(name, str) or Path(name).name != name for name in names)):
        raise ValueError('Require exactly 25 distinct published Bicycle test-image basenames')
    if (not isinstance(references, list) or len(references) != 3 or len(set(references)) != 3
            or set(names) & set(references)):
        raise ValueError('Published test images must exclude all three reference photographs')
    split = root / 'prepared/bicycle/split.json'
    plus_split = split.with_name('split_artifixer3d_plus.json')
    scene, initial = _split(split)
    plus_scene, plus = _split(plus_split)
    if scene != plus_scene:
        raise ValueError('Initial and plus splits have different scene identities')
    transforms_path = _resolve(split, initial, 'transforms_path')
    plus_transforms_path = _resolve(plus_split, plus, 'transforms_path')
    frames = _read(transforms_path)['frames']
    plus_frames = _read(plus_transforms_path)['frames']
    frame_names = _names(frames)
    if frame_names != _names(plus_frames):
        raise ValueError('Initial and plus frame order differs; refusing mismatched image comparisons')
    for left, right in zip(frames, plus_frames):
        if not np.allclose(left['transform_matrix'], right['transform_matrix'], rtol=0, atol=1e-7):
            raise ValueError('Initial and plus camera poses differ')
    indices = {name: index for index, name in enumerate(frame_names)}
    selected = _read(_resolve(split, initial, 'selected_indices_path'))
    plus_selected = _read(_resolve(plus_split, plus, 'selected_indices_path'))
    if (any(type(index) is not int or not 0 <= index < len(frames) for index in selected)
            or selected != plus_selected
            or [frame_names[index] for index in selected] != references):
        raise ValueError('Prepared reference indices differ from the published three photographs')
    directories = {
        'baseline': _resolve(split, initial, 'render_dir'),
        'artifixer': Path(result['prediction_frames']).resolve(),
        'artifixer3d': _resolve(plus_split, plus, 'render_dir'),
        'artifixer3d_plus': Path(result['plus_frames']).resolve(),
    }
    hashes = metadata.get('sha256', {})
    pairs = []
    for name in names:
        if name not in indices:
            raise ValueError(f'Published test image absent from prepared cameras: {name}')
        index = indices[name]
        if index in selected:
            raise ValueError(f'Published test image is a reconstruction reference: {name}')
        relative = f'colmap/images/{name}'
        gt = source / relative
        if relative not in hashes or digest_file(gt) != hashes[relative]:
            raise ValueError(f'Published photographic ground truth changed: {name}')
        with Image.open(gt) as image:
            size = image.size
        paths = {method: directory / f'{index:05d}.png' for method, directory in directories.items()}
        for method, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f'{method} is missing published test prediction {index:05d}: {name}')
            with Image.open(path) as image:
                if image.size != size:
                    raise ValueError(f'{method} shape mismatch for {name}: {image.size} vs original {size}; no implicit resize')
        pairs.append({'name': name, 'prepared_index': index, 'width': size[0], 'height': size[1],
                      'ground_truth': str(gt), 'predictions': {k: str(v) for k, v in paths.items()}})
    return {'root': str(root), 'pairs': pairs, 'references': references,
            'run_sha256': digest_file(root / 'benchmark-run.json'),
            'source_metadata_sha256': digest_file(source / 'benchmark.json'),
            'split_sha256': {str(p): digest_file(p) for p in (split, plus_split, transforms_path, plus_transforms_path)},
            'evaluation_context': run.get('evaluation', {}), 'limitation': run.get('limitation', '')}


def _validate_metric_source(repo):
    """Validate pinned root source contents independently of mount file modes."""
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f'Metric source must be pinned to {UPSTREAM_REVISION}')
    # Metrics are tracked in the root repository. DSS staging may alter root
    # executable bits and submodule permissions; neither changes these metrics.
    # Staged and unstaged root content changes remain disallowed.
    dirty = subprocess.check_output(['git', '-c', 'core.fileMode=false', '-C', str(repo),
        'status', '--porcelain', '--untracked-files=no', '--ignore-submodules=all'], text=True)
    if dirty.strip():
        raise ValueError('Metric source has modified tracked files')
    return revision


def load_official_metrics(repo, device='cpu'):
    """Import the unchanged GenFusion-derived shared author implementation.

    Torch hub weights must already be cached. HF offline flags alone do not
    prevent torchvision/LPIPS from attempting downloads.
    """
    repo = Path(repo).expanduser().resolve()
    revision = _validate_metric_source(repo)
    import torch
    weights = [Path(torch.hub.get_dir()) / 'checkpoints' / name for name in ('vgg16-397923af.pth', 'vgg.pth')]
    missing = [str(path) for path in weights if not path.is_file()]
    if missing:
        raise FileNotFoundError('Precache official torchvision VGG16 and LPIPS v0.1 VGG weights: ' + ', '.join(missing))
    sys.path.insert(0, str(repo))
    from model_eval import metrics_utils
    if not Path(metrics_utils.__file__).resolve().is_relative_to(repo):
        raise RuntimeError('Another model_eval package is already imported; run evaluation in a fresh process')

    def compute(prediction, target):
        def tensor(image):
            return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.inference_mode():
            return metrics_utils.compute_rgb_metrics(tensor(prediction), tensor(target), lpips_net_type='vgg')

    sources = [repo / 'model_eval' / name for name in
               ('metrics_utils.py', 'genfusion_lossutil.py', 'lpipsPytorch/__init__.py',
                'lpipsPytorch/modules/lpips.py', 'lpipsPytorch/modules/networks.py', 'lpipsPytorch/modules/utils.py')]
    provenance = {'implementation': 'model_eval.metrics_utils.compute_rgb_metrics',
                  'upstream_revision': revision, 'repo': str(repo), 'device': str(device),
                  'torch_version': torch.__version__, 'lpips_backbone': 'vgg', 'lpips_version': '0.1',
                  'source_sha256': {str(path.relative_to(repo)): digest_file(path) for path in sources},
                  'weights_sha256': {path.name: digest_file(path) for path in weights}}
    return compute, provenance


def _number(value):
    value = float(value)
    if math.isnan(value):
        raise ValueError('Metric returned NaN')
    return value if math.isfinite(value) else ('Infinity' if value > 0 else '-Infinity')


def evaluate_benchmark(run_dir, *, repo=None, device='cpu', output=None):
    """Return and save per-image and arithmetic mean metrics for exactly 25 views."""
    plan = evaluation_plan(run_dir)
    destination = Path(output) if output else Path(plan['root']) / 'published-test-metrics.json'
    if destination.exists():
        raise FileExistsError(f'Preserving existing evaluation; choose another output: {destination}')
    if repo is None:
        repo = _read(Path(plan['root']) / 'benchmark-run.json')['runtime']['repo']
    compute, provenance = load_official_metrics(repo, device)
    records = []
    values = {method: {metric: [] for metric in METRICS} for method in METHODS}
    for pair in plan['pairs']:
        target = _rgb(pair['ground_truth'])
        record = {**pair, 'sha256': {'ground_truth': digest_file(pair['ground_truth'])}, 'metrics': {}}
        for method, path in pair['predictions'].items():
            prediction = _rgb(path)
            if prediction.shape != target.shape:
                raise ValueError('Image changed dimensions after evaluation preflight')
            scores = compute(prediction, target)
            if set(scores) != set(METRICS):
                raise ValueError('Official metric result is missing PSNR, SSIM, or LPIPS')
            record['metrics'][method] = {key: _number(scores[key]) for key in METRICS}
            record['sha256'][method] = digest_file(path)
            for key in METRICS:
                values[method][key].append(float(scores[key]))
        records.append(record)
    result = {'schema_version': 1, 'published_test_count': len(records), 'reference_images_excluded': plan['references'],
              'aggregation': 'unweighted arithmetic mean of per-image metrics over published 25 photographs only',
              'image_policy': 'original RGB dimensions; [0,1] full image; no resize, mask, crop, or color alignment',
              'official_inference_resize': 'Upstream VAE inputs are aligned to multiples of 16; upstream predictions are saved at original target dimensions.',
              'metrics': provenance, 'benchmark_provenance': {key: value for key, value in plan.items() if key != 'pairs'},
              'per_image': records,
              'aggregate': {method: {key: _number(np.mean(scores)) for key, scores in metrics.items()}
                            for method, metrics in values.items()},
              'scope': 'Published Bicycle test-image scores; not a website-orbit match or verified paper-protocol reproduction.'}
    atomic_json(destination, result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = evaluate_benchmark(args.run_dir, repo=args.repo, device=args.device, output=args.output)
    print(json.dumps(result['aggregate'], indent=2))
