"""Controlled initial-fit comparison using unchanged authors' preparation.

Two fresh 10k fits share source photographs, poses, points, and seed. Only the
COLMAP camera model changes: PINHOLE versus zero-distortion OPENCV, which honors
the stored principal point in the released 3DGRUT loader. No diffusion runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

REVISION = 'a392c4dfe17459ef9952407accdb9fcdcdddba98'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def make_variant(source, destination, model):
    """Keep every source byte except the camera representation unchanged."""
    original = source / 'colmap/sparse/0/cameras.bin'
    with original.open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        camera_id, model_id, width, height = struct.unpack('<iiQQ', stream.read(24))
        if count != 1 or model_id != 1:
            raise ValueError('This control requires the original one-camera PINHOLE Bicycle input')
        intrinsics = struct.unpack('<dddd', stream.read(32))
        if stream.read(1):
            raise ValueError('Unexpected camera records')
    sparse = destination / 'sparse/0'
    sparse.mkdir(parents=True)
    (destination / 'images').symlink_to((source / 'colmap/images').resolve(), target_is_directory=True)
    for name in ('images.bin', 'points3D.bin'):
        (sparse / name).symlink_to((source / 'colmap/sparse/0' / name).resolve())
    camera = sparse / 'cameras.bin'
    if model == 'pinhole':
        camera.write_bytes(original.read_bytes())
    elif model == 'opencv':
        camera.write_bytes(struct.pack('<QiiQQdddddddd', 1, camera_id, 4,
            width, height, *intrinsics, 0., 0., 0., 0.))
    else:
        raise ValueError('Unknown camera model')
    return {'camera_model': model, 'width': width, 'height': height,
            'stored_fx_fy_cx_cy': intrinsics, 'camera_sha256': digest(camera),
            'images_bin_sha256': digest(sparse / 'images.bin'),
            'points3D_bin_sha256': digest(sparse / 'points3D.bin')}


def train_arm(args):
    import numpy as np
    import torch
    sys.path[:0] = [str(args.repo), str(args.repo / 'thirdparty/3DGRUT-ArtiFixer')]
    from data_processing.prepare_colmap_artifixer_inputs import build_parser, prepare_colmap_scene
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    upstream = build_parser().parse_args([
        '--colmap_dir', str(args.output / args.arm / 'colmap'),
        '--output_root', str(args.output / args.arm / 'prepared/bicycle'),
        '--selected_image_names_file', str(args.source / 'selected_images.txt'),
        '--reconstruction_steps', '10000', '--phases', 'prepare,reconstruct,render'])
    prepare_colmap_scene(upstream)


def score_arm(source, root):
    import numpy as np
    from PIL import Image
    metadata = json.loads((source / 'benchmark.json').read_text())
    prepared = root / 'prepared/bicycle'
    frames = json.loads((prepared / '3dgrut_input/bicycle/nerfstudio/transforms.json').read_text())['frames']
    names = [Path(frame['file_path']).name for frame in frames]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate prepared filenames')
    renders = prepared / 'recon_results/bicycle/reconstruction/bicycle/ours_10000/renders'
    groups = {}
    for group, key in (('published_test', 'test_images'), ('training', 'selected_images')):
        records = []
        for name in metadata[key]:
            gt = source / 'colmap/images' / name
            if digest(gt) != metadata['sha256'][f'colmap/images/{name}']:
                raise ValueError(f'Source photograph changed: {name}')
            prediction = renders / f'{names.index(name):05d}.png'
            with Image.open(gt) as image:
                target = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
            with Image.open(prediction) as image:
                actual = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
            if actual.shape != target.shape:
                raise ValueError('Image shapes differ; no implicit resizing')
            mse = float(np.mean((actual - target) ** 2))
            records.append({'name': name, 'index': names.index(name), 'mse': mse,
                            'psnr_db': float(-10 * np.log10(mse)),
                            'prediction_sha256': digest(prediction)})
        groups[group] = {'count': len(records), 'mean_psnr_db': float(np.mean([r['psnr_db'] for r in records])),
                         'per_image': records}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', choices=('pinhole', 'opencv'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.repo, args.source, args.output = (p.resolve() for p in (args.repo, args.source, args.output))
    revision = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Use the pinned authors checkout')
    if args.arm:
        train_arm(args)
        return
    if args.output.is_relative_to(args.source):
        raise ValueError('Output must be outside the preserved source input')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'pid': os.getpid(), 'started_at_unix': time.time(),
             'source': str(args.source), 'upstream_revision': revision, 'seed': args.seed,
             'script_sha256': digest(__file__), 'source_metadata_sha256': digest(args.source / 'benchmark.json'),
             'scope': 'Initial authored 10k fit only; no diffusion or website-orbit replication. Identical seeds do not guarantee deterministic CUDA training.',
             'arms': {}}
    save(args.output / 'comparison.json', state)
    try:
        for arm in ('pinhole', 'opencv'):
            root = args.output / arm
            provenance = make_variant(args.source, root / 'colmap', arm)
            record = {'status': 'running', 'input': provenance, 'started_at_unix': time.time()}
            state['arms'][arm] = record
            save(args.output / 'comparison.json', state)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--repo', str(args.repo),
                       '--source', str(args.source), '--output', str(args.output),
                       '--seed', str(args.seed), '--arm', arm]
            with (root / 'stage.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                save(args.output / 'comparison.json', state)
                try:
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
            if code:
                record.update(status='error', exit_code=code)
                raise RuntimeError(f'{arm} failed with exit {code}; inspect {root / "stage.log"}')
            record.update(status='complete', elapsed_seconds=time.time() - record['started_at_unix'],
                          scores=score_arm(args.source, root))
            save(args.output / 'comparison.json', state)
            print(arm, record['scores']['published_test']['mean_psnr_db'], flush=True)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='error', error=str(exc))
        raise
    finally:
        state['updated_at_unix'] = time.time()
        save(args.output / 'comparison.json', state)


if __name__ == '__main__':
    main()
