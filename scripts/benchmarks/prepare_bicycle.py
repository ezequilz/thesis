"""Import released ReconFusion Bicycle calibration and original COLMAP points.

No photographic pixels are resampled. A verified similarity transforms original
COLMAP points into the released ReconFusion pose coordinate system.
"""
from pathlib import Path
import argparse
import json
import shutil
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.repair import digest_file


def prepare(source, output):
    source, output = Path(source), Path(output)
    released = source / 'reconfusion/mipnerf360/bicycle'
    original = source / 'original/bicycle/sparse/0'
    transforms = json.loads((released / 'transforms.json').read_text())
    split = json.loads((released / 'train_test_split_3.json').read_text())
    with (original / 'cameras.bin').open('rb') as stream:
        camera_header = struct.unpack('<Qi iQQdddd', stream.read(64))
    original_width, original_height = camera_header[3:5]
    records = {}
    with (original / 'images.bin').open('rb') as stream:
        for _ in range(struct.unpack('<Q', stream.read(8))[0]):
            record = struct.unpack('<idddddddi', stream.read(64))
            name = bytearray()
            while (char := stream.read(1)) != b'\0':
                if not char:
                    raise ValueError('Truncated COLMAP image name')
                name.extend(char)
            count = struct.unpack('<Q', stream.read(8))[0]
            observations = stream.read(24 * count)
            w2c = np.eye(4)
            w2c[:3, :3] = Rotation.from_quat([*record[2:5], record[1]]).as_matrix()
            w2c[:3, 3] = record[5:8]
            records[name.decode()] = (record, observations, np.linalg.inv(w2c) @ np.diag([1,-1,-1,1]))
    names = [Path(frame['file_path']).name for frame in transforms['frames']]
    old = np.array([records[name][2] for name in names])
    new = np.array([frame['transform_matrix'] for frame in transforms['frames']])
    rotation = new[0, :3, :3] @ old[0, :3, :3].T
    x, y = old[:, :3, 3] @ rotation.T, new[:, :3, 3]
    scale = np.sum((x-x.mean(0))*(y-y.mean(0))) / np.sum((x-x.mean(0))**2)
    translation = y.mean(0) - scale*x.mean(0)
    pose_error = float(np.max(np.abs(scale*x+translation-y)))
    rotation_error = float(np.max(np.abs(np.einsum('ij,njk->nik', rotation, old[:,:3,:3])-new[:,:3,:3])))
    if max(pose_error, rotation_error) > 1e-8 or scale <= 0:
        raise ValueError('Original COLMAP and released cameras do not share a similarity')
    if set(split['train_ids']) & set(split['test_ids']):
        raise ValueError('Training and test images overlap')
    output.mkdir(parents=True, exist_ok=False)
    sparse = output / 'colmap/sparse/0'
    sparse.mkdir(parents=True)
    images = output / 'colmap/images'
    images.mkdir()
    with (sparse / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<Qi iQQdddd', 1, 1, 1, transforms['w'], transforms['h'],
            transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']))
    with (sparse / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(names)))
        for name, c2w in zip(names, new):
            record, observations, _ = records[name]
            w2c = np.linalg.inv(c2w @ np.diag([1,-1,-1,1]))
            q = Rotation.from_matrix(w2c[:3,:3]).as_quat()
            stream.write(struct.pack('<idddddddi', record[0], q[3], *q[:3], *w2c[:3,3], 1))
            stream.write(name.encode()+b'\0')
            # Metric alignment requires the measured pixel observations. Scale
            # them by the actual released image dimensions (including rounding).
            obs = np.frombuffer(observations, dtype=[('x','<f8'),('y','<f8'),('id','<i8')]).copy()
            obs['x'] *= transforms['w'] / original_width
            obs['y'] *= transforms['h'] / original_height
            stream.write(struct.pack('<Q', len(obs)))
            stream.write(obs.tobytes())
            shutil.copy2(released / 'images_4' / name, images / name)
    with (original / 'points3D.bin').open('rb') as src, (sparse / 'points3D.bin').open('wb') as dst:
        count = struct.unpack('<Q', src.read(8))[0]
        dst.write(struct.pack('<Q', count))
        for _ in range(count):
            point = struct.unpack('<QdddBBBd', src.read(43))
            track_count = struct.unpack('<Q', src.read(8))[0]
            tracks = src.read(track_count*8)
            xyz = scale * rotation @ np.array(point[1:4]) + translation
            dst.write(struct.pack('<QdddBBBdQ', point[0], *xyz, *point[4:7], point[7], track_count))
            dst.write(tracks)
    selected = [names[i] for i in split['train_ids']]
    (output / 'selected_images.txt').write_text('\n'.join(selected)+'\n')
    atomic_json(output / 'transforms.json', transforms)
    manifest = {'schema_version': 1, 'name': 'bicycle', 'label': 'Bicycle — published three-view split',
        'train_ids': split['train_ids'], 'test_ids': split['test_ids'], 'selected_images': selected,
        'test_images': [names[i] for i in split['test_ids']], 'image_count': len(names),
        'sources': {'split': 'https://reconfusion.github.io/', 'colmap': 'https://jonbarron.info/mipnerf360/',
            'reference': 'https://research.nvidia.com/labs/sil/projects/artifixer/vids/bicycle.mp4'},
        'calibration': 'Released ReconFusion intrinsics and OpenGL camera poses; original sparse points transformed to match',
        'sparse_point_count': count, 'similarity': {'scale': float(scale), 'rotation': rotation.tolist(),
            'translation': translation.tolist(), 'position_max_error': pose_error, 'rotation_max_error': rotation_error},
        'website_exact_match_verified': False,
        'limitations': ['Website orbit cameras, prompt, random seed and model variant not released with these inputs.',
            'Source camera trajectory is available; it is not asserted to be the website novel orbit.',
            'Original full-scene COLMAP sparse initialization is retained; author benchmark initialization is not yet verified.'],
        'sha256': {str(p.relative_to(output)): digest_file(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'source_sha256': {str(p.relative_to(source)): digest_file(p) for p in [released/'transforms.json', released/'train_test_split_3.json', original/'cameras.bin', original/'images.bin', original/'points3D.bin']}}
    atomic_json(output / 'benchmark.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
