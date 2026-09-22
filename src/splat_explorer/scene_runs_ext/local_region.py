"""Small, offline ArtiFixer experiments on one calibrated image region.

No scene fitting or publication: first establish that local image supervision
actually improves. Cropping shifts the principal point, never the camera pose.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image


def crop_transforms(transforms, box, count):
    left, top, right, bottom = box
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in box)
            or not 0 <= left < right <= transforms['w']
            or not 0 <= top < bottom <= transforms['h']):
        raise ValueError('Crop must be an integer rectangle inside the source image')
    width, height = right-left, bottom-top
    if width % 16 or height % 16:
        raise ValueError('Crop dimensions must be multiples of 16')
    out = copy.deepcopy(transforms)
    out['frames'] = out['frames'][:count]
    # Upstream accepts per-frame intrinsics; preserve them where present.
    for camera in [out, *out['frames']]:
        camera['cx'] = camera.get('cx', transforms['cx']) - left
        camera['cy'] = camera.get('cy', transforms['cy']) - top
        camera['w'], camera['h'] = width, height
    return out


def choose_reference_frames(transforms, count=5):
    """Greedy translated-view coverage from one continuous local trajectory."""
    poses = np.array([f['transform_matrix'] for f in transforms['frames']])
    positions = poses[:, :3, 3]
    selected = [0]
    while len(selected) < count:
        distances = np.linalg.norm(positions[:, None]-positions[selected][None], axis=-1).min(axis=1)
        index = int(np.argmax(distances))
        if distances[index] < 1e-6:
            raise ValueError('Not enough distinct translated cameras for local references')
        selected.append(index)
    return sorted(selected)


def prepare_region(source, destination, box, reference_count=5):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Use a fresh experiment directory; saved evidence is immutable')
    manifest = json.loads((source/'bundle.json').read_text())
    count = manifest.get('segments', [{'count':len(manifest['transforms']['frames'])}])[0]['count']
    cameras = crop_transforms(manifest['transforms'], box, count)
    indices = choose_reference_frames(cameras, reference_count)
    destination.mkdir(parents=True)
    (destination/'inputs').mkdir()
    for i in range(count):
        with Image.open(source/'inputs'/f'{i:05d}.png') as image:
            image.convert('RGB').crop(box).save(destination/'inputs'/f'{i:05d}.png')
    with Image.open(source/'anchor.png') as image:
        image.convert('RGB').crop(box).save(destination/'anchor.png')
    left, top, right, bottom = box
    opacity = np.load(source/'opacity.npy', mmap_mode='r', allow_pickle=False)
    cropped = np.array(opacity[:count, top:bottom, left:right], dtype=np.float32)
    np.save(destination/'opacity.npy', cropped)
    manifest.update(transforms=cameras, segments=[{'start':0,'count':count}],
                    references=[{'path':'anchor.png','frame_index':0,'kind':'edited_render'}],
                    local_region={'source':str(source),'crop':list(box),'source_size':[manifest['transforms']['w'],manifest['transforms']['h']],
                                  'reference_frames':indices,'opacity_mean':float(cropped.mean()),
                                  'opacity_above_095_fraction':float((cropped>.95).mean())})
    # Full-frame validation cameras no longer describe these cropped images.
    manifest.pop('validation_transforms', None)
    (destination/'bundle.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def install_references(root):
    """Register exactly five or more edited crops at their actual camera poses."""
    root = Path(root)
    manifest = json.loads((root/'bundle.json').read_text())
    references = []
    for i in manifest['local_region']['reference_frames']:
        path = f'references/{i:05d}.png'
        with Image.open(root/path) as image:
            if image.size != (manifest['transforms']['w'],manifest['transforms']['h']):
                raise ValueError('Edited crop dimensions must match its calibrated camera')
        references.append({'path':path,'frame_index':i,'kind':'edited_render'})
    manifest['references'] = references
    (root/'bundle.json').write_text(json.dumps(manifest, indent=2))
