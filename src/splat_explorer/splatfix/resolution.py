"""Calibrated input sizing; never stretch an existing camera to a new aspect."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import shutil
import struct

from PIL import Image

DEFAULT_PROFILE = 'training'
PROFILES = {'training': (832, 480), '720p': (1280, 720), 'early_original': (960, 720)}
PROFILE_LABELS = {
    'early_original': 'v0 · Early original resolution',
    'training': 'v1 832 x 480 (default)',
    '720p': 'v1 1280 x 720',
}
# Invalidate prepared-input caches made with the former 960 x 544 bounds.
POLICY_VERSION = 2


def profile_size(profile=DEFAULT_PROFILE):
    if profile not in PROFILES:
        raise ValueError('resolution_profile must be training, 720p or early_original')
    return PROFILES[profile]


def resize_plan(width, height, profile=DEFAULT_PROFILE):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    bounds = profile_size(profile)
    if profile == 'early_original':
        return {'version': POLICY_VERSION, 'profile': profile, 'bounds': None,
                'source_wh': [width, height], 'output_wh': [width, height],
                'scale': 1., 'offset_xy': [0., 0.],
                'method': 'early original: preserve source dimensions; upstream diffusion alignment only'}
    scale = min(1., bounds[0] / width, bounds[1] / height)
    target = [16 * math.floor((value * scale + 1e-8) / 16) for value in (width, height)]
    if min(target) < 16:
        raise ValueError('ArtiFixer inputs must be at least 16 pixels in each dimension')
    offset = [max(0., (value * scale - size) / 2) for value, size in zip((width, height), target)]
    return {'version': POLICY_VERSION, 'profile': profile, 'bounds': list(bounds),
            'source_wh': [width, height], 'output_wh': target, 'scale': scale,
            'offset_xy': offset, 'method': 'uniform scale, centered subpixel crop to multiples of 16; no upscale'}


def resize_image(image, plan):
    if list(image.size) != plan['source_wh']:
        raise ValueError('Image dimensions differ from recorded resolution plan')
    if plan['source_wh'] == plan['output_wh']:
        return image.copy()
    s = plan['scale']; x, y = plan['offset_xy']; w, h = plan['output_wh']
    # Floating-point roundoff at a full-height/width edge can exceed the source
    # by ~1e-13 pixels; Pillow requires the crop box to lie strictly inside it.
    box = (x/s, y/s, min(image.width, (x+w)/s), min(image.height, (y+h)/s))
    return image.resize((w, h), Image.Resampling.BILINEAR, box=box)


def calibrate(fx, fy, cx, cy, plan):
    s = plan['scale']; x, y = plan['offset_xy']
    return fx*s, fy*s, cx*s-x, cy*s-y


def prepare_colmap(source, destination, profile=DEFAULT_PROFILE):
    """Copy the Bicycle PINHOLE inputs with matching pixels, K and observations.

    Source photographs, poses, point positions, tracks and historical runs remain
    intact. This is an explicit preparation adaptation, not an upstream patch.
    """
    source, destination = Path(source), Path(destination)
    profile_size(profile)
    if profile == 'early_original':
        shutil.copytree(source, destination)
        return {'version': POLICY_VERSION, 'profile': profile, 'images': {},
                'label': PROFILE_LABELS[profile],
                'photographic_policy': 'early original: source pixels and COLMAP files copied unchanged; no resolution cap; upstream diffusion alignment only'}
    sparse = destination / 'sparse/0'
    sparse.mkdir(parents=True, exist_ok=False)
    (destination / 'images').mkdir()
    plans = {}
    with (source / 'sparse/0/cameras.bin').open('rb') as src, (sparse / 'cameras.bin').open('wb') as dst:
        count_bytes = src.read(8); count, = struct.unpack('<Q', count_bytes); dst.write(count_bytes)
        for _ in range(count):
            ident, model, width, height = struct.unpack('<iiQQ', src.read(24))
            if model != 1:
                raise ValueError('Bicycle resolution preparation requires PINHOLE COLMAP cameras')
            params = struct.unpack('<dddd', src.read(32))
            plan = plans[ident] = resize_plan(width, height, profile)
            dst.write(struct.pack('<iiQQdddd', ident, model, *plan['output_wh'], *calibrate(*params, plan)))
    image_plans = {}
    with (source / 'sparse/0/images.bin').open('rb') as src, (sparse / 'images.bin').open('wb') as dst:
        count_bytes = src.read(8); count, = struct.unpack('<Q', count_bytes); dst.write(count_bytes)
        for _ in range(count):
            record = src.read(64); camera_id = struct.unpack('<idddddddi', record)[-1]
            name = bytearray()
            while True:
                char = src.read(1)
                if not char: raise ValueError('Truncated COLMAP image name')
                if char == b'\0': break
                name.extend(char)
            filename = name.decode(); relative = Path(filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe COLMAP image filename')
            plan = plans[camera_id]; image_plans[filename] = plan
            dst.write(record + name + b'\0')
            nbytes = src.read(8); n, = struct.unpack('<Q', nbytes); dst.write(nbytes)
            for _ in range(n):
                x, y, point = struct.unpack('<ddq', src.read(24))
                dst.write(struct.pack('<ddq', x*plan['scale']-plan['offset_xy'][0], y*plan['scale']-plan['offset_xy'][1], point))
            target = destination / 'images' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source / 'images' / relative) as image:
                # Lossless payload retains source basename for published split identity.
                resize_image(image.convert('RGB'), plan).save(target, format='PNG')
    shutil.copy2(source / 'sparse/0/points3D.bin', sparse / 'points3D.bin')
    return {'version': POLICY_VERSION, 'profile': profile, 'images': image_plans,
            'photographic_policy': 'preserve aspect and calibrated rays; crop less than 16 output pixels per dimension',
            'training_evidence': 'Default bounds follow Wan2.1 T2V-1.3B 832x480 configuration, not a measured ArtiFixer training optimum. Both profiles are divisible by 16; existing images use uniform scaling and calibrated cropping, never independent axis resizing.'}


def prepare_checkpoint(checkpoint, output, profile=DEFAULT_PROFILE):
    """Create a calibrated derivative when views exceed bounds or lack alignment."""
    import numpy as np
    from dataclasses import replace
    from .checkpoint import Checkpoint, camera_from_record, camera_to_record, atomic_json
    plans = [resize_plan(v['camera']['width'], v['camera']['height'], profile) for v in checkpoint.views]
    if all(p['source_wh'] == p['output_wh'] for p in plans):
        return checkpoint
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(checkpoint.manifest)
    for view, old, plan in zip(manifest['views'], checkpoint.views, plans):
        camera = camera_from_record(old)
        width, height = plan['output_wh']
        focal = camera.fx * plan['scale']
        camera = replace(camera, width=width, height=height,
                         fov_deg=float(np.degrees(2*np.arctan(width/(2*focal)))))
        view['camera'] = camera_to_record(camera)
        for key in ('original_rgb', 'repaired_rgb'):
            if old.get(key):
                relative = Path('views') / str(view['id']) / (key + '.png')
                target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(checkpoint.root / old[key]) as image:
                    resize_image(image.convert('RGB'), plan).save(target)
                view[key] = str(relative)
    manifest.setdefault('metadata', {})['resolution'] = {'profile': profile, 'plans': plans,
                                                        'source_checkpoint': str(checkpoint.root)}
    atomic_json(root / 'checkpoint.json', manifest)
    return Checkpoint.load(root)


def prepare_repair_checkpoint(checkpoint, output, profile=DEFAULT_PROFILE, *,
                              scene_path=None, should_stop=lambda: False):
    """Prepare exact-size step-4 cameras and recapture their source RGBs.

    Work in a run-local derivative. GPT references start from the original
    response when available, never a previous crop. Legacy checkpoints without
    saved responses fall back to their saved repaired PNG.
    """
    import numpy as np
    from dataclasses import replace
    from PIL import ImageOps
    from .checkpoint import Checkpoint, camera_from_record, camera_to_record, atomic_json, validate_source
    from .viser_capture import capture_rgb

    source = str(Path(scene_path).resolve()) if scene_path else checkpoint.manifest['scene_path']
    validate_source(Checkpoint(checkpoint.root, {**checkpoint.manifest, 'scene_path': source}))
    width, height = profile_size(profile)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(checkpoint.manifest)
    cameras, destinations = [], []
    for view, old in zip(manifest['views'], checkpoint.views):
        if should_stop():
            raise InterruptedError('Repair input preparation stopped')
        camera = camera_from_record(old)
        # A uniform cover transform of the saved camera: same pose, centered
        # crop, square pixels. Recapture rays using the corresponding focal length.
        scale = max(width / camera.width, height / camera.height)
        focal = camera.fx * scale
        camera = replace(camera, width=width, height=height,
                         fov_deg=float(np.degrees(2*np.arctan(width/(2*focal)))))
        view['camera'] = camera_to_record(camera)
        cameras.append(camera)
        directory = root / 'views' / view['id']
        directory.mkdir(parents=True)
        view['original_rgb'] = f"views/{view['id']}/original.png"
        destinations.append(root / view['original_rgb'])
        if old.get('repaired_rgb'):
            response = (old.get('image_repair') or {}).get('response_image')
            if response:
                original = checkpoint.image_path({'original_rgb': response})
                response_relative = f"views/{view['id']}/response.image"
                shutil.copy2(original, root / response_relative)
                view.setdefault('image_repair', {})['response_image'] = response_relative
            else:
                original = checkpoint.image_path(old, repaired=True)
            with Image.open(original) as image:
                image = ImageOps.fit(image.convert('RGB'), (width, height),
                                     Image.Resampling.LANCZOS, centering=(.5, .5))
                view['repaired_rgb'] = f"views/{view['id']}/repaired.png"
                image.save(root / view['repaired_rgb'])
            view.setdefault('image_repair', {})['reconstruction_resize'] = {
                'source': response or old['repaired_rgb'],
                'source_kind': 'full_response' if response else 'legacy_repaired_rgb',
                'method': 'cover_center_crop', 'target_size': [width, height]}
    metadata = manifest.setdefault('metadata', {})
    capture_rgb(source, cameras, destinations, up_axis=metadata.get('up_axis', '-y'),
                lod_level=metadata.get('scene_load', {}).get('lod_level', 0), should_stop=should_stop)
    for path in destinations:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError('Recaptured RGB dimensions differ from selected repair resolution')
    metadata.update(width=width, height=height)
    metadata['resolution'] = {'version': POLICY_VERSION, 'profile': profile,
                              'target_size': [width, height], 'recaptured': True,
                              'source_checkpoint': str(checkpoint.root)}
    atomic_json(root / 'checkpoint.json', manifest)
    return Checkpoint.load(root)
