"""Reusable captures at saved checkpoint poses, with independent FOV sets."""
from pathlib import Path
import copy
import fcntl
import hashlib

import numpy as np
from PIL import Image

from ..checkpoint import Checkpoint, camera_from_record, camera_to_record, utc_now, validate_source
from ...rendering.base import Camera

PROFILE = 'seva-54deg-1024x576-v1'


def capture_camera(view):
    saved = camera_from_record(view)
    return Camera(saved.position.copy(), saved.rotation.copy(), width=1024, height=576, fov_deg=54.)


def asset_path(checkpoint, relative):
    return checkpoint.image_path({'original_rgb': relative})


def validate_capture(checkpoint, view):
    capture = view.get('seva_capture')
    if not capture or capture.get('profile') != PROFILE:
        raise ValueError('Capture SEVA images in Step 2 first (view ' + view['id'] + ')')
    camera = camera_from_record(capture)
    if not np.allclose(camera.intrinsics, capture_camera(view).intrinsics, atol=1e-5) or not np.allclose(camera.c2w, camera_from_record(view).c2w, atol=1e-6):
        raise ValueError('SEVA capture calibration or saved pose changed')
    if (camera.width, camera.height) != (1024, 576) or capture.get('rgb_renderer') != 'viser':
        raise ValueError('Invalid SEVA capture profile')
    if capture.get('source_fingerprint') != checkpoint.manifest.get('source_fingerprint'):
        raise ValueError('SEVA capture source fingerprint changed')
    for key in ('original_rgb', 'source_depth'):
        path = asset_path(checkpoint, capture[key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != capture['sha256'][key]:
            raise ValueError('SEVA capture asset checksum mismatch: ' + key)
    with Image.open(asset_path(checkpoint, capture['original_rgb'])) as image:
        if image.size != (camera.width, camera.height):
            raise ValueError('SEVA capture image dimensions changed')
    with np.load(asset_path(checkpoint, capture['source_depth']), allow_pickle=False) as depth:
        if (depth['depth'].shape != (camera.height, camera.width) or depth['valid'].shape != depth['depth'].shape
                or not np.isfinite(depth['depth']).all() or not depth['valid'].any()
                or not np.allclose(depth['intrinsics'], camera.intrinsics)
                or not np.allclose(depth['c2w'], camera.c2w)):
            raise ValueError('SEVA depth calibration or data is invalid')
    return capture


def select_checkpoint(checkpoint, source='saved', *, edited=False):
    """Read-only image-set projection; never rewrite the original checkpoint."""
    if source == 'saved':
        return checkpoint
    if source != 'seva':
        raise ValueError('Unknown OracleGS capture source')
    views = []
    for view in checkpoint.views:
        capture = validate_capture(checkpoint, view)
        if edited and not capture.get('repaired_rgb'):
            raise ValueError('Repair the SEVA captures in Step 3 first (view ' + view['id'] + ')')
        views.append({**copy.deepcopy(capture), 'id':view['id']})
    manifest = copy.deepcopy(checkpoint.manifest)
    manifest['views'] = views
    manifest.setdefault('metadata', {}).setdefault('renderer', {})['backend'] = 'viser'
    manifest['reference_capture_profile'] = PROFILE
    return Checkpoint(checkpoint.root, manifest)


def save_capture(checkpoint, view, camera, rgb, depth):
    """Write a calibrated RGB/depth pair before publishing its view record."""
    from ..g4splat.source_depth import conservative_mask
    valid = conservative_mask(depth)
    if not valid.any():
        raise ValueError('No reliable SEVA source depth for view ' + view['id'])
    directory = checkpoint.root/'views'/view['id']/'seva'
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(directory/'original.png')
    np.savez_compressed(directory/'depth.npz', depth=np.where(valid, depth, 0).astype(np.float32),
                        valid=valid, intrinsics=camera.intrinsics, c2w=camera.c2w)
    capture = {'profile':PROFILE, 'camera':camera_to_record(camera), 'captured_at':utc_now(),
               'rgb_renderer':'viser', 'depth_renderer':'CpuSplatRenderer.render_depth',
               'depth_convention':'positive_camera_z_scene_units',
               'source_fingerprint':checkpoint.manifest.get('source_fingerprint'),
               'original_rgb':str((directory/'original.png').relative_to(checkpoint.root)),
               'source_depth':str((directory/'depth.npz').relative_to(checkpoint.root)),
               'repaired_rgb':None, 'image_repair':None}
    capture['sha256'] = {key:hashlib.sha256(asset_path(checkpoint, capture[key]).read_bytes()).hexdigest()
                         for key in ('original_rgb', 'source_depth')}
    validate_capture(checkpoint, {**view, 'seva_capture':capture})
    return capture


def capture_checkpoint(checkpoint, *, capture_set='seva', view_id=None, url=None, should_stop=lambda:False, on_progress=lambda **kw:None):
    """Capture missing views in the selected set; preserve other image sets."""
    from ..viser_capture import CaptureSession
    from ...scene import load_scene
    from ...rendering.cpu_splat_renderer import CpuSplatRenderer
    if capture_set not in ('saved', 'seva', '77'):
        raise ValueError('Unknown capture set')
    checkpoint = Checkpoint.load(checkpoint.root if isinstance(checkpoint, Checkpoint) else checkpoint)
    with (checkpoint.root/'.seva-capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = Checkpoint.load(checkpoint.root)
        if not checkpoint.complete:
            raise ValueError('Finish view selection before SEVA capture')
        views = [v for v in checkpoint.views if view_id is None or v['id'] == view_id]
        if not views:
            raise ValueError('Unknown saved view')
        pending = []
        for view in views:
            if capture_set == 'seva' and view.get('seva_capture'):
                validate_capture(checkpoint, view)
            elif capture_set == 'saved' and view.get('original_rgb') and (view.get('rgb_renderer') or meta_renderer(checkpoint)) == 'viser':
                continue
            elif (capture_set == '77' and capture_77(view).get('original_rgb')
                  and (capture_77(view).get('rgb_renderer') or meta_renderer(checkpoint)) == 'viser'):
                continue
            else:
                pending.append(view)
        if not pending:
            return checkpoint
        validate_source(checkpoint)
        meta = checkpoint.manifest.get('metadata', {})
        loading = meta.get('scene_load', {})
        scene_path = checkpoint.manifest['scene_path']
        session = CaptureSession(scene_path, up_axis=meta.get('up_axis', '-y'),
                                 lod_level=int(loading.get('lod_level', 0)), url=url, should_stop=should_stop)
        renderer = (CpuSplatRenderer(load_scene(scene_path, min_opacity=loading.get('min_opacity', 0.),
                                               lod_level=int(loading.get('lod_level', 0)))) if capture_set == 'seva' else None)
        for view in pending:
            if should_stop():
                raise InterruptedError('Saved-pose capture cancelled')
            on_progress(message=f'Capturing {capture_set} images for view ' + view['id'])
            camera = capture_camera(view) if capture_set == 'seva' else camera_from_record(view)
            if capture_set == '77':
                camera = Camera(camera.position.copy(), camera.rotation.copy(), width=camera.width, height=camera.height, fov_deg=77.)
            rgb = session.render(camera)
            if capture_set == 'seva':
                capture = save_capture(checkpoint, view, camera, rgb, renderer.render_depth(camera))
            else:
                directory = checkpoint.root/'views'/view['id']/('fov77' if capture_set == '77' else 'saved-capture')
                directory.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(directory/'original.png')
                capture = {'camera':camera_to_record(camera), 'rgb_renderer':'viser', 'captured_at':utc_now(),
                           'original_rgb':str((directory/'original.png').relative_to(checkpoint.root)),
                           'repaired_rgb':None, 'image_repair':None}
            # Merge into the latest manifest, retaining unrelated original-image edits.
            latest = Checkpoint.load(checkpoint.root)
            target = next(v for v in latest.views if v['id'] == view['id'])
            if target['camera'] != view['camera']:
                raise ValueError('Saved view changed during SEVA capture')
            if capture_set == 'saved':
                target.update(capture)
                if all(v.get('rgb_renderer') == 'viser' for v in latest.views):
                    latest.manifest.setdefault('metadata', {}).setdefault('renderer', {})['backend'] = 'viser'
            else:
                target['seva_capture' if capture_set == 'seva' else 'capture_77'] = capture
            latest.save()
            checkpoint = latest
        return checkpoint


def meta_renderer(checkpoint):
    return checkpoint.manifest.get('metadata', {}).get('renderer', {}).get('backend')


def capture_77(view):
    """Resolve separately captured 77° RGBs or a native 77° original."""
    if view.get('capture_77'):
        return view['capture_77']
    if abs(float(view['camera']['fov_deg']) - 77) < 0.001:
        return view
    return {}
