"""Finish all currently known source Viser captures before LRZ staging."""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ..checkpoint import atomic_json, camera_from_record
from ..persistence import checkpoint_cache_lock
from ..resolution import prepare_repair_checkpoint
from ..viser_capture import CaptureSession
from .pipeline_profile import rgb_backend
from .staging_cache import capture_prepared_rgb, digest


def prepare_source_captures(checkpoint, runtime, root, *, should_stop=lambda: False,
                            progress=lambda **event: None, url=None):
    """Warm immutable input caches, without rendering CUDA geometry locally.

    New authored orbits depend on GPU MoGe scale. Their intermediate poses
    cannot be captured yet; an exact repeat orbit is already known and can.
    Existing geometry is checked in place, avoiding a multi-GB local copy.
    """
    if rgb_backend(runtime) != 'viser':
        return None
    root = Path(root)
    manifest_hash = digest(checkpoint.root / 'checkpoint.json')
    repeat_path = checkpoint.root / 'repeat-repair.json'
    repeat = json.loads(repeat_path.read_text()) if repeat_path.is_file() else None
    if runtime.get('correct_generated_pose') and not repeat:
        from ..pose_correction import corrected_checkpoint
        checkpoint = corrected_checkpoint(checkpoint)
    if should_stop():
        raise InterruptedError('Cancelled before Viser capture preflight')
    progress(phase='viser_preflight', message='Checking cached source RGBs and trajectories before staging')
    session, captured = None, 0

    def capture(source, cameras, destinations, *, up_axis, lod_level, should_stop):
        nonlocal session, captured
        if should_stop():
            raise InterruptedError('Viser capture preflight stopped')
        if session is None:
            progress(phase='viser_preflight', message='Loading Viser for source captures before staging')
            session = CaptureSession(source, up_axis=up_axis, lod_level=lod_level,
                                     url=url, should_stop=should_stop)
        for index, (camera, destination) in enumerate(zip(cameras, destinations)):
            if should_stop():
                raise InterruptedError('Viser capture preflight stopped')
            progress(phase='viser_preflight', message=f'Capturing source RGB before staging: {index + 1}/{len(cameras)}')
            Image.fromarray(session.render(camera)).save(destination)
            captured += 1

    # The same calibrated preparation as the worker includes accepted corrected
    # poses, cover-cropped intrinsics, renderer identity and source fingerprint.
    # Lock against asynchronous result downloads publishing these same caches.
    with checkpoint_cache_lock(checkpoint.root), tempfile.TemporaryDirectory(prefix='viser-preflight-', dir=root) as temporary:
        prepared = prepare_repair_checkpoint(checkpoint, Path(temporary) / 'checkpoint',
            runtime['resolution_profile'], should_stop=should_stop, reuse_artifixer_inputs=True,
            artifixer_runtime=runtime, preserve_repaired_images=bool(repeat),
            restore_cached_geometry=False, capture_rgb_fn=capture)
        cache = Path(prepared.manifest['metadata']['artifixer_input_cache'])
        trajectories = 0
        for marker in sorted((cache / 'trajectories').glob('*/trajectory.json')):
            if should_stop():
                raise InterruptedError('Stopped checking cached trajectories')
            saved = json.loads(marker.read_text())
            for relative, expected in saved['sha256'].items():
                path = cache / relative
                if not path.resolve().is_relative_to(cache.resolve()) or digest(path) != expected:
                    raise ValueError(f'Saved trajectory changed before staging: {relative}')
            trajectories += 1
        orbit = repeat.get('orbit') if repeat else None
        if orbit:
            poses = np.asarray(orbit['poses'], dtype=np.float64)
            if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
                raise ValueError('Invalid repeat orbit before staging')
            anchor = camera_from_record(prepared.views[0])
            cameras = [replace(anchor, position=pose[:3, 3], rotation=pose[:3, :3]) for pose in poses]
            destinations = [Path(temporary) / f'{index:05d}.png' for index in range(len(cameras))]
            metadata = prepared.manifest.get('metadata', {})
            capture_prepared_rgb(prepared, cameras, destinations,
                lambda requested, paths: capture(prepared.manifest['scene_path'], requested, paths,
                    up_axis=metadata.get('up_axis', '-y'),
                    lod_level=metadata.get('scene_load', {}).get('lod_level', 0), should_stop=should_stop),
                cache_root=cache)
        if should_stop():
            raise InterruptedError('Cancelled before staging')
        if digest(checkpoint.root / 'checkpoint.json') != manifest_hash:
            raise ValueError('Checkpoint changed during Viser preflight; start a new run with the updated views')
        result = {'checkpoint_sha256': manifest_hash, 'captured_frames': captured,
                  'anchor_count': len(prepared.views), 'verified_trajectories': trajectories,
                  'input_cache': str(cache), 'known_repeat_orbit': bool(orbit),
                  'remaining_source_captures': 'none for exact repeat' if orbit else 'new intermediate poses after GPU metric alignment'}
        atomic_json(root / 'source-capture-preflight.json', result)
    progress(phase='viser_preflight', message=(f'Source RGB ready before staging · {captured} captured · '
             f'{trajectories} cached trajectories verified'), source_capture_preflight=result)
    return result
