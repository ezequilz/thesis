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
from ..viser_capture import CAPTURE_POLICY, CaptureSession
from .pipeline_profile import rgb_backend
from .staging_cache import capture_prepared_rgb, digest


def prepare_source_captures(checkpoint, runtime, root, *, should_stop=lambda: False,
                            progress=lambda **event: None, url=None, source_plan=None):
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
        from .source_plan import identity, signature, validate
        plan_path = cache / 'source-plans' / (signature(identity(prepared, runtime)) + '.json')
        if source_plan is None and not repeat and plan_path.is_file():
            source_plan = json.loads(plan_path.read_text())
        if source_plan is not None:
            validate(source_plan, prepared, runtime)
            plan_path.parent.mkdir(exist_ok=True)
            atomic_json(plan_path, source_plan)
        orbit = source_plan['orbit'] if source_plan else (repeat.get('orbit') if repeat else None)
        if orbit:
            poses = np.asarray(orbit['poses'], dtype=np.float64)
            if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
                raise ValueError('Invalid repeat orbit before staging')
            anchor = camera_from_record(prepared.views[0])
            cameras = [replace(anchor, position=pose[:3, 3], rotation=pose[:3, :3]) for pose in poses]
            destinations = [Path(temporary) / f'{index:05d}.png' for index in range(len(cameras))]
            metadata = prepared.manifest.get('metadata', {})
            cached_frames = cached_orbit_rgb(cache, prepared, poses)
            from ..checkpoint import camera_to_record
            import shutil
            by_camera = {json.dumps(camera_to_record(camera), sort_keys=True): path
                         for camera, path in zip(cameras, cached_frames)} if cached_frames else {}
            def capture_orbit(requested, paths):
                missing, missing_paths = [], []
                for camera, path in zip(requested, paths):
                    cached = by_camera.get(json.dumps(camera_to_record(camera), sort_keys=True))
                    if cached:
                        shutil.copy2(cached, path)
                    else:
                        missing.append(camera)
                        missing_paths.append(path)
                if missing:
                    capture(prepared.manifest['scene_path'], missing, missing_paths,
                        up_axis=metadata.get('up_axis', '-y'),
                        lod_level=metadata.get('scene_load', {}).get('lod_level', 0), should_stop=should_stop)
            capture_prepared_rgb(prepared, cameras, destinations,
                                 capture_orbit, cache_root=cache)
        if should_stop():
            raise InterruptedError('Cancelled before staging')
        if digest(checkpoint.root / 'checkpoint.json') != manifest_hash:
            raise ValueError('Checkpoint changed during Viser preflight; start a new run with the updated views')
        result = {'checkpoint_sha256': manifest_hash, 'captured_frames': captured,
                  'anchor_count': len(prepared.views), 'verified_trajectories': trajectories,
                  'input_cache': str(cache), 'known_repeat_orbit': bool(repeat and orbit),
                  'trajectory_frames': len(orbit['poses']) if orbit else 0,
                  'source_plan': str(plan_path) if source_plan else None,
                  'remaining_source_captures': 'none' if orbit else 'pending lean GPU metric alignment and trajectory plan'}
        atomic_json(root / 'source-capture-preflight.json', result)
    progress(phase='viser_preflight', message=(f'Source RGB ready before staging · {captured} captured · '
             f'{trajectories} cached trajectories verified'), source_capture_preflight=result)
    return result


def cached_orbit_rgb(cache, checkpoint, poses):
    """Reuse checked Viser trajectory pixels, including pre-plan historical caches."""
    views = [{k: v[k] for k in ('id', 'camera', 'original_rgb')} for v in checkpoint.views]
    source = checkpoint.manifest.get('source_fingerprint', checkpoint.manifest['scene_path'])
    hashes = [digest(checkpoint.image_path(view)) for view in checkpoint.views]
    for path in sorted((cache / 'trajectories').glob('*/trajectory.json')):
        data = json.loads(path.read_text())
        recipe = data.get('recipe', {})
        if (recipe.get('anchor_rgb_policy') != 'viser' or recipe.get('capture_policy') != CAPTURE_POLICY
                or recipe.get('source') != source
                or recipe.get('views') != views or recipe.get('rgb') != hashes
                or recipe.get('scene_load', {}) != checkpoint.manifest.get('metadata', {}).get('scene_load', {})):
            continue
        previous = np.asarray((recipe.get('orbit') or {}).get('poses', []))
        if previous.shape == poses.shape and np.allclose(previous, poses, rtol=0, atol=1e-12):
            return [cache / frame['rgb'] for frame in data['frames']]
    return []
