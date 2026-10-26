"""Optional checkpoint pose correction, with immutable originals and stale-input guards."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .checkpoint import (Checkpoint, atomic_json, camera_from_record, camera_to_record,
                         utc_now, validate_source)

VERSION = 1


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def target_path(cp, view):
    """Use the same reference source/crop as resolution.prepare_repair_checkpoint."""
    relative = (view.get('manual_resized_rgb') or
                (view.get('image_repair') or {}).get('response_image') or view['repaired_rgb'])
    return cp.image_path({'original_rgb': relative})


def inputs(cp, view):
    return copy.deepcopy({'camera': view['camera'], 'source': cp.manifest.get('source_fingerprint'),
            'scene_load': cp.manifest.get('metadata', {}).get('scene_load', {}),
            'up_axis': cp.manifest.get('metadata', {}).get('up_axis', '-y'),
            'original_sha256': digest(cp.image_path(view)),
            'repaired_sha256': digest(cp.image_path(view, repaired=True)),
            'target_sha256': digest(target_path(cp, view))})


def correction_status(cp, view):
    record = view.get('pose_correction') or {}
    if record.get('version') != VERSION or record.get('status') not in ('accepted', 'rejected'):
        return 'missing'
    try:
        if record.get('inputs') != inputs(cp, view):
            return 'stale'
        camera = camera_from_record(record['camera'])
        original = camera_from_record(view)
        if (camera.width, camera.height, camera.fov_deg) != (original.width, original.height, original.fov_deg):
            return 'stale'
    except (OSError, ValueError, KeyError, TypeError):
        return 'stale'
    return record['status']


def corrected_checkpoint(cp):
    """Project accepted poses into a run-local manifest; never mutate source cameras."""
    manifest = copy.deepcopy(cp.manifest)
    required = {view['id'] for view in cp.repair_views}
    accepted = []
    for original, view in zip(cp.views, manifest['views']):
        if view['id'] not in required:
            continue
        status = correction_status(cp, original)
        if status in ('missing', 'stale'):
            raise ValueError(f"View {int(view['id'])+1}: pose correction is {status}; run Correct generated pose in Step 2")
        if status == 'accepted':
            view['camera'] = copy.deepcopy(view['pose_correction']['camera'])
            accepted.append(view['id'])
    manifest.setdefault('metadata', {})['generated_pose_correction'] = {
        'version': VERSION, 'enabled': True, 'accepted_view_ids': accepted,
        'original_cameras': {v['id']: copy.deepcopy(v['camera']) for v in cp.views}}
    return Checkpoint(cp.root, manifest)


def cached_depth(cp, view):
    """Reuse only checked CUDA gsplat camera-z depth in this source coordinate frame."""
    camera = camera_from_record(view)
    manifests = sorted((cp.root/'artifixer-inputs').glob('*/trajectories/*/trajectory.json'))
    manifests += sorted((cp.root/'trajectories').glob('*/trajectory.json'))
    for path in manifests:
        try:
            data = json.loads(path.read_text())
            recipe = data['recipe']
            if (data.get('depth_convention') != 'expected_camera_z'
                    or data.get('camera_convention') != 'opencv_c2w'
                    or recipe.get('source') != cp.manifest.get('source_fingerprint')
                    or recipe.get('scene_load', {}) != cp.manifest.get('metadata', {}).get('scene_load', {})
                    or recipe.get('opacity_policy') == 'native_3dgut_png8'):
                continue
            saved = next(v for v in recipe['views'] if v['id'] == view['id'])
            other = camera_from_record(saved)
            if ((other.width, other.height) != (camera.width, camera.height)
                    or not np.array_equal(other.c2w, camera.c2w)
                    or not np.array_equal(other.intrinsics, camera.intrinsics)):
                continue
            anchor = next(a for a in data['anchors'] if a['view_id'] == view['id'])
            root = path.parent.parent.parent.resolve()
            arrays = []
            for key in ('depth', 'opacity'):
                relative = anchor[key]
                file = (root/relative).resolve()
                if not file.is_relative_to(root) or digest(file) != data['sha256'][relative]:
                    raise ValueError('Cached geometry checksum mismatch')
                array = np.load(file, allow_pickle=False)
                if array.shape != (camera.height, camera.width):
                    raise ValueError('Cached geometry has the wrong dimensions')
                arrays.append(array)
            return *arrays, {'backend': 'cached_cuda_gsplat', 'manifest': str(path.relative_to(cp.root)),
                             'depth_sha256': data['sha256'][anchor['depth']]}
        except (OSError, ValueError, KeyError, StopIteration, TypeError):
            continue
    return None


class DepthProvider:
    def __init__(self, cp):
        self.cp, self.renderer = cp, None

    def __call__(self, view):
        cached = cached_depth(self.cp, view)
        if cached is not None:
            return cached
        # This is the existing Viser depth-only fallback, not a different RGB
        # renderer. ArtiFixer's subsequent RGB/opacity path is unchanged.
        from ..scene import load_scene
        from ..rendering.cpu_splat_renderer import CpuSplatRenderer
        if self.renderer is None:
            scene = load_scene(self.cp.manifest['scene_path'],
                               **self.cp.manifest.get('metadata', {}).get('scene_load', {}))
            self.renderer = CpuSplatRenderer(scene)
        camera = camera_from_record(view)
        scale = min(1., 416/camera.width)
        small = replace(camera, width=round(camera.width*scale), height=round(camera.height*scale))
        depth = self.renderer.render_depth(small)
        from .pose_alignment import opencv
        depth = opencv().resize(depth, (camera.width, camera.height), interpolation=opencv().INTER_NEAREST)
        return depth, None, {'backend': 'viser_cpu_depth_fallback', 'resolution': [small.width, small.height]}


def correct_poses(checkpoint, *, url=None, should_stop=lambda: False,
                  on_progress=lambda **event: None, session_factory=None, depth_provider=None):
    from .pose_alignment import correspondences, fit_pose, structural_scores, render_acceptance, opencv
    from .viser_capture import CaptureSession
    cp = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint.load(checkpoint)
    if not cp.complete or any(not v.get('repaired_rgb') for v in cp.repair_views):
        raise ValueError('Complete the original-FOV image repairs in Step 3 before pose correction')
    cp.require_viser_images()
    opencv()  # Fail before touching the viewer if the optional dependency is absent.
    validate_source(cp)
    root = cp.root/'pose-corrections'/uuid.uuid4().hex
    root.mkdir(parents=True)
    status_path = cp.root/'pose-corrections'/'status.json'
    def progress(message, **fields):
        atomic_json(status_path, {'message': message, 'updated_at': utc_now(), **fields})
        on_progress(message=message, **fields)
    def check_stop():
        if should_stop():
            raise InterruptedError('Pose correction cancelled')
    # Share the repair lock with the six-view transaction. Manifest/input checks
    # additionally catch crop, capture, or legacy editor changes during fitting.
    with (cp.root/'.shared-image-repair.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Image repair or pose correction is already running for this checkpoint') from None
        before = (cp.root/'checkpoint.json').read_bytes()
        if json.loads(before) != cp.manifest:
            raise ValueError('Checkpoint changed; reload it before pose correction')
        try:
            check_stop()
            progress('Connecting to Viser for pose correction', status='running', completed=0)
            session = (session_factory or CaptureSession)(cp.manifest['scene_path'], url=url,
                up_axis=cp.manifest.get('metadata', {}).get('up_axis', '-y'),
                lod_level=cp.manifest.get('metadata', {}).get('scene_load', {}).get('lod_level', 0),
                should_stop=should_stop)
            provider = depth_provider or DepthProvider(cp)
            records = {}
            for index, view in enumerate(cp.repair_views):
                check_stop()
                progress(f"Aligning view {int(view['id'])+1}/{len(cp.views)} against Viser",
                         status='running', completed=index, total=len(cp.repair_views))
                signature = inputs(cp, view)
                camera = camera_from_record(view)
                with Image.open(target_path(cp, view)) as image:
                    target = np.asarray(ImageOps.fit(image.convert('RGB'), (camera.width, camera.height),
                                                    Image.Resampling.LANCZOS))
                original = session.render(camera)
                depth, opacity, depth_info = provider(view)
                p, q = correspondences(original, target)
                candidate, diagnostics = fit_pose(camera, p, q, depth, opacity)
                folder = root/view['id']
                folder.mkdir()
                Image.fromarray(original).save(folder/'original.png')
                Image.fromarray(target).save(folder/'target.png')
                accepted = False
                if candidate is not None:
                    check_stop()
                    rendered = session.render(candidate)
                    old, new = structural_scores(original, target), structural_scores(rendered, target)
                    accepted = render_acceptance(old, new)
                    diagnostics.update(structure_before=old.tolist(), structure_after=new.tolist(),
                        structure_improvement=float(np.mean(new-old)))
                    diagnostics['reason'] = ('Improved held-out geometry and Viser alignment' if accepted else
                                             'Fresh Viser render did not improve broadly without moving aligned regions')
                    Image.fromarray(rendered).save(folder/'candidate.png')
                record = {'version': VERSION, 'status': 'accepted' if accepted else 'rejected',
                          'created_at': utc_now(), 'inputs': signature, 'depth': depth_info,
                          'camera': camera_to_record(candidate if accepted else camera),
                          'diagnostics': diagnostics, 'artifacts': str(folder.relative_to(cp.root))}
                if candidate is not None:
                    record['candidate_camera'] = camera_to_record(candidate)
                atomic_json(folder/'result.json', record)
                records[view['id']] = record
                progress(f"View {int(view['id'])+1}: {record['status']} · {diagnostics['reason']}",
                         status='running', completed=index+1, total=len(cp.repair_views))
            check_stop()
            if ((cp.root/'checkpoint.json').read_bytes() != before or
                    any(records[v['id']]['inputs'] != inputs(cp, v) for v in cp.repair_views)):
                raise ValueError('Checkpoint images or cameras changed during correction; results were not applied')
            for view in cp.views:
                if view['id'] in records:
                    view['pose_correction'] = records[view['id']]
            cp.save()
            accepted_count = sum(r['status'] == 'accepted' for r in records.values())
            progress(f'{accepted_count}/{len(records)} corrected poses saved; remaining original poses retained',
                     status='complete', completed=len(records), accepted=accepted_count, total=len(records))
            return cp
        except Exception as exc:
            progress(str(exc), status='stopped' if isinstance(exc, InterruptedError) else 'error')
            raise
