"""Portable, validated source trajectory plans from the official GPU preparation."""
import hashlib
import json
import math

from ..checkpoint import camera_to_record, camera_from_record
from .pipeline_profile import geometry_backend
from .staging_cache import digest


def identity(checkpoint, runtime):
    from .repair import UPSTREAM_REVISION
    from .official_worker import SCALE_SAMPLING
    from .author_trajectory import validate_trajectory_order
    metadata = checkpoint.manifest.get('metadata', {})
    split = 'no-split' if checkpoint.endpoint_only else ('single-split' if runtime.get('handoff_generation')
                                                      else runtime.get('split_mode', 'double-split'))
    return {**({'teacher_endpoint_extension': runtime['teacher_endpoint_extension'], 'teacher_extension_version': 1}
               if runtime.get('teacher_endpoint_extension') else {}),
            'version': 2, 'upstream_revision': UPSTREAM_REVISION,
            'trajectory_order': ('picked' if checkpoint.endpoint_only else
                                 validate_trajectory_order(runtime.get('trajectory_order', 'picked'))),
            'trajectory_version': 7, 'metric_sampling': SCALE_SAMPLING,
            'source': checkpoint.manifest.get('source_fingerprint', checkpoint.manifest['scene_path']),
            'scene_load': metadata.get('scene_load', {}), 'up_axis': metadata.get('up_axis', '-y'),
            'geometry_backend': geometry_backend(runtime), 'split_mode': split,
            'repair_policy': 'endpoints' if checkpoint.endpoint_only else 'all',
            'resolution_profile': runtime['resolution_profile'],
            'moge_model_path': runtime.get('moge_model_path'),
            'anchors': [{'id': view['id'], 'camera': camera_to_record(camera_from_record(view)),
                         'rgb_sha256': digest(checkpoint.image_path(view))} for view in checkpoint.views]}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate(plan, checkpoint, runtime):
    import numpy as np
    if plan.get('identity') != identity(checkpoint, runtime):
        raise ValueError('Source trajectory plan does not match the source, cameras or geometry settings')
    if plan.get('sha256') != signature({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('Source trajectory plan checksum mismatch')
    if isinstance(plan['camera_scale'], bool) or not math.isfinite(plan['camera_scale']) or plan['camera_scale'] <= 0:
        raise ValueError('Invalid camera scale in source trajectory plan')
    poses = np.asarray(plan['orbit']['poses'], dtype=np.float64)
    extension = plan['orbit'].get('teacher_extensions', {})
    if extension.get('requested_frames', 0) != runtime.get('teacher_endpoint_extension', 0):
        raise ValueError('Endpoint extension plan does not match the requested setting')
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Invalid source trajectory poses')
    for anchor in plan['identity']['anchors']:
        pose = camera_from_record(anchor).c2w
        if not any(np.array_equal(pose, other) for other in poses):
            raise ValueError('Source trajectory plan omitted an anchor')
    if not checkpoint.endpoint_only and plan['identity']['trajectory_order'] == 'picked':
        from .author_trajectory import picked_reference_indices
        picked_reference_indices(poses, [camera_from_record(a).c2w for a in plan['identity']['anchors']],
                                 plan['orbit']['provenance'])
    return plan


def build(checkpoint, runtime, orbit, request):
    plan = {'identity': identity(checkpoint, runtime), 'orbit': orbit,
            'camera_scale': request['camera_scale'],
            'camera_scale_provenance': request['camera_scale_provenance'],
            'scale_estimate': request.get('scale_estimate')}
    plan['sha256'] = signature(plan)
    return validate(plan, checkpoint, runtime)
