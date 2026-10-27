"""Collision-truncated translation extensions, stored separately from wanted poses."""
import copy


def extend_orbit(orbit, anchor_indices, count, collision, *, should_stop=lambda: False):
    import numpy as np
    result = copy.deepcopy(orbit)
    poses = np.asarray(orbit['poses'], dtype=np.float64)
    groups = []
    for seed in anchor_indices:
        for neighbor in (seed - 1, seed + 1):
            if not 0 <= neighbor < len(poses):
                continue
            delta = poses[seed, :3, 3] - poses[neighbor, :3, 3]
            group = {'seed_frame_index': seed, 'interior_frame_index': neighbor,
                     'requested_frames': count, 'poses': [], 'collision_step': None}
            groups.append(group)
            if np.linalg.norm(delta) <= 1e-12:
                group['reason'] = 'stationary endpoint; no translation direction'
                continue
            previous = poses[seed, :3, 3]
            for step in range(1, count + 1):
                if should_stop():
                    raise InterruptedError('Endpoint extension stopped')
                pose = poses[seed].copy()
                pose[:3, 3] += step * delta
                if collision.intersects(previous, pose[:3, 3]):
                    group['collision_step'] = step
                    break  # Never resume beyond an obstacle.
                group['poses'].append(pose.tolist())
                previous = pose[:3, 3]
    result['teacher_extensions'] = {'version': 1, 'requested_frames': count,
        'policy': 'endpoint translation tangent; fixed orientation; continuous 3-sigma Gaussian clearance',
        'groups': groups}
    return result


def render_poses(orbit):
    return [*orbit['poses'], *[pose for group in orbit.get('teacher_extensions', {}).get('groups', [])
                              for pose in group['poses']]]


def extension_manifest(orbit):
    index = len(orbit['poses'])
    groups = []
    for group in orbit.get('teacher_extensions', {}).get('groups', []):
        count = len(group['poses'])
        groups.append({**{k: v for k, v in group.items() if k != 'poses'},
                       'frame_indices': list(range(index, index + count))})
        index += count
    return groups


def validate_clip_path(job, trajectory):
    """Validate the actual tensor path, including the extension/reference join.

    Camera IDs are storage keys, not temporal positions. Reject a reordered or
    opposite-side extension before loading the expensive model.
    """
    import numpy as np
    seed = job['seed_index']
    extra = job.get('extension_frame_indices', [])
    context = job['input_indices']
    indices = [*extra, *([seed] if seed is not None else []), *context]
    if len(indices) != len(set(indices)):
        raise ValueError('Teacher clip repeats a camera')
    if len(context) > 1 and not (all(b - a == 1 for a, b in zip(context, context[1:])) or
                                  all(b - a == -1 for a, b in zip(context, context[1:]))):
        raise ValueError('Teacher target cameras are not a continuous ordered path')
    if seed is not None and (not context or abs(context[0] - seed) != 1):
        raise ValueError('Teacher reference is not adjacent to its target window')
    matrices = trajectory['transforms']['frames']
    if extra:
        group = next((g for g in trajectory.get('teacher_extensions', [])
                      if g['seed_frame_index'] == seed and g['interior_frame_index'] == context[0]), None)
        if group is None or extra != list(reversed(group['frame_indices'][:len(extra)])):
            raise ValueError('Teacher warm-up must approach its reference from the opposite side')
        anchor = np.asarray(matrices[seed]['transform_matrix'], dtype=float)
        neighbor = np.asarray(matrices[context[0]]['transform_matrix'], dtype=float)
        delta = anchor[:3, 3] - neighbor[:3, 3]
        for step, index in enumerate(reversed(extra), 1):
            expected = anchor.copy()
            expected[:3, 3] += step * delta
            if not np.allclose(matrices[index]['transform_matrix'], expected, atol=1e-6, rtol=1e-5):
                raise ValueError('Teacher warm-up camera does not continue the endpoint tangent')
    poses = np.asarray([matrices[i]['transform_matrix'] for i in indices], dtype=float)
    if not np.isfinite(poses).all():
        raise ValueError('Teacher camera path contains non-finite transforms')
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return {'validated': True, 'input_frame_indices': indices,
            'reference_input_position': len(extra) if seed is not None else None,
            'warmup_count': len(extra), 'translation_step_min': float(steps.min()) if len(steps) else 0.,
            'translation_step_max': float(steps.max()) if len(steps) else 0.}


def prepare_extensions(orbit, checkpoint, runtime, *, should_stop=lambda: False):
    count = runtime.get('teacher_endpoint_extension', 0)
    if not count:
        return orbit
    import numpy as np
    from ...scene import load_scene
    from ...agent.orbit_collision import OrbitCollision
    from ..checkpoint import camera_from_record
    from .author_trajectory import picked_reference_indices
    poses = np.asarray(orbit['poses'])
    references = [camera_from_record(v).c2w for v in checkpoint.views]
    if checkpoint.endpoint_only:
        indices = [0, len(poses) - 1]
    elif orbit['provenance'].get('trajectory_order') == 'picked':
        indices = picked_reference_indices(poses, references, orbit['provenance'])
    else:
        indices = [next(i for i, pose in enumerate(poses) if np.array_equal(pose, ref)) for ref in references]
    scene = load_scene(runtime.get('scene_path') or checkpoint.manifest['scene_path'],
                       **checkpoint.manifest.get('metadata', {}).get('scene_load', {}))
    collision = OrbitCollision(scene)
    return extend_orbit(orbit, indices, count, collision, should_stop=should_stop)
