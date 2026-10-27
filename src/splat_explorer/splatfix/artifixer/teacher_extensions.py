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
