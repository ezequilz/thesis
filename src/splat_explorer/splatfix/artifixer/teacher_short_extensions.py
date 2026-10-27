"""Bounded straight extensions at both ends of short captured teacher legs."""
import copy


def extension_counts(budget, leading, trailing):
    """Fill available slots, keeping the leading reference at VAE position 4*n."""
    choices = [(a, min(trailing, budget - a))
               for a in range(0, min(leading, budget) + 1, 4)]
    return max(choices, key=lambda pair: (sum(pair), -abs(pair[0] - pair[1]), pair[0]))


def extend_short_orbit(orbit, anchor_indices, maximum, collision, *, should_stop=lambda: False):
    import numpy as np
    result = copy.deepcopy(orbit)
    poses = np.asarray(orbit['poses'], dtype=float)
    groups = []
    for left, right in zip(sorted(anchor_indices), sorted(anchor_indices)[1:]):
        budget = maximum - (right - left + 1)  # Both real endpoint cameras count.
        if budget <= 0 or right - left < 2:
            continue
        sides = []
        for seed, neighbor in ((left, left + 1), (right, right - 1)):
            group = {'seed_frame_index': seed, 'interior_frame_index': neighbor,
                     'requested_frames': budget, 'poses': [], 'collision_step': None,
                     'leg_endpoints': [left, right]}
            delta = poses[seed, :3, 3] - poses[neighbor, :3, 3]
            if np.linalg.norm(delta) <= 1e-12:
                group['reason'] = 'stationary endpoint; no translation direction'
            else:
                previous = poses[seed, :3, 3]
                for step in range(1, budget + 1):
                    if should_stop():
                        raise InterruptedError('Short-clip extension stopped')
                    pose = poses[seed].copy()
                    pose[:3, 3] += step * delta
                    if collision.intersects(previous, pose[:3, 3]):
                        group['collision_step'] = step
                        break
                    group['poses'].append(pose.tolist())
                    previous = pose[:3, 3]
            sides.append(group)
        a, b = [len(side['poses']) for side in sides]
        forward = extension_counts(budget, a, b)
        backward = extension_counts(budget, b, a)
        # Render only cameras potentially used in forward or reverse inference.
        for side, used in zip(sides, (max(forward[0], backward[1]), max(forward[1], backward[0]))):
            side['safe_frames_checked'] = len(side['poses'])
            side['poses'] = side['poses'][:used]
            groups.append(side)
    result['teacher_extensions'] = {
        'version': 2, 'mode': 'fill-short-clips', 'maximum_frames': maximum,
        'policy': 'both leg endpoints; straight translation; fixed rotation; stop each side at first continuous Gaussian collision',
        'groups': groups}
    return result


def short_clip(leg, references, directory, anchors, maximum, lookup, *, reverse=False):
    """A full short leg, both permitted references, then safe outside context."""
    direction = -1 if reverse else 1
    permitted = {anchors[i] for i in references}
    start = leg[0] - direction
    end = leg[-1] + direction
    start = start if start in permitted else None
    end = end if end in permitted else None
    core_count = len(leg) + int(start is not None) + int(end is not None)
    if core_count > maximum:
        return None
    before = lookup.get((start, leg[0]), [])
    after = lookup.get((end, leg[-1]), [])
    a, b = extension_counts(maximum - core_count, len(before), len(after))
    before, after = list(reversed(before[:a])), list(after[:b])
    return {'indices': list(leg), 'input_indices': list(leg), 'references': list(references),
            'output': directory, 'seed_index': start, 'end_reference_index': end,
            'extension_frame_indices': before, 'trailing_extension_frame_indices': after,
            'context_frame_indices': [*before, *after], 'fill_short_clip': True}
