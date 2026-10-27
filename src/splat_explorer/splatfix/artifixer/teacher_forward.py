"""Forward, reference-bounded teacher windows over the connected picked path."""


def forward_clips(jobs, maximum, anchors, *, seeding, extensions=(), include_warmup=False):
    """Seed references in place, share context, and export each wanted camera once.

    Consecutive repaired cameras normally bound each clip. Longer gaps use
    overlapping windows without inserting a distant reference or reversing.
    Fixed endpoint extensions are capped at eight, never grown to fill a clip.
    """
    if maximum < 5 or (maximum - 1) % 4:
        raise ValueError('Forward teacher budget must be 1 + 4*n, at least five')
    trusted = set(anchors)
    lookup = {(g['seed_frame_index'], g['interior_frame_index']): g['frame_indices']
              for g in extensions}
    exported = set()
    for segment, references, directory in jobs:
        path = list(segment['indices'])
        if not path or any(b - a != 1 for a, b in zip(path, path[1:])):
            raise ValueError('Forward teacher needs a continuous increasing captured path')
        permitted = {anchors[i] for i in references}
        boundaries = sorted({path[0], path[-1], *trusted.intersection(path)})
        for left, right in zip(boundaries, boundaries[1:]):
            core = list(range(left, right + 1))
            before = lookup.get((left, left + 1), []) if left in permitted else []
            after = lookup.get((right, right - 1), []) if right in permitted else []
            a, b = min(8, len(before) // 4 * 4), min(8, len(after) // 4 * 4)
            if len(core) <= maximum:
                budget = maximum - len(core)
                a, b = max(((x, y) for x in range(0, a + 1, 4) for y in range(0, b + 1, 4)
                            if x + y <= budget), key=lambda pair: (sum(pair), -abs(pair[0] - pair[1]), pair[0]))
            else:
                # Keep the starting/ending reference in each boundary window.
                a, b = min(a, (maximum - 5) // 4 * 4), min(b, (maximum - 5) // 4 * 4)
            before, after = list(reversed(before[:a])), list(after[:b])
            sequence = [*before, *core, *after]
            overlap = min(8, (maximum - 1) // 2 // 4 * 4)
            start = 0
            while True:
                window = sequence[start:start + maximum]
                last = start + maximum >= len(sequence)
                next_start = len(sequence) if last else min(start + maximum - overlap, len(sequence) - maximum)
                outside = set(before + after)
                # Split overlap ownership halfway between windows, retaining
                # targets in both reference-bearing boundary windows.
                owned_end = len(sequence) if last else (start + maximum + next_start) // 2
                owned = [i for i in sequence[start:owned_end] if i not in trusted and
                         (include_warmup or i not in outside) and (directory, i) not in exported]
                if owned:
                    exported.update((directory, i) for i in owned)
                    inline = [i for i in window if i in permitted] if seeding else []
                    yield {'indices': owned, 'ordered_input_indices': window,
                           'input_indices': [i for i in window if i not in outside],
                           'inline_reference_indices': inline,
                           'seed_index': inline[0] if inline else None,
                           'extension_frame_indices': [i for i in window if i in before],
                           'trailing_extension_frame_indices': [i for i in window if i in after],
                           'warmup_frame_indices': [i for i in owned if i in outside],
                           'context_frame_indices': [i for i in window if i not in owned and i not in trusted],
                           'references': list(references), 'output': directory,
                           'leg_endpoints': [left, right], 'layout': 'forward'}
                if last:
                    break
                start = next_start


def validate_forward_path(job, trajectory):
    """Check actual input order and every extension's pose, not camera-ID order."""
    import numpy as np
    left, right = job['leg_endpoints']
    indices = job['ordered_input_indices']
    groups = {(g['seed_frame_index'], g['interior_frame_index']): g['frame_indices']
              for g in trajectory.get('teacher_extensions', [])}
    leading = groups.get((left, left + 1), [])
    trailing = groups.get((right, right - 1), [])
    canonical = [*reversed(leading), *range(left, right + 1), *trailing]
    if not indices or len(indices) != len(set(indices)) or indices[0] not in canonical:
        raise ValueError('Invalid forward teacher input path')
    offset = canonical.index(indices[0])
    if canonical[offset:offset + len(indices)] != indices:
        raise ValueError('Forward teacher window reverses or skips captured cameras')
    trusted = {a['frame_index'] for a in trajectory['anchors']}
    inline = job['inline_reference_indices']
    if any(i not in trusted or i not in indices for i in inline):
        raise ValueError('Forward teacher reference must be at its actual camera')
    if set(job['indices']) & trusted:
        raise ValueError('Forward teacher cannot export repaired references as generated targets')
    matrices = trajectory['transforms']['frames']
    for boundary, neighbor, outside in ((left, left + 1, leading), (right, right - 1, trailing)):
        anchor = np.asarray(matrices[boundary]['transform_matrix'], dtype=float)
        adjacent = np.asarray(matrices[neighbor]['transform_matrix'], dtype=float)
        for step, index in enumerate(outside, 1):
            if index not in indices:
                continue
            expected = anchor.copy()
            expected[:3, 3] += step * (anchor[:3, 3] - adjacent[:3, 3])
            if not np.allclose(matrices[index]['transform_matrix'], expected, atol=1e-6, rtol=1e-5):
                raise ValueError('Forward teacher warm-up does not continue the endpoint tangent')
    poses = np.asarray([matrices[i]['transform_matrix'] for i in indices], dtype=float)
    if not np.isfinite(poses).all():
        raise ValueError('Forward teacher camera path contains non-finite transforms')
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return {'validated': True, 'direction': 'forward', 'input_frame_indices': indices,
            'reference_input_positions': [indices.index(i) for i in inline],
            'reference_input_position': indices.index(inline[0]) if inline else None,
            'leading_extension_count': len(job['extension_frame_indices']),
            'trailing_extension_count': len(job['trailing_extension_frame_indices']),
            'warmup_count': len(job['extension_frame_indices']) + len(job['trailing_extension_frame_indices']),
            'translation_step_min': float(steps.min()) if len(steps) else 0.,
            'translation_step_max': float(steps.max()) if len(steps) else 0.}
