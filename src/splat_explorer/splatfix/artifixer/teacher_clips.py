"""Teacher windows over complete captured legs, independent of causal half splits."""


def filled_leg_clips(jobs, maximum, anchors, *, seeding):
    """Export each target once; use overlapping captured views to fill tail context.

    ``maximum`` retains the serialized 1+4*n input-budget convention. The one
    reserved slot holds a repaired starter, or VAE padding when seeding is off.
    No poses are synthesized, and windows never cross an intervening anchor or
    a gap in the captured target sequence. Reverse endpoint outputs stay separate.
    """
    capacity = maximum - 1
    if capacity < 1:
        raise ValueError('Teacher clip must have room for a target and reference')
    trusted = set(anchors)
    groups, ownership = {}, set()
    for segment, references, directory in jobs:
        indices = (list(segment['indices']) if 'indices' in segment else
                   list(range(segment['start'], segment['start'] + segment['count'])))
        targets = [i for i in indices if i not in trusted]
        if not targets:
            continue
        key = directory, tuple(references)
        group = groups.setdefault(key, {'targets': set(), 'reverse': len(indices) > 1 and indices[0] > indices[-1]})
        for index in targets:
            if (directory, index) in ownership:
                raise ValueError('Teacher target has multiple output owners')
            ownership.add((directory, index))
            group['targets'].add(index)

    for (directory, references), group in groups.items():
        ordered = sorted(group['targets'], reverse=group['reverse'])
        legs = []
        for index in ordered:
            if not legs or abs(index - legs[-1][-1]) != 1:
                legs.append([])
            legs[-1].append(index)
        neighbors = [anchors[i] for i in references]
        for leg in legs:
            for offset in range(0, len(leg), capacity):
                owned = set(leg[offset:offset + capacity])
                # End-align a short tail so the model still sees a full window
                # of distinct captured cameras whenever the leg is long enough.
                start = min(offset, max(0, len(leg) - capacity))
                context = leg[start:start + capacity]
                seed = None
                if seeding:
                    if not neighbors:
                        raise ValueError('Teacher starter requires a trusted reference')
                    # Respect single-reference endpoint policy. Otherwise use
                    # the closest repaired boundary and generate inward from it.
                    seed = min(neighbors, key=lambda a: (min(abs(a - i) for i in owned),
                                                         abs(a - leg[0])))
                    context = sorted(context, reverse=seed > max(context))
                yield {'indices': [i for i in context if i in owned],
                       'input_indices': context,
                       'context_frame_indices': [i for i in context if i not in owned],
                       'references': list(references), 'output': directory, 'seed_index': seed}
