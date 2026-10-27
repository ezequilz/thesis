"""Teacher windows over complete captured legs, independent of causal half splits."""


def filled_leg_clips(jobs, maximum, anchors, *, seeding, extensions=(), fill_short=False):
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
    extension_lookup = {(g['seed_frame_index'], g['interior_frame_index']): g['frame_indices'] for g in extensions}
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
            if fill_short:
                if __package__:
                    from .teacher_short_extensions import short_clip
                else:
                    from teacher_short_extensions import short_clip
                clip = short_clip(leg, references, directory, anchors, maximum,
                                  extension_lookup, reverse=group['reverse'])
                if clip is not None:
                    yield clip
                    continue
            # Reserve actual safe context only, not the requested count after
            # a collision. All extension frames remain inside the total budget.
            def extra_for(seed):
                direction = 1 if seed < min(leg) else -1
                safe = extension_lookup.get((seed, seed + direction), [])
                # Put the repaired reference at RGB position 4*n, the end of
                # a temporal VAE group. Its opacity must not max-pool into the
                # first wanted target group's noise initialization.
                return list(reversed(safe[:len(safe) // 4 * 4]))
            reserve = max((len(extra_for(seed)) for seed in neighbors), default=0) if seeding else 0
            owned_capacity = capacity - reserve
            if owned_capacity < 1:
                raise ValueError('Endpoint extension leaves no room for target views')
            for offset in range(0, len(leg), owned_capacity):
                owned = set(leg[offset:offset + owned_capacity])
                # End-align a short tail so the model still sees a full window
                # of distinct captured cameras whenever the leg is long enough.
                seed = None
                extra = []
                if seeding:
                    if not neighbors:
                        raise ValueError('Teacher starter requires a trusted reference')
                    # Respect single-reference endpoint policy. Otherwise use
                    # the closest repaired boundary and generate inward from it.
                    seed = min(neighbors, key=lambda a: (min(abs(a - i) for i in owned),
                                                         abs(a - leg[0])))
                    extra = extra_for(seed)
                window_capacity = capacity - len(extra)
                start = min(offset, max(0, len(leg) - window_capacity))
                context = leg[start:start + window_capacity]
                if seeding:
                    context = sorted(context, reverse=seed > max(context))
                # Extensions are useful only at an adjacent endpoint. Interior
                # windows must not jump back to an endpoint to add warm-up views.
                if extensions and seed is not None and abs(context[0] - seed) != 1:
                    extra = []
                    start = min(offset, max(0, len(leg) - capacity))
                    context = leg[start:start + capacity]
                    context = sorted(context, reverse=seed > max(context))
                    # Keep interior windows continuous too. The permitted
                    # repairs remain cross-attention references, without a
                    # nonadjacent camera being prepended to the video.
                    seed = None
                yield {'indices': [i for i in context if i in owned],
                       'input_indices': context,
                       'extension_frame_indices': extra,
                       'context_frame_indices': [*extra, *[i for i in context if i not in owned]],
                       'references': list(references), 'output': directory, 'seed_index': seed}
