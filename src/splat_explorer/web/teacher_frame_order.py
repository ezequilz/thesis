"""Review teacher outputs in tensor order, never in storage/camera-ID order."""


def frame_order(manifest, output='inference'):
    ordered = {}
    for number, clip in enumerate(manifest.get('clips', []), 1):
        if clip.get('output') != output:
            continue
        owned = set(clip.get('frame_indices', []))
        inputs = clip.get('input_frame_indices', clip.get('frame_indices', []))
        warmup = set(clip.get('warmup_frame_indices', []))
        for position, index in enumerate(inputs, 1):
            if index not in owned or index in ordered:
                continue
            ordered[index] = {'sequence_order': len(ordered), 'clip_number': number,
                              'input_position': position, 'input_count': len(inputs),
                              'warmup': index in warmup,
                              'label': f'Clip {number} · input {position}/{len(inputs)} · '
                                       f'{"warm-up · " if index in warmup else ""}camera {index}'}
    return ordered
