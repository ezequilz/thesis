"""Disjoint inward halves of the existing pairwise camera samples."""
import json
import shutil
from pathlib import Path
from ..checkpoint import atomic_json


def segment_layout(order, n, count, reference_mode, split_mode):
    segments = []
    for pair, (a, b) in enumerate(zip(order[:-1], order[1:])):
        ids = list(range(pair*count, (pair+1)*count))
        halves = [(a, b, ids)]
        if split_mode == 'double-split':
            middle = (count+1)//2
            halves = [(a, b, ids[:middle]), (b, a, ids[middle:][::-1])]
        for source, end, targets in halves:
            index = len(segments)
            refs = [source] if reference_mode == 'single' else list(range(n))
            if split_mode == 'single-split':
                mapping = [i for i in refs if i not in (source,end)]+[source]+[n+i for i in targets]+[end]
            else:
                mapping = [i for i in refs if i != source]+[source]+[n+i for i in targets]
            segment = {'index':index, 'from_reference':source, 'to_reference':end,
                       'pair_index':pair, 'input_subdir':f'segments/{index:03d}',
                       'target_indices':targets, 'frame_order':mapping, 'reference_count':len(refs)}
            if split_mode == 'single-split':
                segment['endpoint_frame_indices'] = [n-2,len(mapping)-1]
            segments.append(segment)
    return segments


def stage_segments(root, frames, segments, n):
    for segment in segments:
        native = root/segment['input_subdir']/'scene'
        (native/'images').mkdir(parents=True)
        selected = [frames[i] for i in segment['frame_order']]
        for frame in selected:
            shutil.copyfile(root/'scene'/frame['file_path'], native/frame['file_path'])
        atomic_json(native/'transforms.json', {'frames':selected})
        atomic_json(native/f"train_test_split_{segment['reference_count']}.json", {
            'train_ids':[i for i,j in enumerate(segment['frame_order']) if j < n],
            'test_ids':[i for i,j in enumerate(segment['frame_order']) if j >= n]})


def validate_split_staging(inputs, manifest):
    n = len(manifest['views'])
    segments = segment_layout(manifest['oracle_path_order'], n, manifest['oracle_target_frames'],
                              manifest['oracle_reference_mode'], manifest['oracle_split_mode'])
    if manifest['seva_task'] != 'img2trajvid' or manifest['seva_segments'] != segments:
        raise ValueError('Invalid inward SEVA segment mapping')
    root = Path(inputs)
    frames = json.loads((root/'scene/transforms.json').read_text())['frames']
    for segment in segments:
        native = root/segment['input_subdir']/'scene'
        mapping = segment['frame_order']
        expected = {'train_ids':[i for i,j in enumerate(mapping) if j < n],
                    'test_ids':[i for i,j in enumerate(mapping) if j >= n]}
        if (json.loads((native/'transforms.json').read_text())['frames'] != [frames[i] for i in mapping]
                or json.loads((native/f"train_test_split_{segment['reference_count']}.json").read_text()) != expected):
            raise ValueError('Inward SEVA cameras disagree with staged trajectory')
