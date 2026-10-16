"""Ordered pair interpolation with off-path RGBs retained only as context.

The first pass remains native joint SEVA sampling. For the dense pass, only
saved endpoints and generated on-path anchors may bound target intervals.
"""
import functools
import inspect
import json
from pathlib import Path

import numpy as np


def ordered_chunks(T, input_ords, test_ords, gt_input_inds, endpoints):
    orders = np.asarray(input_ords)
    targets = np.asarray(test_ords)
    start, end = endpoints
    gt = list(gt_input_inds)
    if (not start < end or len(set(gt)) != len(gt) or
            any(i < 0 or i >= len(orders) for i in gt) or
            len(set(orders.tolist())) != len(orders) or
            not np.isfinite(orders).all() or not np.isfinite(targets).all()):
        raise ValueError('Invalid ordered SEVA pair inputs')
    endpoint_ids = [i for i in gt if orders[i] in (start,end)]
    if len(endpoint_ids) != 2 or any(not start < t < end for t in targets):
        raise ValueError('Ordered SEVA pair must bound every target with saved endpoints')
    on_path = sorted([i for i in range(len(orders)) if i not in gt or i in endpoint_ids],
                     key=lambda i: orders[i])
    if orders[on_path[0]] != start or orders[on_path[-1]] != end:
        raise ValueError('Generated SEVA anchors escaped the ordered pair')
    path_orders = orders[on_path]
    chunks, records = [], []
    selected, tests, bounds = list(gt), [], []

    def flush():
        if not tests:
            return
        chunk = [f'!{i:03d}' for i in selected]+[f'>{i:03d}' for i in tests]
        chunks.append(chunk+['NULL']*(T-len(chunk)))
        records.append({'reference_input_indices':gt, 'conditioning_input_indices':list(selected),
                        'target_indices':list(tests), 'bounds':list(bounds)})

    for j, target in enumerate(targets):
        right = min(int(np.searchsorted(path_orders,target,side='right')),len(on_path)-1)
        left = right-1
        pair = [on_path[left],on_path[right]]
        additions = [i for i in pair if i not in selected]
        if len(selected)+len(additions)+len(tests)+1 > T:
            flush()
            selected, tests, bounds = list(gt), [], []
            additions = [i for i in pair if i not in selected]
        if len(selected)+len(additions)+1 > T:
            raise ValueError('SEVA context cannot fit references, bounding anchors and a target')
        selected.extend(additions)
        tests.append(j)
        bounds.append({'target_order':float(target), 'left_order':float(orders[pair[0]]),
                       'right_order':float(orders[pair[1]])})
    flush()
    input_inds, input_sels, test_inds, test_sels = [], [], [], []
    for chunk in chunks:
        input_inds.append([int(c[1:]) for c in chunk if c.startswith('!')])
        input_sels.append([i for i,c in enumerate(chunk) if c.startswith('!')])
        test_inds.append([int(c[1:]) for c in chunk if c.startswith('>')])
        test_sels.append([i for i,c in enumerate(chunk) if c.startswith('>')])
    return (chunks,input_inds,input_sels,test_inds,test_sels), records


def install_ordered_pair(evaluation, endpoints, report_path, *, demo, data_path):
    frames = json.loads((Path(data_path)/'scene/transforms.json').read_text())['frames']
    start,end = endpoints
    sources = [int(Path(frames[i]['file_path']).stem.removeprefix('ref_')) for i in endpoints]
    axes = np.diag([1.,-1.,-1.,1.])
    target_poses = np.asarray([frames[i]['transform_matrix'] for i in range(start+1,end)]) @ axes
    target_sources = [sources[0 if (i-start)/(end-start) <= .5 else 1] for i in range(start+1,end)]
    report = {'version':1, 'task':'img2vid', 'endpoint_frame_indices':endpoints,
              'endpoint_reference_indices':sources,
              'source_policy':'starting-endpoint-through-midpoint; ending-endpoint-after-midpoint',
              'source_role':'Explicit assignment; all references jointly condition RGB generation. Reprojection uses assigned source when enabled.',
              'policy':'all-references-as-context; only-endpoints-and-generated-anchors-bound-targets',
              'targets':[{'frame_index':i, 'source_reference_index':source}
                         for i,source in zip(range(start+1,end),target_sources)]}
    path = Path(report_path)
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2))
        temporary.replace(path)

    original_parse = demo.parse_task
    @functools.wraps(original_parse)
    def parse(*args, **kwargs):
        parsed = list(original_parse(*args, **kwargs))
        # Tiny diagnostic paths can have fewer targets than SEVA's inferred
        # anchor count. Keep each real camera once instead of duplicate priors.
        keep = list(dict.fromkeys(parsed[4]))
        selections = [parsed[4].index(i) for i in keep]
        parsed[4], parsed[7], parsed[8] = keep, parsed[7][selections], parsed[8][selections]
        report['anchors'] = [{'anchor_index':j, 'frame_index':int(i),
                              'source_reference_index':target_sources[int(i)-start-1]}
                             for j,i in enumerate(parsed[4])]
        save()
        return tuple(parsed)
    demo.parse_task = parse

    original_value = evaluation.get_value_dict
    value_signature = inspect.signature(original_value)
    @functools.wraps(original_value)
    def value(*args, **kwargs):
        bound = value_signature.bind(*args, **kwargs)
        poses = bound.arguments['curr_c2ws'].detach().cpu().numpy().copy()
        conditioned = set(bound.arguments['curr_input_frame_indices'])
        assignment = []
        for slot,pose in enumerate(poses):
            if slot in conditioned:
                assignment.append(None)
                continue
            matches = np.flatnonzero(np.all(np.isclose(target_poses[:,:3],pose[:3],atol=2e-5,rtol=0),axis=(1,2)))
            candidates = {target_sources[i] for i in matches}
            if len(candidates) != 1:
                raise ValueError('Cannot uniquely assign ordered SEVA source reference to target camera')
            assignment.append(candidates.pop())
        result = original_value(*args, **kwargs)
        result['_ordered_pair_sources'] = assignment
        return result
    evaluation.get_value_dict = value

    original = evaluation.chunk_input_and_test
    signature = inspect.signature(original)

    @functools.wraps(original)
    def chunk(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if not values['chunk_strategy'].startswith('interp'):
            return original(*args, **kwargs)
        if values['task'] != 'img2vid':
            raise ValueError('Ordered pair conditioning requires img2vid')
        result, records = ordered_chunks(values['T'], values['input_ords'], values['test_ords'],
                                         values['gt_input_inds'], endpoints)
        report['chunks'] = records
        save()
        return result

    evaluation.chunk_input_and_test = chunk

    original_save = evaluation.save_output
    @functools.wraps(original_save)
    def save_output(samples, save_path, video_save_fps=2):
        original_save(samples, save_path, video_save_fps)
        # Preserve every PNG/export index for calibrated merging, but don't
        # prepend off-path context photographs to the trajectory video.
        if Path(save_path).name == 'scene' and 'samples-rgb/image' in samples:
            rgb = samples['samples-rgb/image']
            if len(rgb) != len(frames):
                raise ValueError('Ordered pair output lost reference frames')
            original_save({'samples-rgb/video':rgb[start:end+1]}, save_path, video_save_fps)
    evaluation.save_output = save_output
