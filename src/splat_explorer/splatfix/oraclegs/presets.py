"""Independent SEVA Basic-demo preset experiments with selectable references."""
import json
import shutil
from pathlib import Path
import numpy as np
from ..checkpoint import atomic_json
from .trajectory import PRESET_SOURCE, preset_poses, preset_assignments


def stage_preset_segments(root, frames, order, targets, presets, reference_mode='all'):
    n = len(frames)-len(targets)
    segments = []
    for j, reference in enumerate(order):
        ids = [t['target_index'] for t in targets if t['segment_index'] == j]
        refs = [reference] if reference_mode == 'single' else [i for i in range(n) if i != reference]+[reference]
        frame_order = refs+[n+i for i in ids]
        relative = f'segments/{j:03d}'
        native = root/relative/'scene'
        (native/'images').mkdir(parents=True)
        selected = [frames[i] for i in frame_order]
        for frame in selected:
            if not frame.get('camera_only', False):
                shutil.copyfile(root/'scene'/frame['file_path'], native/frame['file_path'])
        atomic_json(native/'transforms.json', {'frames':selected})
        atomic_json(native/f'train_test_split_{len(refs)}.json', {'train_ids':list(range(len(refs))), 'test_ids':list(range(len(refs),len(selected)))})
        segments.append({'index':j, 'from_reference':reference, 'to_reference':reference,
                         'input_subdir':relative, 'target_indices':ids, 'frame_order':frame_order,
                         'reference_count':len(refs), 'kind':'per-view-preset', 'preset':presets[j]})
    return segments


def preset_metadata(options, order, presets, seed):
    return {'version':6, 'interpolation':'seva-basic-'+options['oracle_preset'],
            'preset':options['oracle_preset'], 'assignment_seed':seed, 'assigned_presets':presets, 'preset_source':PRESET_SOURCE,
            'preset_scale_scene_units':options['oracle_preset_scale'],
            'preset_parameters':{'orbit_degrees':360, 'focus_distance':10,
                                 'lemniscate_degrees':60, 'spiral_radii':[1,1,.5],
                                 'spiral_rotations':2, 'spiral_zrate':.5, 'spiral_bounds':[1,5]},
            'frame_count_semantics':'per-view-targets-start-included-loop-end-excluded',
            'reference_conditioning':options['oracle_reference_mode']+'-saved-views-per-segment',
            'generation_schedule':'sequential-independent-segments',
            'frames_per_segment':options['oracle_target_frames'], 'closed':True,
            'disconnected':True, 'intrinsics_policy':'preserve-saved-calibration',
            'segments':[{'index':j,'from_reference':i,'to_reference':i,'preset':presets[j]}
                        for j,i in enumerate(order)]}


def validate_preset_staging(inputs, manifest):
    root = Path(inputs)
    n, count = len(manifest['views']), manifest['oracle_target_frames']
    order = manifest['oracle_path_order']
    frames = json.loads((root/'scene/transforms.json').read_text())['frames']
    trajectory = manifest['trajectory']
    expected = len(order)*count
    split = json.loads((root/'scene'/f'train_test_split_{n}.json').read_text())
    if (manifest['seva_task'] != 'img2trajvid' or trajectory.get('version') != 6
            or trajectory.get('target_count') != expected or len(frames) != n+expected
            or len(trajectory['frames']) != expected or len(manifest['seva_segments']) != len(order)
            or split != {'train_ids':list(range(n)), 'test_ids':list(range(n,n+expected))}):
        raise ValueError('Invalid per-view SEVA preset staging')
    assigned = preset_assignments(manifest['oracle_preset'], len(order), trajectory['assignment_seed'])
    if trajectory.get('assigned_presets') != assigned:
        raise ValueError('Invalid balanced SEVA preset assignment')
    axes = np.diag([1.,-1.,-1.,1.])
    for j, reference in enumerate(order):
        segment = manifest['seva_segments'][j]
        ids = list(range(j*count,(j+1)*count))
        refs = [reference] if manifest.get('oracle_reference_mode', 'single') == 'single' else [i for i in range(n) if i != reference]+[reference]
        mapping = refs+[n+i for i in ids]
        if (segment['index'] != j or segment['input_subdir'] != f'segments/{j:03d}'
                or segment['from_reference'] != reference or segment['to_reference'] != reference
                or segment['target_indices'] != ids or segment['frame_order'] != mapping
                or segment.get('reference_count') != len(refs) or segment.get('preset') != assigned[j]):
            raise ValueError('Invalid per-view SEVA reference/target mapping')
        native = root/segment['input_subdir']/'scene'
        if (json.loads((native/'transforms.json').read_text())['frames'] != [frames[i] for i in mapping]
                or json.loads((native/f'train_test_split_{len(refs)}.json').read_text()) !=
                {'train_ids':list(range(len(refs))), 'test_ids':list(range(len(refs),len(refs)+count))}):
            raise ValueError('Per-view SEVA cameras disagree with staged preset')
        poses = preset_poses(np.asarray(frames[reference]['transform_matrix'])@axes,
                             assigned[j], count, manifest['oracle_preset_scale'])
        for local, i in enumerate(ids):
            frame, record = frames[n+i], trajectory['frames'][i]
            if (not np.allclose(frame['transform_matrix'],poses[local]@axes,atol=1e-8)
                    # Older staging interpolated identical intrinsics, causing
                    # roundoff (~6e-14 pixels at cy=288). Still reject real drift.
                    or not np.allclose([frame[k] for k in ('fl_x','fl_y','cx','cy')],
                                       [frames[reference][k] for k in ('fl_x','fl_y','cx','cy')],
                                       rtol=0, atol=1e-8)
                    or record['generation_source_reference_index'] != reference
                    or record['segment_index'] != j):
                raise ValueError('Per-view SEVA preset path or calibration changed')
