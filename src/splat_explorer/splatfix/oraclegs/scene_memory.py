"""Scene-coordinate memory for sparse SEVA conditioning, never reconstruction GT."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..checkpoint import atomic_json
from ..worldexplorer.bundle import load_staged
from .inputs import fit_reference


def rotation_distances(sources, targets):
    """WorldExplorer/SEVA SO(3) geodesic angle, in degrees."""
    # SEVA exports 3x4 poses; the WorldExplorer scaffold uses homogeneous
    # 4x4 poses. Extract their shared rotation block before stacking the pool.
    source_rotations = np.stack([np.asarray(pose)[:3, :3] for pose in sources])
    target_rotations = np.stack([np.asarray(pose)[:3, :3] for pose in targets])
    relative = source_rotations[:, None] @ target_rotations[None].transpose(0, 1, 3, 2)
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=2, axis2=3)-1)/2, -1, 1)))


def select_context(candidates, targets, direction, count=5):
    # Generalize upstream's three excluded opposite scaffold sectors to arbitrary
    # saved cameras: exclude trajectory origins more than 90 degrees away.
    eligible = [f for f in candidates if np.dot(np.asarray(f['trajectory_pose'])[:3, 2],
                                                np.asarray(direction)[:3, 2]) >= -1e-6]
    if len(eligible) < count:
        raise ValueError('Scene memory needs five eligible prior views; select Repaired images only as fallback')
    distances = rotation_distances([f['transform_matrix'] for f in eligible], targets)
    nearest = np.argsort(distances, axis=0, kind='stable')[:count].T
    # Upstream votes over nearest rotations for the pending target cameras.
    indices = [i for i, _ in Counter(nearest.flatten().tolist()).most_common(count)]
    return [eligible[i] for i in indices]


def world_memory(root, manifest):
    if not manifest.get('worldexplorer_source'):
        raise ValueError('Scene memory requires a posed WorldExplorer scaffold; select Repaired images only as fallback')
    data = load_staged(Path(root)/manifest['worldexplorer_source'])
    scaffold, generated = {}, []
    for frame in data['frames']:
        if frame.get('source_role') == 'scaffold':
            scaffold.setdefault(frame['source_frame_index'], frame)
        elif frame.get('source_role') == 'generated' and (frame.get('source_frame_index') or 0) >= 2:
            generated.append(frame)
    if set(scaffold) != set(range(8)):
        raise ValueError('Scene memory requires all eight posed scaffold images; restage source or select Repaired images only')
    fixed = [scaffold[i] for i in range(8)]
    candidates = []
    for frame in generated:
        prefix = frame.get('source_trajectory', '').split('_')[0]
        if not prefix.isdigit() or int(prefix) not in scaffold:
            raise ValueError('Cannot identify WorldExplorer trajectory direction')
        candidates.append({**frame, 'memory_source':'worldexplorer',
                           'trajectory_pose':scaffold[int(prefix)]['transform_matrix']})
    return fixed, candidates


def validate_memory_source(root, manifest):
    if len(manifest['views']) != 6:
        raise ValueError('Scene memory requires six saved references (8 + 5 + 6 = 19); select Repaired images only for other counts')
    fixed, candidates = world_memory(root, manifest)
    frames = json.loads((Path(root)/'scene/transforms.json').read_text())['frames']
    for segment in manifest['seva_segments'] or [None]:
        ids = [6+i for i in segment['target_indices']] if segment else list(range(6, len(frames)))
        direction = frames[segment['from_reference'] if segment else 0]['transform_matrix']
        select_context(candidates, [frames[i]['transform_matrix'] for i in ids[:8]], direction)
    return fixed, candidates


def prepare_memory(inputs, output, manifest, segment, destination):
    inputs, output, destination = Path(inputs), Path(output), Path(destination)
    fixed, candidates = world_memory(inputs, manifest)
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames']
    n = len(manifest['views'])
    for previous in manifest['seva_segments']:
        if segment is None or previous['index'] >= segment['index']:
            break
        folder = output/'seva-segments'/f"{previous['index']:03d}"/'seva'/manifest['seva_task']/'scene'
        exported = json.loads((folder/'transforms.json').read_text())['frames']
        from .adapter import verify_exported_cameras
        verify_exported_cameras([frames[i] for i in previous['frame_order']], exported, manifest)
        for target in previous['target_indices'][2:]:
            frame = exported[previous['frame_order'].index(n+target)]
            image = (folder/frame['file_path']).resolve()
            if not image.is_relative_to(folder.resolve()) or not image.is_file():
                raise ValueError('Invalid completed SEVA memory image')
            # Strip the diffusion stride padding before fitting to the next run.
            candidates.append({**frame, 'file_path':str(image), 'memory_source':'seva',
                'source_trajectory':str(previous['index']), 'source_frame_index':target,
                'trajectory_pose':frames[previous['from_reference']]['transform_matrix'],
                'unpadded_size':manifest['resolution']})
    ids = [n+i for i in segment['target_indices']] if segment else list(range(n, len(frames)))
    direction = frames[segment['from_reference'] if segment else 0]['transform_matrix']
    selected = select_context(candidates, [frames[i]['transform_matrix'] for i in ids], direction)
    records = []
    width, height = manifest['resolution']
    folder = destination/'scene-memory'
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(fixed+selected):
        k = np.array([[frame['fl_x'], 0, frame['cx']], [0, frame['fl_y'], frame['cy']], [0, 0, 1.]])
        with Image.open(frame['file_path']) as image:
            image = image.convert('RGB')
            if image.size != (frame['w'], frame['h']):
                raise ValueError('Memory image dimensions disagree with calibration')
            if 'unpadded_size' in frame:
                w, h = frame['unpadded_size']
                top = (image.height-h)//2
                image = image.crop((0, top, w, top+h))
                k[1, 2] -= top
            camera = SimpleNamespace(intrinsics=k, width=image.width, height=image.height)
            fitted, calibrated, _ = fit_reference(image, camera, (width, height), manifest['seva_fit'])
            path = folder/f'{i:03d}.png'
            fitted.save(path)
        records.append({'file_path':str(path.resolve()), 'transform_matrix':frame['transform_matrix'],
                        'K':calibrated.tolist(), 'role':'scaffold' if i < 8 else 'memory',
                        'memory_source':frame.get('memory_source', 'worldexplorer'),
                        'source_image':frame['file_path'], 'source_trajectory':frame.get('source_trajectory'),
                        'source_frame_index':frame.get('source_frame_index')})
    request = {'frames':records, 'resolution':[width, height], 'report_path':str(folder/'conditioning.json'),
               'reference_count':19, 'saved_reference_mode':manifest['mode'],
               'selection':'SO(3) nearest-rotation votes; exclude opposite trajectory origins; omit first two generated frames'}
    atomic_json(folder/'selection.json', request)
    return request
