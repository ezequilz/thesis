"""Combine independently generated pairs without changing camera or GT identity."""
import json
import shutil
from pathlib import Path
from PIL import Image
from .adapter import verify_exported_cameras
from ..checkpoint import atomic_json


def merge_segments(inputs, output, manifest, *, should_stop=lambda:False):
    inputs, output = Path(inputs), Path(output)
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames']
    n = len(manifest['views'])
    destination = output/'seva'/manifest['seva_task']/'scene'
    destination.mkdir(parents=True, exist_ok=False)
    merged = [None]*len(frames)
    provenance = []
    for segment in manifest['seva_segments']:
        if should_stop():
            raise InterruptedError('Cancelled while collecting SEVA sub-trajectories')
        index = segment['index']
        source = output/'seva-segments'/f'{index:03d}'/'seva'/manifest['seva_task']/'scene'
        exported = json.loads((source/'transforms.json').read_text())['frames']
        canonical = segment['frame_order']
        verify_exported_cameras([frames[i] for i in canonical], exported, manifest)
        for local, global_index in enumerate(canonical):
            if should_stop():
                raise InterruptedError('Cancelled while collecting SEVA sub-trajectories')
            raw = Path(exported[local]['file_path'])
            image = (source/raw).resolve()
            if raw.is_absolute() or not image.is_relative_to(source.resolve()):
                raise ValueError('SEVA segment image escapes its output directory')
            with Image.open(image) as im:
                if list(im.size) != manifest['seva_internal_resolution']:
                    raise ValueError('SEVA segment image dimensions disagree with camera calibration')
            if global_index < n and merged[global_index] is not None:
                continue
            if merged[global_index] is not None:
                raise ValueError('SEVA segment target appears more than once')
            relative = (f'input/{global_index:06d}.png' if global_index < n else
                        f'samples-rgb/{global_index-n:06d}.png')
            path = destination/relative
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(image, path)
            merged[global_index] = {**exported[local], 'file_path':relative}
            if global_index >= n:
                provenance.append({'target_index':global_index-n, 'segment_index':index,
                                   'segment_target_index':segment['target_indices'].index(global_index-n), 'source':str(image.relative_to(output.resolve()))})
    # Explicit view subsets still retain every saved GT image for reconstruction.
    if manifest['oracle_adapter'] in ('per_view_preset', 'trajectory'):
        width, height = manifest['resolution']
        internal = tuple(manifest['seva_internal_resolution'])
        top = (internal[1]-height)//2
        for i, frame in enumerate(frames[:n]):
            if merged[i] is not None:
                continue
            relative = f'input/{i:06d}.png'
            (destination/'input').mkdir(exist_ok=True)
            with Image.open(inputs/'scene'/frame['file_path']) as image:
                padded = Image.new('RGB',internal)
                padded.paste(image,(0,top))
                padded.save(destination/relative)
            merged[i] = {**frame, 'file_path':relative, 'w':internal[0],
                         'h':internal[1], 'cy':frame['cy']+top}
    if any(frame is None for frame in merged):
        raise ValueError('SEVA segments did not generate every target camera')
    atomic_json(destination/'transforms.json', {'frames':merged})
    atomic_json(output/'seva-segments.json', {'completed_segments':len(manifest['seva_segments']),
        'target_count':len(provenance), 'frames':provenance})
