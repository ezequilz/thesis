"""WorldExplorer reconstruction supervision in the exported splat's coordinates."""
import json
import errno
import os
import uuid
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..checkpoint import atomic_json
from .bundle import digest_file, load_staged


def scene_prompt(scene):
    """Recover the exact submitted prompt from the archived WorldExplorer request."""
    from ...scene.catalog import openable_scene_path
    path = openable_scene_path(scene).resolve()
    root = next((p for p in path.parents if p.name == 'results' and p.parent.name.startswith('we_')), None)
    if root is None:
        return None
    for request in (root / 'request.json', root.parent / 'request.json'):
        if request.is_file():
            prompt = json.loads(request.read_text()).get('prompt')
            if isinstance(prompt, str) and prompt.strip():
                return prompt
    return None


def checkpoint_prompt(checkpoint):
    prompt = checkpoint.manifest.get('metadata', {}).get('worldexplorer_prompt')
    if not prompt:
        prompt = scene_prompt(checkpoint.manifest['scene_path'])
        if prompt:
            checkpoint.manifest.setdefault('metadata', {})['worldexplorer_prompt'] = prompt
            # Preserve canonical views when called on an endpoint-only run projection.
            saved = checkpoint.root / 'checkpoint.json'
            if saved.is_file():
                manifest = json.loads(saved.read_text())
                manifest.setdefault('metadata', {})['worldexplorer_prompt'] = prompt
                atomic_json(saved, manifest)
            else:
                checkpoint.save()
    return prompt


def source_frames(scene):
    """Return every posed source image; never silently discard missing inputs."""
    from ...scene.catalog import openable_scene_path
    ply = openable_scene_path(scene).resolve()
    root = next((p for p in ply.parents if p.name == 'results' and p.parent.name.startswith('we_')), None)
    if root is None:
        return None
    if ply.relative_to(root).parts[0] != 'nerfstudio_output':
        return None
    transforms = list(root.glob('scenes/*/img2trajvid/transforms.json'))
    if len(transforms) != 1:
        raise ValueError('WorldExplorer source requires exactly one saved camera dataset')
    parser_path = ply.parents[2] / 'dataparser_transforms.json'
    for path in [transforms[0], parser_path]:
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing WorldExplorer camera normalization metadata')
    data = json.loads(transforms[0].read_text())
    parser = json.loads(parser_path.read_text())
    transform = np.asarray(parser['transform'], dtype=float)
    scale = float(parser['scale'])
    if (transform.shape != (3, 4) or not np.isfinite(transform).all() or
            not np.isfinite(scale) or scale <= 0 or
            not np.allclose(transform[:, :3].T @ transform[:, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(transform[:, :3]), 1, atol=1e-4) or
            'applied_transform' in data or 'applied_scale' in data):
        raise ValueError('Unsupported WorldExplorer camera normalization')
    frames = []
    for frame in data['frames']:
        raw = Path(frame['file_path'])
        if raw.is_absolute():
            # Archived paths refer to the GPU run, whose contents live in results/.
            prefix = Path('/workspace/worldexplorer-runs') / root.parent.name
            try:
                path = root / raw.relative_to(prefix)
            except ValueError as exc:
                raise ValueError('Unknown WorldExplorer image path') from exc
        else:
            path = transforms[0].parent / raw
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing WorldExplorer source image: {path}')
        pose = np.asarray(frame['transform_matrix'], dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1]) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
            raise ValueError('Invalid WorldExplorer camera pose')
        calibrated = {k: frame.get(k, data.get(k)) for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')}
        if not np.isfinite(list(calibrated.values())).all() or any(calibrated[k] <= 0 for k in ('w', 'h', 'fl_x', 'fl_y')):
            raise ValueError('Invalid WorldExplorer camera intrinsics')
        with Image.open(path) as image:
            if image.size != (calibrated['w'], calibrated['h']):
                raise ValueError('WorldExplorer image dimensions disagree with camera calibration')
        pose[:3, :] = transform[:, :3] @ pose[:3, :]
        pose[:3, 3] = scale * (pose[:3, 3] + transform[:, 3])
        frames.append({'source_image': str(path.relative_to(root)),
                       'source_trajectory': path.parent.parent.name,
                       'source_role': 'scaffold' if path.parent.name == 'input' and path.stem.isdigit() and int(path.stem) < 8 else 'generated' if path.parent.name == 'samples-rgb' else 'context',
                       'source_frame_index': int(path.stem) if path.stem.isdigit() else None,
                       **calibrated, 'transform_matrix': pose.tolist(), 'file_path': str(path.resolve())})
    if not frames:
        raise ValueError('WorldExplorer source contains no posed images')
    trajectories = {}
    for index, frame in enumerate(frames):
        trajectories.setdefault(frame['source_trajectory'], []).append(index)
    return {'trajectories': [{'index': i, 'name': name, 'frame_indices': indices}
                             for i, (name, indices) in enumerate(trajectories.items())],
            'run_id': root.parent.name, 'scene_path': str(ply), 'camera_convention': 'opengl_c2w',
            'normalization': parser, 'frames': frames}


def link_source_image(source, destination):
    """Share immutable archived RGB bytes; retain a portable regular-file path.

    Consumers must never edit staged source images in place. Renaming/deleting
    either run is safe because hard links retain the bytes independently.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() and source.samefile(destination):
        return
    temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.link')
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        # Cross-device staging cannot share an inode. Copy to a new file so
        # replacing a previously linked destination cannot mutate its source.
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_source(scene, destination):
    data = source_frames(scene)
    if data is None:
        return None
    destination = Path(destination)
    (destination / 'images').mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(data['frames']):
        source = Path(frame['file_path'])
        name = f'images/worldexplorer_{i:06d}{source.suffix.lower()}'
        link_source_image(source, destination / name)
        frame.update(file_path=name, sha256=digest_file(destination / name))
    atomic_json(destination / 'transforms.json', data)
    return str(destination.resolve())


def link_bundle(source, destination):
    """Portable snapshots sharing immutable image AND camera-metadata inodes."""
    source, destination = Path(source), Path(destination)
    data = load_staged(source)
    destination.mkdir(parents=True, exist_ok=True)
    for frame in data['frames']:
        relative = Path(frame['file_path']).relative_to(source.resolve())
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        link_source_image(source / relative, target)
    link_source_image(source / 'transforms.json', destination / 'transforms.json')
    return str(destination.resolve())


def stage_checkpoint_source(checkpoint, destination):
    """Save source trajectories once, reuse them across checkpoints and jobs.

    The checkpoint-local hard links survive archive/cache deletion and remain
    portable to remote workers. Old checkpoints are upgraded lazily.
    """
    import hashlib
    import tempfile
    import fcntl
    root = Path(checkpoint.root)
    local = root / 'worldexplorer-source'
    if not (local / 'transforms.json').is_file():
        scene = checkpoint.manifest['scene_path']
        # Ordinary source scenes do not have a WorldExplorer bundle.
        if not any(p.name == 'results' and p.parent.name.startswith('we_')
                   for p in Path(scene).resolve().parents):
            return stage_source(scene, destination)
        key = hashlib.sha256(str(Path(scene).resolve()).encode()).hexdigest()
        cache_root = root.parent / 'worldexplorer-sources'
        cache_root.mkdir(parents=True, exist_ok=True)
        shared = cache_root / key
        with (cache_root / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (shared / 'transforms.json').is_file():
                temporary = Path(tempfile.mkdtemp(prefix='.' + key, dir=cache_root))
                try:
                    if not stage_source(scene, temporary):
                        return None
                    temporary.rename(shared)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            link_bundle(shared, local)
    # Record only a portable relative reference, never machine-specific cache paths.
    if checkpoint.manifest.get('worldexplorer_source') != 'worldexplorer-source':
        checkpoint.manifest['worldexplorer_source'] = 'worldexplorer-source'
        # A run may hold an endpoint-only interpretation of the checkpoint.
        # Persist only this pointer, never overwrite its saved views/policy.
        saved = json.loads((root / 'checkpoint.json').read_text())
        saved['worldexplorer_source'] = 'worldexplorer-source'
        atomic_json(root / 'checkpoint.json', saved)
    if Path(destination).resolve() == local.resolve():
        return str(local.resolve())
    return link_bundle(local, destination)



def discover(output):
    from ...scene.catalog import SceneSpec
    scenes = []
    for run in sorted((Path(output) / 'worldexplorer').glob('we_*')):
        if run.is_symlink():
            continue
        for ply in sorted((run / 'results').glob('nerfstudio_output/*/*/*/exports/splat/splat.ply')):
            if not ply.resolve().is_relative_to(run.resolve()):
                continue
            if (ply.parents[2] / 'dataparser_transforms.json').is_file() and list((run / 'results').glob('scenes/*/img2trajvid/transforms.json')):
                scenes.append(SceneSpec(f'{run.name}-{ply.parents[2].name}',
                    f'WorldExplorer · {run.name} · {ply.parents[2].name}', ply, up_axis='+z'))
    return scenes
