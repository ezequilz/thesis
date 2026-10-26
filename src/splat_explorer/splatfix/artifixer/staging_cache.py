"""Checkpoint-owned source captures and geometry, independent of edited targets."""
import errno
import hashlib
import json
import shutil
import tempfile

from PIL import Image
from pathlib import Path

from ..checkpoint import atomic_json, camera_to_record


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def compatible_trajectory(cache, recipe):
    """Find an existing orbit despite insignificant interpolation roundoff.

    The exact signature remains the fast path. Rebuilding the same spline can
    differ at double-precision rounding, including its derived diagnostics.
    Everything except those diagnostics and the pose roundoff must still match,
    including source, calibration, renderer policy and inference segmentation.
    The caller must verify the returned trajectory's artifact checksums.
    """
    import numpy as np

    orbit = recipe.get('orbit')
    if not isinstance(orbit, dict):
        return None

    def identity(value):
        result = {key: item for key, item in value.items() if key != 'orbit'}
        saved_orbit = value['orbit']
        result['orbit'] = {key: item for key, item in saved_orbit.items()
                           if key not in ('poses', 'provenance')}
        result['orbit']['provenance'] = {
            key: item for key, item in saved_orbit.get('provenance', {}).items()
            if key != 'motion_diagnostics'}
        return result

    expected = identity(recipe)
    poses = np.asarray(orbit['poses'], dtype=np.float64)
    for path in sorted(Path(cache).glob('*/trajectory.json')):
        saved = json.loads(path.read_text()).get('recipe', {})
        if not isinstance(saved.get('orbit'), dict) or identity(saved) != expected:
            continue
        previous = np.asarray(saved['orbit']['poses'], dtype=np.float64)
        # No relative tolerance: a large coordinate must not widen the match.
        if (previous.shape == poses.shape and np.isfinite(poses).all()
                and np.isfinite(previous).all()
                and np.allclose(previous, poses, rtol=0, atol=1e-12)):
            return path
    return None


def capture_inputs(checkpoint, cameras, destinations, capture, *, profile, renderer='viser'):
    if len(cameras) != len(destinations) or not cameras:
        raise ValueError('Capture cameras and destinations must match and be nonempty')
    metadata = checkpoint.manifest.get('metadata', {})
    recipe = {'version': 1, 'profile': profile, 'renderer': renderer,
              'source': checkpoint.manifest.get('source_fingerprint', checkpoint.manifest['scene_path']),
              'cameras': [camera_to_record(camera) for camera in cameras],
              'up_axis': metadata.get('up_axis', '-y'),
              'scene_load': metadata.get('scene_load', {})}
    if renderer == '3dgut':
        from .repair import UPSTREAM_REVISION
        recipe['upstream_revision'] = UPSTREAM_REVISION
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    root = checkpoint.root / 'artifixer-inputs' / signature
    marker = root / 'captures.json'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved['recipe'] != recipe or len(saved['sha256']) != len(destinations):
            raise ValueError('Invalid saved ArtiFixer capture recipe')
        for index, (destination, expected) in enumerate(zip(destinations, saved['sha256'])):
            source = root / f'{index:05d}.png'
            if digest(source) != expected:
                raise ValueError('Saved ArtiFixer capture checksum mismatch')
            shutil.copy2(source, destination)
    else:
        capture()
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + signature, dir=root.parent))
        try:
            hashes = []
            for index, (source, camera) in enumerate(zip(destinations, cameras)):
                with Image.open(source) as image:
                    if image.size != (camera.width, camera.height):
                        raise ValueError('Invalid ArtiFixer capture dimensions')
                target = temporary / f'{index:05d}.png'
                shutil.copy2(source, target)
                hashes.append(digest(target))
            atomic_json(temporary / 'captures.json', {'recipe': recipe, 'sha256': hashes})
            try:
                temporary.rename(root)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                if not marker.is_file():
                    raise ValueError('Incomplete saved ArtiFixer capture cache') from exc
                # Another repair published the same immutable camera recipe.
                return capture_inputs(checkpoint, cameras, destinations, capture, profile=profile, renderer=renderer)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return root


def capture_prepared_rgb(checkpoint, cameras, destinations, capture, *, cache_root=None):
    """Reuse calibrated Viser anchors; cache only the remaining source cameras.

    Selection thumbnails are deliberately ineligible. Only the exact-size
    preparation path certifies that these source images have matching rays.
    The remaining batch is checkpoint-owned, so known repeat orbits can also
    be captured by the desktop before staging and consumed by the GPU worker.
    """
    if len(cameras) != len(destinations):
        raise ValueError('Capture cameras and destinations must match')
    metadata = checkpoint.manifest.get('metadata', {})
    if (metadata.get('conditioning_rgb_renderer') != 'viser'
            or not metadata.get('resolution', {}).get('recaptured')):
        capture(cameras, destinations)
        return
    def key(camera):
        return json.dumps(camera_to_record(camera), sort_keys=True)
    from ..checkpoint import camera_from_record
    anchors = {key(camera_from_record(view)): checkpoint.image_path(view)
               for view in checkpoint.views}
    missing_cameras, missing_paths = [], []
    for camera, destination in zip(cameras, destinations):
        source = anchors.get(key(camera))
        if source is None:
            missing_cameras.append(camera)
            missing_paths.append(destination)
        else:
            shutil.copy2(source, destination)
    if missing_cameras:
        from ..checkpoint import Checkpoint
        owner = Checkpoint(Path(cache_root), checkpoint.manifest) if cache_root else checkpoint
        capture_inputs(owner, missing_cameras, missing_paths,
                       lambda: capture(missing_cameras, missing_paths), profile='trajectory')


def restore_geometry(cache, prepared):
    from .execution import persist_checkpoint_caches
    persist_checkpoint_caches(cache, prepared)


def publish_geometry(checkpoint):
    cache = checkpoint.manifest.get('metadata', {}).get('artifixer_input_cache')
    if cache:
        from .execution import persist_checkpoint_caches
        persist_checkpoint_caches(checkpoint.root, Path(cache))
