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
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture_inputs(checkpoint, cameras, destinations, capture, *, profile):
    metadata = checkpoint.manifest.get('metadata', {})
    recipe = {'version': 1, 'profile': profile, 'renderer': 'viser',
              'source': checkpoint.manifest.get('source_fingerprint', checkpoint.manifest['scene_path']),
              'cameras': [camera_to_record(camera) for camera in cameras],
              'up_axis': metadata.get('up_axis', '-y'),
              'scene_load': metadata.get('scene_load', {})}
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
                return capture_inputs(checkpoint, cameras, destinations, capture, profile=profile)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return root


def restore_geometry(cache, prepared):
    from .execution import persist_checkpoint_caches
    persist_checkpoint_caches(cache, prepared)


def publish_geometry(checkpoint):
    cache = checkpoint.manifest.get('metadata', {}).get('artifixer_input_cache')
    if cache:
        from .execution import persist_checkpoint_caches
        persist_checkpoint_caches(checkpoint.root, Path(cache))
