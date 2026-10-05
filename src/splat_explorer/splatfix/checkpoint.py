"""Portable selected views shared by image editing and independent repair runs.

Only selected RGBs are persisted, once. Camera poses use OpenCV camera-to-world
coordinates; all image paths are relative to the checkpoint directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ..rendering.base import Camera

SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, body: dict):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_fingerprint(scene_path) -> dict:
    """Hash source bytes without depending on mount path or modification times.

    Unbundled SOG assets are identified by sorted paths relative to their scene
    directory. Hidden files are excluded; keep generated outputs outside this
    directory so that the fingerprint describes only the source asset bundle.
    """
    from ..scene.catalog import openable_scene_path
    source = openable_scene_path(Path(scene_path).expanduser()).resolve()
    if not source.exists():
        raise FileNotFoundError(f'Source scene is unavailable: {source}')
    directory = source.parent if source.name in ('meta.json', 'lod-meta.json') else source
    is_directory = directory.is_dir()
    files = (sorted((p for p in directory.rglob('*') if p.is_file()
                     and not any(part.startswith('.') for part in p.relative_to(directory).parts)),
                    key=lambda p: p.relative_to(directory).as_posix())
             if is_directory else [source])
    if not files:
        raise ValueError(f'Source scene directory has no assets: {directory}')
    digest = hashlib.sha256()
    for path in files:
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                content.update(block)
        if is_directory:
            digest.update(path.relative_to(directory).as_posix().encode('utf-8'))
            digest.update(b'\0')
        digest.update(content.digest())
    return {'kind': 'directory' if is_directory else 'file',
            'sha256': (digest if is_directory else content).hexdigest(), 'file_count': len(files)}


def validate_source(checkpoint):
    """Guard first trajectory rendering; cached replay need not read the source."""
    expected = checkpoint.manifest.get('source_fingerprint')
    if expected is not None and source_fingerprint(checkpoint.manifest['scene_path']) != expected:
        raise ValueError('Source scene changed since view selection; create a new checkpoint')


def camera_to_record(camera: Camera) -> dict:
    return {'position': np.asarray(camera.position).tolist(),
            'rotation': np.asarray(camera.rotation).tolist(),
            'width': int(camera.width), 'height': int(camera.height),
            'fov_deg': float(camera.fov_deg),
            'intrinsics': camera.intrinsics.tolist(), 'c2w': camera.c2w.tolist()}


def camera_from_record(record: dict) -> Camera:
    body = record.get('camera', record)
    position = np.asarray(body['position'], dtype=np.float32)
    rotation = np.asarray(body['rotation'], dtype=np.float32)
    if any(isinstance(body[k], bool) or not isinstance(body[k], (int, np.integer))
           for k in ('width', 'height')):
        raise ValueError('Saved camera dimensions must be integers')
    width, height, fov = int(body['width']), int(body['height']), float(body['fov_deg'])
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or width <= 0 or height <= 0 or not 0 < fov < 180):
        raise ValueError('Invalid saved camera')
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError('Saved camera rotation must be a proper orthonormal rotation')
    camera = Camera(position, rotation, width=width, height=height, fov_deg=fov)
    for key, expected in (('intrinsics', camera.intrinsics), ('c2w', camera.c2w)):
        if key in body:
            saved = np.asarray(body[key], dtype=np.float32)
            if saved.shape != expected.shape or not np.allclose(saved, expected, atol=1e-5):
                raise ValueError(f'Saved camera {key} disagrees with its pose or calibration')
    return camera


class Checkpoint:
    """An incrementally saved view selection, safe to reuse without API calls."""

    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root).resolve()
        self.manifest = manifest

    @classmethod
    def create(cls, output_root, scene_path, target_views=6, metadata=None):
        if isinstance(target_views, bool) or not isinstance(target_views, int) or target_views < 1:
            raise ValueError('target_views must be a positive integer')
        from ..scene.catalog import openable_scene_path
        source = openable_scene_path(Path(scene_path).expanduser()).resolve()
        fingerprint = source_fingerprint(source) if source.exists() else None
        root = Path(output_root).resolve() / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                                           + '_' + uuid.uuid4().hex[:8])
        root.mkdir(parents=True, exist_ok=False)
        (root / 'views').mkdir()
        checkpoint = cls(root, {'schema_version': SCHEMA_VERSION, 'created_at': utc_now(),
                               'scene_path': str(Path(scene_path).resolve()),
                               'camera_convention': 'opencv_c2w', 'target_views': target_views,
                               'metadata': metadata or {}, 'views': []})
        if fingerprint is not None:
            checkpoint.manifest['source_fingerprint'] = fingerprint
        checkpoint.save()
        return checkpoint

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest_path = path if path.is_file() else path / 'checkpoint.json'
        body = json.loads(manifest_path.read_text())
        if body.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported splatfix checkpoint schema')
        if body.get('camera_convention') != 'opencv_c2w':
            raise ValueError('Unsupported checkpoint camera convention')
        target = body.get('target_views')
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError('Invalid checkpoint target_views')
        checkpoint = cls(manifest_path.parent, body)
        views = body.get('views')
        if not isinstance(views, list) or len(views) > target:
            raise ValueError('Invalid checkpoint view list')
        ids = set()
        for index, view in enumerate(views):
            if view.get('id') != f'{index:03d}':
                raise ValueError('Invalid checkpoint view id')
            if view['id'] in ids:
                raise ValueError('Duplicate checkpoint view id')
            ids.add(view['id'])
            camera = camera_from_record(view)
            with Image.open(checkpoint.image_path(view)) as image:
                if image.size != (camera.width, camera.height):
                    raise ValueError('Original RGB dimensions do not match saved camera')
            if view.get('repaired_rgb'):
                checkpoint.image_path(view, repaired=True)
        return checkpoint

    @property
    def views(self):
        return self.manifest['views']

    @property
    def target_views(self):
        return self.manifest['target_views']

    @property
    def complete(self):
        return len(self.views) == self.target_views

    def save(self):
        atomic_json(self.root / 'checkpoint.json', self.manifest)

    def image_path(self, view, repaired=False):
        key = 'repaired_rgb' if repaired else 'original_rgb'
        relative = Path(view[key])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.root):
            raise ValueError('Checkpoint images must remain inside the run directory')
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def add_view(self, rgb, camera: Camera, metadata=None):
        if self.complete:
            raise ValueError('Requested view selection is already complete')
        camera_record = camera_to_record(camera)
        camera_from_record(camera_record)
        if isinstance(rgb, Image.Image):
            image = rgb.convert('RGB')
        else:
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError('Selected RGB must be an H×W×3 uint8 image')
            image = Image.fromarray(rgb)
        if image.size != (camera.width, camera.height):
            raise ValueError('Selected RGB dimensions must match the camera')
        view_id = f'{len(self.views):03d}'
        relative = f'views/{view_id}/original.png'
        record = {'id': view_id, 'original_rgb': relative, 'camera': camera_record,
                  'metadata': metadata or {}}
        # Validate metadata before writing an image or mutating the selection.
        json.dumps(record, allow_nan=False)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format='PNG')
        self.views.append(record)
        try:
            self.save()
        except Exception:
            self.views.pop()
            raise
        return record
