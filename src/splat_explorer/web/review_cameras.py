"""Saved review cameras in the exported scene coordinate frame (no normalization)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..splatfix.jobs import read_json


def camera_view(camera, label):
    """Application saved cameras are OpenCV c2w: +z forward, -y up."""
    from ..splatfix.checkpoint import camera_from_record
    cam = camera_from_record({'camera': camera})
    return {'center': cam.position.astype(float), 'forward': cam.rotation[:, 2].astype(float),
            'up': -cam.rotation[:, 1].astype(float), 'vfov': cam.vertical_fov_rad(),
            'label': str(label)}


def checkpoint_views(path):
    body = read_json(Path(path) / 'checkpoint.json', {})
    if body.get('camera_convention') != 'opencv_c2w':
        return []
    result = []
    for view in body.get('views', []):
        try:
            result.append(camera_view(view['camera'], view.get('id', len(result))))
        except (ValueError, KeyError, TypeError):
            continue
    return result


def historical_views(root):
    """Only cameras of persisted repair requests, not an invented overview."""
    result = []
    for path in sorted(Path(root).glob('requests/*-repair/request.json')):
        if not path.resolve().is_relative_to(Path(root).resolve()):
            continue
        body = read_json(path, {})
        try:
            result.append(camera_view(body['camera'], f"Step {body['step']}"))
        except (ValueError, KeyError, TypeError):
            continue
    return result


def benchmark_views(manifest):
    """Use official prepared OpenGL transforms and selected photograph names.

    These transforms are used by the original renderer and share the checkpoint /
    PLY frame. metric_scale is diffusion conditioning only, never a scene transform.
    """
    root = Path(manifest).parent
    result = read_json(manifest, {})
    names = result.get('selected_images', [])
    candidates = sorted(root.glob('prepared/*/split.json'))
    for split in candidates:
        data = read_json(split, {})
        for entry in data.get('test', {}).values():
            raw = entry.get('transforms_path', '')
            path = (split.parent / raw).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                continue
            transforms = read_json(path, {})
            frames = {Path(f.get('file_path', '')).name: f for f in transforms.get('frames', [])}
            views = []
            for name in names:
                frame = frames.get(name)
                if frame is None:
                    continue
                try:
                    pose = np.asarray(frame['transform_matrix'], dtype=float)
                    fy = float(frame.get('fl_y', transforms['fl_y']))
                    height = float(frame.get('h', transforms['h']))
                    rotation = pose[:3, :3]
                    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or fy <= 0 or height <= 0
                        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
                        continue
                    views.append({'center': pose[:3, 3], 'forward': -pose[:3, 2],
                                  'up': pose[:3, 1], 'vfov': float(2*np.arctan(height/(2*fy))),
                                  'label': name})
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
            if views:
                return views
    return []
