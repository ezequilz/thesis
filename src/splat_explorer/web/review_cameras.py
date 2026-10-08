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
    # Image-up includes camera pitch. Viser also uses up_direction as the
    # gravity axis for first-person movement, so use the captured scene axis
    # when available instead of tilting the walking plane with every view.
    from ..rendering.base import up_vector
    try:
        world_up = up_vector(body.get('metadata', {}).get('up_axis'))
    except (ValueError, TypeError):
        world_up = None
    result = []
    for view in body.get('views', []):
        try:
            camera = camera_view(view['camera'], view.get('id', len(result)))
            if world_up is not None:
                camera['up'] = world_up.copy()
            result.append(camera)
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
    """Read the run's OpenGL inference cameras, reference inputs first.

    Interrupted/preview manifests need not contain selected_images. Their split
    and index files are the authoritative record of the reconstruction inputs.
    Poses already share the exported PLY frame; metric_scale is conditioning only.
    """
    root = Path(manifest).parent.resolve()
    result = read_json(manifest, {})

    def local_path(base, raw):
        if not isinstance(raw, str) or not raw:
            return None
        path = (base / raw).resolve()
        return path if path.is_relative_to(root) and path.is_file() else None

    candidates = []
    split = local_path(root, result.get('inference_split'))
    if split:
        candidates.append(split)
    candidates.extend(p for p in sorted(root.glob('prepared/*/split.json')) if p not in candidates)
    for split in candidates:
        if not split.resolve().is_relative_to(root):
            continue
        data = read_json(split, {})
        for entry in data.get('test', {}).values():
            path = local_path(split.parent, entry.get('transforms_path'))
            if path is None:
                continue
            transforms = read_json(path, {})
            if transforms.get('camera_convention', 'opengl_c2w') != 'opengl_c2w':
                continue
            frames = transforms.get('frames', [])
            selected = local_path(split.parent, entry.get('selected_indices_path'))
            indices = read_json(selected, []) if selected else []
            if not indices:
                names = result.get('selected_images', [])
                by_name = {Path(f.get('file_path', '')).name: i for i, f in enumerate(frames)}
                indices = [by_name[name] for name in names if name in by_name]
            targets = local_path(split.parent, entry.get('target_indices_path'))
            indices = [*indices, *(read_json(targets, []) if targets else [])]
            views = []
            seen = set()
            for index in indices:
                if type(index) is not int or not 0 <= index < len(frames) or index in seen:
                    continue
                seen.add(index)
                frame = frames[index]
                try:
                    pose = np.asarray(frame['transform_matrix'], dtype=float)
                    fy = float(frame.get('fl_y', transforms.get('fl_y')))
                    height = float(frame.get('h', transforms.get('h')))
                    if pose.shape != (4, 4):
                        continue
                    rotation = pose[:3, :3]
                    if (not np.isfinite(pose).all() or not np.isfinite([fy, height]).all()
                        or fy <= 0 or height <= 0
                        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
                        continue
                    views.append({'center': pose[:3, 3], 'forward': -pose[:3, 2],
                                  'up': pose[:3, 1], 'vfov': float(2*np.arctan(height/(2*fy))),
                                  'label': Path(frame.get('file_path', '')).name or f'Frame {index + 1}'})
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
            if views:
                return views
    return []
