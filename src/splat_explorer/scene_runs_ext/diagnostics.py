"""Cheap observations, not a geometric acceptance score or ground truth."""
from __future__ import annotations
import numpy as np


def trajectory_diagnostics(transforms):
    poses = np.asarray([f['transform_matrix'] for f in transforms['frames']], dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError('Expected finite 4x4 camera poses')
    rotations, positions = poses[:, :3, :3], poses[:, :3, 3]
    def angles(a, b):
        relative = np.swapaxes(a, -1, -2) @ b
        return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    return {
        'frames': len(poses),
        'max_anchor_rotation_degrees': float(angles(rotations[0], rotations).max()),
        'max_step_rotation_degrees': float(angles(rotations[:-1], rotations[1:]).max()) if len(poses)>1 else 0.,
        'max_anchor_translation_scene_units': float(np.linalg.norm(positions-positions[0], axis=1).max()),
        'max_step_translation_scene_units': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()) if len(poses)>1 else 0.,
        'rotation_orthogonality_max_error': float(np.abs(np.swapaxes(rotations, -1, -2)@rotations-np.eye(3)).max()),
        'rotation_determinant_max_error': float(np.abs(np.linalg.det(rotations)-1).max()),
        'note': 'Scene units are not necessarily metres. Smooth poses do not establish image/camera agreement.',
    }


def image_diagnostics(targets):
    """Separate contrast loss from closure drift; do not equate either with geometry."""
    first = np.asarray(targets[0], dtype=np.float32)/255
    rows = []
    for i, target in enumerate(targets):
        value = np.asarray(target, dtype=np.float32)/255
        if value.shape != first.shape or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError('Targets must have matching RGB dimensions')
        gray = value @ np.array([.2126, .7152, .0722], dtype=np.float32)
        rows.append({'frame_index': i, 'mean_luminance': float(gray.mean()),
                     'luminance_std': float(gray.std()),
                     'anchor_rgb_mae': float(np.abs(value-first).mean())})
    return {'frames': rows, 'note': 'Unwarped image statistics; viewpoint/content changes affect these values. Not a quality gate.'}
