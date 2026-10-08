"""Geometry-only preflight for OpenGL C2W inference paths.

These measurements describe camera motion, not visibility or image quality.
Distances remain in scene units; a reconstruction's scale is not assumed metric.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def summarize_trajectory(poses, *, up=None):
    """Measure a path without changing poses, sampling, or frame identities.

    Supply a scene up vector when known. Otherwise use the mean camera +Y
    direction (OpenGL); ambiguous means omit height measurements. Closure is
    excluded: only consecutive input frames contribute to travel.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
            or not np.isfinite(poses).all()):
        raise ValueError('poses must be a nonempty array of finite 4x4 OpenGL C2W poses')
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
        raise ValueError('poses must be rigid homogeneous C2W matrices')
    positions = poses[:, :3, 3]
    steps = np.diff(positions, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    angles = (np.degrees(Rotation.from_matrix(
        rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).magnitude())
        if len(poses) > 1 else np.empty(0))
    result = {
        'frame_count': len(poses), 'distance_units': 'scene units',
        'closed_segment_included': False,
        'translation_total': float(lengths.sum()),
        'translation_step_max': float(lengths.max(initial=0)),
        'rotation_total_degrees': float(angles.sum()),
        'rotation_step_max_degrees': float(angles.max(initial=0)),
        'up_source': 'explicit' if up is not None else 'mean_opengl_camera_y',
    }
    direction = np.array(up if up is not None else rotations[:, :, 1].mean(axis=0), dtype=float, copy=True)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('up must be a finite three-vector')
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        if up is not None:
            raise ValueError('up must be nonzero')
        result.update(up=None, height=None)
        return result
    direction /= norm
    heights = (positions - positions[0]) @ direction
    changes = np.diff(heights)
    # Ignore roundoff and plateaus; height reversals survive added interpolation.
    tolerance = max(float(lengths.sum()) * 1e-10, 1e-12)
    nonzero = np.flatnonzero(np.abs(changes) > tolerance)
    reversals = nonzero[1:][np.diff(np.sign(changes[nonzero])) != 0]
    result.update(up=direction.tolist(), height={
        'range': float(np.ptp(heights)),
        'total_travel': float(np.abs(changes).sum()),
        'reversal_count': len(reversals),
        'reversal_frame_indices': reversals.tolist(),
        'measurement_tolerance': tolerance,
    })
    return result
