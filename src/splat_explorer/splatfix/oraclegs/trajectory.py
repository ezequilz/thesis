"""SEVA GUI keyframe interpolation, without its viewer or GPU dependencies."""
import numpy as np
from scipy.spatial.transform import Rotation


def keyframe_path(cameras, order, frames_per_segment):
    """Use SEVA gui.CameraPath's natural Kochanek–Bartels splines (tension 0).

    Each pair includes both endpoints. Junctions intentionally appear twice,
    with distinct output identities, so every pair has exactly the requested
    number of generated images. No implicit closing segment is added.
    """
    # Only input staging constructs paths; prepared GPU workers do not need
    # this dependency in their orchestration environment.
    import splines
    import splines.quaternion

    positions = splines.KochanekBartels(
        [cameras[i].position.astype(float) for i in order],
        tcb=(0., 0., 0.), endconditions='natural')
    orientations = splines.quaternion.KochanekBartels(
        [splines.quaternion.UnitQuaternion.from_unit_xyzw(
            Rotation.from_matrix(cameras[i].rotation).as_quat()) for i in order],
        tcb=(0., 0., 0.), endconditions='natural')
    for segment in range(len(order)-1):
        for local_index, alpha in enumerate(np.linspace(0., 1., frames_per_segment)):
            t = segment + alpha
            pose = np.eye(4)
            pose[:3,3] = positions.evaluate(t)
            q = orientations.evaluate(t)
            pose[:3,:3] = Rotation.from_quat([*q.vector, q.scalar]).as_matrix()
            reference = order[segment] if local_index == 0 else (
                order[segment+1] if local_index == frames_per_segment-1 else None)
            if reference is not None:
                pose = cameras[reference].c2w.astype(float).copy()
            if not np.isfinite(pose).all():
                raise ValueError('SEVA keyframe spline produced a nonfinite camera')
            yield segment, local_index, float(alpha), reference, pose
