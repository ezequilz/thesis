"""SEVA GUI keyframe interpolation, without its viewer or GPU dependencies."""
import numpy as np
from scipy.spatial.transform import Rotation


def keyframe_path(cameras, order, frames_per_segment, *, include_endpoints=True):
    """Use SEVA gui.CameraPath's natural Kochanek–Bartels splines (tension 0).

    Ordered native paths sample strictly interior targets: saved reference
    keyframes are inserted separately. The unordered adapter can still request
    both endpoints. No implicit closing segment is added.
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
    alphas = (np.linspace(0., 1., frames_per_segment) if include_endpoints else
              np.linspace(0., 1., frames_per_segment+2)[1:-1])
    for segment in range(len(order)-1):
        for local_index, alpha in enumerate(alphas):
            t = segment + alpha
            pose = np.eye(4)
            pose[:3,3] = positions.evaluate(t)
            q = orientations.evaluate(t)
            pose[:3,:3] = Rotation.from_quat([*q.vector, q.scalar]).as_matrix()
            reference = order[segment] if local_index == 0 else (
                order[segment+1] if local_index == frames_per_segment-1 else None)
            if not include_endpoints:
                reference = None
            if reference is not None:
                pose = cameras[reference].c2w.astype(float).copy()
            if not np.isfinite(pose).all():
                raise ValueError('SEVA keyframe spline produced a nonfinite camera')
            yield segment, local_index, float(alpha), reference, pose


def pairwise_path(cameras, order, frames_per_segment):
    """Cut into pairs first; other keyframes cannot influence either spline."""
    for segment, (a,b) in enumerate(zip(order[:-1], order[1:])):
        for _, local, alpha, reference, pose in keyframe_path(
                cameras, [a,b], frames_per_segment, include_endpoints=False):
            yield segment, local, alpha, reference, pose


PRESET_SOURCE = 'https://github.com/Stability-AI/stable-virtual-camera/blob/fe19948e9b7bea261ab2db780a59656131404a83/seva/geometry.py'


def preset_poses(start_c2w, preset, count, scale=1.):
    """NumPy equivalent of SEVA's Basic demo camera presets, in OpenCV c2w.

    Uses get_preset_pose_fov's orbit/lemniscate and the Gradio spiral radii
    [1,1,.5]. Apply the canonical path in the saved camera's local frame.
    Scale converts one demo unit to scene units; calibration stays unchanged.
    Sampling includes the starting pose and excludes the repeated loop end.
    """
    if preset not in ('orbit', 'spiral', 'spiral_expanding', 'lemniscate'):
        raise ValueError('Unknown SEVA preset')
    if type(count) is not int or count < 2 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid SEVA preset size or frame count')
    theta = np.arange(count)*2*np.pi/count
    def lookat(positions, focus, up):
        forward = focus-positions
        forward /= np.linalg.norm(forward, axis=1)[:,None]
        right = np.cross(forward, up)
        right /= np.linalg.norm(right, axis=1)[:,None]
        down = np.cross(forward, right)
        poses = np.tile(np.eye(4), (count,1,1))
        poses[:,:3,:3] = np.stack([right,down,forward],axis=-1)
        poses[:,:3,3] = positions
        return poses
    if preset == 'orbit':
        positions = np.column_stack([10*np.sin(theta),np.zeros(count),10*(1-np.cos(theta))])
        poses = lookat(positions, np.array([0.,0.,10.]), np.array([0.,-1.,0.]))
    elif preset == 'lemniscate':
        t = theta+np.pi/2
        a = 10*np.tan(np.pi/6)  # author preset: degree=60, focus distance=10
        x = a*np.cos(t)/(1+np.sin(t)**2)
        positions = np.column_stack([x,x*np.sin(t),np.zeros(count)])
        poses = lookat(positions, np.array([0.,0.,10.]), np.array([0.,-1.,0.]))
    else:
        # generate_spiral_path: bounds=[1,5], n_rots=2, zrate=.5.
        t = 2*theta
        radius = np.ones(count)
        if preset == 'spiral_expanding':
            # First turn unchanged; smooth growth during turn two, reaching
            # +25% at the last sampled frame (the loop endpoint is excluded).
            progress = np.clip((np.arange(count)-count/2)/max(count-1-count/2, 1), 0, 1)
            radius += .25*progress**2*(3-2*progress)
        positions = np.column_stack([radius*np.cos(t),radius*np.sin(t),.5*np.sin(t*.5)])
        focal = 1/(.25/.9+.75/25)
        poses = lookat(positions, np.array([0.,0.,focal]), np.array([0.,-1.,0.]))
        origin = np.linalg.inv(poses[0])
        poses = origin[None] @ poses
        if preset == 'spiral_expanding':
            # Z is forward/backward in the saved camera's OpenCV frame.
            # Scale after anchoring so lateral movement is unaffected.
            positions = poses[:,:3,3].copy()
            positions[:,2] *= 1.5
            focus = (origin @ np.array([0.,0.,focal,1.]))[:3]
            up = origin[:3,:3] @ np.array([0.,-1.,0.])
            poses = lookat(positions, focus, up)
    poses[:,:3,3] *= scale
    poses = np.asarray(start_c2w,dtype=float)[None] @ poses
    poses[0] = start_c2w
    return poses


def preset_assignments(preset, count, seed=23):
    if preset != 'shuffle_orbit':
        return [preset]*count
    import random
    rng = random.Random(seed)
    kinds = ['orbit', 'spiral', 'lemniscate']
    rng.shuffle(kinds)
    assigned = [kinds[i % 3] for i in range(count)]
    rng.shuffle(assigned)
    return assigned


def per_view_preset_path(cameras, order, frames_per_segment, presets, scale):
    for segment, reference in enumerate(order):
        poses = preset_poses(cameras[reference].c2w, presets[segment], frames_per_segment, scale)
        for local, pose in enumerate(poses):
            yield segment, local, local/frames_per_segment, None, pose
