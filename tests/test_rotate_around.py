"""Orbit geometry, opacity-aware picking, and local candidate integration."""
from types import SimpleNamespace

import numpy as np
import pytest

from splat_explorer.agent.actions import Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.orbit import pick_pivot
from splat_explorer.navigation import MotionContext
from splat_explorer.scene import GaussianScene
from splat_explorer.scene_runs_ext.local_loop import LocalRepairLoop


def scene_at(points, opacities=None, scales=None):
    n = len(points)
    return GaussianScene(np.array(points, dtype=np.float32),
                         np.full((n, 3), .25, np.float32) if scales is None else np.array(scales, np.float32),
                         np.tile([1., 0., 0., 0.], (n, 1)),
                         np.array(opacities if opacities is not None else [1.] * n),
                         np.ones((n, 3)))


def orbit(rig, scene, **args):
    return rig.apply(Action('rotate_around', {'pixel_x':.5, 'pixel_y':.5,
                     'azimuth_pi':.1, **args}),
                     MotionContext(camera=rig.camera(100, 100, 75), scene=scene))


@pytest.mark.parametrize('axis', ['+x', '-x', '+y', '-y', '+z', '-z'])
def test_quarter_circle_moves_right_and_aims_at_pivot(axis):
    rig = CameraRig(np.zeros(3), up_axis=axis)
    forward = rig.view_direction()
    right = np.cross(forward, rig.up)
    pivot = forward * 5
    result = orbit(rig, scene_at([pivot]), azimuth_pi=.5)
    np.testing.assert_allclose(rig.position, pivot + right * 5, atol=1e-6)
    np.testing.assert_allclose(rig.view_direction(), -right, atol=1e-6)
    assert result['baseline'] == pytest.approx(5 * np.sqrt(2))
    assert result['radius'] == pytest.approx(5)


def test_birds_eye_absolute_elevation_and_noop():
    rig = CameraRig(np.zeros(3))
    scene = scene_at([[0, 0, -5]])
    result = orbit(rig, scene, azimuth_pi=0, elevation_pi=.4)
    assert not result['blocked']
    assert rig.position[1] == pytest.approx(5 * np.sin(.4 * np.pi))
    assert rig.pitch_deg == pytest.approx(-72)
    before = rig.position.copy()
    result = orbit(rig, scene, azimuth_pi=0, elevation_pi=.4)
    np.testing.assert_allclose(rig.position, before, atol=1e-6)
    assert result['baseline'] < 1e-6
    result = orbit(rig, scene, azimuth_pi=0)
    assert result['baseline'] < 1e-6


def test_picker_passes_faint_foreground_and_returns_gaussian_center():
    rig = CameraRig(np.zeros(3))
    scene = scene_at([[0, 0, -1], [.1, 0, -5]], [.05, 1])
    pivot, index = pick_pivot(scene, rig.camera(100, 100, 75), .5, .5)
    assert index == 1
    np.testing.assert_allclose(pivot, scene.means[1])


def test_picker_uses_accumulated_opacity_and_respects_occlusion():
    rig = CameraRig(np.zeros(3))
    camera = rig.camera(100, 100, 75)
    scene = scene_at([[0,0,-1], [0,0,-2], [0,0,-5]], [.3,.3,1])
    assert pick_pivot(scene, camera, .5, .5)[1] == 1
    scene.opacities[0] = 1
    assert pick_pivot(scene, camera, .5, .5)[1] == 0


def test_anisotropic_rotated_gaussian_and_offcenter_pixel():
    rig = CameraRig(np.zeros(3))
    camera = rig.camera(100, 100, 75)
    scene = scene_at([[1,0,-5]], scales=[[.05, 2, .05]])
    assert pick_pivot(scene, camera, .5, .5) is None
    scene.quats[0] = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
    assert pick_pivot(scene, camera, .5, .5)[1] == 0
    scene.quats[0] = [1,0,0,0]
    px = (camera.fx / 5 + 50 - .5) / 99
    assert pick_pivot(scene, camera, px, .5)[1] == 0


@pytest.mark.parametrize('args', [{'pixel_x':2}, {'azimuth_pi':float('nan')},
                                  {'elevation_pi':.5}, {'pixel_y':'bad'}])
def test_invalid_inputs_leave_pose_unchanged(args):
    rig = CameraRig(np.zeros(3))
    assert 'error' in orbit(rig, scene_at([[0,0,-5]]), **args)
    np.testing.assert_array_equal(rig.position, np.zeros(3))


def test_empty_ray_and_transparent_fog_fail_without_moving():
    rig = CameraRig(np.zeros(3))
    for scene in (scene_at([[10,0,-5]]), scene_at([[0,0,-1]], [.05]),
                  scene_at(np.empty((0,3)))):
        assert 'error' in orbit(rig, scene)
        np.testing.assert_array_equal(rig.position, np.zeros(3))


def test_collision_checks_arc_and_retains_safe_radius():
    rig = CameraRig(np.zeros(3))
    calls = []
    class Wall:
        def clamp_motion(self, start, direction, distance):
            calls.append(start.copy())
            return (0., True) if start[0] + direction[0] * distance > 1 else (distance, False)
    result = rig.apply(Action('rotate_around', {'pixel_x':.5,'pixel_y':.5,'azimuth_pi':.5}),
                       MotionContext(camera=rig.camera(100,100,75),
                                     scene=scene_at([[0,0,-5]]), world=Wall()))
    assert result['blocked'] and 0 < result['completed_fraction'] < 1
    assert 0 < rig.position[0] <= 1
    assert len(calls) > 2
    assert np.linalg.norm(rig.position - [0,0,-5]) == pytest.approx(5)
    np.testing.assert_allclose(rig.view_direction(), (np.array([0,0,-5]) - rig.position)/5, atol=1e-6)


def test_local_orbit_records_one_candidate_without_separate_rotate():
    rig = CameraRig(np.zeros(3))
    loop = LocalRepairLoop(SimpleNamespace(), Action('report_artifact'), 0, rig, None, 2,
                           candidates=5)
    scene = scene_at([[0,0,-5]])
    frame = np.zeros((48,64,3), np.uint8)
    for i in range(5):
        action, _ = loop.handle(Action('rotate_around', {'pixel_x':.5, 'pixel_y':.5,
                                                       'azimuth_pi':.1}), i, rig)
        assert action.args['azimuth_pi'] == .1
        outcome = rig.apply(action, MotionContext(scene=scene, camera=rig.camera(100,100,75)))
        assert outcome['baseline'] > 1
        loop.observe(frame, i+1, rig)
    assert loop.steps == [1,2,3,4,5]
    assert loop.selecting


def test_failed_or_noop_orbit_does_not_record_candidate():
    rig = CameraRig(np.zeros(3))
    loop = LocalRepairLoop(SimpleNamespace(), Action('report_artifact'), 0, rig, None, 2)
    for amount in [0, .1]:
        action, _ = loop.handle(Action('rotate_around', {'pixel_x':.5, 'pixel_y':.5,
                                                       'azimuth_pi':amount}), 0, rig)
        rig.apply(action, MotionContext(scene=scene_at([[0,0,-5]]),
                  camera=rig.camera(100,100,75)) if amount == 0 else None)
        loop.observe(np.zeros((48,64,3), np.uint8), 1, rig)
    assert not loop.steps
