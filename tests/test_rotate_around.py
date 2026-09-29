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


@pytest.mark.parametrize('mode', [None, 'off', 'low', 'full'])
def test_orbit_stops_at_splat_even_with_collision_disabled(mode):
    from splat_explorer.navigation import CollisionWorld
    from splat_explorer.agent.orbit_collision import ORBIT_CLEARANCE
    rig = CameraRig(np.zeros(3))
    angle = .2
    obstacle = np.array([5*np.sin(angle), 0, -5+5*np.cos(angle)])
    scene = scene_at([[0, 0, -5], obstacle], scales=[[.1]*3, [.02]*3])
    world = None if mode is None else CollisionWorld(scene, collision=mode)
    result = rig.apply(Action('rotate_around', {'pixel_x':.5, 'pixel_y':.5,
                       'azimuth_pi':.5}), MotionContext(
                           camera=rig.camera(100,100,75), scene=scene, world=world))
    assert result['blocked']
    assert 0 < result['completed_fraction'] < angle / (np.pi / 2)
    gap = np.linalg.norm(rig.position - obstacle) - .06
    assert gap >= ORBIT_CLEARANCE - 1e-7
    if mode in (None, 'off'):
        assert gap > ORBIT_CLEARANCE + .05
        assert result['shortened']
    assert np.linalg.norm(rig.position - [0,0,-5]) == pytest.approx(5)


def test_swept_collision_catches_thin_rotated_splat_between_clear_endpoints():
    from splat_explorer.agent.orbit_collision import OrbitCollision
    scene = scene_at([[0,0,0]], scales=[[1, .0001, .1]])
    scene.quats[0] = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
    collision = OrbitCollision(scene)
    a, b = np.array([-1.,0,0]), np.array([1.,0,0])
    assert not collision.intersects(a, a)
    assert not collision.intersects(b, b)
    assert collision.intersects(a, b)
    # A max-scale sphere would incorrectly block this parallel passage.
    assert not collision.intersects(np.array([.5,-1.,0]), np.array([.5,1.,0]))


def test_orbit_initial_overlap_stops_without_moving():
    rig = CameraRig(np.zeros(3))
    scene = scene_at([[0,0,-5], [0,0,0]], [.99, .1], [[.1]*3, [.01]*3])
    result = orbit(rig, scene)
    assert result['blocked']
    assert result['completed_fraction'] == 0
    np.testing.assert_array_equal(rig.position, np.zeros(3))


def test_collision_ignores_invisible_and_invalid_splats():
    from splat_explorer.agent.orbit_collision import OrbitCollision
    scene = scene_at([[0,0,0], [0,0,0], [np.nan,0,0]], [0, 1, 1])
    scene.scales[1] = 0
    collision = OrbitCollision(scene)
    assert not collision.intersects(np.array([-1.,0,0]), np.array([1.,0,0]))


@pytest.mark.parametrize('azimuth,elevation', [(-.5, 0), (.3, .25), (0, -.25)])
def test_collision_follows_negative_and_elevated_orbits(azimuth, elevation):
    rig = CameraRig(np.zeros(3))
    theta, phi = .4 * azimuth * np.pi, .4 * elevation * np.pi
    obstacle = np.array([5*np.cos(phi)*np.sin(theta), 5*np.sin(phi),
                         -5+5*np.cos(phi)*np.cos(theta)])
    scene = scene_at([[0,0,-5], obstacle], scales=[[.1]*3, [.01]*3])
    result = orbit(rig, scene, azimuth_pi=azimuth, elevation_pi=elevation)
    if azimuth and elevation:
        assert result['elevation_adjusted']
        assert not result['blocked']
        assert result['completed_fraction'] == 1
    else:
        assert result['blocked']
        assert 0 < result['completed_fraction'] < .4
    assert np.linalg.norm(rig.position - obstacle) >= .04 - 1e-7
    assert np.linalg.norm(rig.position - [0,0,-5]) == pytest.approx(5)


def test_arc_error_covers_obstacle_off_the_chord():
    from splat_explorer.agent.orbit_collision import OrbitCollision
    collision = OrbitCollision(scene_at([[0,.03,0]], scales=[[.001]*3]))
    a, b = np.array([-.1,0,0]), np.array([.1,0,0])
    assert not collision.intersects(a, b)
    assert collision.intersects(a, b, arc_error=.03)


def test_shortened_orbit_keeps_requested_curve_and_records_view():
    from splat_explorer.agent.loop import _motion_note
    rig = CameraRig(np.zeros(3))
    loop = LocalRepairLoop(SimpleNamespace(), Action('report_artifact'), 0, rig, None, 2)
    class Wall:
        def clamp_motion(self, start, direction, distance):
            return (0., True) if start[0] + direction[0] * distance > 1 else (distance, False)
    action, _ = loop.handle(Action('rotate_around', {
        'pixel_x': .5, 'pixel_y': .5, 'azimuth_pi': .5, 'elevation_pi': .25}), 0, rig)
    result = rig.apply(action, MotionContext(camera=rig.camera(100,100,75),
        scene=scene_at([[0,0,-5]]), world=Wall()))
    assert result['shortened']
    f = result['completed_fraction']
    theta, phi = f * .5 * np.pi, f * .25 * np.pi
    np.testing.assert_allclose(rig.position, [5*np.cos(phi)*np.sin(theta),
        5*np.sin(phi), -5+5*np.cos(phi)*np.cos(theta)], atol=1e-6)
    assert .7 < rig.position[0] < .85  # deliberate margin from the x=1 boundary
    assert 'shorter safe orbit' in _motion_note(result)
    loop.observe(np.zeros((48,64,3), np.uint8), 1, rig)
    assert loop.steps == [1]
    assert loop.failed_orbits == 0


def test_blocked_orbit_feedback_requires_translation():
    from splat_explorer.agent.loop import _motion_note
    rig = CameraRig(np.zeros(3))
    result = orbit(rig, scene_at([[0,0,-5], [0,0,0]], [.99,.1], [[.1]*3,[.01]*3]))
    note = _motion_note(result)
    assert 'Use move' in note
    assert 'heading is unchanged' in note
    assert 'aimed at Gaussian' not in note
    assert not result['shortened']


@pytest.mark.parametrize('sign', [-1, 1])
def test_elevation_fallback_interpolates_from_current_elevation(sign):
    from splat_explorer.agent.loop import _motion_note
    pivot = np.array([0., 0., -5.])
    initial = sign * .1 * np.pi
    rig = CameraRig(pivot + [0, 5*np.sin(initial), 5*np.cos(initial)])
    rig.aim_at(pivot)
    start = rig.position.copy()
    class HeightLimit:
        def clamp_motion(self, source, direction, distance):
            end = source + direction * distance
            blocked = sign * (end[1] - start[1]) > .6
            return (0., True) if blocked else (distance, False)
    result = rig.apply(Action('rotate_around', {'pixel_x': .5, 'pixel_y': .5,
        'azimuth_pi': .3, 'elevation_pi': sign * .3}), MotionContext(
        scene=scene_at([pivot]), camera=rig.camera(100,100,75), world=HeightLimit()))
    assert result['elevation_adjusted']
    assert result['requested_elevation_pi'] == pytest.approx(sign * .3)
    assert .1 <= sign * result['elevation_pi'] < .3
    assert not result['blocked']
    assert result['completed_fraction'] == 1
    assert sign * (rig.position[1] - start[1]) <= .6
    phi = result['elevation_pi'] * np.pi
    np.testing.assert_allclose(rig.position, pivot + [5*np.cos(phi)*np.sin(.3*np.pi),
        5*np.sin(phi), 5*np.cos(phi)*np.cos(.3*np.pi)], atol=1e-6)
    assert 'Elevation target reduced' in _motion_note(result)


def test_clear_elevation_request_is_not_adjusted():
    rig = CameraRig(np.zeros(3))
    result = orbit(rig, scene_at([[0,0,-5]]), elevation_pi=.25)
    assert not result['elevation_adjusted']
    assert result['elevation_pi'] == .25
    assert result['completed_fraction'] == 1
