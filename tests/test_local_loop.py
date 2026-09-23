from types import SimpleNamespace
import numpy as np
import pytest
from splat_explorer.agent.actions import Action, filter_tools
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.scene_runs_ext.local_loop import LocalRepairLoop
from splat_explorer.scene_runs_ext.config import configure_policy, validate_options
from splat_explorer.tasks import artifact_hunt_3


def make_loop(candidates=10):
    policy = SimpleNamespace(_task=None, _tools=[])
    configure_policy(policy)
    rig = CameraRig(np.zeros(3))
    loop = LocalRepairLoop(policy, Action('report_artifact'), 0, rig, None, 2,
                           candidates=candidates)
    return policy, rig, loop


def collect(loop, rig):
    for i in range(loop.candidates):
        action, state = loop.handle(Action('move', {'direction':'right','distance':1}), i, rig)
        rig.apply(action)
        loop.observe(np.full((48,64,3), i, np.uint8), 2 * i + 1, rig)
        action, state = loop.handle(Action('rotate', {'yaw_degrees': -10}), 2 * i + 1, rig)
        rig.apply(action)
        image = loop.observe(np.full((48,64,3), i, np.uint8), 2 * i + 2, rig)
    return image


def test_movement_only_then_numbered_selection_restores_outer_policy():
    policy, rig, loop = make_loop()
    assert {t['function']['name'] for t in policy._tools} == {'move','move_toward','rotate'}
    image = collect(loop, rig)
    assert image.shape == (192, 192, 3)  # full-resolution tiles, numbers inside
    tile = image[0:48, 64:128]
    assert tile.max() == 255 and (tile == 0).any()  # white ink on a dark backing
    assert tile[0, 50, 0] == 1  # no full-width bar; the view fills the tile
    assert loop.steps == list(range(2,21,2))
    assert policy._tools[0]['function']['name'] == 'select_repair_views'
    action, state = loop.handle(Action('select_repair_views', {'views':[2,4,6,8,9]}), 10, rig)
    assert state == 'ready'
    assert loop.selected_steps == [4,8,12,16,18]
    assert action.args['view_steps'] == [8,12,16,18]
    assert np.all(loop.anchor_frame == 1)
    np.testing.assert_allclose(loop.anchor_rig.position, loop.rigs[1].position)
    assert policy._task is artifact_hunt_3
    assert policy._tools == filter_tools(artifact_hunt_3.HIDDEN_TOOLS)


@pytest.mark.parametrize('numbers', [[1,1,2,3,4], [1,2,3,4,11], [1,2,3,4], [True,2,3,4,5]])
def test_invalid_selection_never_repairs(numbers):
    policy, rig, loop = make_loop()
    collect(loop,rig)
    for attempt in range(3):
        action,state = loop.handle(Action('select_repair_views', {'views':numbers}), 11, rig)
    assert state == 'cancelled'
    assert policy._task is artifact_hunt_3


def test_report_capture_and_jump_do_not_record_views():
    policy, rig, loop = make_loop()
    for name in ('report_artifact','capture_repair_view','jump_to_waypoint','view_map'):
        action,state = loop.handle(Action(name),1,rig)
        assert action.name == 'local_noop' and state == 'collecting'
        loop.observe(np.zeros((48,64,3),np.uint8),1,rig)
    assert loop.steps == []


def test_normal_navigation_and_absolute_pitch():
    _, rig, loop = make_loop()
    rig.pitch_deg = -80
    action,state = loop.handle(Action('rotate',{'pitch_degrees':60}),1,rig)
    rig.apply(action)
    assert rig.pitch_deg == 60
    action,state = loop.handle(Action('move',{'direction':'right','distance':100}),2,rig)
    assert action.args['distance'] == 2


def test_candidate_options_and_original_outer_prompt():
    assert validate_options()['local_candidate_count'] == 10
    assert validate_options({'local_candidate_count':5})['local_candidate_count'] == 5
    with pytest.raises(ValueError): validate_options({'local_candidate_count':11})
    assert 'local_step_fraction' not in validate_options({'local_step_fraction':.025})


def test_translation_requires_rotation_before_recording_or_more_travel():
    _, rig, loop = make_loop()
    frame = np.zeros((48,64,3), np.uint8)
    action, _ = loop.handle(Action('move', {'direction':'right', 'distance':1}), 0, rig)
    rig.apply(action)
    loop.observe(frame, 1, rig)
    assert not loop.steps and loop.needs_rotation
    action, _ = loop.handle(Action('move', {'direction':'right', 'distance':1}), 1, rig)
    assert action.name == 'local_noop'
    action, _ = loop.handle(Action('rotate', {'yaw_degrees':0}), 2, rig)
    rig.apply(action)
    loop.observe(frame, 3, rig)
    assert not loop.steps and loop.needs_rotation
    action, _ = loop.handle(Action('rotate', {'yaw_degrees':-10}), 3, rig)
    rig.apply(action)
    loop.observe(frame, 4, rig)
    assert loop.steps == [4] and not loop.needs_rotation
    assert loop.rigs[0].yaw_deg == 350


def test_blocked_translation_and_rotation_only_do_not_count():
    _, rig, loop = make_loop()
    frame = np.zeros((48,64,3), np.uint8)
    loop.handle(Action('move', {'direction':'right', 'distance':1}), 0, rig)
    # Collision clamp left the rig at its original position.
    loop.observe(frame, 1, rig)
    assert not loop.needs_rotation
    action, _ = loop.handle(Action('rotate', {'yaw_degrees':20}), 1, rig)
    rig.apply(action)
    loop.observe(frame, 2, rig)
    assert not loop.steps


def test_revisited_camera_center_does_not_count_even_with_new_heading():
    _, rig, loop = make_loop()
    frame = np.zeros((48,64,3), np.uint8)
    loop.rigs.append(CameraRig(np.array([1., 0., 0.])))
    action, _ = loop.handle(Action('move', {'direction':'right', 'distance':1}), 0, rig)
    rig.apply(action)
    loop.observe(frame, 1, rig)
    action, _ = loop.handle(Action('rotate', {'yaw_degrees':-10}), 1, rig)
    rig.apply(action)
    loop.observe(frame, 2, rig)
    assert not loop.steps and not loop.needs_rotation


def test_converging_views_keep_target_centered_with_translational_parallax():
    _, rig, loop = make_loop(candidates=5)
    target = np.array([0., 0., -5.])
    frame = np.zeros((48,64,3), np.uint8)
    for i in range(5):
        action, _ = loop.handle(Action('move', {'direction':'right', 'distance':.5}), 2*i, rig)
        rig.apply(action)
        loop.observe(frame, 2*i+1, rig)
        desired = CameraRig.from_look_at(rig.position, target)
        yaw_delta = (desired.yaw_deg - rig.yaw_deg + 180) % 360 - 180
        action, _ = loop.handle(Action('rotate', {'yaw_degrees':yaw_delta,
                                                 'pitch_degrees':desired.pitch_deg}), 2*i+1, rig)
        rig.apply(action)
        loop.observe(frame, 2*i+2, rig)
    assert loop.selecting
    for camera in loop.rigs:
        ray = target - camera.position
        np.testing.assert_allclose(camera.view_direction(), ray / np.linalg.norm(ray), atol=1e-8)
    angle = np.degrees(np.arccos(np.dot(loop.rigs[0].view_direction(),
                                       loop.rigs[-1].view_direction())))
    assert angle > 20  # different target sightlines, not just an in-place pan


def test_selection_numbers_are_large_on_full_views():
    _, rig, loop = make_loop(candidates=1)
    frame = np.full((720, 960, 3), 40, np.uint8)
    action, _ = loop.handle(Action('move', {'direction':'right', 'distance':1}), 0, rig)
    rig.apply(action)
    loop.observe(frame, 1, rig)
    action, _ = loop.handle(Action('rotate', {'yaw_degrees':-10}), 1, rig)
    rig.apply(action)
    sheet = loop.observe(frame, 2, rig)
    ink = np.any(sheet != 40, axis=-1)
    rows = np.flatnonzero(ink.any(axis=1))
    assert rows[-1] - rows[0] >= 70  # full-view tiles use a much larger number than size 18


def test_selection_overview_is_sent_at_high_detail_without_resizing():
    import base64
    import io
    from PIL import Image
    from splat_explorer.agent.cli_relay import CliRelayPolicy, _png_data_url
    policy, rig, loop = make_loop()
    sheet = collect(loop, rig)
    relay = object.__new__(CliRelayPolicy)
    relay._task = policy._task
    relay.model = 'test'
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
    relay.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    relay._ask('Choose five', [(relay._task.observation_label, _png_data_url(sheet))])
    image = calls[0]['messages'][0]['content'][2]['image_url']
    assert image['detail'] == 'high'
    decoded = Image.open(io.BytesIO(base64.b64decode(image['url'].split(',')[1])))
    assert decoded.size == (sheet.shape[1], sheet.shape[0])


@pytest.mark.parametrize('up_axis', ['+y', '-y', '+z', '-z'])
@pytest.mark.parametrize('direction,sign', [('up', 1), ('down', -1)])
def test_vertical_translation_then_pitch_tracks_target(up_axis, direction, sign):
    policy = SimpleNamespace(_task=None, _tools=[])
    rig = CameraRig(np.zeros(3), up_axis=up_axis)
    target = rig.view_direction() * 5
    loop = LocalRepairLoop(policy, Action('report_artifact'), 0, rig, None, 2)
    frame = np.zeros((48,64,3), np.uint8)
    action, _ = loop.handle(Action('move', {'direction':direction, 'distance':.5}), 0, rig)
    rig.apply(action)
    loop.observe(frame, 1, rig)
    np.testing.assert_allclose(rig.position, sign * .5 * rig.up)
    assert loop.current_height == pytest.approx(sign * .5)
    assert not loop.steps  # height change must be followed by re-aiming
    desired = CameraRig.from_look_at(rig.position, target, up_axis=up_axis)
    assert desired.pitch_deg * sign < 0
    action, _ = loop.handle(Action('rotate', {'pitch_degrees':desired.pitch_deg}), 1, rig)
    rig.apply(action)
    loop.observe(frame, 2, rig)
    assert loop.steps == [2]
    ray = target - rig.position
    np.testing.assert_allclose(rig.view_direction(), ray / np.linalg.norm(ray), atol=1e-8)
    missing = 'lower' if sign > 0 else 'higher'
    recorded = 'higher' if sign > 0 else 'lower'
    assert f'Still need a {missing} vantage' in loop.context()
    assert f'Still need a {recorded} vantage' not in loop.context()
