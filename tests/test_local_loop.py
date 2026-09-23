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
        image = loop.observe(np.full((48,64,3), i, np.uint8), i + 1, rig)
    return image


def test_movement_only_then_numbered_selection_restores_outer_policy():
    policy, rig, loop = make_loop()
    assert {t['function']['name'] for t in policy._tools} == {'move','move_toward','rotate'}
    image = collect(loop, rig)
    assert image.shape == (192, 192, 3)  # full-resolution tiles, numbers inside
    tile = image[0:48, 64:128]
    assert tile.max() == 255 and (tile == 0).any()  # white ink on a dark backing
    assert tile[0, 50, 0] == 1  # no full-width bar; the view fills the tile
    assert loop.steps == list(range(1,11))
    assert policy._tools[0]['function']['name'] == 'select_repair_views'
    action, state = loop.handle(Action('select_repair_views', {'views':[2,4,6,8,9]}), 10, rig)
    assert state == 'ready'
    assert loop.selected_steps == [2,4,6,8,9]
    assert action.args['view_steps'] == [4,6,8,9]
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
