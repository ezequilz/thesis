"""Contract tests for persisted cameras and independent VLM image inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.agent.actions import Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.view_loop import (
    ACTION_NAMES, ViewFinderPolicy, run_view_finding, selected_view_sheet, view_tools,
)


class Renderer:
    def render(self, camera):
        return np.full((camera.height, camera.width, 3), int(camera.position[0]) + 20, np.uint8)


class Map:
    def add_pose(self, *args):
        pass

    def render(self):
        return np.zeros((20, 20, 3), np.uint8)

    def render_coverage(self):
        return np.ones((20, 20, 3), np.uint8)


class Policy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.inputs = []

    def decide(self, observation, pose_description, step, **kwargs):
        self.inputs.append((observation.copy(), kwargs))
        return next(self.actions)


def test_finished_persists_current_camera_once_and_dynamic_count(tmp_path):
    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=8)
    rig = CameraRig(np.zeros(3))
    actions = [Action('finished')]
    for _ in range(7):
        actions += [Action('move', {'direction': 'right', 'distance': 1}), Action('finished')]
    policy = Policy(actions)
    run_view_finding(Renderer(), rig, policy, checkpoint, views=8, width=40, height=30,
                     exploration_map=Map())
    assert checkpoint.complete
    assert len(list(checkpoint.root.rglob('*.png'))) == 8
    assert len(list(checkpoint.root.rglob('original.png'))) == 8
    for index, record in enumerate(checkpoint.views):
        assert record['camera']['position'][0] == index
        with Image.open(checkpoint.root / record['original_rgb']) as image:
            assert np.asarray(image)[0, 0, 0] == index + 20
    # The first finished tile stays identical as the ACTIVE camera moves.
    first = policy.inputs[1][1]['selected_views_image']
    later = policy.inputs[2][1]['selected_views_image']
    np.testing.assert_array_equal(first[24:264, :320], later[24:264, :320])
    assert np.any(first[24:264, 323:637] != later[24:264, 323:637])
    assert all(item[1]['map_image'] is not None for item in policy.inputs)


def test_budget_retains_partial_checkpoint_and_rejects_old_done(tmp_path):
    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=2)
    policy = Policy([Action('finished'), Action('done'), Action('report_artifact')])
    with pytest.raises(RuntimeError, match='partial checkpoint saved'):
        run_view_finding(Renderer(), CameraRig(np.zeros(3)), policy, checkpoint, views=2,
                         max_steps_per_view=2, width=40, height=30, exploration_map=Map())
    loaded = Checkpoint.load(checkpoint.root)
    assert len(loaded.views) == 1
    records = [json.loads(line) for line in (checkpoint.root / 'actions.jsonl').read_text().splitlines()]
    assert 'Unsupported action' in records[-1]['outcome']['error']


def test_coverage_requested_for_next_observation(tmp_path):
    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=1)
    policy = Policy([Action('view_coverage_map'), Action('finished')])
    run_view_finding(Renderer(), CameraRig(np.zeros(3)), policy, checkpoint, views=1,
                     width=40, height=30, exploration_map=Map())
    assert policy.inputs[0][1]['coverage_image'] is None
    assert policy.inputs[1][1]['coverage_image'] is not None


def test_policy_labels_extra_images_and_exposes_only_new_action_space():
    policy = ViewFinderPolicy.__new__(ViewFinderPolicy)
    policy._tools = view_tools()
    policy._history = []
    captured = []
    policy._ask = lambda prompt, images: (captured.append((prompt, images)) or
                                         ('{"action":"finished","args":{}}', None))
    image = np.zeros((20, 20, 3), np.uint8)
    action = policy.decide(image, 'View 1/6', 0,
                           selected_views_image=selected_view_sheet([], image), map_image=image)
    assert action.name == 'finished'
    assert [tool['function']['name'] for tool in policy._tools] == list(ACTION_NAMES)
    labels = [label for label, _ in captured[0][1]]
    assert len(labels) == 3
    assert 'CURRENT RGB' in labels[0]
    assert 'selected views' in labels[1]
    assert 'MAP' in labels[2]
    assert 'multiple angles' in captured[0][0]


@pytest.mark.parametrize('action', [
    Action('move', {'direction': 'right', 'distance': 'one'}),
    Action('move', {'direction': 'right', 'distance': float('nan')}),
    Action('move', {'direction': 'sideways', 'distance': 1}),
    Action('rotate_around', {'pixel_x': .5, 'pixel_y': .5}),
])
def test_invalid_arguments_become_feedback_and_next_action_can_finish(tmp_path, action):
    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=1)
    policy = Policy([action, Action('finished')])
    rig = CameraRig(np.zeros(3))
    run_view_finding(Renderer(), rig, policy, checkpoint, views=1,
                     width=40, height=30, exploration_map=Map())
    assert checkpoint.complete
    np.testing.assert_array_equal(rig.position, np.zeros(3))
    records = [json.loads(line) for line in (checkpoint.root / 'actions.jsonl').read_text().splitlines()]
    assert 'error' in records[0]['outcome']


def test_move_toward_renders_depth_only_when_needed(tmp_path):
    class DepthRenderer(Renderer):
        calls = 0

        def render_depth(self, camera):
            self.calls += 1
            return np.full((camera.height, camera.width), 5., np.float32)

    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=1)
    renderer = DepthRenderer()
    policy = Policy([Action('move_toward', {'pixel_x': 20, 'pixel_y': 15, 'amount': .5}),
                     Action('finished')])
    rig = CameraRig(np.zeros(3))
    run_view_finding(renderer, rig, policy, checkpoint, views=1,
                     width=40, height=30, exploration_map=Map())
    assert renderer.calls == 1
    assert np.linalg.norm(rig.position) > 2
    assert rig.position[1] == 0


def test_renderer_without_depth_reports_motion_feedback(tmp_path):
    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=1)
    policy = Policy([Action('move_toward', {'pixel_x': 20, 'pixel_y': 15, 'amount': .5}),
                     Action('finished')])
    run_view_finding(Renderer(), CameraRig(np.zeros(3)), policy, checkpoint, views=1,
                     width=40, height=30, exploration_map=Map())
    first = json.loads((checkpoint.root / 'actions.jsonl').read_text().splitlines()[0])
    assert 'no depth' in first['outcome']['error']
    assert checkpoint.complete
