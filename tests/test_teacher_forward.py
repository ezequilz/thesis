"""Forward teacher windows follow camera order and seed references in place."""
from copy import deepcopy

import numpy as np
import pytest

from splat_explorer.splatfix.artifixer.inference_policy import MODELS, model_runtime, new_job_options
from splat_explorer.splatfix.artifixer.official_worker import inference_jobs
from splat_explorer.splatfix.artifixer.teacher_extensions import clip_input_indices, validate_clip_path
from splat_explorer.splatfix.artifixer.teacher_inference import planned_clips


def trajectory(anchors, extension=8):
    count = anchors[-1] + 1
    poses = []
    for index in range(count):
        pose = np.eye(4)
        pose[0, 3] = index
        poses.append({'transform_matrix': pose.tolist()})
    groups = []
    for seed in anchors:
        for neighbor in (seed - 1, seed + 1):
            if not 0 <= neighbor < count:
                continue
            indices = []
            for step in range(1, extension + 1):
                pose = np.eye(4)
                pose[0, 3] = seed + step * (seed - neighbor)
                indices.append(len(poses))
                poses.append({'transform_matrix': pose.tolist()})
            groups.append({'seed_frame_index': seed, 'interior_frame_index': neighbor, 'frame_indices': indices})
    return {'anchors': [{'frame_index': i} for i in anchors], 'frames': [{} for _ in poses],
            'base_frame_count': count, 'teacher_extensions': groups, 'transforms': {'frames': poses},
            # A saved reverse segment must not influence the forward planner.
            'segments': [{'indices': list(range(count - 1, -1, -1))}]}


def plan(data, maximum=81, plus=False):
    request = {'runtime': new_job_options({}), 'repair_policy': 'all'}
    return list(planned_clips(inference_jobs(request, data, plus=plus), maximum,
                             [a['frame_index'] for a in data['anchors']], layout='forward',
                             seeding=True, extensions=data['teacher_extensions']))


@pytest.mark.parametrize('plus', [False, True])
def test_six_views_bound_five_forward_clips_with_every_target_exported_once(plus):
    anchors = [0, 40, 105, 170, 211, 258]
    data = trajectory(anchors)
    clips = plan(data, plus=plus)
    assert len(clips) == 5
    exported = [i for job in clips for i in job['indices']]
    assert exported == [i for i in range(259) if i not in anchors]
    for job, left, right in zip(clips, anchors, anchors[1:]):
        inputs = clip_input_indices(job)
        assert len(inputs) <= 81
        assert job['output'] == ('plus' if plus else 'inference')
        assert job['references'] == list(range(6))
        assert job['inline_reference_indices'] == [left, right]
        assert inputs[inputs.index(left):inputs.index(right) + 1] == list(range(left, right + 1))
        assert set(job['context_frame_indices']).isdisjoint(job['indices'])
        assert not job['warmup_frame_indices']
        check = validate_clip_path(job, data)
        assert check['validated'] and check['direction'] == 'forward'
        assert check['reference_input_positions'] == [inputs.index(left), inputs.index(right)]


@pytest.mark.parametrize('distance,leading,trailing,expected', [
    (72, 8, 8, (4, 4)), (64, 8, 8, (8, 8)), (80, 8, 8, (0, 0)),
    (40, 6, 3, (4, 0)), (40, 0, 8, (0, 8)),
])
def test_warmups_fit_the_budget_and_collision_clearance(distance, leading, trailing, expected):
    data = trajectory([0, distance])
    data['teacher_extensions'][0]['frame_indices'] = data['teacher_extensions'][0]['frame_indices'][:leading]
    data['teacher_extensions'][1]['frame_indices'] = data['teacher_extensions'][1]['frame_indices'][:trailing]
    job, = plan(data)
    assert (len(job['extension_frame_indices']), len(job['trailing_extension_frame_indices'])) == expected
    assert validate_clip_path(job, data)['validated']


@pytest.mark.parametrize('maximum', [5, 9, 17, 81])
def test_long_intervals_overlap_without_restarting_at_a_distant_reference(maximum):
    data = trajectory([0, 210])
    clips = plan(data, maximum)
    assert len(clips) > 1
    assert [i for c in clips for i in c['indices']] == list(range(1, 210))
    assert clips[0]['inline_reference_indices'] == [0]
    assert clips[-1]['inline_reference_indices'] == [210]
    assert any(not c['inline_reference_indices'] for c in clips[1:-1])
    for clip in clips:
        assert len(clip_input_indices(clip)) <= maximum
        assert validate_clip_path(clip, data)['translation_step_min'] == 1
    if maximum > 5:
        assert all(set(clip_input_indices(a)) & set(clip_input_indices(b)) for a, b in zip(clips, clips[1:]))


def test_validation_rejects_reverse_input_and_misplaced_extension_pose():
    data = trajectory([0, 40])
    job, = plan(data)
    reverse = deepcopy(job)
    reverse['ordered_input_indices'].reverse()
    with pytest.raises(ValueError, match='reverses or skips'):
        validate_clip_path(reverse, data)
    index = job['extension_frame_indices'][0]
    data['transforms']['frames'][index]['transform_matrix'][0][3] += 2
    with pytest.raises(ValueError, match='tangent'):
        validate_clip_path(job, data)


def test_forward_preservation_is_rejected_but_explicit_legacy_fallback_survives():
    with pytest.raises(ValueError, match='legacy layout'):
        new_job_options({'teacher_clip_layout': 'forward', 'image_cache_insertion': True})
    fallback = new_job_options({'teacher_clip_layout': 'full-leg', 'image_cache_insertion': True})
    assert fallback['image_cache_insertion'] and fallback['teacher_endpoint_extension'] == 0
    saved = model_runtime({'checkpoint': '/cache/artifixer-s1-14b.pt', 'model_id': MODELS['14b'][1],
                           'teacher_inference': True, 'initial_image_seeding': True, 'image_cache_insertion': True})
    assert saved['teacher_clip_layout'] == 'segment' and saved['teacher_max_frames'] == 37
    assert saved['image_cache_insertion'] and saved['teacher_endpoint_extension'] == 0
