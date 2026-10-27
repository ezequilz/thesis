"""Two-sided extension budgets, collision cutoffs, exact references and fitting."""
from copy import deepcopy
import json
import numpy as np
import pytest

from splat_explorer.splatfix.artifixer.teacher_short_extensions import extend_short_orbit
from splat_explorer.splatfix.artifixer.teacher_extensions import render_poses, extension_manifest, validate_clip_path
from splat_explorer.splatfix.artifixer.teacher_inference import planned_clips
from splat_explorer.splatfix.artifixer.inference_policy import new_job_options


def pose(x):
    p = np.eye(4); p[0, 3] = x
    return p.tolist()


class Clear:
    def intersects(self, *args):
        return False


def make_path(end, anchors, collision=Clear()):
    orbit = {'poses': [pose(i) for i in range(end + 1)], 'provenance': {}}
    before = deepcopy(orbit)
    expanded = extend_short_orbit(orbit, anchors, 81, collision)
    assert orbit == before
    return {'anchors': [{'frame_index': i} for i in anchors],
            'frames': [{} for _ in render_poses(expanded)], 'base_frame_count': end + 1,
            'context_only_indices': list(range(end + 1, len(render_poses(expanded)))),
            'teacher_extensions': extension_manifest(expanded),
            'transforms': {'w':16,'h':16,'fl_x':12,'fl_y':12,'cx':8,'cy':8,
                           'frames': [{'transform_matrix': p} for p in render_poses(expanded)]}}


@pytest.mark.parametrize('reverse', [False, True])
def test_both_sides_of_every_leg_fill_81_with_continuous_cameras(reverse):
    anchors = [0, 32, 64]
    t = make_path(64, anchors)
    jobs = [({'indices': list(range(a+1,b))[::(-1 if reverse else 1)]}, [n,n+1], 'inference')
            for n,(a,b) in enumerate(zip(anchors,anchors[1:]))]
    clips = list(planned_clips(jobs,81,anchors,seeding=True,layout='full-leg',
                              extensions=t['teacher_extensions'],include_warmup=True,fill_short=True))
    owned = [i for c in clips for i in c['indices']]
    assert len(owned) == len(set(owned)) == 62 + 96
    assert set(range(65)) - set(anchors) <= set(owned)
    for c in clips:
        audit = validate_clip_path(c,t)
        assert len(audit['input_frame_indices']) == 81
        assert audit['leading_extension_count'] == audit['trailing_extension_count'] == 24
        assert audit['reference_input_position'] == 24
        assert audit['end_reference_input_position'] == 56
        positions = [t['transforms']['frames'][i]['transform_matrix'][0][3] for i in audit['input_frame_indices']]
        assert np.all(np.diff(positions) == (-1 if reverse else 1))
        assert set(c['indices']).isdisjoint(anchors)
    bad = deepcopy(clips[0]); bad['trailing_extension_frame_indices'].reverse()
    with pytest.raises(ValueError, match='opposite side'):
        validate_clip_path(bad,t)


def test_collision_stops_one_side_and_other_side_fills_remaining_budget():
    class Wall:
        def intersects(self,start,end):
            return end[0] <= -7
    t = make_path(32,[0,32],Wall())
    assert t['teacher_extensions'][0]['collision_step'] == 7
    clip, = planned_clips([({'indices':list(range(1,32))},[0,1],'inference')],81,[0,32],
        seeding=True,layout='full-leg',extensions=t['teacher_extensions'],include_warmup=True,fill_short=True)
    audit = validate_clip_path(clip,t)
    assert len(audit['input_frame_indices']) == 81
    assert (audit['leading_extension_count'], audit['trailing_extension_count']) == (4,44)
    assert all(t['transforms']['frames'][i]['transform_matrix'][0][3] > -7 for i in clip['indices'])


def test_two_blocked_sides_stay_short_and_never_resume_after_collision():
    class Walls:
        def intersects(self,start,end):
            return end[0] <= -6 or end[0] >= 39
    t = make_path(32,[0,32],Walls())
    clip, = planned_clips([({'indices':list(range(1,32))},[0,1],'inference')],81,[0,32],
        seeding=True,layout='full-leg',extensions=t['teacher_extensions'],include_warmup=True,fill_short=True)
    audit = validate_clip_path(clip,t)
    assert len(audit['input_frame_indices']) == 43
    assert (audit['leading_extension_count'], audit['trailing_extension_count']) == (4,6)
    assert [g['collision_step'] for g in t['teacher_extensions']] == [6,7]


def test_tail_extensions_supervise_reconstruction_once(tmp_path):
    from splat_explorer.splatfix.artifixer.official_worker import reconstruction_supervision
    t = make_path(32,[0,32])
    clip, = planned_clips([({'indices':list(range(1,32))},[0,1],'inference')],81,[0,32],
        seeding=True,layout='full-leg',extensions=t['teacher_extensions'],include_warmup=True,fill_short=True)
    (tmp_path/'inference-teacher.json').write_text(json.dumps({'clips':[{
        'output':'inference','warmup_frame_indices':clip['warmup_frame_indices']}]}))
    request={'runtime':{'teacher_extend_short_clips':True},'references':[str(tmp_path/'a'),str(tmp_path/'b')]}
    transforms, supervision = reconstruction_supervision(tmp_path,request,t)
    assert len(transforms['frames']) == 81
    assert len(supervision['warmup_supervision_indices']) == 48
    assert set(clip['trailing_extension_frame_indices']) <= set(supervision['warmup_supervision_indices'])


def test_new_setting_is_optional_unlocks_references_and_rejects_invalid_modes():
    assert not new_job_options({})['teacher_extend_short_clips']
    assert new_job_options({})['teacher_reference_rgb_replacement']
    cfg = new_job_options({'teacher_extend_short_clips':True,'teacher_reference_rgb_replacement':True})
    assert cfg['initial_image_seeding'] and not cfg['image_cache_insertion']
    for bad in [{'teacher_extend_short_clips':1},
                {'teacher_extend_short_clips':True,'teacher_clip_layout':'segment'},
                {'teacher_reference_rgb_replacement':1},
                {'teacher_reference_rgb_replacement':False,'image_cache_insertion':True}]:
        with pytest.raises(ValueError): new_job_options(bad)
    assert not new_job_options({'teacher_extend_short_clips':True,'teacher_reference_rgb_replacement':False})['teacher_reference_rgb_replacement']
