"""Experimental warm-ups stay continuous, collision bounded, and fit once."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from splat_explorer.splatfix.artifixer.teacher_extensions import extend_orbit, render_poses, extension_manifest
from splat_explorer.splatfix.artifixer.teacher_inference import planned_clips
from splat_explorer.splatfix.artifixer.inference_policy import new_job_options


def pose(x):
    p = np.eye(4); p[0, 3] = x
    return p.tolist()


def test_extension_uses_endpoint_step_fixed_rotation_and_cuts_at_first_collision():
    from splat_explorer.agent.orbit_collision import OrbitCollision
    scene = SimpleNamespace(means=np.array([[-2., 0, 0]]), scales=np.full((1,3), .1),
                            quats=np.array([[1.,0,0,0]]), opacities=np.ones(1))
    orbit = {'poses': [pose(x) for x in range(5)], 'provenance': {}}
    before=deepcopy(orbit)
    extended=extend_orbit(orbit,[0,4],5,OrbitCollision(scene))
    assert orbit == before
    groups=extended['teacher_extensions']['groups']
    assert [p[0][3] for p in groups[0]['poses']]==[-1]
    assert groups[0]['collision_step']==2
    assert [p[0][3] for p in groups[1]['poses']]==[5,6,7,8,9]
    assert all(np.array_equal(np.array(p)[:3,:3],np.eye(3)) for g in groups for p in g['poses'])
    assert render_poses(extended)[:5]==orbit['poses']
    assert extension_manifest(extended)[0]['frame_indices']==[5]
    assert extension_manifest(extended)[1]['frame_indices']==list(range(6,11))


def test_collision_checks_whole_step_and_stationary_endpoint_adds_nothing():
    class Collision:
        def intersects(self,start,end):
            return start[0] > -.5 and end[0] < -.5
    result=extend_orbit({'poses':[pose(0),pose(1)],'provenance':{}},[0],10,Collision())
    assert result['teacher_extensions']['groups'][0]['poses']==[]
    assert result['teacher_extensions']['groups'][0]['collision_step']==1
    result=extend_orbit({'poses':[pose(0),pose(0)],'provenance':{}},[0],5,Collision())
    assert not result['teacher_extensions']['groups'][0]['poses']
    assert 'stationary' in result['teacher_extensions']['groups'][0]['reason']


def test_extension_windows_are_continuous_with_reference_at_its_real_position():
    class Clear:
        def intersects(self,*args):return False
    orbit=extend_orbit({'poses':[pose(i) for i in range(67)],'provenance':{}},[0,66],5,Clear())
    positions=[p[0][3] for p in render_poses(orbit)]
    jobs=[({'indices':list(range(1,34)), 'seed_index':0},[0,1],'inference'),
          ({'indices':list(range(65,33,-1)), 'seed_index':66},[0,1],'inference')]
    clips=list(planned_clips(jobs,25,[0,66],seeding=True,layout='full-leg',extensions=extension_manifest(orbit)))
    assert sorted(i for c in clips for i in c['indices'])==list(range(1,66))
    assert any(c['seed_index'] is None for c in clips)  # No distant-reference jump in interior clips.
    assert clips[0]['extension_frame_indices']
    for c in clips:
        inputs=[*c['extension_frame_indices'], *([c['seed_index']] if c['seed_index'] is not None else []), *c['input_indices']]
        assert len(inputs)<=25
        differences=np.diff([positions[i] for i in inputs])
        assert np.all(differences==1) or np.all(differences==-1)
        assert not set(c['indices']).intersection(c['context_frame_indices'])


def test_extension_unlocks_reference_without_affecting_normal_defaults():
    normal=new_job_options({})
    assert normal['image_cache_insertion'] and normal['teacher_opacity_scale']==1
    experiment=new_job_options({'teacher_endpoint_extension':8,'teacher_opacity_scale':.75})
    assert experiment['initial_image_seeding'] and not experiment['image_cache_insertion']
    assert experiment['teacher_opacity_scale']==.75
    assert normal['teacher_reference_opacity'] == 1
    assert new_job_options({'teacher_endpoint_extension':8, 'teacher_reference_opacity':.8})['teacher_reference_opacity'] == .8
    for change in [{'teacher_endpoint_extension':True},{'teacher_endpoint_extension':6},
                   {'teacher_endpoint_extension':8,'teacher_max_frames':9},
                   {'teacher_endpoint_extension':8,'teacher_clip_layout':'segment'},
                   {'teacher_opacity_scale':-1},{'teacher_opacity_scale':1.1},
                   {'teacher_opacity_scale':float('nan')},{'teacher_opacity_scale':True},
                   {'teacher_reference_opacity': -.1}, {'teacher_reference_opacity': float('nan')},
                   {'teacher_reference_opacity': True}, {'teacher_reference_opacity': .8}]:
        with pytest.raises(ValueError):new_job_options(change)


def test_context_extension_is_excluded_from_endpoint_inference_and_reconstruction(tmp_path):
    from splat_explorer.splatfix.artifixer.official_worker import inference_jobs, reconstruction_supervision
    trajectory={'anchors':[{'frame_index':0},{'frame_index':4}], 'base_frame_count':5,
                'frames':[{}]*7, 'context_only_indices':[5,6], 'camera_convention':'opengl_c2w',
                'transforms': {'w':16,'h':16,'fl_x':12,'fl_y':12,'cx':8,'cy':8,
                               'frames':[{'transform_matrix':pose(i)} for i in range(7)]}}
    request={'repair_policy':'endpoints','runtime':{},'references':[str(tmp_path/'a'),str(tmp_path/'b')]}
    jobs=inference_jobs(request,trajectory)
    assert jobs[0][0]['indices']==list(range(5))
    assert jobs[1][0]['indices']==list(range(4,-1,-1))
    transforms,supervision=reconstruction_supervision(tmp_path,request,trajectory)
    assert len(transforms['frames'])==5
    assert supervision['original_to_distillation'][-2:]==[None,None]
    assert [g['source_index'] for g in supervision['groups']]==list(range(5))


def test_warmup_export_has_one_owner_and_enters_reconstruction(tmp_path):
    import json
    from splat_explorer.splatfix.artifixer.official_worker import reconstruction_supervision
    jobs=[({'indices':[1,2,3], 'seed_index':0},[0,1],'inference')]
    extension=[{'seed_frame_index':0,'interior_frame_index':1,'frame_indices':[5,6,7,8]}]
    clips=list(planned_clips(jobs,9,[0,4],seeding=True,layout='full-leg',extensions=extension,include_warmup=True))
    assert clips[0]['indices']==[8,7,6,5,1,2,3]
    assert len(set(i for c in clips for i in c['indices']))==7
    trajectory={'anchors':[{'frame_index':0},{'frame_index':4}], 'base_frame_count':5,
        'frames':[{}]*10,'context_only_indices':[5,6,7,8,9], 'camera_convention':'opengl_c2w',
        'transforms':{'w':16,'h':16,'fl_x':12,'fl_y':12,'cx':8,'cy':8,
                      'frames':[{'transform_matrix':pose(i)} for i in range(10)]}}
    (tmp_path/'inference-teacher.json').write_text(json.dumps({'clips':[
        {'output':'inference','warmup_frame_indices':[8,7,6,5]}]}))
    request={'runtime':{'teacher_endpoint_extension':4},'references':[str(tmp_path/'a'),str(tmp_path/'b')]}
    transforms,supervision=reconstruction_supervision(tmp_path,request,trajectory)
    assert len(transforms['frames'])==9
    assert supervision['warmup_supervision_indices']==[5,6,7,8]
    assert supervision['excluded_extension_indices']==[9]
    assert supervision['original_to_distillation'][9] is None


def test_path_gate_rejects_wrong_side_reversal_and_reference_teleport():
    from splat_explorer.splatfix.artifixer.teacher_extensions import validate_clip_path
    class Clear:
        def intersects(self, *args): return False
    orbit = extend_orbit({'poses': [pose(i) for i in range(33)], 'provenance': {}}, [0, 32], 8, Clear())
    trajectory = {'teacher_extensions': extension_manifest(orbit), 'transforms': {
        'frames': [{'transform_matrix': p} for p in render_poses(orbit)]}}
    clips = list(planned_clips([({'indices': list(range(1, 32))}, [0, 1], 'inference')],
                              25, [0, 32], seeding=True, layout='full-leg',
                              extensions=trajectory['teacher_extensions'], include_warmup=True))
    for clip in clips:
        audit = validate_clip_path(clip, trajectory)
        assert audit['validated'] and audit['translation_step_max'] == 1
        assert audit['reference_input_position'] == 8
    first = clips[0]
    assert first['indices'][:8] == list(range(40, 32, -1))
    bad = deepcopy(first); bad['extension_frame_indices'].reverse()
    with pytest.raises(ValueError, match='opposite side'): validate_clip_path(bad, trajectory)
    bad = deepcopy(first); bad['input_indices'] = list(range(2, 18))
    with pytest.raises(ValueError, match='adjacent'): validate_clip_path(bad, trajectory)
    bad = deepcopy(trajectory)
    bad['transforms']['frames'][33]['transform_matrix'] = pose(1)
    with pytest.raises(ValueError, match='tangent'): validate_clip_path(first, bad)


def test_path_gate_allows_interior_windows_without_distant_reference():
    from splat_explorer.splatfix.artifixer.teacher_extensions import validate_clip_path
    job = {'seed_index': None, 'extension_frame_indices': [], 'input_indices': list(range(24, 48))}
    trajectory = {'transforms': {'frames': [{'transform_matrix': pose(i)} for i in range(70)]}}
    assert validate_clip_path(job, trajectory)['reference_input_position'] is None
    job['input_indices'][16] = 60
    with pytest.raises(ValueError, match='continuous'): validate_clip_path(job, trajectory)
