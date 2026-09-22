from types import SimpleNamespace
import numpy as np
from splat_explorer.agent.actions import Action
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.scene_runs_ext.local_loop import LocalRepairLoop


def test_local_loop_blocks_jumps_duplicates_and_restores_outer_policy():
    task=object();tools=[{'function':{'name':n}} for n in ['move','rotate','jump_to_waypoint','report_artifact']]
    policy=SimpleNamespace(_task=task,_tools=tools)
    rig=CameraRig(np.zeros(3))
    loop=LocalRepairLoop(policy,Action('report_artifact',{'description':'broken railing'}),0,rig,np.ones((8,8)),1)
    assert 'jump_to_waypoint' not in [t['function']['name'] for t in policy._tools]
    action,state=loop.handle(Action('jump_to_waypoint',{'target':'waypoint 2'}),1,rig)
    assert action.name=='view_depth' and state=='collecting'
    loop.handle(Action('capture_repair_view'),2,rig)
    assert loop.steps==[0]
    move,_=loop.handle(Action('move',{'direction':'right','distance':100}),3,rig)
    assert move.args['distance']==.025
    for i in range(1,5):
        rig.position[0]=i*.025
        action,state=loop.handle(Action('capture_repair_view'),i+3,rig)
    assert state=='ready' and action.args['view_steps']==[0,4,5,6]
    assert policy._task is task and policy._tools is tools


def test_inner_loop_cancel_restores_policy_and_cannot_repair():
    policy=SimpleNamespace(_task=object(),_tools=[]); original=policy._task
    rig=CameraRig(np.zeros(3));loop=LocalRepairLoop(policy,Action('report_artifact'),0,rig,None,1,max_turns=1)
    loop.handle(Action('view_depth'),1,rig)
    action,state=loop.handle(Action('capture_repair_view'),2,rig)
    assert state=='cancelled' and action.name=='cancel_local_repair'
    assert policy._task is original
