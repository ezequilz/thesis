import json
import subprocess
from pathlib import Path

import pytest

from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.artifixer.progress import ProgressObserver, mark_active, summarize, planned_count
from splat_explorer.splatfix.artifixer.progress_probe import snapshot


def packet():
    return {'state': {'status': 'running', 'phase': 'infer'},
            'teachers': {'inference': {'planned_clip_count': 10, 'teacher_steps': 50,
                'expected_frame_count': 253, 'clips': []}},
            'logs': {'inference': 'ARTIFIXER_MEMORY {"phase":"model_loaded"}\n 48%|xx | 24/50 [05:30<05:58]\r'},
            'reconstruction_complete': False}


def test_current_step_contributes_to_overall_without_claiming_saved_frames():
    p = packet()
    stages = summarize(p, observed_at=100)['stages']
    assert stages['inference']['percent'] == 4.8
    assert stages['inference']['completed_frames'] == 0
    assert stages['inference']['detail'] == 'Clip 1 / 10 · denoising 24 / 50'
    assert stages['reconstruction']['status'] == 'pending'
    assert stages['reconstruction']['percent'] == 0


def test_clip_transition_does_not_double_count_previous_50_steps():
    p = packet()
    p['teachers']['inference']['clips'] = [{'output':'inference','frame_indices':[1,2]}]
    p['logs']['inference'] += '100%|xx | 50/50 [11:00<00:00]\nARTIFIXER_MEMORY {"phase":"clip_complete"}\n'
    stage = summarize(p, observed_at=100)['stages']['inference']
    assert stage['percent'] == 10
    assert stage['completed_frames'] == 2
    assert 'Clip 2 / 10' in stage['detail']


def test_finished_denoising_waits_for_decode_before_100_percent():
    p = packet()
    p['teachers']['inference']['planned_clip_count'] = 1
    p['logs']['inference'] += '100%|xx | 50/50 [11:00<00:00]'
    assert summarize(p, observed_at=100)['stages']['inference']['percent'] == 99.9
    p['teachers']['inference']['complete'] = True
    assert summarize(p, observed_at=100)['stages']['inference']['percent'] == 100


def test_reconstruction_and_plus_are_independent():
    p = packet()
    p['state']['phase'] = 'distill'
    p['logs']['reconstruction'] = 'Training progress: 40%|xx | 12000/30000 [00:30<01:00, Loss=0.5]\r'
    stages = summarize(p, observed_at=100)['stages']
    assert stages['inference']['percent'] == 100
    assert stages['reconstruction']['percent'] == 40
    assert stages['plus']['status'] == 'pending'
    p['state']['phase'] = 'plus'
    p['reconstruction_complete'] = True
    p['teachers']['plus'] = {'planned_clip_count':10,'teacher_steps':50,'clips':[]}
    p['logs']['plus'] = ' 10%|xx | 5/50 [00:30<01:00]'
    stages = summarize(p, observed_at=100)['stages']
    assert stages['reconstruction']['percent'] == 100
    assert stages['plus']['percent'] == 1


def test_disabled_plus_pass_is_skipped_before_and_after_reconstruction():
    p = packet()
    p['plan'] = {'runtime': {'artifixer_plus_pass': False}}
    for complete in (False, True):
        p['reconstruction_complete'] = complete
        p['state'].update(phase='finished' if complete else 'distill', status='completed' if complete else 'running')
        stages = summarize(p, observed_at=100)['stages']
        assert stages['plus']['status'] == 'skipped'
        assert 'final output' in stages['plus']['detail']
        if complete:
            assert stages['reconstruction']['status'] == 'completed'


def test_failure_retains_observed_progress_and_ignores_weight_loading_bars():
    p = packet();p['state'].update(status='error',phase='finished',failed_phase='infer')
    stage = summarize(p, observed_at=100)['stages']['inference']
    assert stage['status'] == 'error' and stage['percent'] == 4.8
    p['state'].update(status='running',phase='infer');p['state'].pop('failed_phase')
    p['logs']['inference'] = 'Loading weights: | xx | 120/242 [00:01<00:02]'
    assert summarize(p, observed_at=100)['stages']['inference']['percent'] == 0


def test_existing_run_plan_uses_seeded_capacity_and_preserves_reverse_endpoints():
    p = {'plan': {'frame_count':74,'repair_policy':'all','trajectory_mode':'authors_orbit',
        'runtime': {'inference_model':'14b-teacher','teacher_max_frames':37,'initial_image_seeding':True},
        'trajectory': {'anchors':[{'frame_index':0},{'frame_index':73}],
                       'segments':[{'indices':list(range(1,73)), 'seed_index':0}]}}}
    assert planned_count(p) == 2  # 36 targets + reference per clip.
    p['plan']['repair_policy']='endpoints'
    assert planned_count(p) == 4
    assert planned_count(p, plus=True) == 2


def test_shared_cadence_survives_tabs_visibility_changes_and_monitor_restart(tmp_path):
    root=tmp_path/'run';root.mkdir();clock=[100.]
    observer=ProgressObserver(root,'/workspace/job','/dss/job',clock=lambda:clock[0])
    mark_active(tmp_path,now=100)
    observer.wait(lambda:False)
    for now in (101,120,159.9):
        clock[0]=now;mark_active(tmp_path,now=now)
        assert not observer.due()
    restarted=ProgressObserver(root,'/workspace/job','/dss/job',clock=lambda:clock[0])
    assert not restarted.due()
    clock[0]=160;assert restarted.due()
    restarted.wait(lambda:False)
    clock[0]=210;assert restarted.interval()==300
    clock[0]=220;assert not restarted.due()  # No browser activity.
    mark_active(tmp_path,now=220);assert restarted.due()


def test_cancellation_wakes_once_and_does_not_start_a_fast_poll_loop(tmp_path):
    root=tmp_path/'run';root.mkdir();clock=[100.]
    mark_active(tmp_path,now=100)
    observer=ProgressObserver(root,'/workspace/job','/dss/job',clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    observer.wait(lambda:False)
    observer.wait(lambda:clock[0]>=102)
    assert clock[0]==102
    assert not observer.due()
    mark_active(tmp_path,now=130);clock[0]=130
    observer.wait(lambda:True)
    assert clock[0]==162


def test_probe_is_one_bounded_safe_read_and_observer_uses_one_remote_call(tmp_path):
    root=tmp_path/'job';out=root/'results/pass';out.mkdir(parents=True)
    atomic_json(root/'worker-status.json',{'status':'running','phase':'infer','output_dir':'/workspace/job/results/pass'})
    atomic_json(out/'request.json', {'trajectory':'/workspace/job/results/pass/trajectory.json','runtime':{}})
    atomic_json(out/'trajectory.json', {'frames':[{},{}], 'anchors':[], 'segments':[]})
    (out/'infer.log').write_text('x'*70000)
    outside=tmp_path/'secret';outside.write_text('must not be read')
    (out/'distill.log').symlink_to(outside)
    data=snapshot(root,'/workspace/job')
    assert len(data['logs']['inference'])==65536
    assert data['logs']['reconstruction']==''
    local=tmp_path/'local';local.mkdir();calls=[]
    observer=ProgressObserver(local,'/workspace/job',str(root))
    def ssh(command):
        calls.append(command)
        return subprocess.check_output(command,shell=True,text=True)
    observer.poll(ssh)
    assert len(calls)==1
    assert json.loads((local/'progress.json').read_text())['stages']['reconstruction']['status']=='pending'


def test_probe_terminal_snapshot_recovers_result_and_distillation_completion(tmp_path):
    root=tmp_path/'job';out=root/'results/pass';out.mkdir(parents=True)
    atomic_json(root/'worker-status.json',{'status':'completed','phase':'finished','result':{'output_dir':'/workspace/job/results/pass'}})
    atomic_json(out/'distillation.json',{'splat_path':'artifixer3d.ply'})
    atomic_json(out/'inference-teacher.json',{'complete':True,'clips':[]})
    (root/'launcher-exit').write_text('0')
    data=snapshot(root,'/workspace/job')
    assert data['exit']=='0' and data['reconstruction_complete']
    assert summarize(data,observed_at=100)['stages']['reconstruction']['percent']==100


def test_browser_progress_is_accessible_and_hidden_tabs_do_not_heartbeat():
    script=Path('src/splat_explorer/web/static/splatfix_progress.js').read_text()
    harness=r'''
const vm=require('vm'),assert=require('assert');
let calls=0,interval,visible;
const document={hidden:true,head:{append(){}},createElement(){return {}},addEventListener(n,fn){visible=fn}};
const context={document,window:{},fetch:async()=>{calls++},setInterval(fn){interval=fn},Date,Number,String};
vm.runInNewContext(SCRIPT,context);
(async()=>{
 assert.equal(calls,0);await interval();assert.equal(calls,0);
 document.hidden=false;await visible();assert.equal(calls,1);
 document.hidden=true;await interval();assert.equal(calls,1);
 const html=context.window.SplatfixProgress.html({stages:{inference:{label:'14B inference',percent:40,status:'active',detail:'Clip <1>'}}});
 assert(html.includes('aria-label="14B inference"'));
 assert(html.includes('value="40"'));
 assert(html.includes('Clip &lt;1&gt;'));
 assert(!html.includes('3D reconstruction')); // Inference-only replay.
})().catch(e=>{console.error(e);process.exitCode=1});
'''.replace('SCRIPT',json.dumps(script))
    subprocess.run(['node','-e',harness],check=True)


@pytest.mark.parametrize('fail', [False, True])
def test_training_report_tracks_upstream_iterations_and_restores_trainer(tmp_path, monkeypatch, fail):
    import sys
    from types import SimpleNamespace
    from splat_explorer.splatfix.artifixer.training_progress import report_training_progress
    calls=[]
    class Trainer:
        global_step=0
        def log_training_iter(self, value):
            calls.append(value)
            return 'upstream result'
    training=SimpleNamespace(Trainer3DGRUT=Trainer)
    monkeypatch.setitem(sys.modules,'data_processing',SimpleNamespace(threedgrut_training=training))
    path=tmp_path/'reconstruction-progress.json'
    try:
        with report_training_progress(path,30000):
            trainer=training.Trainer3DGRUT()
            trainer.global_step=12000
            assert trainer.log_training_iter('metrics')=='upstream result'
            if fail:raise RuntimeError('training failed')
    except RuntimeError:
        assert fail
    assert training.Trainer3DGRUT is Trainer
    assert calls==['metrics']
    data=json.loads(path.read_text())
    assert data['iteration']==(12000 if fail else 30000)
    assert data['status']==('interrupted' if fail else 'trained')
    p=packet();p['state']['phase']='distill';p['training']=data
    stages=summarize(p,observed_at=100)['stages']
    assert stages['reconstruction']['percent']==(40 if fail else 99.9)


def test_student_progress_uses_causal_sampling_not_teacher_clip_schedule():
    p = {'state': {'status': 'running', 'phase': 'infer'},
         'plan': {'runtime': {'inference_model': '14b', 'artifixer_plus_pass': False}},
         'logs': {'inference': 'SPLATFIX_STUDENT_MEMORY: {"phase":"cache_allocation"}\n'}}
    progress = summarize(p, observed_at=100)['stages']
    assert progress['inference']['label'] == '14B student inference'
    assert '4 steps per block' in progress['inference']['detail']
    assert progress['inference']['total'] is None
    assert progress['plus']['status'] == 'skipped'
    p['state']['phase'] = 'distill'
    assert summarize(p, observed_at=110)['stages']['inference']['percent'] == 100
    p['plan']['runtime']['artifixer_plus_pass'] = True
    p['state'].update(phase='finished', status='completed')
    progress = summarize(p, observed_at=120)['stages']
    assert progress['plus']['status'] == 'completed'
    assert progress['plus']['percent'] == 100
