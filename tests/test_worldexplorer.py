"""CPU contracts for remote independence, failure handling and shared assets."""
import json
from pathlib import Path
import shlex
import subprocess
import pytest
from splat_explorer.splatfix.worldexplorer import DEFAULT_PROMPT
from splat_explorer.splatfix.worldexplorer import service, worker, runner

ID = 'we_20261019_120000_abcdef123456'
CFG = dict(user='u',host='host',workspace='/dss/test',job_id='123',cpus=4,mem='64G',container='/dss/test/image.sqsh')


def job(tmp_path, status='submitted'):
    api = service.WorldExplorerService(tmp_path)
    root = tmp_path/ID
    root.mkdir()
    body = dict(id=ID,request=service.validate({}),remote='/dss/test/worldexplorer-runs/'+ID,
                connection=CFG,status=status)
    worker.atomic(root/'job.json',body)
    return api, root, body


@pytest.mark.parametrize('body', [{'mode':'invalid'},{'prompt':''},{'translation_scaling':float('nan')},
                                  {'translation_scaling':0},{'num_images_for_vggt':101}, {'prompt':'a\nb'}])
def test_invalid_request(body):
    with pytest.raises(ValueError): service.validate(body)


def test_prompt_is_literal_argument_and_safe_path():
    prompt = '--help $(touch /tmp/nope); / " `x` 🌌'
    request = service.validate({'prompt':prompt})
    argv = runner.generation_args(request)
    assert argv[-2:] == ['--',prompt]
    assert request['mode'] == 'automatic'
    assert '/' not in runner.safe_theme(prompt)
    assert len(runner.safe_theme('x'*2000)) < 80
    assert service.validate({})['prompt'] == DEFAULT_PROMPT


def test_detach_command_uses_reserved_gpu_and_no_prompt():
    command = service.launch_command(CFG,ID)
    assert 'nohup' in command and '< /dev/null &' in command
    assert '--jobid=123' in command and '--overlap' in command and '--gres=gpu:1' in command
    assert 'control/worker.py' in command and 'exit.json' in command
    assert 'flock -n' in command
    assert 'sbatch' not in command and DEFAULT_PROMPT not in command
    subprocess.run(['bash','-n'],input=command,text=True,check=True)


def test_disconnect_does_not_fail_remote_job(tmp_path,monkeypatch):
    api, root, body = job(tmp_path)
    def offline(*args): raise RuntimeError('offline')
    monkeypatch.setattr(api,'_ssh',offline)
    result = api.refresh(ID)
    assert result['status'] == 'submitted'
    assert result['connection_error'] == 'offline'
    assert service.WorldExplorerService(tmp_path).jobs()[0]['id'] == ID


def test_launcher_failure_overrides_stale_running_state(tmp_path,monkeypatch):
    api, root, body = job(tmp_path)
    monkeypatch.setattr(api,'_ssh',lambda *a: json.dumps({'status.json':{'status':'running'},'exit.json':{'returncode':137}}))
    assert api.refresh(ID)['status'] == 'failed'


def test_missing_launcher_exit_is_interrupted(tmp_path,monkeypatch):
    api, root, body = job(tmp_path)
    monkeypatch.setattr(api,'_ssh',lambda *a: json.dumps({'alive':False}))
    assert api.refresh(ID)['status'] == 'interrupted'


def test_shared_authenticated_cache_and_isolation(tmp_path,monkeypatch):
    import pwd
    monkeypatch.setenv('PYTHONPATH','oracle-only-packages')
    env = worker.model_environment(tmp_path)
    assert env['HF_HOME'] == str(Path(pwd.getpwuid(__import__('os').getuid()).pw_dir)/'.cache/huggingface')
    assert env['HF_TOKEN_PATH'] == env['HF_HOME']+'/token'
    assert 'PYTHONPATH' not in env
    assert str(tmp_path/'worldexplorer-env/bin') in env['PATH']


def test_file_path_cannot_escape(tmp_path):
    api, root, body = job(tmp_path)
    results = root/'results'; results.mkdir()
    (results/'safe.ply').write_text('ply')
    (results/'escape').symlink_to(root/'job.json')
    assert api.file(ID,'safe.ply').name == 'safe.ply'
    for relative in ('../job.json','escape','/etc/passwd'):
        with pytest.raises(ValueError): api.file(ID,relative)
    with pytest.raises(ValueError): api.directory('../')


def test_inventory_and_verified_download(tmp_path,monkeypatch):
    import hashlib
    api, root, body = job(tmp_path,'completed')
    results = root/'results'
    ply = results/'nerfstudio_output/scene/splatfacto-big/run/exports/splat/splat.ply'
    ply.parent.mkdir(parents=True); ply.write_bytes(b'ply\n'+b'x'*200)
    files = worker.inventory(results)
    worker.atomic(results/'artifacts.json',{'files':files})
    worker.atomic(results/'.archive-manifest.json',{'files':files})
    digest=hashlib.sha256((results/'.archive-manifest.json').read_bytes()).hexdigest()
    monkeypatch.setattr(api,'refresh',lambda _:dict(body))
    calls=[]; commands=[]
    monkeypatch.setattr(service.lrz,'_mux_run',lambda args:calls.append(args))
    def ssh(cfg,command,**kwargs): commands.append(command); return digest
    monkeypatch.setattr(api,'_ssh',ssh)
    result=api.download(ID)
    assert result['downloaded_at'] and result['remote_cleaned_at']
    assert '--exclude=/model' in calls[0]
    assert 'shutil.rmtree' in commands[-1]
    commands.clear()
    ply.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='verification failed'): api.download(ID)
    assert all('shutil.rmtree' not in c for c in commands)


def test_worker_missing_export_fails(tmp_path,monkeypatch):
    run=tmp_path/'run'; run.mkdir()
    worker.atomic(run/'request.json',service.validate({}))
    worker.atomic(tmp_path/'worldexplorer-ready.json',{})
    repo=tmp_path/'third_party/worldexplorer'; repo.mkdir(parents=True)
    (repo/'worldexplorer.py').write_text('')
    monkeypatch.setattr(worker.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr('sys.argv',['worker','--run-dir',str(run),'--base',str(tmp_path)])
    assert worker.main()==1
    state=json.loads((run/'status.json').read_text())
    assert state['status']=='failed' and 'without an exported splat' in state['error']


def test_no_scheduler_calls_on_refresh(tmp_path,monkeypatch):
    api,root,body=job(tmp_path)
    commands=[]
    def ssh(cfg,command): commands.append(command); return '{}'
    monkeypatch.setattr(api,'_ssh',ssh)
    api.refresh(ID)
    assert all('squeue' not in c and 'sinfo' not in c for c in commands)


def test_submission_timeout_is_recoverable_without_relaunch(tmp_path,monkeypatch):
    api=service.WorldExplorerService(tmp_path)
    monkeypatch.setattr(api,'request_reconcile',lambda **kwargs:None)
    monkeypatch.setattr(service.lrz,'load_lrz_config',lambda:CFG)
    monkeypatch.setattr(service.lrz,'lrz_session_alive',lambda _:True)
    monkeypatch.setattr(service.lrz,'_mux_run',lambda _:None)
    def ssh(cfg,command):
        if command.startswith('squeue'): return 'R 09:00:00'
        if command.startswith('nohup'): raise RuntimeError('SSH timed out')
        return ''
    monkeypatch.setattr(api,'_ssh',ssh)
    result=api.create({})
    assert result['status']=='launch_unknown'
    assert api.jobs()[0]['id']==result['id']


def test_http_routes_and_streamed_artifact(tmp_path):
    from io import BytesIO
    from types import SimpleNamespace
    from splat_explorer.web.server import DashboardHandler
    api, root, body = job(tmp_path)
    target = root/'results/splat.ply'; target.parent.mkdir(); target.write_bytes(b'ply\n')
    handler = object.__new__(DashboardHandler)
    handler.app = SimpleNamespace(worldexplorer=api)
    handler.command = 'GET'
    messages=[]
    handler._send_json = lambda *args: messages.append(args)
    handler._serve_splatfix('/api/splatfix/worldexplorer')
    assert messages[0][0]['jobs'][0]['id'] == ID
    handler.path = f'/api/splatfix/worldexplorer/file?id={ID}&path=splat.ply'
    handler.send_response = lambda _:None
    handler.send_header = lambda *a:None
    handler.end_headers = lambda:None
    handler.wfile = BytesIO()
    handler._serve_splatfix('/api/splatfix/worldexplorer/file')
    assert handler.wfile.getvalue() == b'ply\n'


def test_short_allocation_rejected_before_upload(tmp_path,monkeypatch):
    api=service.WorldExplorerService(tmp_path)
    monkeypatch.setattr(api,'request_reconcile',lambda **kwargs:None)
    monkeypatch.setattr(service.lrz,'load_lrz_config',lambda:CFG)
    monkeypatch.setattr(service.lrz,'lrz_session_alive',lambda _:True)
    monkeypatch.setattr(api,'_ssh',lambda *a:'R 00:30:00')
    with pytest.raises(RuntimeError,match='at least 8 hours'): api.create({})
    assert not api.jobs()


def manual_job(tmp_path):
    api, root, body = job(tmp_path,'awaiting_selection')
    body['request']['mode']='manual'
    results=root/'results'
    scaffold=results/'panoramas/theme/20261019'
    (scaffold/'final').mkdir(parents=True)
    for i in (0,2,4,6): (scaffold/'final'/f'{i:03}.png').write_bytes(f'original-{i}'.encode())
    for i in (1,3,5,7):
        for seed in (0,42): (scaffold/f'{i:03}_blend_7.0_{seed}.png').write_bytes(f'{i}-{seed}'.encode())
    body['candidates']=worker.manual_candidates(results)
    worker.atomic(results/'artifacts.json',{'files':worker.inventory(results)})
    worker.atomic(root/'job.json',body)
    return api,root,body


def test_manual_scaffold_preserves_four_originals_and_groups(tmp_path):
    api,root,body=manual_job(tmp_path)
    assert set(body['candidates']['originals'])=={'000','002','004','006'}
    assert set(body['candidates']['groups'])=={'001','003','005','007'}
    assert all(len(v)==2 for v in body['candidates']['groups'].values())
    args=runner.generation_args(body['request'])
    assert args[:4]==['worldexplorer.py','scaffold','--mode','manual']
    assert '--translation-scaling' not in args


def test_manual_selection_submits_eight_verified_images(tmp_path,monkeypatch):
    api,root,body=manual_job(tmp_path)
    chosen={k:v[1] for k,v in body['candidates']['groups'].items()}
    received={}
    def create(request,**kwargs): received.update(request=request,**kwargs); return {'id':'new-run'}
    monkeypatch.setattr(api,'create',create)
    assert api.expand(ID,chosen)['id']=='new-run'
    assert received['request']['operation']=='expand'
    assert len(received['scaffold'])==8
    assert received['selection']['source_run']==ID
    assert received['scaffold']['000'].read_bytes()==b'original-0'
    assert received['scaffold']['001'].read_bytes()==b'1-42'
    assert runner.generation_args(received['request'])[1:3]==['expand','input_scaffold']


def test_manual_selection_rejects_wrong_group_and_mutation(tmp_path,monkeypatch):
    api,root,body=manual_job(tmp_path)
    chosen={k:v[0] for k,v in body['candidates']['groups'].items()}
    with pytest.raises(ValueError,match='exactly one'): api.expand(ID,{'001':chosen['001']})
    with pytest.raises(ValueError,match='not a candidate'): api.expand(ID,{**chosen,'001':chosen['003']})
    (root/'results'/chosen['001']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed since generation'): api.expand(ID,chosen)


def test_manual_wait_is_not_failed_launcher(tmp_path,monkeypatch):
    api,root,body=job(tmp_path)
    monkeypatch.setattr(api,'_ssh',lambda *a:json.dumps({'status.json':{'status':'awaiting_selection'},'exit.json':{'returncode':0},'alive':False}))
    assert api.refresh(ID)['status']=='awaiting_selection'


def test_remote_archive_removes_only_finished_job_tree(tmp_path):
    import sys
    from splat_explorer.splatfix.worldexplorer.archive import MANIFEST_SCRIPT, DELETE_SCRIPT
    shared=tmp_path/'shared-models'; shared.mkdir(); (shared/'weights').write_bytes(b'keep')
    run=tmp_path/'worldexplorer-runs'/ID; run.mkdir(parents=True)
    (run/'model').symlink_to(shared,target_is_directory=True)
    (run/'output.ply').write_bytes(b'generated')
    result=subprocess.run([sys.executable,'-c',MANIFEST_SCRIPT,str(run)],capture_output=True,text=True)
    assert result.returncode and 'exit is unconfirmed' in result.stderr
    worker.atomic(run/'exit.json',{'returncode':0})
    digest=subprocess.check_output([sys.executable,'-c',MANIFEST_SCRIPT,str(run)],text=True).strip()
    manifest=json.loads((run/'.archive-manifest.json').read_text())
    assert all(not r['path'].startswith('model') for r in manifest['files'])
    (run/'output.ply').write_bytes(b'changed-after-archive')
    result=subprocess.run([sys.executable,'-c',DELETE_SCRIPT,str(run),digest],capture_output=True,text=True)
    assert result.returncode and run.exists()
    digest=subprocess.check_output([sys.executable,'-c',MANIFEST_SCRIPT,str(run)],text=True).strip()
    subprocess.run([sys.executable,'-c',DELETE_SCRIPT,str(run),digest],check=True,capture_output=True)
    assert not run.exists() and (shared/'weights').read_bytes()==b'keep'


def test_auto_reconcile_after_restart(tmp_path,monkeypatch):
    api,root,body=job(tmp_path)
    seen=[]
    monkeypatch.setattr(api,'refresh',lambda _:dict(body,status='completed'))
    monkeypatch.setattr(api,'download',lambda run:seen.append(run))
    api.reconcile()
    assert seen==[ID]
    worker.atomic(root/'job.json',dict(body,remote_cleaned_at=1))
    api.reconcile()
    assert seen==[ID]


def test_cleanup_reply_loss_recovers_from_verified_archive(tmp_path,monkeypatch):
    api,root,body=job(tmp_path,'completed')
    worker.atomic(root/'job.json',dict(body,downloaded_at=1,cleanup_requested_at=2))
    monkeypatch.setattr(api,'_ssh',lambda *a:json.dumps({'missing':True}))
    result=api.refresh(ID)
    assert result['remote_cleaned_at'] and result['status']=='completed'
