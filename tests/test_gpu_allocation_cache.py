from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest
from splat_explorer.scene_runs import allocation_cache as cache


CFG = dict(job_id='123', host='gpu.example', user='user', workspace='/work')


def allocation():
    return dict(job_id='123', state='R', expected_end='2036-10-06T17:00:00', node='a100', partition='gpu', mem='64G')


def test_cache_is_durable_job_bound_and_expires(tmp_path):
    path=tmp_path/'allocation.json'
    saved=cache.save_loaded_allocation(CFG,allocation(),path=path)
    expected=datetime.fromisoformat('2036-10-06T17:00:00+02:00').timestamp()
    assert saved['deadline_epoch']==expected
    assert cache.loaded_allocation(CFG,path=path,now=expected-1)['deadline_epoch']==expected
    for key in CFG:
        with pytest.raises(ValueError,match='Load GPU setup'):
            cache.loaded_allocation({**CFG,key:'different'},path=path,now=expected-1)
    with pytest.raises(ValueError,match='expired'):
        cache.loaded_allocation(CFG,path=path,now=expected)


def test_time_left_fallback_uses_setup_observation_not_each_read(tmp_path,monkeypatch):
    path=tmp_path/'allocation.json'
    monkeypatch.setattr(cache.time,'time',lambda:1000)
    cache.save_loaded_allocation(CFG,{**allocation(),'expected_end':'Unknown','time_left':'00:20:00','observed_at_epoch':900},path=path)
    assert cache.loaded_allocation(CFG,path=path,now=1100)['deadline_epoch']==2100
    assert cache.loaded_allocation(CFG,path=path,now=1500)['deadline_epoch']==2100


def test_setup_captures_deadline_from_its_one_probe(tmp_path,monkeypatch):
    from splat_explorer import repair_lrz as lrz
    path=tmp_path/'allocation.json'
    monkeypatch.setattr(cache,'cache_path',lambda:path)
    calls=[]
    def probe(cfg):
        calls.append('probe')
        lrz.remember_live_allocation(**allocation())
        return 'R'
    monkeypatch.setattr(lrz,'lrz_session_alive',lambda cfg:True)
    monkeypatch.setattr(lrz,'gpu_work_owner',lambda:'')
    monkeypatch.setattr(lrz,'probe_job',probe)
    monkeypatch.setattr(lrz,'_setup_lrz_gpu_locked',lambda *a,**kw:{'ok':True})
    result=lrz.setup_lrz_gpu(CFG)
    assert calls==['probe']
    assert result['allocation_deadline_epoch']==cache.loaded_allocation(CFG)['deadline_epoch']


def test_manager_never_probes_slurm_for_readiness_or_deadline(tmp_path,monkeypatch):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.manager import SceneRunManager
    from splat_explorer.config import Config
    path=tmp_path/'allocation.json'
    monkeypatch.setattr(cache,'cache_path',lambda:path)
    monkeypatch.setattr(lrz,'load_lrz_config',lambda:CFG)
    monkeypatch.setattr(lrz,'_ssh_run',lambda *a,**kw:pytest.fail('Must not query Slurm'))
    manager=SceneRunManager(Config({'output':{'dir':str(tmp_path)}}))
    assert not manager._gpu_ready()['ready']
    saved=cache.save_loaded_allocation(CFG,allocation())
    assert manager._gpu_ready(force=True)['deadline_epoch']==saved['deadline_epoch']
    # A new manager process reads the same local record.
    other=SceneRunManager(Config({'output':{'dir':str(tmp_path)}}))
    assert other._gpu_ready()['ready']
    monkeypatch.setattr(lrz,'load_lrz_config',lambda:{**CFG,'job_id':'456'})
    assert not other._gpu_ready()['ready']


def test_probe_job_collects_deadline_in_same_scheduler_query(monkeypatch):
    from splat_explorer import repair_lrz as lrz
    calls=[]
    monkeypatch.setattr(lrz,'lrz_session_alive',lambda cfg:True)
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0,stdout='R|64G|gpu|node|2036-10-06T17:00:00|1:00:00|6:00:00\n',stderr='')
    monkeypatch.setattr(lrz.subprocess,'run',run)
    lrz.probe_job(CFG)
    assert len(calls)==1 and '%e|%L|%l' in calls[0][-1]
    assert lrz.live_allocation('123')['expected_end']=='2036-10-06T17:00:00'


def test_launch_validation_uses_setup_cache_without_scheduler_query(tmp_path,monkeypatch):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    monkeypatch.setattr(cache,'cache_path',lambda:tmp_path/'allocation.json')
    saved=cache.save_loaded_allocation(CFG,allocation())
    transport=object.__new__(LrzSceneRunTransport)
    transport.cfg=CFG
    transport._require_session=lambda:None
    monkeypatch.setattr(lrz,'_ssh_run',lambda *a,**kw:pytest.fail('Unexpected scheduler query'))
    monkeypatch.setattr(lrz,'read_remote_setup_marker',lambda cfg:{'ok':True,'job_id':'123'})
    assert transport.validate()['allocation']['deadline_epoch']==saved['deadline_epoch']


def test_failed_setup_does_not_publish_deadline(tmp_path,monkeypatch):
    from splat_explorer import repair_lrz as lrz
    path=tmp_path/'allocation.json'
    monkeypatch.setattr(cache,'cache_path',lambda:path)
    monkeypatch.setattr(lrz,'lrz_session_alive',lambda cfg:True)
    monkeypatch.setattr(lrz,'gpu_work_owner',lambda:'')
    monkeypatch.setattr(lrz,'probe_job',lambda cfg:lrz.remember_live_allocation(**allocation()))
    def failed(*a,**kw):raise RuntimeError('setup failed')
    monkeypatch.setattr(lrz,'_setup_lrz_gpu_locked',failed)
    with pytest.raises(RuntimeError,match='setup failed'):lrz.setup_lrz_gpu(CFG)
    assert not path.exists()
