import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from splat_explorer.splatfix.oraclegs import graceful
from splat_explorer.splatfix.oraclegs.partial import salvage


def test_stop_saves_current_iteration_before_any_scheduled_snapshot(tmp_path,monkeypatch):
    events=[]
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('sync'))))
    def save(path):
        events.append('save')
        Path(path).write_bytes(b'ply\ncurrent gaussian state')
    scene=SimpleNamespace(gaussians=SimpleNamespace(save_ply=save))
    control=tmp_path/'SAVE_STOP'
    graceful.iteration_boundary(scene,17,control)
    assert not events
    control.touch()
    with pytest.raises(SystemExit) as exc:
        graceful.iteration_boundary(scene,18,control)
    assert exc.value.code==0
    assert events==['sync','save']
    result=salvage(tmp_path,'User stop')
    assert result['saved_iteration']==18 and result['saved_on_stop']
    assert Path(result['splat_path']).read_bytes()==b'ply\ncurrent gaussian state'


def test_failed_stop_write_preserves_previous_snapshot(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None)))
    prior=tmp_path/'model/point_cloud/iteration_3000/point_cloud.ply'
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'ply\nprevious')
    (tmp_path/'model/saved-snapshot.json').write_text(json.dumps({'iteration':3000,'size_bytes':prior.stat().st_size}))
    def fail(path):
        Path(path).write_bytes(b'partial')
        raise OSError('disk full')
    control=tmp_path/'SAVE_STOP'; control.touch()
    with pytest.raises(OSError):
        graceful.iteration_boundary(SimpleNamespace(gaussians=SimpleNamespace(save_ply=fail)),3210,control)
    assert not (tmp_path/'stop-snapshot.ply').exists()
    result=salvage(tmp_path,'Save failed')
    assert result['saved_iteration']==3000 and not result['saved_on_stop']


def test_parent_defers_termination_and_times_out(tmp_path,monkeypatch):
    now=[10.]; stop=[False]; events=[]
    monkeypatch.setattr(graceful.time,'monotonic',lambda:now[0])
    poll=graceful.stop_poll(tmp_path,lambda:stop[0],events.append,timeout=240)
    assert not poll() and not (tmp_path/'SAVE_STOP').exists()
    stop[0]=True
    assert not poll() and (tmp_path/'SAVE_STOP').exists()
    assert len(events)==1
    now[0]=249.
    assert not poll()
    now[0]=250.
    assert poll() and (tmp_path/'stop-timeout.json').exists()


def test_hook_runs_after_updates_and_does_not_modify_source(tmp_path,monkeypatch):
    source=b'''events = []
scene = object()
def training():
    for iteration in range(1, 4):
        events.append(("update", iteration))
training()
'''
    (tmp_path/'train.py').write_bytes(source)
    (tmp_path/'upstream-lock.json').write_text(json.dumps({'oraclegs':{'files':{'train.py':hashlib.sha256(source).hexdigest()}}}))
    monkeypatch.setattr(graceful,'__file__',str(tmp_path/'graceful.py'))
    scope={}
    scope['_splatfix_stop_boundary']=lambda scene,i:scope['events'].append(('boundary',i))
    exec(graceful.training_code(tmp_path),scope)
    assert scope['events']==[(kind,i) for i in range(1,4) for kind in ('update','boundary')]
    assert (tmp_path/'train.py').read_bytes()==source
    (tmp_path/'train.py').write_bytes(source+b'\n')
    with pytest.raises(ValueError,match='Unexpected OracleGS'):
        graceful.training_code(tmp_path)


def test_actual_pinned_trainer_accepts_hook():
    assert graceful.training_code(Path(graceful.__file__).parent/'upstream/oraclegs')
