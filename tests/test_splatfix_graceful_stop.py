import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from splat_explorer.splatfix import graceful_worker as graceful
from splat_explorer.splatfix.repair import run_worker


def test_hook_saves_once_at_boundary_only_when_stopped(tmp_path):
    calls = []
    class Trainer:
        def render_gui(self, updated):
            calls.append(('normal', updated))
    control = tmp_path / 'stop'
    graceful.install_hook(Trainer, control, tmp_path / 'snapshot', saver=lambda t, d: calls.append(('saved', d)))
    trainer = Trainer()
    trainer.render_gui(True)
    assert calls == [('normal', True)]
    control.touch()
    with pytest.raises(SystemExit) as exc:
        trainer.render_gui(False)
    assert exc.value.code == 75
    assert calls == [('normal', True), ('saved', tmp_path / 'snapshot')]


def test_stop_waits_for_real_subprocess_save_and_keeps_ply(tmp_path):
    package = tmp_path / 'threedgrut'
    (package / 'export').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'export/__init__.py').write_text('')
    (tmp_path / 'torch.py').write_text('from types import SimpleNamespace\ncuda=SimpleNamespace(synchronize=lambda: None)\n')
    (package / 'export/ply_exporter.py').write_text('class PLYExporter:\n def export(self, model, path):\n  path.write_text("ply\\nintermediate model\\n")\n')
    (package / 'trainer.py').write_text('''from pathlib import Path
from types import SimpleNamespace
class Trainer3DGRUT:
 def __init__(self):
  self.global_step=0
  self.model=object()
  self.tracking=SimpleNamespace(output_dir='training',writer=None)
 def render_gui(self, updated): pass
 def save_checkpoint(self):
  p=Path(self.tracking.output_dir)/f'ours_{self.global_step}'/f'ckpt_{self.global_step}.pt'
  p.parent.mkdir(parents=True,exist_ok=True)
  p.write_bytes(b'current model')
''')
    (tmp_path / 'train.py').write_text('''from threedgrut.trainer import Trainer3DGRUT
from pathlib import Path
import time,signal
signal.signal(signal.SIGTERM,lambda *a: Path('forced-termination').touch())
t=Trainer3DGRUT()
while True:
 t.global_step+=1
 Path('ready').touch()
 t.render_gui(True)
 time.sleep(.02)
''')
    with pytest.raises(InterruptedError, match='preserved'):
        run_worker([sys.executable, str(tmp_path / 'train.py')], cwd=tmp_path,
                   env={**os.environ, 'SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS': '5'},
                   log_path=tmp_path / 'artifixer3d.log', should_stop=lambda: (tmp_path / 'ready').exists())
    status=json.loads((tmp_path / 'artifixer3d.interrupted/stop-state.json').read_text())
    assert status['status']=='saved' and status['step']>0
    assert Path(status['checkpoint']).read_bytes()==b'current model'
    assert Path(status['splat_path']).read_text().startswith('ply')
    assert not (tmp_path / 'forced-termination').exists()


def test_export_failure_preserves_checkpoint_and_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    class Exporter:
        def export(self, *args):raise RuntimeError('export failed')
    monkeypatch.setitem(sys.modules, 'threedgrut.export.ply_exporter', SimpleNamespace(PLYExporter=Exporter))
    checkpoint=tmp_path/'ours_17/ckpt_17.pt'
    def save():
        checkpoint.parent.mkdir();checkpoint.write_bytes(b'weights')
    trainer=SimpleNamespace(global_step=17,tracking=SimpleNamespace(output_dir=tmp_path,writer=None),save_checkpoint=save,model=None)
    status=graceful.save_state(trainer,tmp_path/'snapshot')
    assert status['status']=='save_failed'
    assert status['checkpoint']==str(checkpoint)
    assert checkpoint.exists() and status['splat_path'] is None


def test_remote_worker_stops_at_deadline_without_local_manager(tmp_path, monkeypatch):
    from splat_explorer.splatfix import job_worker
    from splat_explorer.splatfix.artifixer import worker as artifixer_worker
    (tmp_path/'worker-request.json').write_text(json.dumps(dict(checkpoint='saved', output=str(tmp_path/'out'),
        mode='baseline',frames=25,span_fraction=.04,runtime={},stop_at_epoch=100)))
    now=[99]
    monkeypatch.setattr(artifixer_worker.time,'time',lambda:now[0])
    def run(*args,**kwargs):
        assert not kwargs['should_stop']()
        now[0]=100
        assert kwargs['should_stop']()
        raise InterruptedError('deadline save completed')
    monkeypatch.setattr(artifixer_worker,'run_repair',run)
    job_worker.execute(tmp_path)
    assert (tmp_path/'STOP').exists()
    assert json.loads((tmp_path/'worker-status.json').read_text())['status']=='stopped'


def test_manager_gpu_deadline_is_five_minutes_early_and_survives_reattach(tmp_path):
    from splat_explorer.scene_runs.manager import SceneRunManager
    from splat_explorer.scene_runs.store import SceneRunStore
    from splat_explorer.config import Config
    from datetime import datetime
    store=SceneRunStore(tmp_path/'scene-runs')
    run=store.create_run({'scene_id':'bicycle','duration_seconds':86400})
    manager=SceneRunManager(Config({'output':{'dir':str(tmp_path)}}),store=store)
    manager._apply_gpu_deadline(run.run_id,{'expected_end':'2026-10-06T17:05:56+02:00'})
    expected=datetime.fromisoformat('2026-10-06T17:00:56+02:00').timestamp()
    assert store.get_run(run.run_id).state.details['effective_deadline']==expected
    manager._apply_gpu_deadline(run.run_id,{'ready':True})
    assert store.get_run(run.run_id).state.details['effective_deadline']==expected


def test_hung_training_gets_bounded_graceful_timeout(tmp_path, monkeypatch):
    from splat_explorer.splatfix import repair
    killed=[]
    class Process:
        pid=123
        returncode=None
        def poll(self):return None
        def wait(self,timeout):return 0
    monkeypatch.setattr(repair.subprocess,'Popen',lambda *a,**kw:Process())
    monkeypatch.setattr(repair.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    clock=[0]
    def monotonic():clock[0]+=1;return clock[0]
    monkeypatch.setattr(repair.time,'monotonic',monotonic)
    monkeypatch.setattr(repair.time,'sleep',lambda t:None)
    polls=[0]
    def stop():polls[0]+=1;return polls[0]>1
    with pytest.raises(InterruptedError,match='timed out'):
        repair.run_worker(['python','-m','train'],cwd=tmp_path,
            env={'SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS':'2'},log_path=tmp_path/'artifixer3d.log',should_stop=stop)
    assert (tmp_path/'artifixer3d.save-stop').exists()
    assert json.loads((tmp_path/'artifixer3d.stop-timeout.json').read_text())['status']=='save_timeout'
    assert killed


def test_interruption_publishes_actual_saved_model_as_incomplete(tmp_path):
    from splat_explorer.splatfix.interrupted import preserve_interrupted_results
    root=tmp_path/'results/benchmark_test';snapshot=root/'artifixer3d.interrupted';snapshot.mkdir(parents=True)
    ply=snapshot/'interrupted.ply';ply.write_text('ply')
    checkpoint=snapshot/'weights.pt';checkpoint.write_bytes(b'weights')
    (snapshot/'stop-state.json').write_text(json.dumps(dict(status='saved',step=42,splat_path=str(ply),checkpoint=str(checkpoint))))
    preserve_interrupted_results({'output':str(root.parent),'runtime':{'model_variant':'1.3b'}},'deadline')
    result=json.loads((root/'partial-result.json').read_text())
    assert result['incomplete'] and result['splat_path']==str(ply)
    assert result['interruption']['last_logged_training_step']==42
    assert not (root/'result.json').exists()


def test_manager_deadline_falls_back_to_observed_time_left(tmp_path):
    from splat_explorer.scene_runs.manager import SceneRunManager
    from splat_explorer.scene_runs.store import SceneRunStore
    from splat_explorer.config import Config
    store=SceneRunStore(tmp_path/'scene-runs')
    run=store.create_run({'scene_id':'bicycle','duration_seconds':86400})
    manager=SceneRunManager(Config({'output':{'dir':str(tmp_path)}}),store=store)
    manager._apply_gpu_deadline(run.run_id,{'expected_end':'Unknown','time_left':'00:10:00','observed_at_epoch':1000})
    assert store.get_run(run.run_id).state.details['effective_deadline']==1300
