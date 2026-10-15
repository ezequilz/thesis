import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from splat_explorer.splatfix.g4splat import graceful


def test_iteration_boundary_saves_only_on_stop(tmp_path, monkeypatch):
    control = tmp_path / 'SAVE_STOP'
    monkeypatch.setenv('SPLATFIX_G4_STOP', str(control))
    calls = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: calls.append('sync')),
        save=lambda data, path: Path(path).write_text(str(data))))
    model = SimpleNamespace(save_ply=lambda path: Path(path).write_text('native'), capture=lambda: 'weights')
    graceful.iteration_boundary(model, 1)
    assert not calls
    control.touch()
    with pytest.raises(SystemExit) as exc:
        graceful.iteration_boundary(model, 42)
    assert exc.value.code == 75
    state = json.loads((tmp_path / 'stop-state.json').read_text())
    assert state['status'] == 'saved' and state['step'] == 42
    assert Path(state['native_splat_path']).read_text() == 'native'
    assert Path(state['checkpoint']).exists()


def test_failed_checkpoint_keeps_native(tmp_path, monkeypatch):
    monkeypatch.setenv('SPLATFIX_G4_STOP', str(tmp_path / 'SAVE_STOP'))
    (tmp_path / 'SAVE_STOP').touch()
    def fail(*args):
        raise RuntimeError('checkpoint unavailable')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None), save=fail))
    model = SimpleNamespace(save_ply=lambda path: Path(path).write_text('native'), capture=fail)
    with pytest.raises(SystemExit):
        graceful.iteration_boundary(model, 3)
    state = json.loads((tmp_path / 'stop-state.json').read_text())
    assert state['status'] == 'saved' and state['checkpoint_error']
    assert Path(state['native_splat_path']).exists()


def test_stop_waits_for_training_and_bounds_hung_save(tmp_path, monkeypatch):
    now = [0]
    monkeypatch.setattr(graceful.time, 'monotonic', lambda: now[0])
    events = []
    active = tmp_path / 'training-active'
    active.touch()
    poll = graceful.stop_poll(tmp_path, lambda: True, events.append, timeout=10)
    assert not poll() and (tmp_path / 'SAVE_STOP').exists()
    now[0] = 11
    assert poll()
    assert json.loads((tmp_path / 'stop-timeout.json').read_text())['status'] == 'save_timeout'
    assert len(events) == 1
    active.unlink()
    assert graceful.stop_poll(tmp_path, lambda: True, events.append)()


@pytest.mark.parametrize('snapshot', ['on_stop', 'last_saved', 'unavailable'])
def test_partial_publication(tmp_path, monkeypatch, snapshot):
    from splat_explorer.splatfix.g4splat import official_runner
    monkeypatch.setattr(official_runner, 'export_viewer_ply', lambda source, target: Path(target).write_text('viewer'))
    if snapshot != 'unavailable':
        native = tmp_path / ('interrupted-native.ply' if snapshot == 'on_stop' else
                             'official/free_gaussians/point_cloud/iteration_7000/point_cloud.ply')
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_text('native')
    result = graceful.preserve_partial(tmp_path, {'mode': 'baseline'}, 'Stopped')
    assert result['incomplete'] and result['interruption']['snapshot'] == snapshot
    assert bool(result['splat_path']) == (snapshot != 'unavailable')
    assert (tmp_path / 'partial-result.json').exists()
    assert not (tmp_path / 'result.json').exists()


def test_patch_rejects_unknown_trainer_without_mutation(tmp_path):
    trainer = tmp_path / '2d-gaussian-splatting/train_with_refine_depth.py'
    trainer.parent.mkdir()
    trainer.write_text('unknown')
    with pytest.raises(ValueError, match='Unexpected pinned'):
        graceful.patch_runtime(tmp_path)
    assert trainer.read_text() == 'unknown'


def test_patch_detaches_overlay_and_runs_after_optimizer(tmp_path, monkeypatch):
    import hashlib
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    source = '''def training():
    for iteration in range(1, 4):
        gaussians.optimizer.step()
        with torch.no_grad():        
            if network_gui.conn == None:
                pass
'''
    trainer = upstream / 'train_with_refine_depth.py'
    trainer.write_text(source)
    overlay = tmp_path / 'overlay'
    overlay.mkdir()
    (overlay / '2d-gaussian-splatting').symlink_to(upstream, target_is_directory=True)
    monkeypatch.setattr(graceful, 'TRAINER_SHA256', hashlib.sha256(source.encode()).hexdigest())
    graceful.patch_runtime(overlay)
    patched = (overlay / '2d-gaussian-splatting/train_with_refine_depth.py').read_text()
    assert patched.index('optimizer.step()') < patched.index('iteration_boundary(gaussians, iteration)')
    assert trainer.read_text() == source
    assert not (upstream / 'splatfix_graceful.py').exists()


def test_real_subprocess_saves_before_group_termination(tmp_path):
    from splat_explorer.splatfix.executor import run_process
    root = str(Path(graceful.__file__).parent)
    (tmp_path / 'torch.py').write_text('from types import SimpleNamespace\ncuda=SimpleNamespace(synchronize=lambda: None)\ndef save(data,path): path.write_text("weights")\n')
    script = tmp_path / 'trainer.py'
    script.write_text('''import os, sys, signal, time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from graceful import iteration_boundary
root = Path(sys.argv[2])
os.environ['SPLATFIX_G4_STOP'] = str(root / 'SAVE_STOP')
signal.signal(signal.SIGTERM, lambda *args: (root / 'killed').touch())
model = SimpleNamespace(save_ply=lambda path: Path(path).write_text('native'), capture=lambda: 'weights')
(root / 'training-active').touch()
(root / 'ready').touch()
try:
    for step in range(1000):
        iteration_boundary(model, step)
        time.sleep(.01)
finally:
    (root / 'training-active').unlink()
''')
    poll = graceful.stop_poll(tmp_path, lambda: (tmp_path / 'ready').exists(), lambda event: None, timeout=5)
    with pytest.raises((RuntimeError, InterruptedError)):
        run_process([sys.executable, str(script), root, str(tmp_path)], tmp_path / 'log', poll)
    state = json.loads((tmp_path / 'stop-state.json').read_text())
    assert state['status'] == 'saved' and Path(state['checkpoint']).exists()
    assert not (tmp_path / 'killed').exists()


def test_pipeline_converts_cooperative_exit_to_partial_result(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat import repair
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'inputs.json').write_text(json.dumps({'views': [{}, {}], 'mode': 'baseline'}))
    monkeypatch.setattr(repair, 'validate_runtime', lambda rt: (tmp_path, Path(sys.executable)))
    stopped = [False]
    def run(command, log, poll, *args):
        if Path(log).name == 'preflight.log':
            return
        stopped[0] = True
        assert poll()
        raise RuntimeError('Stage exited with code 75')
    monkeypatch.setattr(repair, 'run_process', run)
    output = tmp_path / 'output'
    with pytest.raises(InterruptedError, match='preserved'):
        repair.run_prepared(inputs, output, should_stop=lambda: stopped[0])
    result = json.loads((output / 'partial-result.json').read_text())
    assert result['incomplete'] and result['interruption']['snapshot'] == 'unavailable'
    assert not (output / 'result.json').exists()


def test_stopped_remote_downloads_before_finishing(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat.execution import execute_remote
    from splat_explorer import repair_lrz as lrz
    details = {'remote_dir': '/dss/splatfix-jobs/run'}
    actions = []
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss', 'user': 'user', 'host': 'host'})
    def ssh(cfg, command, timeout):
        actions.append(command)
        return SimpleNamespace(returncode=0, stderr='', stdout='launched' if 'printf launched' in command else
                               json.dumps({'status': 'stopped', 'phase': 'finished'}))
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: actions.append('download'))
    store = SimpleNamespace(get_run=lambda _: SimpleNamespace(state=SimpleNamespace(details=details)))
    def update(**fields):
        if fields.get('remote_finished'):
            assert actions[-1] == 'download'
            actions.append('finished')
    with pytest.raises(InterruptedError, match='downloaded'):
        execute_remote(SimpleNamespace(store=store, cfg={}), 'run', {}, tmp_path, lambda: True, update)
    assert any(command.startswith('touch ') for command in actions)
    assert actions[-1] == 'finished'


def test_partial_remote_paths_exposed_in_results(tmp_path):
    from test_splatfix_results import setup
    studio, run, output = setup(tmp_path)
    (output / 'result.json').unlink()
    (output / 'interrupted-viewer.ply').write_text('ply')
    (output / 'interrupted-native.ply').write_text('ply')
    prefix = '/workspace/splatfix-jobs/run/results/' + output.name
    (output / 'partial-result.json').write_text(json.dumps({
        'incomplete': True, 'reconstruction_method': 'g4splat',
        'splat_path': prefix + '/interrupted-viewer.ply',
        'native_splat_path': prefix + '/interrupted-native.ply'}))
    row = studio.results.entries()[0]
    assert row['incomplete'] and row['ply_url'] and row['native_ply_url']


def test_runner_does_not_launch_more_stages_after_save(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat import official_runner as runner
    repo = tmp_path / 'repo'
    repo.mkdir()
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    (inputs / 'inputs.json').write_text('{}')
    output = tmp_path / 'result/official'
    output.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', ['runner', str(repo), str(inputs), str(output), '2'])
    monkeypatch.setattr(runner, 'prepare_runtime', lambda *args: repo)
    monkeypatch.setattr(runner.graceful, 'patch_runtime', lambda *args: None)
    monkeypatch.setenv('SPLATFIX_G4_STOP', 'previous-value')
    def system(command):
        assert (output.parent / 'training-active').exists()
        assert os.environ['SPLATFIX_G4_STOP'] == str(output.parent / 'SAVE_STOP')
        (output.parent / 'SAVE_STOP').touch()
        return 256
    monkeypatch.setattr(runner.os, 'system', system)
    def train(*args, **kwargs):
        runner.os.system('python scripts/refine_free_gaussians.py')
        pytest.fail('later stages must not run after stop')
    monkeypatch.setattr(runner.runpy, 'run_path', train)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 75
    assert not (output.parent / 'training-active').exists()
    assert os.environ['SPLATFIX_G4_STOP'] == 'previous-value'
