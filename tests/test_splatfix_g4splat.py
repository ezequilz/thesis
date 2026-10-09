"""CPU contract tests for official G4Splat integration."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation
from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.g4splat import backend, repair, worker
from splat_explorer.splatfix.g4splat.inputs import export_inputs
from splat_explorer.splatfix.jobs import validate_job


def checkpoint(tmp_path):
    cp = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=2,
                           metadata={'renderer': {'backend': 'viser'}})
    for x in (0., 2.):
        camera = Camera(np.array([x, 1., -3.]), Rotation.from_euler('xyz', [12, 34, 56], degrees=True).as_matrix(),
                        width=80, height=48, fov_deg=67.)
        view = cp.add_view(Image.new('RGB', (80, 48), 'red'), camera)
        image = cp.root / 'views' / view['id'] / 'repaired.png'
        Image.new('RGB', (80, 48), 'blue').save(image)
        view['repaired_rgb'] = str(image.relative_to(cp.root))
    cp.save()
    return cp


@pytest.mark.parametrize('mode,color', [('baseline', (255,0,0)), ('edited', (0,0,255))])
def test_export_projection_and_rgb(tmp_path, mode, color):
    cp = checkpoint(tmp_path)
    output = tmp_path / 'inputs'
    manifest = export_inputs(cp, output, mode)
    cameras = (output / 'sparse/0/cameras.txt').read_text().splitlines()
    poses = (output / 'sparse/0/images.txt').read_text().splitlines()
    assert len(poses) == 4 and poses[1] == '' and poses[3] == ''
    for index, view in enumerate(cp.views):
        parts = poses[index*2].split()
        q = np.array(parts[1:5], float)
        r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        t = np.array(parts[5:8], float)
        c2w = np.array(view['camera']['c2w'])
        point_camera = np.array([.2, -.1, 3.])
        world = c2w[:3,:3] @ point_camera + c2w[:3,3]
        np.testing.assert_allclose(r @ world + t, point_camera, atol=1e-6)
        fx, fy, cx, cy = map(float, cameras[index].split()[4:])
        k = np.array(view['camera']['intrinsics'])
        np.testing.assert_allclose([fx,fy,cx,cy], [k[0,0],k[1,1],k[0,2],k[1,2]])
        assert Image.open(output / manifest['views'][index]['image']).getpixel((0,0)) == color
    assert not manifest['source_splat_used_for_initialization']
    assert json.loads((output / 'split-2views.json').read_text()) == {'train': [0,1], 'test': []}


def test_method_options_and_readiness(tmp_path):
    job = validate_job({'stage': 'repair', 'checkpoint': 'saved', 'reconstruction_method': 'g4splat',
                        'frames': 0, 'regularization_profile': 'irrelevant'})
    assert 'frames' not in job and 'regularization_profile' not in job
    with pytest.raises(ValueError, match='does not support'):
        validate_job({'stage': 'benchmark', 'source': 'saved', 'reconstruction_method': 'g4splat'})
    cp = checkpoint(tmp_path)
    assert backend.readiness(cp)['ready']
    cp.views[1]['camera'] = cp.views[0]['camera']
    assert not backend.readiness(cp)['ready']


def test_wrong_image_dimensions_rejected(tmp_path):
    cp = checkpoint(tmp_path)
    Image.new('RGB', (20,20)).save(cp.image_path(cp.views[0], repaired=True))
    with pytest.raises(ValueError, match='dimensions'):
        export_inputs(cp, tmp_path / 'inputs', 'edited')


def test_worker_deadline_and_failure(tmp_path, monkeypatch):
    request = {'inputs': 'inputs', 'output': 'output', 'runtime': {}, 'stop_at_epoch': 1}
    (tmp_path / 'worker-request.json').write_text(json.dumps(request))
    monkeypatch.setattr(worker, 'run_prepared', lambda *a, **k: pytest.fail('expired worker executed'))
    worker.execute(tmp_path)
    assert json.loads((tmp_path / 'worker-status.json').read_text())['status'] == 'stopped'
    request['stop_at_epoch'] = None
    (tmp_path / 'worker-request.json').write_text(json.dumps(request))
    def fail(*a, **k):
        raise RuntimeError('missing weights')
    monkeypatch.setattr(worker, 'run_prepared', fail)
    worker.execute(tmp_path)
    assert json.loads((tmp_path / 'worker-status.json').read_text())['status'] == 'error'


def test_pipeline_requires_output_and_records_recipe(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path)
    export_inputs(cp, tmp_path / 'inputs', 'baseline')
    monkeypatch.setattr(repair, 'validate_runtime', lambda rt: (tmp_path, Path(sys.executable)))
    monkeypatch.setattr(repair, 'run_process', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='did not produce'):
        repair.run_prepared(tmp_path / 'inputs', tmp_path / 'result')
    assert not (tmp_path / 'result/result.json').exists()
    recipe = json.loads((tmp_path / 'result/recipe.json').read_text())
    assert recipe['generation_rounds'] == 3 and recipe['sfm_config'] == 'posed'
    assert recipe['revision'] == repair.REVISION


def test_native_conversion_preserves_surfel_parameters(tmp_path):
    from splat_explorer.splatfix.g4splat.official_runner import export_viewer_ply
    names = ['x','y','z','scale_0','scale_1','rot_0','rot_1','rot_2','rot_3','opacity','f_dc_0','f_dc_1','f_dc_2']
    vertex = np.zeros(2, dtype=[(name, '<f4') for name in names])
    vertex['scale_0'] = [-1,-2]
    vertex['scale_1'] = [-3,-1]
    vertex['rot_0'] = 1
    native = tmp_path / 'native.ply'
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex 2\n' + ''.join('property float ' + n + '\n' for n in names) + 'end_header\n'
    native.write_bytes(header.encode() + vertex.tobytes())
    original = native.read_bytes()
    export_viewer_ply(native, tmp_path / 'viewer.ply')
    from splat_explorer.scene.ply_loader import _parse_header, load_ply
    with (tmp_path / 'viewer.ply').open('rb') as stream:
        count, props, offset = _parse_header(stream)
        result = np.fromfile(stream, dtype=props, count=count)
    assert load_ply(tmp_path / 'viewer.ply').means.shape == (2,3)
    for name in names:
        np.testing.assert_array_equal(result[name], vertex[name])
    np.testing.assert_allclose(np.exp(result['scale_2']), np.exp([-3,-2]) * .01, rtol=1e-6)
    assert native.read_bytes() == original


def test_official_entrypoint_retains_commands_and_skips_only_evaluation(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat import official_runner as runner
    repo, output = tmp_path / 'repo', tmp_path / 'result/official'
    repo.mkdir()
    output.parent.mkdir()
    calls = []
    monkeypatch.setattr(sys, 'argv', ['runner', str(repo), 'inputs', str(output), '6'])
    monkeypatch.setattr(runner.os, 'system', lambda value: calls.append(value) or 0)
    def upstream(path, run_name):
        assert sys.argv[sys.argv.index('--config_view_num') + 1] == '6'
        assert sys.argv[sys.argv.index('--sfm_config') + 1] == 'posed'
        runner.os.system('python scripts/run_sfm.py --config posed')
        for stage in (1, 2, 3):
            runner.os.system(f'python scripts/see3d_inpaint.py --see3d_stage {stage}')
            runner.os.system('python scripts/refine_free_gaussians.py')
        runner.os.system('python 2d-gaussian-splatting/eval/eval.py --source_path inputs')
        native = output / 'free_gaussians/point_cloud/iteration_7000/point_cloud.ply'
        native.parent.mkdir(parents=True)
        native.touch()
    monkeypatch.setattr(runner.runpy, 'run_path', upstream)
    monkeypatch.setattr(runner, 'export_viewer_ply', lambda source, target: target.touch())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    runner.main()
    assert len(calls) == 7
    records = [json.loads(line) for line in (output.parent / 'commands.jsonl').read_text().splitlines()]
    assert len(records) == 8 and records[-1]['skipped']
    assert all(not row['skipped'] for row in records[:-1])
    assert (output.parent / 'g4splat-viewer.ply').is_file()


def test_remote_reattach_downloads_before_releasing_lease(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from splat_explorer.splatfix.g4splat.execution import execute_remote
    from splat_explorer import repair_lrz as lrz
    details = {'remote_dir': '/dss/splatfix-jobs/run'}
    updates, transfers = [], []
    state = {'status': 'completed', 'phase': 'finished'}
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss', 'user': 'user', 'host': 'host'})
    def ssh(cfg, command, timeout):
        return SimpleNamespace(returncode=0, stderr='', stdout='launched' if 'printf launched' in command else json.dumps(state))
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: transfers.append(argv))
    store = SimpleNamespace(get_run=lambda _: SimpleNamespace(state=SimpleNamespace(details=details)))
    def update(**fields):
        if fields.get('remote_finished'):
            assert transfers
        updates.append(fields)
    result = execute_remote(SimpleNamespace(store=store, cfg={}), 'run',
                            {'mode': 'baseline', 'checkpoint': 'saved'}, tmp_path, lambda: False, update)
    assert result['reconstruction_method'] == 'g4splat'
    assert updates[-1]['remote_finished']


def test_auto_setup_runs_before_preflight_and_training(tmp_path, monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/foreign/torch/lib')
    cp = checkpoint(tmp_path)
    export_inputs(cp, tmp_path / 'inputs', 'baseline')
    validations = []
    def validate(runtime):
        validations.append(runtime)
        if len(validations) == 1:
            raise FileNotFoundError('not installed')
        return tmp_path / 'repo', tmp_path / 'env/bin/python'
    monkeypatch.setattr(repair, 'validate_runtime', validate)
    commands = []
    def run(argv, log, *args):
        commands.append(argv)
        Path(log).touch()
    monkeypatch.setattr(repair, 'run_process', run)
    events = []
    with pytest.raises(RuntimeError, match='did not produce'):
        repair.run_prepared(tmp_path / 'inputs', tmp_path / 'result',
                            runtime={'auto_setup': True, 'python': str(tmp_path / 'env/bin/python'),
                                     'repo': str(tmp_path / 'repo')}, on_progress=events.append)
    assert len(validations) == 2
    assert commands[0][0] == 'bash' and commands[0][1].endswith('provision.sh')
    assert commands[0][-1] == repair.REVISION
    assert any(x.endswith('runtime_probe.py') for x in commands[1])
    assert any(x.endswith('official_runner.py') for x in commands[2])
    assert events[0]['phase'] == 'setup'
    assert 'PYTHONNOUSERSITE=1' in commands[2]
    assert not any('/foreign/torch/lib' in arg for arg in commands[2])


def test_auto_setup_preserves_incompatible_checkout(tmp_path, monkeypatch):
    def validate(runtime):
        raise ValueError('tracked source must be unmodified')
    monkeypatch.setattr(repair, 'validate_runtime', validate)
    monkeypatch.setattr(repair, 'run_process', lambda *args: pytest.fail('must not overwrite source'))
    with pytest.raises(ValueError, match='unmodified'):
        repair.run_prepared('unused', tmp_path / 'result', runtime={'auto_setup': True})


def test_empty_see3d_directory_is_not_a_checkpoint(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'train.py').touch()
    (repo / 'checkpoint/MVD_weights').mkdir(parents=True)
    monkeypatch.setattr(repair.subprocess, 'check_output',
                        lambda argv, **kwargs: repair.REVISION if 'rev-parse' in argv else '')
    with pytest.raises(FileNotFoundError, match='unet/sparse/ema-checkpoint/diffusion_pytorch_model.safetensors'):
        repair.validate_runtime({'repo': str(repo), 'python': sys.executable})


def test_preflight_failure_never_launches_training(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path)
    export_inputs(cp, tmp_path / 'inputs', 'baseline')
    monkeypatch.setattr(repair, 'validate_runtime', lambda rt: (tmp_path, Path(sys.executable)))
    def run(argv, *args):
        assert any(x.endswith('runtime_probe.py') for x in argv)
        raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(repair, 'run_process', run)
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        repair.run_prepared(tmp_path / 'inputs', tmp_path / 'result')
    assert not (tmp_path / 'result/launch.json').exists()
    assert not (tmp_path / 'result/result.json').exists()


def test_runtime_ignores_only_mode_changes_not_source_edits(tmp_path, monkeypatch):
    import subprocess
    repo = tmp_path / 'upstream'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init', '-q')
    (repo / 'train.py').write_text('# official source\n')
    (repo / 'train.py').chmod(0o755)
    git('add', 'train.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'upstream')
    monkeypatch.setattr(repair, 'REVISION', git('rev-parse', 'HEAD'))
    (repo / 'train.py').chmod(0o644)
    runtime = {'repo': str(repo), 'python': sys.executable}
    with pytest.raises(FileNotFoundError, match='weights'):
        repair.validate_runtime(runtime)
    (repo / 'train.py').write_text('# changed algorithm\n')
    with pytest.raises(ValueError, match='unmodified'):
        repair.validate_runtime(runtime)
