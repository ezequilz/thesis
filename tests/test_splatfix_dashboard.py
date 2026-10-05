from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.config import Config
from splat_explorer.rendering.base import Camera
from splat_explorer.scene_runs.manager import SceneRunManager
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.jobs import validate_job
from splat_explorer.web.splatfix_studio import SplatfixStudio


@pytest.fixture
def studio(tmp_path):
    cfg = Config({'output': {'dir': str(tmp_path)}, 'agent': {'model': 'test'}})
    store = SceneRunStore(tmp_path / 'scene-runs')
    result = SplatfixStudio(SimpleNamespace(cfg=cfg), store)
    result.scenes = lambda: [{'id': 'room', 'label': 'Room'}]
    return result


def checkpoint(studio, count=1, complete=True):
    cp = Checkpoint.create(studio.checkpoint_root, '/fake/scene.ply', target_views=count)
    if complete:
        camera = Camera(np.zeros(3), np.eye(3), width=16, height=16, fov_deg=60)
        for _ in range(count):
            cp.add_view(Image.new('RGB', (16, 16)), camera)
    return cp


def test_stage_validation():
    assert validate_job({'stage': 'select'})['views'] == 6
    for value in (True, 0, 25, 3.5):
        with pytest.raises(ValueError):
            validate_job({'stage': 'select', 'views': value})
    with pytest.raises(ValueError, match='timezone'):
        validate_job({'stage': 'select', 'scheduled_at': '2026-10-01T12:30'})
    with pytest.raises(ValueError, match='frames'):
        validate_job({'stage': 'repair', 'checkpoint': 'x', 'frames': 12})
    with pytest.raises(ValueError, match='span_fraction'):
        validate_job({'stage': 'repair', 'checkpoint': 'x', 'span_fraction': .2})


def test_schedule_persists_and_manager_skips_future(studio):
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    later = studio.create({'stage': 'select', 'scene_id': 'room', 'scheduled_at': future})
    manager = SceneRunManager(studio.cfg, store=studio.store)
    assert manager._next_queued() is None
    immediate = studio.create({'stage': 'select', 'scene_id': 'room', 'views': 10})
    assert manager._next_queued()['run_id'] == immediate['run_id']
    reloaded = SceneRunStore(studio.root).get_run(later['run_id'])
    assert reloaded.config.splatfix['scheduled_at'] == future


def test_local_stage_does_not_probe_gpu(studio, monkeypatch):
    job = studio.create({'stage': 'select', 'scene_id': 'room'})
    called = []
    class Executor:
        def __init__(self, cfg, store):
            pass
        def execute(self, run_id):
            called.append(run_id)
    manager = SceneRunManager(studio.cfg, store=studio.store, executor_factory=Executor)
    monkeypatch.setattr(manager, '_gpu_ready', lambda **_: pytest.fail('Local stage must not wait for GPU'))
    assert manager.run_once()
    assert called == [job['run_id']]
    assert studio.store.gpu_lease_owner() is None


def test_completed_candidate_not_reexecuted_after_lease(studio, monkeypatch):
    job = studio.create({'stage': 'select', 'scene_id': 'room'})
    manager = SceneRunManager(studio.cfg, store=studio.store, executor_factory=lambda *_: pytest.fail('Already completed'))
    acquire = manager._acquire_lease
    def raced(run_id, gpu):
        studio.store.update_status(run_id, status='completed')
        return acquire(run_id, gpu)
    monkeypatch.setattr(manager, '_acquire_lease', raced)
    assert not manager.run_once()


def test_checkpoint_dependencies_and_cancel(studio):
    cp = checkpoint(studio)
    original = json.loads((cp.root / 'checkpoint.json').read_text())
    baseline = studio.create({'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root)})
    assert baseline['config']['splatfix']['mode'] == 'baseline'
    with pytest.raises(ValueError, match='Repair all'):
        studio.create({'stage': 'repair', 'mode': 'edited', 'checkpoint': str(cp.root)})
    edit = studio.create({'stage': 'edit', 'checkpoint': str(cp.root)})
    studio.cancel(edit['run_id'])
    assert studio.store.get_run(edit['run_id']).state.status.value == 'stopped'
    assert next(j for j in studio.jobs() if j['run_id'] == edit['run_id'])['state']['status'] == 'cancelled'
    assert original == json.loads((cp.root / 'checkpoint.json').read_text())


def test_partial_checkpoint_cannot_edit_or_reconstruct(studio):
    cp = checkpoint(studio, complete=False)
    for stage in ('edit', 'repair'):
        with pytest.raises(ValueError, match='Finish selecting'):
            studio.create({'stage': stage, 'checkpoint': str(cp.root)})


def test_artifacts_cannot_escape_output_roots(studio, tmp_path):
    cp = checkpoint(studio)
    assert studio.allowed_file(cp.image_path(cp.views[0])).is_file()
    outside = tmp_path / 'secret.json'
    outside.write_text('{}')
    link = cp.root / 'linked.json'
    link.symlink_to(outside)
    for path in (outside, link, '/etc/passwd'):
        with pytest.raises(ValueError):
            studio.allowed_file(path)


def test_gpu_worker_passes_only_saved_inputs_and_cooperative_stop(tmp_path, monkeypatch):
    from splat_explorer.splatfix import job_worker
    request = {'checkpoint': 'saved', 'output': 'out', 'mode': 'baseline', 'frames': 25,
               'span_fraction': .04, 'runtime': {'repo': 'official'}}
    (tmp_path / 'worker-request.json').write_text(json.dumps(request))
    def fake_repair(cp, out, **kwargs):
        assert cp == 'saved' and out == 'out'
        assert kwargs['mode'] == 'baseline'
        assert not kwargs['should_stop']()
        kwargs['on_progress']({'phase': 'distill'})
        (tmp_path / 'STOP').touch()
        assert kwargs['should_stop']()
        raise InterruptedError('cancelled')
    monkeypatch.setattr(job_worker, 'run_repair', fake_repair)
    job_worker.execute(tmp_path)
    assert json.loads((tmp_path / 'worker-status.json').read_text())['status'] == 'stopped'


def test_restart_reattaches_gpu_job_instead_of_replaying(studio):
    cp = checkpoint(studio)
    job = studio.create({'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root)})
    studio.store.update_status(job['run_id'], status='running', remote_dir='/workspace/job')
    manager = SceneRunManager(studio.cfg, store=studio.store)
    manager._recover_interrupted()
    recovered = studio.store.get_run(job['run_id'])
    assert recovered.state.status.value == 'queued'
    assert recovered.state.details['remote_dir'] == '/workspace/job'


def test_local_selection_dispatches_separate_cli_without_image_edits(studio, monkeypatch):
    from splat_explorer.config import load_config
    from splat_explorer.splatfix import executor
    cfg = load_config()
    cfg['output']['dir'] = str(studio.root.parent)
    studio.cfg = cfg
    from splat_explorer.scene.catalog import list_scenes
    scene = list_scenes(cfg)[0]
    studio.scenes = lambda: [scene.to_json()]
    job = studio.create({'stage': 'select', 'scene_id': scene.id, 'views': 8})
    commands = []
    monkeypatch.setattr(executor, 'run_process', lambda argv, *_: commands.append(argv))
    executor.SplatfixExecutor(cfg, studio.store).execute(job['run_id'])
    argv = commands[0]
    assert 'splat_explorer.splatfix.local_worker' in argv
    assert '--select-only' in argv
    assert argv[argv.index('--views') + 1] == '8'
    config_path = Path(argv[argv.index('--config') + 1])
    import yaml
    assert yaml.safe_load(config_path.read_text())['agent']['vlm_backend'] == 'cli_relay'
    assert studio.store.get_run(job['run_id']).state.status.value == 'completed'


def test_lrz_reconnect_does_not_launch_a_second_reconstruction(studio, monkeypatch, tmp_path):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.splatfix.executor import SplatfixExecutor
    cp = checkpoint(studio)
    source = tmp_path / 'source.ply'
    source.write_text('offline source fixture')
    cp.manifest['scene_path'] = str(source)
    cp.save()
    job = studio.create({'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root)})
    commands, transfers, launches = [], [], []
    connection_fails = [True]
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', lambda self: {})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: transfers.append(argv))
    monkeypatch.setattr(lrz, '_remote_pythonpath_exports', lambda cfg: '')
    monkeypatch.setattr(lrz, 'container_srun_prefix', lambda cfg: 'srun ')
    def ssh(cfg, command, **kwargs):
        commands.append(command)
        if 'nohup' in command:
            launches.append(command)
        if 'worker-status.json' in command:
            if connection_fails[0]:
                connection_fails[0] = False
                raise OSError('temporary SSH interruption')
            return SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'completed', 'phase': 'finished'}), stderr='')
        return SimpleNamespace(returncode=0, stdout='launched' if 'printf launched' in command else '', stderr='')
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    worker = SplatfixExecutor(studio.cfg, studio.store)
    worker._execute_attempt(job['run_id'])
    assert studio.store.get_run(job['run_id']).state.status.value == 'queued'
    worker._execute_attempt(job['run_id'])
    assert studio.store.get_run(job['run_id']).state.status.value == 'completed'
    assert len(launches) == 1
    assert 'splat_explorer.splatfix.job_worker' in launches[0]
    request = json.loads((studio.store.run_path(job['run_id']) / 'worker-request.json').read_text())
    assert request['mode'] == 'baseline'
    assert request['checkpoint'].endswith('/checkpoint')
    assert request['runtime']['scene_path'].endswith('/source/source.ply')
    assert any('--exclude=source/' in call for call in transfers)


def test_cancel_pending_remote_job_still_reattaches_without_gpu_probe(studio, monkeypatch):
    cp = checkpoint(studio)
    job = studio.create({'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root)})
    studio.store.update_status(job['run_id'], status='queued', remote_dir='/dss/job')
    studio.cancel(job['run_id'])
    assert studio.store.get_run(job['run_id']).state.status.value == 'queued'
    assert studio.store.stop_requested(job['run_id'])
    called = []
    class Executor:
        def __init__(self, *_):
            pass
        def execute(self, run_id):
            called.append(run_id)
    manager = SceneRunManager(studio.cfg, store=studio.store, executor_factory=Executor)
    monkeypatch.setattr(manager, '_gpu_ready', lambda **_: pytest.fail('Reattachment must not need an active GPU allocation'))
    assert manager.run_once()
    assert called == [job['run_id']]


def test_unresolved_remote_lease_cannot_be_reclaimed_by_other_work(studio, monkeypatch):
    import socket
    from splat_explorer.scene_runs import store as store_module
    cp = checkpoint(studio)
    job = studio.create({'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root)})
    other = studio.create({'stage': 'select', 'scene_id': 'room'})
    lease = studio.store.acquire_gpu_lease(job['run_id'])
    studio.store.update_status(job['run_id'], status='running', remote_dir='/dss/job')
    monkeypatch.setattr(store_module, '_pid_alive', lambda _: False)
    assert studio.store.acquire_gpu_lease(other['run_id']) is None
    assert studio.store.acquire_setup_lease() is None
    recovered = studio.store.acquire_gpu_lease(job['run_id'])
    assert recovered is not None
    recovered.release()


def benchmark_input(studio):
    source = studio.benchmark_root / 'bicycle' / 'input'
    (source / 'colmap' / 'images').mkdir(parents=True)
    (source / 'colmap' / 'sparse' / '0').mkdir(parents=True)
    (source / 'selected_images.txt').write_text('one.JPG\ntwo.JPG\nthree.JPG\n')
    (source / 'benchmark.json').write_text(json.dumps({'name': 'Bicycle 3-view', 'source': 'published dataset'}))
    return source


def test_registered_benchmark_queues_without_checkpoint(studio):
    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source)})
    options = job['config']['splatfix']
    assert options == {'stage': 'benchmark', 'source': str(source), 'mode': 'baseline', 'model': '1.3b'}
    assert studio.benchmarks()[0]['ready']
    assert 'not confirmed' in studio.benchmarks()[0]['provenance_note']
    from splat_explorer.splatfix.jobs import requires_gpu
    assert requires_gpu(job['config'])
    with pytest.raises(ValueError, match='baseline'):
        studio.create({'stage': 'benchmark', 'source': str(source), 'mode': 'edited'})
    with pytest.raises(ValueError, match='model'):
        studio.create({'stage': 'benchmark', 'source': str(source), 'model': 'unknown'})
    large = studio.create({'stage': 'benchmark', 'source': str(source), 'model': '14b'})
    assert large['config']['splatfix']['model'] == '14b'


def test_benchmark_registration_rejects_incomplete_and_escaping_sources(studio, tmp_path):
    source = benchmark_input(studio)
    (source / 'selected_images.txt').unlink()
    assert not studio.benchmarks()[0]['ready']
    with pytest.raises(ValueError, match='incomplete'):
        studio.create({'stage': 'benchmark', 'source': str(source)})
    (source / 'selected_images.txt').write_text('one.JPG\n')
    secret = tmp_path / 'private.json'
    secret.write_text('{}')
    (source / 'escape.json').symlink_to(secret)
    with pytest.raises(ValueError, match='symbolic links'):
        studio.create({'stage': 'benchmark', 'source': str(source)})
    with pytest.raises(ValueError):
        studio.allowed_file(source / 'escape.json')
    with pytest.raises(ValueError):
        studio.create({'stage': 'benchmark', 'source': str(tmp_path)})
    assert studio.allowed_file(source / 'benchmark.json').name == 'benchmark.json'
    with pytest.raises(ValueError, match='limited'):
        studio.allowed_file(source / 'selected_images.txt')


def test_benchmark_worker_uses_official_dataset_entrypoint(tmp_path, monkeypatch):
    import sys
    from splat_explorer.splatfix import job_worker
    calls = []
    def benchmark(source, output, **kwargs):
        calls.append((source, output, kwargs['runtime']))
        kwargs['on_progress']({'phase': 'colmap_preparation'})
        assert not kwargs['should_stop']()
        return {'splat_path': 'official.ply'}
    monkeypatch.setitem(sys.modules, 'splat_explorer.splatfix.benchmark', SimpleNamespace(run_benchmark=benchmark))
    monkeypatch.setattr(job_worker, 'run_repair', lambda *a, **kw: pytest.fail('Benchmark must not use saved-view repair'))
    (tmp_path / 'worker-request.json').write_text(json.dumps({'stage': 'benchmark', 'source': '/benchmark-input',
        'output': '/results', 'runtime': {'model_variant': '14b'}}))
    job_worker.execute(tmp_path)
    assert calls == [('/benchmark-input', '/results', {'model_variant': '14b'})]
    assert json.loads((tmp_path / 'worker-status.json').read_text())['status'] == 'completed'


def test_benchmark_remote_stages_dataset_and_matched_model(studio, monkeypatch):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.splatfix.executor import SplatfixExecutor
    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source), 'model': '14b'})
    transfers, commands = [], []
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', lambda self: {})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: transfers.append(argv))
    monkeypatch.setattr(lrz, '_remote_pythonpath_exports', lambda cfg: '')
    monkeypatch.setattr(lrz, 'container_srun_prefix', lambda cfg: 'srun ')
    def ssh(cfg, command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps({'status': 'completed', 'phase': 'finished'}) if 'worker-status.json' in command else '')
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    result = SplatfixExecutor(studio.cfg, studio.store).execute(job['run_id'])
    request = json.loads((studio.store.run_path(job['run_id']) / 'worker-request.json').read_text())
    assert request['stage'] == 'benchmark'
    assert request['source'].endswith('/benchmark-input')
    assert 'checkpoint' not in request
    assert request['runtime']['model_variant'] == '14b'
    assert request['runtime']['checkpoint'].endswith('artifixer-14b.pt')
    assert '14B' in request['runtime']['model_id']
    assert any(str(source) + '/' in call for call in transfers)
    assert any('--exclude=benchmark-input/' in call for call in transfers)
    assert result['source'] == str(source)


def test_benchmark_restart_reattaches_existing_gpu_run(studio):
    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source)})
    studio.store.update_status(job['run_id'], status='running', remote_dir='/dss/benchmark-job')
    SceneRunManager(studio.cfg, store=studio.store)._recover_interrupted()
    run = studio.store.get_run(job['run_id'])
    assert run.state.status.value == 'queued'
    assert run.state.details['remote_dir'] == '/dss/benchmark-job'


def test_benchmark_registry_does_not_read_symlinked_metadata(studio, tmp_path):
    source = benchmark_input(studio)
    secret = tmp_path / 'private.json'
    secret.write_text(json.dumps({'private': 'do not expose'}))
    (source / 'benchmark.json').unlink()
    (source / 'benchmark.json').symlink_to(secret)
    assert studio.benchmarks() == []


def test_result_gallery_uses_recorded_output_not_an_arbitrary_ply(studio):
    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source)})
    result_dir = studio.store.run_path(job['run_id']) / 'gpu/results/benchmark_demo'
    (result_dir / 'official').mkdir(parents=True)
    (result_dir / 'initialization.ply').write_text('wrong output')
    (result_dir / 'official/fresh.ply').write_text('correct output')
    (result_dir / 'result.json').write_text(json.dumps({'splat_path': '/workspace/job/results/benchmark_demo/official/fresh.ply'}))
    result = studio.jobs()[0]['results'][0]
    assert 'fresh.ply' in result['ply_url']
    assert 'initialization' not in result['ply_url']


def test_current_phase_log_available_before_job_completion(studio):
    from splat_explorer.splatfix.live_logs import PhaseLogMirror
    cp = checkpoint(studio)
    job = studio.create({'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root)})
    root = studio.store.run_path(job['run_id'])
    studio.store.update_status(job['run_id'], status='running', phase='reconstruct')
    mirror = PhaseLogMirror(root, '/workspace/job', '/dss/job', lambda command: 'training iteration 1500\n')
    assert mirror.poll({'phase': 'reconstruct', 'log': '/workspace/job/results/reconstruct.log'})
    row = studio.jobs()[0]
    assert row['state']['status'] == 'running'
    assert 'live-stage.log' in row['log_url']
    assert row['log_label'] == 'reconstruct log · latest 64 KiB'
    assert not row['results']
    full = root / 'gpu/results/reconstruct.log'
    full.parent.mkdir()
    full.write_text('complete phase log')
    assert 'live-stage.log' in studio.jobs()[0]['log_url']
    studio.store.update_status(job['run_id'], status='completed')
    row = studio.jobs()[0]
    assert 'reconstruct.log' in row['log_url']
    assert row['log_label'] == 'reconstruct log'


def test_executor_mirrors_phase_log_before_completion_without_output_rsync(studio, monkeypatch):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.splatfix import executor
    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source)})
    root = studio.store.run_path(job['run_id'])
    transfers, polls = [], []
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', lambda self: {})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: transfers.append(argv))
    monkeypatch.setattr(lrz, '_remote_pythonpath_exports', lambda cfg: '')
    monkeypatch.setattr(lrz, 'container_srun_prefix', lambda cfg: 'srun ')
    monkeypatch.setattr(executor.time, 'sleep', lambda _: None)
    def ssh(cfg, command, **kwargs):
        body = ''
        if 'worker-status.json' in command:
            polls.append(command)
            body = json.dumps({'status': 'running', 'phase': 'reconstruct',
                'log': '/workspace/splatfix-jobs/' + job['run_id'] + '/results/reconstruct.log'} if len(polls) == 1 else {'status': 'completed'})
        elif 'tail -c' in command:
            body = 'training 1000/30000\n'
        elif 'launcher-exit' in command and 'nohup' not in command:
            assert (root / 'gpu/live-stage.log').read_text() == 'training 1000/30000\n'
            assert studio.store.get_run(job['run_id']).state.status.value == 'running'
            assert not any('--exclude=source/' in transfer for transfer in transfers)
        return SimpleNamespace(returncode=0, stderr='', stdout=body)
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    executor.SplatfixExecutor(studio.cfg, studio.store).execute(job['run_id'])
    assert len(polls) == 2


def failed_benchmark(studio):
    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source)})
    studio.store.update_status(job['run_id'], status='error', remote_finished=True,
                               remote_dir='/dss/work/splatfix-jobs/' + job['run_id'])
    return source, job


def test_benchmark_resume_creates_new_job_preserving_previous(studio):
    source, prior = failed_benchmark(studio)
    before = studio.store.get_run(prior['run_id']).to_dict()
    resumed = studio.create({'stage': 'benchmark', 'source': str(source), 'resume_from': prior['run_id']})
    assert resumed['run_id'] != prior['run_id']
    assert resumed['config']['splatfix']['resume_from'] == prior['run_id']
    assert studio.store.get_run(prior['run_id']).to_dict() == before


def test_benchmark_resume_rejects_unsafe_mismatched_active_or_unstaged_jobs(studio):
    source, prior = failed_benchmark(studio)
    body = {'stage': 'benchmark', 'source': str(source), 'resume_from': prior['run_id']}
    for bad in ('../run_20261004_155421', '/tmp/run_20261004_155421', 12):
        with pytest.raises(ValueError, match='job id'):
            studio.create({**body, 'resume_from': bad})
    with pytest.raises(ValueError, match='match'):
        studio.create({**body, 'model': '14b'})
    other_source = studio.benchmark_root / 'other' / 'input'
    import shutil
    shutil.copytree(source, other_source)
    with pytest.raises(ValueError, match='match'):
        studio.create({**body, 'source': str(other_source)})
    for status in ('queued', 'running', 'completed'):
        studio.store.update_status(prior['run_id'], status=status)
        with pytest.raises(ValueError, match='Only failed'):
            studio.create(body)
    studio.store.set_status(prior['run_id'], status='error')
    with pytest.raises(ValueError, match='remote job directory'):
        studio.create(body)


def test_benchmark_resume_resolves_only_a_valid_prior_result(studio):
    from splat_explorer.splatfix.resume import resolve_remote_resume
    source, prior = failed_benchmark(studio)
    options = {'source': str(source), 'model': '1.3b', 'resume_from': prior['run_id']}
    commands = []
    def ssh(command):
        commands.append(command)
        return json.dumps({'relative': 'results/benchmark_1.3b_abc123'})
    result = resolve_remote_resume(studio.store, options, {'workspace': '/dss/work'}, ssh)
    assert result == '/workspace/splatfix-jobs/' + prior['run_id'] + '/results/benchmark_1.3b_abc123'
    assert 'benchmark-run.json' in commands[0]
    for escaped in ('../private', '/private/results/benchmark_x', 'results/../benchmark_x', 'results/benchmark_x/file'):
        with pytest.raises(ValueError, match='escaped'):
            resolve_remote_resume(studio.store, options, {'workspace': '/dss/work'}, lambda _: json.dumps({'relative': escaped}))
    with pytest.raises(ValueError, match='remote workspace'):
        resolve_remote_resume(studio.store, options, {'workspace': '/different'}, ssh)


def test_benchmark_resume_discovery_rejects_multiple_results_and_symlinks(tmp_path):
    import subprocess
    import sys
    from splat_explorer.splatfix.resume import _DISCOVER_RESULT
    root = tmp_path / 'prior'
    result = root / 'results/benchmark_first'
    result.mkdir(parents=True)
    supported = {'status': 'failed', 'stages': [{'phase': p, 'status': 'complete'} for p in ('prepare', 'reconstruct', 'render', 'scale')]}
    (result / 'prepared/bicycle').mkdir(parents=True)
    (result / 'benchmark-run.json').write_text(json.dumps(supported))
    found = subprocess.run([sys.executable, '-c', _DISCOVER_RESULT, str(root)], capture_output=True, text=True)
    assert found.returncode == 0
    assert json.loads(found.stdout)['relative'] == 'results/benchmark_first'
    outside = tmp_path / 'private'
    outside.mkdir()
    (outside / 'benchmark-run.json').write_text(json.dumps({'status': 'failed'}))
    (root / 'results/benchmark_escape').symlink_to(outside)
    found = subprocess.run([sys.executable, '-c', _DISCOVER_RESULT, str(root)], capture_output=True, text=True)
    assert found.returncode == 0
    second = root / 'results/benchmark_second'
    second.mkdir()
    (second / 'prepared/bicycle').mkdir(parents=True)
    (second / 'benchmark-run.json').write_text(json.dumps(supported))
    ambiguous = subprocess.run([sys.executable, '-c', _DISCOVER_RESULT, str(root)], capture_output=True, text=True)
    assert ambiguous.returncode != 0
    assert 'exactly one' in ambiguous.stderr


def test_benchmark_worker_passes_validated_resume_runtime(tmp_path, monkeypatch):
    import sys
    from splat_explorer.splatfix import job_worker
    calls = []
    def benchmark(source, output, **kwargs):
        calls.append(kwargs['runtime'])
        return {}
    monkeypatch.setitem(sys.modules, 'splat_explorer.splatfix.benchmark', SimpleNamespace(run_benchmark=benchmark))
    resume = '/workspace/splatfix-jobs/run_20261004_155421/results/benchmark_1.3b_abc'
    (tmp_path / 'worker-request.json').write_text(json.dumps({'stage': 'benchmark', 'source': '/new/input',
        'output': '/new/results', 'runtime': {'resume_from': resume}}))
    job_worker.execute(tmp_path)
    assert calls == [{'resume_from': resume}]


def test_resume_button_readiness_matches_supported_failed_preparation(studio):
    source, prior = failed_benchmark(studio)
    result = studio.store.run_path(prior['run_id']) / 'gpu/results/benchmark_1.3b_old'
    (result / 'prepared/bicycle').mkdir(parents=True)
    manifest = {'status': 'failed', 'stages': [{'phase': p, 'status': 'complete'} for p in ('prepare', 'reconstruct', 'render', 'scale')]}
    path = result / 'benchmark-run.json'
    path.write_text(json.dumps(manifest))
    assert studio.jobs()[0]['resume_supported']
    path.write_text(json.dumps({**manifest, 'resume_supported': False}))
    assert not studio.jobs()[0]['resume_supported']
    path.write_text(json.dumps({**manifest, 'status': 'cancelled', 'resume_supported': True}))
    assert not studio.jobs()[0]['resume_supported']


@pytest.mark.parametrize('reuse_field', ['resume_from', 'preparation_from'])
def test_benchmark_executor_dispatches_resume_into_new_job(studio, monkeypatch, reuse_field):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.splatfix.executor import SplatfixExecutor
    source, prior = failed_benchmark(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source), reuse_field: prior['run_id'],
                         'model': '14b' if reuse_field == 'preparation_from' else '1.3b'})
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', lambda self: {})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_mux_run', lambda argv: None)
    monkeypatch.setattr(lrz, '_remote_pythonpath_exports', lambda cfg: '')
    monkeypatch.setattr(lrz, 'container_srun_prefix', lambda cfg: 'srun ')
    def ssh(cfg, command, **kwargs):
        body = ''
        if 'worker-status.json' in command:
            body = json.dumps({'status': 'completed'})
        elif 'benchmark-run.json' in command:
            body = json.dumps({'relative': 'results/benchmark_1.3b_old'})
        return SimpleNamespace(returncode=0, stderr='', stdout=body)
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    SplatfixExecutor(studio.cfg, studio.store).execute(job['run_id'])
    request = json.loads((studio.store.run_path(job['run_id']) / 'worker-request.json').read_text())
    assert prior['run_id'] in request['runtime'][reuse_field]
    assert job['run_id'] in request['output']
    assert job['run_id'] not in request['runtime'][reuse_field]
    assert studio.store.get_run(prior['run_id']).state.status.value == 'error'


def test_saved_checkpoint_controls_scene_provenance_not_scene_dropdown(studio):
    cp = checkpoint(studio)
    studio.scenes = lambda: [{'id': 'original-scene', 'path': '/fake/scene.ply'},
                            {'id': 'unrelated-scene', 'path': '/fake/other.ply'}]
    for stage in ('edit', 'repair'):
        job = studio.create({'stage': stage, 'mode': 'baseline', 'checkpoint': str(cp.root),
                             'scene_id': 'unrelated-scene'})
        assert job['config']['scene_id'] == 'original-scene'
        assert job['config']['splatfix']['checkpoint'] == str(cp.root)


def test_independent_repair_schedules_preserve_same_cameras_and_input_modes(studio):
    cp = checkpoint(studio)
    Image.new('RGB', (16, 16)).save(cp.root / 'views/000/repaired.png')
    cp.views[0]['repaired_rgb'] = 'views/000/repaired.png'
    cp.save()
    before = (cp.root / 'checkpoint.json').read_bytes()
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    baseline = studio.create({'stage': 'repair', 'checkpoint': str(cp.root), 'mode': 'baseline', 'scheduled_at': future})
    improved = studio.create({'stage': 'repair', 'checkpoint': str(cp.root), 'mode': 'edited', 'scheduled_at': future})
    assert baseline['run_id'] != improved['run_id']
    assert baseline['config']['splatfix']['mode'] == 'baseline'
    assert improved['config']['splatfix']['mode'] == 'edited'
    assert all(j['config']['splatfix']['checkpoint'] == str(cp.root) for j in (baseline, improved))
    assert (cp.root / 'checkpoint.json').read_bytes() == before
    assert SceneRunManager(studio.cfg, store=studio.store)._next_queued() is None


def test_shared_preparation_allows_cross_model_without_changing_prior(studio):
    source, prior = failed_benchmark(studio)
    studio.store.update_status(prior['run_id'], status='completed')
    before = studio.store.get_run(prior['run_id']).to_dict()
    job = studio.create({'stage': 'benchmark', 'source': str(source), 'model': '14b',
                        'preparation_from': prior['run_id']})
    options = job['config']['splatfix']
    assert options['model'] == '14b'
    assert options['preparation_from'] == prior['run_id']
    assert 'resume_from' not in options
    assert job['run_id'] != prior['run_id']
    assert studio.store.get_run(prior['run_id']).to_dict() == before


def test_shared_preparation_rejects_unsafe_ambiguous_or_nonterminal_requests(studio):
    source, prior = failed_benchmark(studio)
    body = {'stage': 'benchmark', 'source': str(source), 'preparation_from': prior['run_id']}
    for bad in ('../run_20261004_155421', '/tmp/run_20261004_155421', False):
        with pytest.raises(ValueError, match='job id'):
            studio.create({**body, 'preparation_from': bad})
    with pytest.raises(ValueError, match='mutually exclusive'):
        studio.create({**body, 'resume_from': prior['run_id']})
    for status in ('queued', 'running', 'stopped'):
        studio.store.update_status(prior['run_id'], status=status)
        with pytest.raises(ValueError, match='completed or failed'):
            studio.create(body)
    studio.store.update_status(prior['run_id'], status='completed')
    import shutil
    other = studio.benchmark_root / 'other' / 'input'
    shutil.copytree(source, other)
    with pytest.raises(ValueError, match='source must match'):
        studio.create({**body, 'source': str(other)})


def test_shared_preparation_requires_caption_and_supports_completed_manifest(studio, tmp_path):
    import subprocess
    import sys
    from splat_explorer.splatfix.resume import _DISCOVER_RESULT, resolve_remote_preparation
    source, prior = failed_benchmark(studio)
    studio.store.update_status(prior['run_id'], status='completed')
    root = studio.store.run_path(prior['run_id']) / 'gpu'
    result = root / 'results/benchmark_1.3b_previous'
    (result / 'prepared/bicycle').mkdir(parents=True)
    body = {'status': 'complete', 'stages': [{'phase': p, 'status': 'complete'} for p in ('prepare', 'reconstruct', 'render', 'scale')]}
    manifest = result / 'benchmark-run.json'
    manifest.write_text(json.dumps(body))
    assert not studio.jobs()[0]['preparation_supported']
    missing = subprocess.run([sys.executable, '-c', _DISCOVER_RESULT, str(root), 'preparation'], capture_output=True, text=True)
    assert missing.returncode != 0
    body['stages'].append({'phase': 'caption', 'status': 'complete'})
    manifest.write_text(json.dumps(body))
    assert studio.jobs()[0]['preparation_supported']
    ready = subprocess.run([sys.executable, '-c', _DISCOVER_RESULT, str(root), 'preparation'], capture_output=True, text=True)
    assert ready.returncode == 0, ready.stderr
    options = {'source': str(source), 'model': '14b', 'preparation_from': prior['run_id']}
    commands = []
    def ssh(command):
        commands.append(command)
        return ready.stdout
    resolved = resolve_remote_preparation(studio.store, options, {'workspace': '/dss/work'}, ssh)
    assert resolved.endswith('/results/benchmark_1.3b_previous')
    assert commands[0].endswith(' preparation')
    with pytest.raises(ValueError, match='escaped'):
        resolve_remote_preparation(studio.store, options, {'workspace': '/dss/work'}, lambda _: json.dumps({'relative': '/outside'}))
    with pytest.raises(ValueError, match='remote workspace'):
        resolve_remote_preparation(studio.store, options, {'workspace': '/other'}, ssh)
    body['preparation_supported'] = False
    manifest.write_text(json.dumps(body))
    assert not studio.jobs()[0]['preparation_supported']


def test_completed_remote_transfer_failure_retains_diagnostics_and_artifacts(studio, monkeypatch):
    import subprocess
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.splatfix.executor import SplatfixExecutor

    source = benchmark_input(studio)
    job = studio.create({'stage': 'benchmark', 'source': str(source)})
    root = studio.store.run_path(job['run_id'])
    saved = root / 'gpu/results/official.ply'
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b'previously downloaded PLY')
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', lambda self: {})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_remote_pythonpath_exports', lambda cfg: '')
    monkeypatch.setattr(lrz, 'container_srun_prefix', lambda cfg: 'srun ')
    commands = []
    def ssh(cfg, command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps({'status': 'completed', 'phase': 'finished'}) if 'worker-status.json' in command else '')
    monkeypatch.setattr(lrz, '_ssh_run', ssh)
    diagnostic = 'initial SSH failure\n' + 'x' * 900 + '\nrsync code 12'
    def transfer(argv):
        if '--exclude=benchmark-input/' in argv:
            raise lrz.RemoteCommandError(argv, subprocess.CompletedProcess(argv, 255, 'partial transfer', diagnostic))
    monkeypatch.setattr(lrz, '_mux_run', transfer)
    assert SplatfixExecutor(studio.cfg, studio.store).execute(job['run_id']) == {}
    run = studio.store.get_run(job['run_id'])
    assert run.state.status.value == 'error'
    assert run.state.details['remote_finished'] is True
    assert diagnostic in (root / 'artifact-transfer.log').read_text()
    assert 'partial transfer' in (root / 'artifact-transfer.log').read_text()
    assert saved.read_bytes() == b'previously downloaded PLY'
    assert not any('/STOP' in command for command in commands)
    assert studio.jobs()[0]['transfer_log_url'] == studio.file_url(root / 'artifact-transfer.log')
