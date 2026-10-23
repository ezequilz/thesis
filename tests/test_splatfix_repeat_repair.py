from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from splat_explorer.splatfix.checkpoint import atomic_json, camera_from_record, Checkpoint, validate_source
from splat_explorer.splatfix.artifixer.repeat_repair import snapshot_repeat, repeat_inputs
from splat_explorer.splatfix import repair
from test_splatfix_repair import make_checkpoint, mock_scene, FakeRenderer, write_caption_fixture


def fixture(tmp_path):
    cp = make_checkpoint(tmp_path)
    cp.manifest['target_views'] = 2
    cp.save()
    camera = camera_from_record(cp.views[0])
    cp.add_view(np.full((32, 32, 3), 91, np.uint8), replace(camera, position=camera.position + [1, 0, 0]))
    poses = [camera_from_record(v).c2w.tolist() for v in cp.views]
    orbit = {'poses': poses, 'provenance': {'segments': []}}
    result = tmp_path / 'result'; result.mkdir()
    atomic_json(result / 'result.json', {})
    (result / 'output.ply').write_bytes(cp.manifest['scene_path'].encode())
    atomic_json(cp.root / 'trajectory.json', {'recipe': {'orbit': orbit}})
    atomic_json(result / 'request.json', {'checkpoint_root': str(cp.root),
        'trajectory': str(cp.root / 'trajectory.json'), 'trajectory_mode': 'authors_orbit',
        'runtime': {**repair.DEFAULT_RUNTIME, 'initial_image_seeding': False},
        'seed': 123, 'camera_scale': .027, 'mode': 'baseline'})
    row = {'id': 'result-id', 'run_id': 'prior', 'kind': 'ArtiFixer3D',
           '_manifest': result / 'result.json', '_paths': {'repaired': result / 'output.ply'}, 'repair_pass': 2}
    studio = SimpleNamespace(result_artifact=lambda manifest, raw: Path(raw) if raw and Path(raw).exists() else None)
    return cp, row, studio, orbit


def test_snapshot_is_independent_and_keeps_only_replay_inputs(tmp_path):
    cp, row, studio, orbit = fixture(tmp_path)
    before = deepcopy(cp.manifest)
    (cp.root / 'captions').mkdir()
    (cp.root / 'captions/stale').write_text('old')
    root = tmp_path / 'repeat'; root.mkdir()
    info = snapshot_repeat(studio, row, root)
    saved = Checkpoint.load(root)
    validate_source(saved)
    assert info['repair_pass'] == 3
    assert info['orbit'] == orbit
    assert info['runtime']['initial_image_seeding'] is False
    assert saved.manifest['scene_path'] == str(root / 'original.ply')
    assert (root / 'original.ply').read_bytes() == row['_paths']['repaired'].read_bytes()
    assert not (root / 'captions').exists()
    assert not (root / 'trajectories').exists()
    assert cp.manifest == before
    row['_paths']['repaired'].unlink()
    assert (root / 'original.ply').is_file()


def test_missing_orbit_rejects_repeat(tmp_path):
    cp, row, studio, _ = fixture(tmp_path)
    atomic_json(cp.root / 'trajectory.json', {})
    with pytest.raises(ValueError, match='Exact orbit'):
        repeat_inputs(studio, row)


def test_repeat_rerenders_saved_poses_and_skips_interpolation(tmp_path, monkeypatch):
    cp, row, studio, orbit = fixture(tmp_path)
    # Keep a valid source PLY for the rendering path.
    row['_paths']['repaired'].write_bytes(Path(cp.manifest['scene_path']).read_bytes())
    root = tmp_path / 'repeat'; root.mkdir()
    info = snapshot_repeat(studio, row, root)
    mock_scene(monkeypatch)
    monkeypatch.setattr('splat_explorer.splatfix.rendering.BundleRenderer', FakeRenderer)
    monkeypatch.setattr(repair, 'validate_runtime', lambda runtime: {**repair.DEFAULT_RUNTIME, **runtime})
    captures, phases = [], []
    monkeypatch.setattr(repair, 'capture_rgb', lambda source, cameras, paths, **kw: captures.append((source, [c.c2w.tolist() for c in cameras])))
    def worker(command, **kwargs):
        phase = command[-1]; phases.append(phase)
        assert phase not in ('orbit', 'scale')
        path = Path(command[command.index('--request') + 1])
        req = json.loads(path.read_text())
        assert req['seed'] == 123 and req['camera_scale'] == .027
        assert req['runtime']['initial_image_seeding'] is False
        trajectory = json.loads(Path(req['trajectory']).read_text())
        assert trajectory['recipe']['orbit'] == orbit
        assert all((root / frame['opacity']).is_file() for frame in trajectory['frames'])
        if phase == 'caption':
            atomic_json(path.parent / 'caption-result.json', write_caption_fixture(path.parent / 'caption.h5'))
        if phase == 'plus':
            output = path.parent / 'artifixer3d.ply'; output.write_bytes(b'new result')
            atomic_json(path.parent / 'result.json', {'splat_path': str(output)})
    monkeypatch.setattr(repair, 'run_worker', worker)
    result = repair.run_repair(root, tmp_path / 'outputs', mode='baseline', runtime=info['runtime'])
    assert phases == ['caption', 'infer', 'distill', 'plus']
    assert captures == [(str(root / 'original.ply'), orbit['poses'])]
    assert result['repeat_repair']['repair_pass'] == 3
    assert (Path(result['output_dir']) / 'original.ply').read_bytes() == (root / 'original.ply').read_bytes()


def test_repeat_button_precedes_oracle_action():
    html = Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()
    assert html.index('id="repeatStart"') < html.index('id="oracleStart"')
    assert '/api/splatfix/results/repeat-artifixer' in html


def test_queue_inherits_settings_and_cleans_failed_snapshot(tmp_path, monkeypatch):
    from test_splatfix_results import setup
    from splat_explorer.web.splatfix_results import ResultCatalog
    cp, row, source_studio, _ = fixture(tmp_path)
    studio, run_root, _ = setup(tmp_path / 'catalog')
    config = studio.store.get_run(run_root.name).to_dict()['config']
    config['splatfix'] = dict(stage='repair', checkpoint=str(cp.root), mode='baseline',
        reconstruction_method='artifixer', resolution_profile='720p', frames=29,
        span_fraction=.03, initial_image_seeding=False, image_cache_insertion=False,
        reconstruction_images='distilled', regularization_profile='base_mcmc')
    row['config'] = config
    monkeypatch.setattr(studio, 'result_artifact', source_studio.result_artifact)
    monkeypatch.setattr(studio.results, 'detail', lambda _: row)
    queued = studio.results.repeat_artifixer(row['id'])
    options = queued['config']['splatfix']
    assert queued['state']['status'] == 'queued'
    assert options['frames'] == 29 and options['regularization_profile'] == 'base_mcmc'
    assert options['initial_image_seeding'] is False
    assert Path(options['checkpoint']) != cp.root
    settings = json.loads((studio.root / queued['run_id'] / 'settings.json').read_text())
    assert settings['submitted']['repeat_repair']['seed'] == 123
    before = set(studio.checkpoint_root.iterdir())
    monkeypatch.setattr(studio.store, 'create_run', lambda *a, **k: (_ for _ in ()).throw(ValueError('queue failed')))
    with pytest.raises(ValueError, match='queue failed'):
        studio.results.repeat_artifixer(row['id'])
    assert set(studio.checkpoint_root.iterdir()) == before


def test_repeat_uses_viewer_ply_from_interrupted_run(tmp_path):
    _, row, studio, _ = fixture(tmp_path)
    root = row['_manifest'].parent
    row['_manifest'] = root / 'partial-result.json'
    atomic_json(row['_manifest'], {'incomplete': True})
    row['incomplete'] = True
    interrupted = root / 'distill.interrupted' / 'interrupted.ply'
    interrupted.parent.mkdir()
    row['_paths']['repaired'].rename(interrupted)
    row['_paths']['repaired'] = interrupted
    destination = tmp_path / 'repeat'; destination.mkdir()
    info = snapshot_repeat(studio, row, destination)
    assert (destination / 'original.ply').read_bytes() == interrupted.read_bytes()
    assert info['parent_incomplete'] is True
    assert info['parent_splat'] == 'distill.interrupted/interrupted.ply'
    interrupted.unlink()
    with pytest.raises(ValueError, match='saved reconstruction PLY'):
        repeat_inputs(studio, row)


def test_repeat_preserves_custom_gaussian_limit(tmp_path):
    _, row, studio, _ = fixture(tmp_path)
    request_path = row['_manifest'].parent / 'request.json'
    request = json.loads(request_path.read_text())
    request['runtime']['max_gaussians'] = 2_500_000
    atomic_json(request_path, request)
    destination = tmp_path / 'repeat'; destination.mkdir()
    info = snapshot_repeat(studio, row, destination)
    assert info['runtime']['max_gaussians'] == 2_500_000
