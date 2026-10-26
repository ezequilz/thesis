import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint, atomic_json, camera_from_record, camera_to_record
from splat_explorer.splatfix.artifixer import preflight
from splat_explorer.splatfix.artifixer.staging_cache import capture_prepared_rgb
from splat_explorer.splatfix.resolution import prepare_repair_checkpoint


@pytest.fixture
def source(tmp_path, monkeypatch):
    scene = tmp_path / 'source.ply'
    scene.write_text('source fingerprint fixture')
    cp = Checkpoint.create(tmp_path / 'saved', scene, target_views=2,
                           metadata={'renderer': {'backend': 'viser'}, 'up_axis': '-y',
                                     'scene_load': {'lod_level': 2}})
    for index in range(2):
        cp.add_view(Image.new('RGB', (32, 32), 'red'),
                    Camera(np.array([float(index), 0., 0.]), np.eye(3), width=32, height=32))
    captures = []
    sessions = []
    class Session:
        def __init__(self, source, **kwargs):
            sessions.append((str(source), kwargs))
        def render(self, camera):
            captures.append(camera)
            return np.full((camera.height, camera.width, 3), 123, np.uint8)
    monkeypatch.setattr(preflight, 'CaptureSession', Session)
    root = tmp_path / 'run'; root.mkdir()
    runtime = {'resolution_profile': 'training', 'artifixer_pipeline': 'legacy_gsplat'}
    return cp, root, runtime, captures, sessions


def test_preflight_cache_hit_needs_no_viewer_and_worker_reuses_anchors(source, monkeypatch):
    cp, root, runtime, captures, sessions = source
    before = (cp.root / 'checkpoint.json').read_bytes()
    events = []
    first = preflight.prepare_source_captures(cp, runtime, root, progress=lambda **e: events.append(e), url='http://viewer')
    assert first['captured_frames'] == 2
    assert [(c.width, c.height) for c in captures] == [(832, 480)] * 2
    assert sessions[0][1]['lod_level'] == 2 and sessions[0][1]['url'] == 'http://viewer'
    assert events[0]['phase'] == 'viser_preflight'
    monkeypatch.setattr(preflight, 'CaptureSession', lambda *a, **kw: pytest.fail('Cache hit must not need Viser'))
    assert preflight.prepare_source_captures(cp, runtime, root)['captured_frames'] == 0
    monkeypatch.setattr('splat_explorer.splatfix.viser_capture.capture_rgb', lambda *a, **kw: pytest.fail('Worker must reuse preflight'))
    prepared = prepare_repair_checkpoint(cp, root / 'worker', 'training', reuse_artifixer_inputs=True, artifixer_runtime=runtime)
    paths = [root / f'geometry-{i}.png' for i in range(2)]
    capture_prepared_rgb(prepared, [camera_from_record(v) for v in prepared.views], paths,
                         lambda *a: pytest.fail('Geometry must reuse calibrated anchor RGB'))
    assert all(path.read_bytes() == prepared.image_path(view).read_bytes() for path, view in zip(paths, prepared.views))
    assert (cp.root / 'checkpoint.json').read_bytes() == before
    assert not list(root.glob('viser-preflight-*'))


def test_preflight_uses_corrected_cameras_without_mutating_checkpoint(source, monkeypatch):
    cp, root, runtime, captures, _ = source
    corrected = Checkpoint(cp.root, copy.deepcopy(cp.manifest))
    from dataclasses import replace
    camera = camera_from_record(corrected.views[0])
    corrected.views[0]['camera'] = camera_to_record(replace(camera, position=camera.position + [0., .2, 0.]))
    monkeypatch.setattr('splat_explorer.splatfix.pose_correction.corrected_checkpoint', lambda _: corrected)
    preflight.prepare_source_captures(cp, {**runtime, 'correct_generated_pose': True}, root)
    assert captures[0].position[1] == .2
    assert cp.views[0]['camera']['position'][1] == 0


def test_exact_repeat_captures_intermediate_views_before_staging_and_reuses(source, monkeypatch):
    cp, root, runtime, captures, _ = source
    first, last = [camera_from_record(v).c2w for v in cp.views]
    middle = first.copy(); middle[0, 3] = .5
    orbit = {'poses': [first.tolist(), middle.tolist(), last.tolist()]}
    atomic_json(cp.root / 'repeat-repair.json', {'orbit': orbit})
    result = preflight.prepare_source_captures(cp, runtime, root)
    assert result['known_repeat_orbit'] and len(captures) == 3
    assert len(list(Path(result['input_cache']).glob('artifixer-inputs/*/captures.json'))) == 1
    monkeypatch.setattr(preflight, 'CaptureSession', lambda *a, **kw: pytest.fail('Repeat cache must be reused'))
    assert preflight.prepare_source_captures(cp, runtime, root)['captured_frames'] == 0
    monkeypatch.setattr('splat_explorer.splatfix.viser_capture.capture_rgb', lambda *a, **kw: pytest.fail('No late anchor capture'))
    prepared = prepare_repair_checkpoint(cp, root / 'worker', 'training', reuse_artifixer_inputs=True,
                                         artifixer_runtime=runtime, preserve_repaired_images=True)
    from dataclasses import replace
    anchor = camera_from_record(prepared.views[0])
    cameras = [replace(anchor, position=p[:3, 3], rotation=p[:3, :3]) for p in (first, middle, last)]
    paths = [root / f'orbit-{i}.png' for i in range(3)]
    capture_prepared_rgb(prepared, cameras, paths, lambda *a: pytest.fail('No late repeat orbit capture'))
    assert all(Image.open(path).size == (832, 480) for path in paths)


def test_corrupt_cached_trajectory_fails_locally_without_copying_geometry(source, monkeypatch):
    cp, root, runtime, _, _ = source
    result = preflight.prepare_source_captures(cp, runtime, root)
    cache = Path(result['input_cache'])
    trajectory = cache / 'trajectories/fixture'; trajectory.mkdir(parents=True)
    frame = trajectory / '00000.png'; frame.write_bytes(b'broken')
    atomic_json(trajectory / 'trajectory.json', {'sha256': {'trajectories/fixture/00000.png': 'wrong'}})
    monkeypatch.setattr(preflight, 'CaptureSession', lambda *a, **kw: pytest.fail('Cached anchors must not be captured'))
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.staging_cache.restore_geometry',
                        lambda *a: pytest.fail('Preflight must not copy multi-GB geometry'))
    with pytest.raises(ValueError, match='trajectory changed before staging'):
        preflight.prepare_source_captures(cp, runtime, root)


@pytest.mark.parametrize('cancel', [False, True])
def test_capture_failure_or_cancellation_does_not_commit_partial_batch(source, monkeypatch, cancel):
    cp, root, runtime, _, _ = source
    class Broken:
        def __init__(self, *a, **kw):
            pass
        def render(self, camera):
            raise (InterruptedError('Stopped capture') if cancel else RuntimeError('Viser disconnected'))
    monkeypatch.setattr(preflight, 'CaptureSession', Broken)
    with pytest.raises(InterruptedError if cancel else RuntimeError):
        preflight.prepare_source_captures(cp, runtime, root)
    assert not list(cp.root.glob('artifixer-inputs/*/captures.json'))
    assert not (root / 'source-capture-preflight.json').exists()
    assert not list(root.glob('viser-preflight-*'))


@pytest.mark.parametrize('fail_capture', [False, True])
def test_capture_precedes_allocation_validation_and_any_upload(source, monkeypatch, fail_capture):
    cp, root, runtime, captures, _ = source
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.splatfix.artifixer.execution import execute_remote
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda *a: pytest.fail('No uploads before capture and validation'))
    monkeypatch.setattr(lrz, '_mux_run', lambda *a: pytest.fail('No uploads before capture and validation'))
    monkeypatch.setattr(lrz, '_ssh_run', lambda *a, **kw: pytest.fail('No remote launch before capture and validation'))
    def validate(_):
        assert len(captures) == 2
        raise RuntimeError('Reached GPU validation after captures')
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', validate)
    if fail_capture:
        def fail(*a, **kw):
            raise RuntimeError('Viser disconnected before staging')
        monkeypatch.setattr(preflight, 'CaptureSession', fail)
    executor = SimpleNamespace(cfg={}, store=SimpleNamespace(get_run=lambda _: SimpleNamespace(state=SimpleNamespace(details={}))))
    options = {'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root),
               'regularization_profile': 'artifixer', **runtime}
    with pytest.raises(RuntimeError, match='Viser disconnected' if fail_capture else 'Reached GPU validation'):
        execute_remote(executor, 'test', options, root, lambda: False, lambda **kw: None)


def test_native_pipeline_does_not_require_viser(source, monkeypatch):
    cp, root, runtime, captures, _ = source
    monkeypatch.setattr(preflight, 'CaptureSession', lambda *a, **kw: pytest.fail('Native pipeline must not need Viser'))
    assert preflight.prepare_source_captures(cp, {**runtime, 'artifixer_pipeline': 'authors_3dgut'}, root) is None
    assert not captures


def test_remote_reattachment_does_not_capture_or_stage_again(source, monkeypatch):
    cp, root, runtime, _, _ = source
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.splatfix.artifixer.execution import execute_remote
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(lrz, '_ssh_run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='launched', stderr=''))
    monkeypatch.setattr(preflight, 'prepare_source_captures', lambda *a, **kw: pytest.fail('Reattach must not capture'))
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda *a: pytest.fail('Reattach must not stage'))
    def mirroring(*a, **kw):
        raise RuntimeError('Reached existing remote job mirroring')
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.live_logs.PhaseLogMirror', mirroring)
    executor = SimpleNamespace(cfg={}, store=SimpleNamespace(get_run=lambda _: SimpleNamespace(
        state=SimpleNamespace(details={'remote_dir': '/dss/work/splatfix-jobs/test'}))))
    with pytest.raises(RuntimeError, match='existing remote job mirroring'):
        execute_remote(executor, 'test', {'stage': 'repair', 'mode': 'baseline', 'checkpoint': str(cp.root), **runtime},
                       root, lambda: False, lambda **kw: None)
