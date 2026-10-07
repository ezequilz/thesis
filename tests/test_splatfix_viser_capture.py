"""The manager/worker handoff must deliver Viser pixels, never proxy RGB."""
import json
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix import viser_capture as capture
from splat_explorer.splatfix.checkpoint import Checkpoint, camera_to_record
from splat_explorer.splatfix.image_repair import repair_images
from splat_explorer.splatfix.official_worker import validate_render_sources


def camera():
    return Camera(np.array([1., 2., 3.]), np.eye(3), width=32, height=16)


@pytest.mark.parametrize('backend', ['cpu_splats', 'cpu_points', 'gsplat', None])
def test_proxy_and_unverified_edits_are_rejected_before_backend_call(tmp_path, backend):
    cp = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=1,
                           metadata={'renderer': {'backend': backend}})
    cp.add_view(np.zeros((16, 32, 3), np.uint8), camera())
    with pytest.raises(ValueError, match='Viser captures'):
        repair_images(cp, backend=SimpleNamespace(edit=lambda *a: pytest.fail('No paid call')))


def test_old_trajectory_and_substituted_baseline_references_rejected(tmp_path):
    request = {'checkpoint_root': str(tmp_path), 'mode': 'baseline',
               'references': [str(tmp_path / 'views/000/original.png')]}
    trajectory = {'recipe': {'anchor_rgb_policy': 'source_scene_bundle_renderer'},
                  'anchors': [{'original_rgb': 'trajectories/00000.png'}]}
    with pytest.raises(ValueError, match='Unverified trajectory renderer'):
        validate_render_sources(request, trajectory)
    trajectory['recipe']['anchor_rgb_policy'] = 'viser'
    with pytest.raises(ValueError, match='Viser trajectory captures'):
        validate_render_sources(request, trajectory)
    request['references'] = [str(tmp_path / 'trajectories/00000.png')]
    validate_render_sources(request, trajectory)


@pytest.mark.parametrize('fail', [False, True])
def test_remote_capture_roundtrip_and_failure_never_falls_back(tmp_path, monkeypatch, fail):
    remote, local = tmp_path / 'remote', tmp_path / 'local'
    mailbox = remote / 'viser-capture'
    mailbox.mkdir(parents=True)
    monkeypatch.setenv('SPLATFIX_VISER_MAILBOX', str(mailbox))
    received = []
    class Session:
        def __init__(self, scene, **kwargs):
            received.append(str(scene))
        def render(self, pose):
            if fail:
                raise RuntimeError('Viser disconnected')
            received.append(camera_to_record(pose))
            return np.full((pose.height, pose.width, 3), 217, np.uint8)
    monkeypatch.setattr(capture, 'CaptureSession', Session)
    def ssh(command):
        parts = shlex.split(command)
        if parts[0] == 'if':
            path = Path(parts[3])
            return path.read_text() if path.is_file() else ''
        assert parts[:2] == ['mkdir', '-p']
        Path(parts[2]).mkdir(parents=True, exist_ok=True)
        return ''
    def sync(source, destination):
        source, destination = str(source).removeprefix('host:'), str(destination).removeprefix('host:')
        if source.endswith('/'):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, destination)
    cp = SimpleNamespace(manifest={'scene_path': '/local/scene.ply'})
    mirror = capture.CaptureMirror(local, str(remote), '/workspace/job', cp, 'host', ssh, sync)
    output = tmp_path / 'RGB.png'
    Image.new('RGB', (32, 16), (10, 10, 10)).save(output)
    with ThreadPoolExecutor() as pool:
        task = pool.submit(capture.capture_rgb, '/workspace/job/source/scene.ply', [camera()], [output])
        for _ in range(200):
            if (mailbox / 'request.json').is_file():
                break
            time.sleep(.01)
        mirror.poll()
        if fail:
            with pytest.raises(RuntimeError, match='Viser disconnected'):
                task.result(timeout=3)
            assert np.asarray(Image.open(output))[0, 0, 0] == 10
        else:
            task.result(timeout=3)
            assert np.all(np.asarray(Image.open(output)) == 217)
            assert received[1] == camera_to_record(camera())
        # Restarting the manager must not repeat a published batch.
        count = len(received)
        capture.CaptureMirror(local, str(remote), '/workspace/job', cp, 'host', ssh, sync).poll()
        assert len(received) == count


def test_plus_uses_reconstructed_scene_and_original_opencv_poses(tmp_path, monkeypatch):
    pose = camera()
    tf = {'w': pose.width, 'h': pose.height, 'fl_x': pose.fx,
          'frames': [{'transform_matrix': pose.c2w.tolist()}]}
    (tmp_path / 'distillation.json').write_text(json.dumps({
        'splat_path': '/job/results/artifixer3d.ply', 'render_dir': str(tmp_path / 'renders')}))
    calls = []
    monkeypatch.setattr(capture, 'capture_rgb', lambda *a, **kw: calls.append((a, kw)))
    capture.capture_plus_rgb(tmp_path, {'transforms': tf})
    args, _ = calls[0]
    assert args[0] == '/job/results/artifixer3d.ply'
    np.testing.assert_allclose(args[1][0].c2w, pose.c2w)
    np.testing.assert_allclose(args[1][0].intrinsics, pose.intrinsics)
    assert args[2] == [tmp_path / 'renders/renders/00000.png']


def test_scene_change_during_capture_is_rejected(monkeypatch):
    session = object.__new__(capture.CaptureSession)
    session.should_stop = lambda: False
    checks = iter([True, False])
    monkeypatch.setattr(session, 'ready', lambda: next(checks))
    session.renderer = SimpleNamespace(render=lambda pose: np.zeros((16, 32, 3), np.uint8))
    with pytest.raises(RuntimeError, match='scene changed'):
        session.render(camera())
