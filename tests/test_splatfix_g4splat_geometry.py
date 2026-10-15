"""Regressions for calibrated pixels, restored poses and the depth-stage gate."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from splat_explorer.splatfix.g4splat import corrected_runtime, geometry, official_runner


@pytest.mark.parametrize('wh,expected', [((832, 480), (832, 468)), ((1280, 720), (1280, 720)),
                                        ((960, 720), (960, 720))])
def test_crop_preserves_calibrated_pixel_rays(tmp_path, wh, expected):
    w, h = wh
    # Distinct rows expose a calibration-only edit or incorrect crop direction.
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(h, dtype=np.uint16)[:, None] % 256
    source, target = tmp_path/'original.png', tmp_path/'training.png'
    Image.fromarray(pixels).save(source)
    k = np.array([[417., 0, w/2], [0, 417., h/2], [0, 0, 1]])
    pw, ph = (512, 384) if wh == (960, 720) else (512, 288)
    adjusted, record = geometry.prepare_calibrated_image(source, target, k, (pw, ph))
    image = np.asarray(Image.open(target))
    x, y = record['crop_xy']
    assert Image.open(target).size == expected
    np.testing.assert_array_equal(image, pixels[y:y+expected[1], x:x+expected[0]])
    np.testing.assert_array_equal(np.asarray(Image.open(source)), pixels)
    np.testing.assert_allclose(adjusted[:2, 2], np.array(expected)/2)
    # An arbitrary physical ray is unchanged after translating pixel coordinates.
    pixel = np.array([expected[0]*.37, expected[1]*.61, 1])
    np.testing.assert_allclose(np.linalg.inv(adjusted) @ pixel,
                               np.linalg.inv(k) @ (pixel + [x, y, 0]))
    if wh == (832, 480):
        assert y == 6 and adjusted[1, 2] == 234 and image.shape[0] == 468
    assert expected[0]/pw == expected[1]/ph


def test_crop_rejects_unhandled_off_center_camera(tmp_path):
    source = tmp_path/'source.png'
    Image.new('RGB', (832, 480)).save(source)
    k = np.array([[417, 0, 416], [0, 417, 234], [0, 0, 1]])
    with pytest.raises(ValueError, match='centered square-pixel'):
        geometry.prepare_calibrated_image(source, tmp_path/'output.png', k, (512, 288))


def scene_fixture(tmp_path):
    scene = tmp_path/'sfm'
    (scene/'images').mkdir(parents=True)
    (scene/'pointmaps').mkdir()
    (scene/'sparse/0').mkdir(parents=True)
    k = np.array([[24., 0, 16], [0, 24., 8], [0, 0, 1]])
    yy, xx = np.mgrid[:16, :32]
    depths = 2 + xx/32 + yy/16
    local = (np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(k).T) * depths[..., None]
    poses, filenames, calibration, lines = [], [], [], []
    for i, angle in enumerate([0, 180]):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('y', angle, degrees=True).as_matrix()
        pose[:3, 3] = [i*2, .3, -1]
        points = local.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]
        name = f'{i:06}.png'
        Image.new('RGB', (64, 32)).save(scene/'images'/name)
        full_k = k.copy(); full_k[:2] *= 2
        calibration.append({'source_wh': [64, 32], 'output_wh': [64, 32], 'crop_xy': [0, 0],
                            'intrinsics': full_k.tolist()})
        (scene/'pointmaps'/f'{i:06}.json').write_text(json.dumps({'points': points.tolist(),
                                                             'confs': np.ones((16,32)).tolist()}))
        poses.append(pose.tolist()); filenames.append(name)
        lines.append(f'{i+1} PINHOLE 64 32 48 48 32 16')
    body = {'filepaths': filenames, 'cams2world': poses, 'intrinsics': [k.tolist()]*2,
            'image_calibration': calibration}
    (scene/'cameras.json').write_text(json.dumps(body))
    (scene/'sparse/0/cameras.txt').write_text('\n'.join(lines)+'\n')
    return scene, body


def test_reanchor_preserves_depth_for_opposite_rotated_camera(tmp_path):
    scene, body = scene_fixture(tmp_path)
    path = scene/'pointmaps/000001.json'
    pointmap = json.loads(path.read_text())
    new = np.array(body['cams2world'][1])
    old = new.copy()
    old[:3, 3] += [1.2, -.3, -.9]
    old[:3, :3] = Rotation.from_euler('xyz', [10, 170, -4], degrees=True).as_matrix()
    correct = np.array(pointmap['points'])
    drifted = geometry.reanchor_points(correct, new, old)
    pointmap['points'] = drifted.tolist(); path.write_text(json.dumps(pointmap))
    with pytest.raises(ValueError, match='stopped depth alignment'):
        geometry.validate_pointmaps(scene, tmp_path/'failed.json')
    assert not json.loads((tmp_path/'failed.json').read_text())['passed']
    restored = geometry.reanchor_points(drifted, old, new)
    np.testing.assert_allclose(restored, correct, atol=1e-12)
    np.testing.assert_allclose((drifted-old[:3,3]) @ old[:3,:3],
                               (restored-new[:3,3]) @ new[:3,:3], atol=1e-12)
    pointmap['points'] = restored.tolist(); path.write_text(json.dumps(pointmap))
    report = geometry.validate_pointmaps(scene, tmp_path/'passed.json')
    assert report['passed'] and max(v['max_error_px'] for v in report['views']) < 1e-10


@pytest.mark.parametrize('fault', ['nan', 'behind', 'one_bad_pixel', 'principal_point', 'uncropped_image', 'missing_metadata'])
def test_gate_rejects_invalid_geometry_and_writes_report(tmp_path, fault):
    scene, body = scene_fixture(tmp_path)
    path = scene/'pointmaps/000000.json'
    pointmap = json.loads(path.read_text())
    if fault == 'nan':
        pointmap['points'][0][0] = float('nan')
    elif fault == 'behind':
        pointmap['points'][0] = [0, .3, -2]
    elif fault == 'one_bad_pixel':
        pointmap['points'][0][0] += 1  # Do not hide tail errors behind percentiles.
    elif fault == 'principal_point':
        p = scene/'sparse/0/cameras.txt'
        p.write_text(p.read_text().replace('48 48 32 16', '48 48 32 10'))
    elif fault == 'uncropped_image':
        Image.new('RGB', (64, 44)).save(scene/'images/000000.png')
    else:
        del body['intrinsics']
        (scene/'cameras.json').write_text(json.dumps(body))
    path.write_text(json.dumps(pointmap))
    with pytest.raises(ValueError, match='geometry gate'):
        geometry.validate_pointmaps(scene, tmp_path/'report.json')
    report = json.loads((tmp_path/'report.json').read_text())
    assert not report['passed'] and report['error']


def test_failed_gate_does_not_invoke_alignment_or_publish(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir()
    output = tmp_path/'result/official'; output.parent.mkdir()
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/inputs.json').write_text('{}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', ['runner', str(repo), 'inputs', str(output), '2'])
    monkeypatch.setattr(official_runner, 'prepare_runtime', lambda *args: repo)
    monkeypatch.setattr(official_runner.graceful, 'patch_runtime', lambda *args: None)
    calls = []
    original_system = lambda value: calls.append(value) or 0
    monkeypatch.setattr(official_runner.os, 'system', original_system)
    def train(*args, **kwargs):
        official_runner.os.system('python scripts/run_sfm.py')
        official_runner.os.system('python scripts/align_charts.py')
        pytest.fail('bad geometry reached training')
    monkeypatch.setattr(official_runner.runpy, 'run_path', train)
    with pytest.raises(ValueError, match='geometry gate'):
        official_runner.main()
    assert calls == ['python scripts/run_sfm.py']
    assert official_runner.os.system is original_system
    assert not (output.parent/'g4splat-viewer.ply').exists()


def test_patch_rejects_unknown_source():
    with pytest.raises(ValueError, match='Unexpected pinned'):
        corrected_runtime.corrected_source('changed upstream source')


def test_overlay_applies_patch_without_modifying_checkout(tmp_path, monkeypatch):
    # Minimal source with the exact patch sites; actual pinned-source application
    # is also verified during development against its complete SHA-256.
    source = '''def main():
        # Modify images and intrinsics so that the principal points become the center
        pass
        # Modify intrinsics according to MASt3R resizing
        if fix_rotation and fix_translation:
            cams2world = restored
        print_debug('output extrinsics:')
        output_cameras = {
        'focals': focals.numpy().tolist(),
        }
'''
    monkeypatch.setattr(corrected_runtime, 'UPSTREAM_SHA256', hashlib.sha256(source.encode()).hexdigest())
    repo = tmp_path/'repo'; (repo/'mast3r').mkdir(parents=True)
    (repo/'mast3r/run_mast3r.py').write_text(source)
    (repo/'mast3r/weights').mkdir()
    (repo/'train.py').write_text('# unchanged')
    overlay = corrected_runtime.prepare_runtime(repo, tmp_path/'output/runtime')
    patched = (overlay/'mast3r/run_mast3r.py').read_text()
    assert 'reanchor_points' in patched and 'prepare_calibrated_image' in patched
    assert (repo/'mast3r/run_mast3r.py').read_text() == source
    assert (overlay/'train.py').resolve() == repo/'train.py'
    assert (overlay/'mast3r/weights').resolve() == repo/'mast3r/weights'
    assert (overlay/'mast3r/splatfix_geometry.py').is_file()
    manifest = json.loads((overlay.parent/'geometry-correction.json').read_text())
    assert manifest['patched_source_sha256'] == hashlib.sha256(patched.encode()).hexdigest()
    assert not manifest['shared_checkout_modified']


@pytest.mark.parametrize('certificate', [None, {'passed': False, 'version': geometry.CORRECTION_VERSION},
                                         {'passed': True, 'version': 'obsolete'}])
def test_completed_result_requires_current_successful_report(tmp_path, monkeypatch, certificate):
    from splat_explorer.splatfix.g4splat import repair
    inputs = tmp_path/'inputs'; inputs.mkdir()
    (inputs/'inputs.json').write_text(json.dumps({'mode': 'baseline', 'views': [{}, {}]}))
    monkeypatch.setattr(repair, 'validate_runtime', lambda runtime: (tmp_path, Path(sys.executable)))
    def run(argv, *args):
        if not any(x.endswith('official_runner.py') for x in argv):
            return
        output = tmp_path/'result'
        native = output/'official/free_gaussians/point_cloud/iteration_7000/point_cloud.ply'
        native.parent.mkdir(parents=True); native.touch()
        (output/'g4splat-viewer.ply').touch()
        if certificate is not None:
            (output/'reprojection-check.json').write_text(json.dumps(certificate))
    monkeypatch.setattr(repair, 'run_process', run)
    with pytest.raises(RuntimeError, match='camera-consistency report'):
        repair.run_prepared(inputs, tmp_path/'result')
    assert not (tmp_path/'result/result.json').exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_runner_depth_extension_precedes_gate_and_isolates_environment(tmp_path, monkeypatch, enabled):
    repo = tmp_path/'repo'; repo.mkdir()
    source = tmp_path/'inputs'; source.mkdir()
    (source/'inputs.json').write_text(json.dumps({'source_depth': {'enabled': enabled}}))
    output = tmp_path/'result/official'; output.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(sys, 'argv', ['runner', str(repo), str(source), str(output), '2'])
    monkeypatch.setattr(official_runner, 'prepare_runtime', lambda *args: repo)
    monkeypatch.setattr(official_runner.graceful, 'patch_runtime', lambda *args: None)
    events = []
    helper = official_runner.source_depth_runtime
    monkeypatch.setattr(helper, 'patch_source_depth_runtime', lambda *a: events.append('patch'))
    monkeypatch.setattr(helper, 'apply_source_depth_pointmaps', lambda *a: events.append('depth'))
    monkeypatch.setattr(official_runner, 'validate_pointmaps', lambda *a: events.append('gate'))
    monkeypatch.setattr(official_runner.os, 'system', lambda *a: events.append('align') or 0)
    keys = ['SPLATFIX_SOURCE_DEPTH_INPUTS', 'SPLATFIX_SOURCE_DEPTH_SCENE']
    for key in keys:
        monkeypatch.setenv(key, 'unrelated-run')
    def train(*a, **kw):
        for key in keys:
            if enabled:
                assert official_runner.os.environ[key] != 'unrelated-run'
            else:
                assert key not in official_runner.os.environ
        official_runner.os.system('python scripts/align_charts.py')
        raise RuntimeError('stop after alignment')
    monkeypatch.setattr(official_runner.runpy, 'run_path', train)
    with pytest.raises(RuntimeError, match='stop after alignment'):
        official_runner.main()
    assert events == (['patch', 'depth', 'gate', 'align'] if enabled else ['gate', 'align'])
    assert all(official_runner.os.environ[key] == 'unrelated-run' for key in keys)
