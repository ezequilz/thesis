import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from splat_explorer.splatfix.g4splat.source_depth import conservative_mask
from splat_explorer.splatfix.g4splat.source_depth_runtime import calibrated_depth, patch_source_depth_runtime, HASHES


def test_mask_rejects_holes_edges_and_depth_boundary():
    z = np.ones((9, 12), dtype=np.float32) * 2
    z[:, 6:] = 10
    z[4, 3] = np.inf
    mask = conservative_mask(z)
    assert not mask[0].any() and not mask[:, 0].any()
    assert not mask[3:6, 2:5].any()
    assert not mask[:, 5:7].any()
    assert mask[2, 8] and mask[2, 2]


def depth_fixture(tmp_path, monkeypatch):
    z = np.repeat(np.arange(480, dtype=np.float32)[:,None]+1, 832, axis=1)
    k = np.array([[500.,0,416],[0,500,240],[0,0,1]])
    pose = np.eye(4)
    file = tmp_path / 'depth.npz'
    np.savez(file, depth=z, valid=np.ones_like(z, dtype=bool), confidence=np.ones_like(z), intrinsics=k, c2w=pose)
    (tmp_path/'inputs.json').write_text(json.dumps({'source_depth': {'views':[{'image':'000000.png','path':'depth.npz','sha256':hashlib.sha256(file.read_bytes()).hexdigest()}]}}))
    cropped = k.copy(); cropped[1,2] -= 6
    (tmp_path/'cameras.json').write_text(json.dumps({'filepaths':['000000.png'], 'cams2world':[pose.tolist()], 'image_calibration':[{'source_wh':[832,480], 'output_wh':[832,468], 'crop_xy':[0,6], 'source_intrinsics':k.tolist(), 'intrinsics':cropped.tolist()}]}))
    monkeypatch.setenv('SPLATFIX_SOURCE_DEPTH_INPUTS',str(tmp_path)); monkeypatch.setenv('SPLATFIX_SOURCE_DEPTH_SCENE',str(tmp_path))
    return file


def test_exact_crop_and_resize_and_integrity(tmp_path, monkeypatch):
    file = depth_fixture(tmp_path, monkeypatch)
    depth, mask = calibrated_depth('000000.png',(468,832))
    assert depth[0,0] == 7 and depth[-1,-1] == 474 and mask.all()
    small, _ = calibrated_depth('000000', (234,416))
    assert small[0,0] == 8 and small[-1,0] == 474
    file.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        calibrated_depth('000000',(468,832))


def test_overlay_patch_never_mutates_symlink_target(tmp_path):
    upstream = Path('/tmp/g4splat-audit-upstream')
    if not upstream.exists():
        pytest.skip('Optional pinned upstream source checkout unavailable')
    overlay = tmp_path / 'overlay'; overlay.mkdir()
    for name in ('matcha', '2d-gaussian-splatting'):
        (overlay/name).symlink_to(upstream/name, target_is_directory=True)
    patch_source_depth_runtime(overlay)
    for relative, expected in HASHES.items():
        assert hashlib.sha256((upstream/relative).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((overlay/relative).read_bytes()).hexdigest() != expected
        compile((overlay/relative).read_text(), relative, 'exec')
    assert (tmp_path/'source-depth-extension.json').exists()


def test_metric_depth_and_pose_preserved_under_rotation_and_scale(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat.source_depth_runtime import calibrated_rays
    depth_fixture(tmp_path, monkeypatch)
    body = json.loads((tmp_path/'cameras.json').read_text())
    pose = np.array([[ -1.,0,0,4],[0,1,0,2],[0,0,-1,8],[0,0,0,1]])
    body['cams2world'] = [pose.tolist()]
    (tmp_path/'cameras.json').write_text(json.dumps(body))
    rays, center = calibrated_rays('000000', (234,416), scale=3)
    points = rays * 6 + center
    local = (points - pose[:3,3]*3) @ pose[:3,:3]
    np.testing.assert_allclose(local[:,:,2], 6)
    np.testing.assert_allclose(local[117,208,:2], 0, atol=1e-6)

    k = np.array([[250.,0,208],[0,250,117],[0,0,1]])
    uvw = local @ k.T
    yy, xx = np.mgrid[:234,:416]
    np.testing.assert_allclose(uvw[:,:,:2]/uvw[:,:,2:], np.stack([xx,yy],-1), atol=1e-6)


def test_source_injection_rescues_only_trusted_points(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat.source_depth_runtime import apply_source_depth_pointmaps
    depth_fixture(tmp_path, monkeypatch)
    (tmp_path/'pointmaps').mkdir()
    shape = (234,416)
    pointfile = tmp_path/'pointmaps/000000.json'
    pointfile.write_text(json.dumps({'confs': np.ones(shape).tolist(), 'points': np.full((234*416,3), -5.).tolist()}))
    report = apply_source_depth_pointmaps(tmp_path,tmp_path/'source-depth-check.json')
    points = np.asarray(json.loads(pointfile.read_text())['points']).reshape(234,416,3)
    depth,_ = calibrated_depth('000000',shape)
    np.testing.assert_allclose(points[:,:,2],depth)
    uv = points[:,:,:2] / points[:,:,2:]*250 + [208,117]
    yy,xx=np.mgrid[:234,:416]
    np.testing.assert_allclose(uv,np.stack([xx,yy],-1),atol=1e-6)
    assert report['views'][0]['trusted_fraction'] == 1.


def test_opt_in_export_records_extension_and_metric_maps(tmp_path, monkeypatch):
    from splat_explorer.rendering.base import Camera
    from splat_explorer.splatfix.checkpoint import Checkpoint
    from splat_explorer.splatfix.g4splat.inputs import export_inputs
    from PIL import Image
    cp = Checkpoint.create(tmp_path/'checkpoint', tmp_path/'scene.ply', target_views=2,
                           metadata={'renderer': {'backend':'viser'}})
    for x in (0., 1.):
        cp.add_view(Image.new('RGB',(32,32)), Camera(np.array([x,0.,0.]),np.eye(3),width=32,height=32,fov_deg=60.))
    cp.save()
    monkeypatch.setattr('splat_explorer.scene.load_scene', lambda path: object())
    import splat_explorer.splatfix.g4splat.source_depth as module
    monkeypatch.setattr(module,'validate_source',lambda cp:None)
    monkeypatch.setattr('splat_explorer.rendering.cpu_splat_renderer.CpuSplatRenderer.__init__',lambda self,scene:None)
    root=tmp_path/'inputs'
    progress = []
    def render_depth(self, camera):
        published = json.loads((root/'inputs.json').read_text())
        assert published['source_depth']['status'] == 'exporting'
        ready = published['source_depth']['views']
        progress.append(len(ready))
        for view in ready:
            with np.load(root/view['path'], allow_pickle=False) as data:
                assert data['depth'].shape == (32, 32)
        return np.full((camera.height,camera.width),7.,dtype=np.float32)
    monkeypatch.setattr('splat_explorer.rendering.cpu_splat_renderer.CpuSplatRenderer.render_depth',render_depth)
    manifest=export_inputs(cp,root,'baseline',source_depth=True)
    assert progress == [0, 1]
    assert manifest['source_depth']['status'] == 'ready'
    assert manifest['mode']=='baseline' and manifest['method_variant']=='source-splat-depth-extension'
    assert manifest['source_splat_used_for_initialization']
    assert manifest['source_depth']['enabled']
    with np.load(root/manifest['source_depth']['views'][0]['path']) as data:
        assert np.all(data['depth'][data['valid']]==7.)
        assert not data['valid'][0].any()


def test_source_disparity_alignment_recovers_metric_scale_with_outliers():
    from splat_explorer.splatfix.g4splat.source_depth_runtime import fit_source_disparity
    disp = np.linspace(.2, 2, 1600).reshape(40, 40)
    truth = 1 / (.3 + 1.7 * disp)
    depth = truth.copy()
    depth.flat[::17] *= 8
    valid = np.ones(disp.shape, bool)
    valid[:, -4:] = False
    result, info = fit_source_disparity(disp, depth, valid)
    np.testing.assert_allclose(result, truth, rtol=1e-7)
    assert info['fit'] == 'robust-affine-inverse-depth'


def test_negative_affine_extrapolation_uses_positive_scale_without_clamping():
    from splat_explorer.splatfix.g4splat.source_depth_runtime import fit_source_disparity
    disp = np.linspace(.01, 2, 1600).reshape(40, 40)
    valid = disp > 1
    depth = np.zeros_like(disp)
    depth[valid] = 1 / (disp[valid] - .8)
    result, info = fit_source_disparity(disp, depth, valid)
    assert info['fit'] == 'positive-scale-inverse-depth'
    assert np.isfinite(result).all() and (result > 0).all()
    np.testing.assert_allclose(result * disp, 1 / info['beta'])


@pytest.mark.parametrize('fault', ['nan', 'negative', 'no_anchors', 'zero'])
def test_source_disparity_rejects_unusable_prediction(fault):
    from splat_explorer.splatfix.g4splat.source_depth_runtime import fit_source_disparity
    disp = np.ones((10, 10)); depth = disp * 2; valid = np.ones_like(disp, bool)
    if fault == 'nan':
        disp[0, 0] = np.nan
    elif fault == 'negative':
        disp[0, 0] = -1
    elif fault == 'zero':
        disp[0, 0] = 0
    else:
        valid[:] = False
    with pytest.raises(ValueError):
        fit_source_disparity(disp, depth, valid)


def test_invalid_fallback_completed_but_not_used_as_chart_reference(tmp_path, monkeypatch):
    from splat_explorer.splatfix.g4splat.source_depth_runtime import (
        apply_source_depth_pointmaps, calibrated_rays, reference_mask,
    )
    file = depth_fixture(tmp_path, monkeypatch)
    with np.load(file) as data:
        maps = {k: data[k].copy() for k in data.files}
    maps['valid'][100:120, 100:120] = False
    np.savez(file, **maps)
    manifest = json.loads((tmp_path/'inputs.json').read_text())
    manifest['source_depth']['views'][0]['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path/'inputs.json').write_text(json.dumps(manifest))
    shape = (234, 416)
    depth, valid = calibrated_depth('000000', shape)
    rays, center = calibrated_rays('000000', shape)
    points = rays * depth[..., None] + center
    hole = np.argwhere(~valid)
    a, b = hole[0], hole[1]
    points[tuple(a)] = center - rays[tuple(a)] * 2  # behind camera
    points[tuple(b)] = np.nan
    # A finite but misprojected fallback must remain for the gate to reject.
    c = hole[2]; points[tuple(c)][0] += 20
    original = points.copy()
    (tmp_path/'pointmaps').mkdir()
    target = tmp_path/'pointmaps/000000.json'
    target.write_text(json.dumps({'confs': np.ones(shape).tolist(), 'points': points.reshape(-1,3).tolist()}))
    calls = []
    def predict(name, requested_shape):
        calls.append((name, requested_shape))
        return 1/depth  # Oracle predictor tests the integration, not model accuracy.
    report = apply_source_depth_pointmaps(tmp_path, tmp_path/'check.json', predict)
    result = np.asarray(json.loads(target.read_text())['points']).reshape(*shape, 3)
    assert report['views'][0]['completed_invalid_fallback_pixels'] == 2
    assert len(calls) == 1
    np.testing.assert_allclose(result[tuple(a)], rays[tuple(a)]*depth[tuple(a)]+center)
    np.testing.assert_allclose(result[valid], original[valid])
    np.testing.assert_allclose(result[tuple(c)], original[tuple(c)])
    mask = reference_mask('000000.png', shape)
    assert (~mask).sum() == 2 and mask[valid].all()
    # Replay preserves provenance and does not rerun prediction.
    apply_source_depth_pointmaps(tmp_path, tmp_path/'check.json', predict)
    assert len(calls) == 1 and (~reference_mask('000000.png', shape)).sum() == 2
