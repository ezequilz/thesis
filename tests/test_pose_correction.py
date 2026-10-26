"""Camera registration, conservative acceptance, and checkpoint/inference contracts."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint, camera_from_record, camera_to_record
from splat_explorer.splatfix import pose_correction as correction
from splat_explorer.splatfix.pose_alignment import fit_pose, render_acceptance


def checkpoint(tmp_path, count=2):
    cp = Checkpoint.create(tmp_path, tmp_path/'scene.ply', target_views=count,
                           metadata={'renderer': {'backend': 'viser'}})
    for i in range(count):
        cp.add_view(np.zeros((240, 416, 3), np.uint8),
                    Camera(np.array([i, 2., 3.]), np.eye(3), width=416, height=240))
        v = cp.views[-1]
        v['repaired_rgb'] = f"views/{v['id']}/repaired.png"
        Image.new('RGB', (416, 240), 'white').save(cp.root/v['repaired_rgb'])
    cp.save()
    return cp


def record(cp, view, status='accepted'):
    camera = camera_from_record(view)
    return {'version': correction.VERSION, 'status': status,
            'inputs': correction.inputs(cp, view),
            'camera': camera_to_record(replace(camera, position=camera.position+[.01, 0, 0]))}


def test_recovers_3d_pose_with_outliers_in_world_coordinates():
    cv = pytest.importorskip('cv2')
    rot = cv.Rodrigues(np.array([.4, -.2, .15]))[0]
    camera = Camera(np.array([4., -7., 2.]), rot, width=832, height=480)
    yy, xx = np.mgrid[:480, :832]
    depth = 4 + .7*np.sin(xx/140) + .8*np.cos(yy/90)
    yy, xx = np.mgrid[32:448:24, 32:800:24]
    p = np.c_[xx.ravel(), yy.ravel()].astype(float)
    z = depth[yy, xx].ravel()
    xyz = (np.c_[p, np.ones(len(p))] @ np.linalg.inv(camera.intrinsics).T)*z[:, None]
    rv, tv = np.array([.025, -.012, .005]), np.array([.03, -.02, .01])
    q = cv.projectPoints(xyz, rv, tv, camera.intrinsics.astype(float), None)[0][:, 0]
    rng = np.random.default_rng(31)
    q += rng.normal(0, .15, q.shape)
    q[::5] = rng.uniform([0, 0], [832, 480], (len(q[::5]), 2))
    result, metrics = fit_pose(camera, p, q, depth, np.ones_like(depth))
    assert result is not None, metrics
    relative = cv.Rodrigues(rv)[0]
    np.testing.assert_allclose(result.position, camera.position + rot @ (-relative.T @ tv), atol=.003)
    np.testing.assert_allclose(result.rotation, rot @ relative.T, atol=.001)
    np.testing.assert_array_equal(result.intrinsics, camera.intrinsics)
    assert metrics['held_out_error_after'] < metrics['held_out_error_before']


def test_local_object_motion_cannot_define_camera():
    cv = pytest.importorskip('cv2')
    c = Camera(np.zeros(3), np.eye(3), width=832, height=480)
    yy, xx = np.mgrid[160:240:5, 200:280:5]
    p = np.c_[xx.ravel(), yy.ravel()].astype(float)
    candidate, report = fit_pose(c, p, p+[10, -6], np.full((480, 832), 4.))
    assert candidate is None
    assert 'too little' in report['reason'] or 'rigid' in report['reason']


def test_image_acceptance_preserves_already_aligned_region():
    before = np.full(12, .5)
    assert render_acceptance(before, before+.03)
    after = before+.05
    after[3] = .1
    assert not render_acceptance(before, after)
    assert not render_acceptance(before, before+.005)


def test_corrected_projection_preserves_original_and_rejected_views(tmp_path):
    cp = checkpoint(tmp_path)
    for v, status in zip(cp.views, ('accepted', 'rejected')):
        v['pose_correction'] = record(cp, v, status)
    cp.save()
    original = (cp.root/'checkpoint.json').read_bytes()
    projected = correction.corrected_checkpoint(cp)
    assert projected.views[0]['camera'] != cp.views[0]['camera']
    assert projected.views[1]['camera'] == cp.views[1]['camera']
    assert projected.manifest['metadata']['generated_pose_correction']['accepted_view_ids'] == ['000']
    assert (cp.root/'checkpoint.json').read_bytes() == original


@pytest.mark.parametrize('change', ['repair', 'camera', 'response', 'source', 'load_settings'])
def test_modified_inputs_invalidate_saved_pose(tmp_path, change):
    cp = checkpoint(tmp_path, 1)
    v = cp.views[0]
    v['pose_correction'] = record(cp, v)
    if change == 'repair':
        Image.new('RGB', (416, 240), 'red').save(cp.image_path(v, repaired=True))
    elif change == 'camera':
        v['camera'] = camera_to_record(replace(camera_from_record(v), position=np.array([1., 0, 0])))
    elif change == 'response':
        Image.new('RGB', (416, 240), 'blue').save(cp.root/'response.png')
        v['image_repair'] = {'response_image': 'response.png'}
    elif change == 'source':
        cp.manifest['source_fingerprint'] = {'sha256': 'changed'}
    else:
        cp.manifest['metadata']['scene_load'] = {'lod_level': 1}
    assert correction.correction_status(cp, v) == 'stale'
    with pytest.raises(ValueError, match='stale'):
        correction.corrected_checkpoint(cp)


def test_correction_saves_transaction_and_keeps_calibration(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path)
    originals = [copy.deepcopy(v['camera']) for v in cp.views]
    from splat_explorer.splatfix import pose_alignment
    monkeypatch.setattr(pose_alignment, 'opencv', lambda: None)
    monkeypatch.setattr(pose_alignment, 'correspondences', lambda *a: ([], []))
    monkeypatch.setattr(pose_alignment, 'fit_pose', lambda c, *a: (replace(c, position=c.position+[.01, 0, 0]), {'reason': 'test'}))
    monkeypatch.setattr(pose_alignment, 'structural_scores', lambda a, b: np.full(12, a.mean()/255))
    class Session:
        def __init__(self, *a, **kw):
            self.count = 0
        def render(self, c):
            self.count += 1
            return np.full((240, 416, 3), 100 if self.count % 2 else 110, np.uint8)
    result = correction.correct_poses(cp, session_factory=Session,
        depth_provider=lambda v: (np.ones((240, 416)), None, {'backend': 'test'}))
    assert [v['camera'] for v in result.views] == originals
    restored = Checkpoint.load(cp.root)
    assert all(correction.correction_status(restored, v) == 'accepted' for v in restored.views)
    assert all((cp.root/v['pose_correction']['artifacts']/'candidate.png').is_file() for v in restored.views)


def test_concurrent_image_change_never_publishes_correction(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path, 1)
    from splat_explorer.splatfix import pose_alignment
    monkeypatch.setattr(pose_alignment, 'opencv', lambda: None)
    monkeypatch.setattr(pose_alignment, 'correspondences', lambda *a: ([], []))
    monkeypatch.setattr(pose_alignment, 'fit_pose', lambda *a: (None, {'reason': 'no fit'}))
    class Session:
        def __init__(self, *a, **kw): pass
        def render(self, c):
            Image.new('RGB', (416, 240), 'blue').save(cp.image_path(cp.views[0], repaired=True))
            return np.zeros((240, 416, 3), np.uint8)
    with pytest.raises(ValueError, match='changed during correction'):
        correction.correct_poses(cp, session_factory=Session,
            depth_provider=lambda v: (np.ones((240, 416)), None, {}))
    assert 'pose_correction' not in Checkpoint.load(cp.root).views[0]


def test_cached_depth_requires_matching_pose_and_checksum(tmp_path):
    cp = checkpoint(tmp_path, 1)
    cp.manifest['source_fingerprint'] = {'sha256': 'test'}
    root = cp.root/'artifixer-inputs'/'cache'
    folder = root/'trajectories'/'anchors'
    folder.mkdir(parents=True)
    v = cp.views[0]
    depth = np.ones((240, 416), np.float32)
    np.save(folder/'depth.npy', depth)
    np.save(folder/'opacity.npy', depth)
    depth_path, alpha_path = 'trajectories/anchors/depth.npy', 'trajectories/anchors/opacity.npy'
    meta = {'depth_convention': 'expected_camera_z', 'camera_convention': 'opencv_c2w',
        'recipe': {'source': cp.manifest['source_fingerprint'], 'views': [v]},
        'anchors': [{'view_id': '000', 'depth': depth_path, 'opacity': alpha_path}],
        'sha256': {p: correction.digest(root/p) for p in (depth_path, alpha_path)}}
    (folder/'trajectory.json').write_text(json.dumps(meta))
    assert correction.cached_depth(cp, v)[2]['backend'] == 'cached_cuda_gsplat'
    v2 = copy.deepcopy(v)
    v2['camera'] = camera_to_record(replace(camera_from_record(v), position=np.zeros(3)))
    assert correction.cached_depth(cp, v2) is None
    np.save(folder/'depth.npy', depth*2)
    assert correction.cached_depth(cp, v) is None


def test_staging_captures_corrected_pose_and_same_normalized_target(tmp_path, monkeypatch):
    from splat_explorer.splatfix import resolution, viser_capture
    cp = checkpoint(tmp_path, 1)
    v = cp.views[0]
    response = np.arange(416*416*3, dtype=np.uint8).reshape(416, 416, 3)
    Image.fromarray(response).save(cp.root/'response.png')
    v['image_repair'] = {'response_image': 'response.png'}
    v['pose_correction'] = record(cp, v)
    projected = correction.corrected_checkpoint(cp)
    seen = []
    def capture(source, cameras, destinations, **kwargs):
        seen.extend(cameras)
        for c, path in zip(cameras, destinations):
            Image.new('RGB', (c.width, c.height)).save(path)
    monkeypatch.setattr(viser_capture, 'capture_rgb', capture)
    staged = resolution.prepare_repair_checkpoint(projected, tmp_path/'staged', 'training')
    np.testing.assert_allclose(seen[0].position, camera_from_record(v).position+[.01, 0, 0], atol=1e-6)
    assert staged.views[0]['image_repair']['reconstruction_resize']['source_kind'] == 'pose_correction_calibrated_target'
    assert cp.views[0]['camera'] != staged.views[0]['camera']


def test_local_worker_wires_correction_command(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from splat_explorer.config import Config
    from splat_explorer.splatfix import executor
    cp = checkpoint(tmp_path/'inputs')
    root = tmp_path/'job'
    root.mkdir()
    cfg = Config({'agent': {'vlm_backend': 'cli_relay'}, 'image_edit': {'backend': 'gpt-image-2'}})
    run = SimpleNamespace(run_id='pose-job', config=SimpleNamespace(image_edit_backend='gpt-image-2',
        splatfix={'stage': 'pose_correction', 'checkpoint': str(cp.root), 'resolution_profile': 'training'}))
    seen = []
    def process(argv, path, stop, tick):
        seen.extend(argv)
        tick()
    monkeypatch.setattr(executor, 'run_process', process)
    store = SimpleNamespace(get_run=lambda _: SimpleNamespace(state=SimpleNamespace(details={})))
    executor.SplatfixExecutor(cfg, store)._local(run, root, lambda: False, lambda **kw: None)
    assert seen[-2:] == ['correct-poses', str(cp.root)]
    assert 'edit' not in seen


def test_cancellation_never_publishes_partial_camera_set(tmp_path, monkeypatch):
    cp = checkpoint(tmp_path, 1)
    before = (cp.root/'checkpoint.json').read_bytes()
    from splat_explorer.splatfix import pose_alignment
    monkeypatch.setattr(pose_alignment, 'opencv', lambda: None)
    with pytest.raises(InterruptedError, match='cancelled'):
        correction.correct_poses(cp, should_stop=lambda: True)
    assert (cp.root/'checkpoint.json').read_bytes() == before


def test_missing_depth_can_use_existing_viser_depth_renderer(tmp_path, monkeypatch):
    pytest.importorskip('cv2')
    from splat_explorer.rendering import cpu_splat_renderer
    import splat_explorer.scene
    cp = checkpoint(tmp_path, 1)
    seen = []
    monkeypatch.setattr(splat_explorer.scene, 'load_scene', lambda *a, **kw: object())
    class Renderer:
        def __init__(self, scene): pass
        def render_depth(self, camera):
            seen.append(camera)
            return np.full((camera.height, camera.width), 3.5, np.float32)
    monkeypatch.setattr(cpu_splat_renderer, 'CpuSplatRenderer', Renderer)
    depth, alpha, metadata = correction.DepthProvider(cp)(cp.views[0])
    assert depth.shape == (240, 416) and np.all(depth == 3.5)
    assert alpha is None and metadata['backend'] == 'viser_cpu_depth_fallback'
    np.testing.assert_array_equal(seen[0].c2w, camera_from_record(cp.views[0]).c2w)
