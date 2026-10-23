"""Capture once, isolate repaired images, and stage calibrated SEVA RGB/depth."""
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from test_splatfix_oraclegs import checkpoint
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.oraclegs.captures import capture_checkpoint, select_checkpoint, validate_capture
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from splat_explorer.splatfix.image_repair import repair_images
from splat_explorer.splatfix.jobs import validate_job, requires_gpu


@pytest.fixture
def capture_backend(monkeypatch):
    import splat_explorer.splatfix.viser_capture as capture
    import splat_explorer.scene as scene
    import splat_explorer.rendering.cpu_splat_renderer as cpu
    calls = []
    class Session:
        def __init__(self, *args, **kwargs):
            calls.append(('session', kwargs))
        def render(self, camera):
            calls.append(('rgb', camera))
            return np.full((camera.height, camera.width, 3), 80, dtype=np.uint8)
    class Depth:
        def __init__(self, scene):
            pass
        def render_depth(self, camera):
            calls.append(('depth', camera))
            return np.full((camera.height, camera.width), 3, dtype=np.float32)
    monkeypatch.setattr(capture, 'CaptureSession', Session)
    monkeypatch.setattr(cpu, 'CpuSplatRenderer', Depth)
    monkeypatch.setattr(scene, 'load_scene', lambda *a, **kw: object())
    return calls


def test_capture_preserves_originals_and_resumes_only_missing_views(tmp_path, capture_backend):
    cp = checkpoint(tmp_path, 2)
    original = copy.deepcopy(cp.manifest)
    partial = capture_checkpoint(cp, view_id='001')
    assert 'seva_capture' not in partial.views[0]
    with pytest.raises(ValueError, match='Step 2'):
        select_checkpoint(partial, 'seva')
    cp = capture_checkpoint(cp)
    before = len(capture_backend)
    capture_checkpoint(cp)
    assert len(capture_backend) == before
    assert len([x for x in capture_backend if x[0] == 'rgb']) == 2
    for old, view in zip(original['views'], cp.views):
        assert {k:v for k,v in view.items() if k != 'seva_capture'} == old
        record = validate_capture(cp, view)
        assert record['camera']['fov_deg'] == 54
        assert record['camera']['width'] == 1024 and record['camera']['height'] == 576
        np.testing.assert_array_equal(record['camera']['c2w'], old['camera']['c2w'])
        assert record['camera']['intrinsics'][0][0] == pytest.approx(1004.85658)
    # Original-image repairs cannot satisfy the new image set's edited mode.
    with pytest.raises(ValueError, match='Step 3'):
        select_checkpoint(cp, 'seva', edited=True)


def test_seva_edit_is_separate_resumable_and_staged_with_cached_depth(tmp_path, capture_backend, monkeypatch):
    cp = capture_checkpoint(checkpoint(tmp_path, 2))
    old_paths = [cp.image_path(v, True).read_bytes() for v in cp.views]
    result = io.BytesIO()
    Image.new('RGB', (1024,576), 'green').save(result, format='PNG')
    edits = []
    class Editor:
        name = 'test'
        def edit_with_references(self, path, references, prompt):
            assert references == [cp.image_path(cp.views[0]['seva_capture'], True)]
            return self.edit(path, prompt)
        def edit(self, path, prompt):
            edits.append(path)
            return SimpleNamespace(error=None, images=[result.getvalue()])
    repair_images(cp, image_set='seva', view_id='000', backend=Editor())
    first = cp.image_path(cp.views[0]['seva_capture'], True)
    saved_bytes, saved_mtime = first.read_bytes(), first.stat().st_mtime_ns
    cp = Checkpoint.load(cp.root)
    repair_images(cp, image_set='seva', backend=Editor())
    assert len(edits) == 2
    assert edits[-1] == cp.image_path(cp.views[1]['seva_capture'])
    assert first.read_bytes() == saved_bytes and first.stat().st_mtime_ns == saved_mtime
    repair_images(cp, image_set='seva', backend=Editor())
    assert len(edits) == 2 and all('/seva/' in str(p) for p in edits)
    assert old_paths == [cp.image_path(v, True).read_bytes() for v in cp.views]
    # Reuse cached depth: staging must not touch either renderer or load a scene.
    import splat_explorer.scene as scene
    monkeypatch.setattr(scene, 'load_scene', lambda *a, **kw: pytest.fail('Cached depth must not rerender'))
    root = tmp_path/'staged'
    manifest = export_inputs(cp, root, 'edited', runtime={'oracle_capture_source':'seva',
        'seva_geometry_guidance':'source_splat', 'oracle_target_frames':2})
    assert manifest['reference_capture_profile'] == 'seva-54deg-1024x576-v1'
    assert manifest['source_depth']['reused_checkpoint_depth']
    frames = json.loads((root/'scene/transforms.json').read_text())['frames']
    assert frames[0]['fl_x'] == pytest.approx(1004.85658)
    assert Image.open(root/'scene/images/ref_000000.png').getpixel((10,10)) == (0,128,0)
    with np.load(root/'source-depth/ref_000000.npz') as depth:
        assert depth['invdepth'][100,100] == pytest.approx(1/3)
    assert len([x for x in capture_backend if x[0] == 'depth']) == 2


def test_capture_failure_never_publishes_incomplete_view(tmp_path, capture_backend, monkeypatch):
    import splat_explorer.rendering.cpu_splat_renderer as cpu
    cp = checkpoint(tmp_path, 2)
    calls = []
    def depth(self, camera):
        calls.append(camera)
        if len(calls) == 2:
            raise RuntimeError('interrupted depth')
        return np.ones((576,1024), dtype=np.float32)
    monkeypatch.setattr(cpu.CpuSplatRenderer, 'render_depth', depth)
    with pytest.raises(RuntimeError, match='interrupted depth'):
        capture_checkpoint(cp)
    saved = Checkpoint.load(cp.root)
    assert saved.views[0].get('seva_capture') and not saved.views[1].get('seva_capture')
    capture_checkpoint(cp)
    assert len(calls) == 3


def test_capture_rejects_corruption_and_changed_pose(tmp_path, capture_backend):
    cp = capture_checkpoint(checkpoint(tmp_path, 2))
    cp.views[0]['seva_capture']['camera']['fov_deg'] = 50
    with pytest.raises(ValueError):
        select_checkpoint(cp, 'seva')
    cp = Checkpoint.load(cp.root)
    path = cp.root/cp.views[0]['seva_capture']['source_depth']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        select_checkpoint(cp, 'seva')


def test_capture_stage_is_local_and_image_sets_survive_job_validation():
    capture = validate_job({'stage':'seva_capture', 'checkpoint':'/saved', 'view_id':'001'})
    assert not requires_gpu({'pipeline':'splatfix', 'splatfix':capture})
    assert capture['view_id'] == '001'
    assert validate_job({'stage':'edit', 'checkpoint':'/saved', 'image_set':'seva'})['image_set'] == 'seva'
    assert validate_job({'stage':'repair', 'reconstruction_method':'oraclegs', 'checkpoint':'/saved',
                         'oracle_capture_source':'seva'})['oracle_capture_source'] == 'seva'
    for view_id in ('../', 1, '99'):
        with pytest.raises(ValueError):
            validate_job({'stage':'seva_capture', 'checkpoint':'/saved', 'view_id':view_id})


@pytest.mark.parametrize('size,accepted', [((1024,576),True), ((1536,864),True), ((1672,941),True), ((1536,1024),True)])
def test_seva_output_dimensions_preserve_full_frame(tmp_path, capture_backend, size, accepted):
    cp = capture_checkpoint(checkpoint(tmp_path, 2))
    raw = io.BytesIO()
    Image.new('RGB', size, 'green').save(raw, format='PNG')
    class Editor:
        name = 'test'
        def edit_with_references(self, path, references, prompt):
            assert references == [cp.image_path(cp.views[0]['seva_capture'], True)]
            return self.edit(path, prompt)
        def edit(self, path, prompt):
            from splat_explorer.splatfix.image_repair import DEFAULT_PROMPT
            assert prompt == DEFAULT_PROMPT
            return SimpleNamespace(error=None, images=[raw.getvalue()])
    if accepted:
        repair_images(cp, image_set='seva', view_id='000', backend=Editor())
    else:
        with pytest.raises(ValueError, match='refusing to crop'):
            repair_images(cp, image_set='seva', view_id='000', backend=Editor())
    capture = Checkpoint.load(cp.root).views[0]['seva_capture']
    audit = capture['image_repair']['dimension_check']
    assert audit['exact_dimensions'] == (size == (1024,576))
    assert audit['same_aspect_ratio'] == (size[0]*576 == size[1]*1024)
    assert bool(capture.get('repaired_rgb')) == accepted
    assert (cp.root/capture['image_repair']['response_image']).read_bytes() == raw.getvalue()
    if accepted:
        assert Image.open(cp.image_path(capture, True)).size == (1024,576)
        assert capture['image_repair']['resize']['source_crop_box'] == [0,0,*size]

from test_splatfix_dashboard import studio


def test_dashboard_defaults_to_seva_and_capture_jobs_are_deduplicated(studio, capture_backend):
    cp = checkpoint(studio.checkpoint_root, 2)
    with pytest.raises(ValueError, match='Step 2'):
        studio.create({'stage':'repair', 'checkpoint':str(cp.root), 'reconstruction_method':'oraclegs', 'mode':'baseline'})
    job = studio.create({'stage':'seva_capture', 'checkpoint':str(cp.root), 'view_id':'000'})
    assert job['config']['splatfix']['view_id'] == '000'
    with pytest.raises(ValueError, match='already queued'):
        studio.create({'stage':'seva_capture', 'checkpoint':str(cp.root)})
    cp = capture_checkpoint(cp)
    row = next(c for c in studio.checkpoints() if c['id'] == str(cp.root))
    assert row['seva_captured'] == 2 and row['seva_edited'] == 0
    assert row['views'][0]['seva_original_url']
    run = studio.create({'stage':'repair', 'checkpoint':str(cp.root), 'reconstruction_method':'oraclegs', 'mode':'baseline'})
    assert run['config']['splatfix']['oracle_capture_source'] == 'seva'
    with pytest.raises(ValueError, match='Step 3'):
        studio.create({'stage':'repair', 'checkpoint':str(cp.root), 'reconstruction_method':'oraclegs', 'mode':'edited'})
    run = studio.create({'stage':'edit', 'checkpoint':str(cp.root), 'image_set':'seva'})
    assert run['config']['splatfix']['image_set'] == 'seva'


def test_seva_only_checkpoint_is_listed_and_original_jobs_are_rejected(studio, capture_backend):
    cp = capture_checkpoint(checkpoint(studio.checkpoint_root, 2))
    for view in cp.views:
        view['original_rgb'] = None
        view['repaired_rgb'] = None
        view['camera'] = copy.deepcopy(view['seva_capture']['camera'])
    cp.manifest['metadata']['capture_set'] = 'seva'
    cp.save()
    row = next(c for c in studio.checkpoints() if c['id'] == str(cp.root))
    assert row['complete'] and row['seva_captured'] == 2
    assert all(v['original_url'] is None and v['seva_original_url'] for v in row['views'])
    with pytest.raises(ValueError, match='no original-FOV'):
        studio.create({'stage':'edit','checkpoint':str(cp.root),'image_set':'saved'})
    job=studio.create({'stage':'edit','checkpoint':str(cp.root),'image_set':'seva'})
    assert job['config']['splatfix']['image_set']=='seva'


def test_77_capture_preserves_other_sets_and_resumes(tmp_path, capture_backend):
    cp = capture_checkpoint(checkpoint(tmp_path, 2))
    original = copy.deepcopy(cp.manifest)
    cp = capture_checkpoint(cp, capture_set='77', view_id='001')
    assert 'capture_77' not in cp.views[0]
    cp = capture_checkpoint(cp, capture_set='77')
    before = len(capture_backend)
    capture_checkpoint(cp, capture_set='77')
    assert len(capture_backend) == before
    for old, view in zip(original['views'], cp.views):
        assert {k:v for k,v in view.items() if k != 'capture_77'} == old
        wide = view['capture_77']
        assert wide['camera']['fov_deg'] == 77
        assert wide['camera']['c2w'] == old['camera']['c2w']
        assert wide['camera']['width'] == old['camera']['width']
        assert wide['camera']['height'] == old['camera']['height']
        assert wide['camera']['intrinsics'] != old['camera']['intrinsics']
        assert cp.image_path(wide).exists()
    assert len([c for c in capture_backend if c[0] == 'depth']) == 2


def test_original_capture_restores_seva_only_checkpoint(tmp_path, capture_backend):
    cp = capture_checkpoint(checkpoint(tmp_path, 2))
    for view in cp.views:
        view['original_rgb'] = None
        view['repaired_rgb'] = None
        view['camera'] = copy.deepcopy(view['seva_capture']['camera'])
    cp.save()
    before = copy.deepcopy(cp.views)
    cp = capture_checkpoint(cp, capture_set='saved')
    for old, view in zip(before, cp.views):
        assert view['camera'] == old['camera']
        assert view['seva_capture'] == old['seva_capture']
        assert cp.image_path(view).exists()
    count = len(capture_backend)
    capture_checkpoint(cp, capture_set='saved')
    assert len(capture_backend) == count


def test_77_capture_repair_and_dashboard(studio, capture_backend):
    cp = capture_checkpoint(checkpoint(studio.checkpoint_root, 2), capture_set='77')
    originals = copy.deepcopy(cp.views)
    result = io.BytesIO()
    Image.new('RGB', (32, 24), 'green').save(result, format='PNG')
    class Editor:
        name = 'test'
        def edit(self, path, prompt):
            assert '/fov77/' in str(path)
            return SimpleNamespace(error=None, images=[result.getvalue()])
    job = studio.create({'stage':'edit', 'checkpoint':str(cp.root), 'image_set':'77'})
    assert job['config']['splatfix']['image_set'] == '77'
    repair_images(cp, image_set='77', backend=Editor(), endpoint_repair_context='rgb_only')
    cp = Checkpoint.load(cp.root)
    for old, view in zip(originals, cp.views):
        assert view['original_rgb'] == old['original_rgb']
        assert view['repaired_rgb'] == old['repaired_rgb']
        assert cp.image_path(view['capture_77'], True).exists()
    row = next(c for c in studio.checkpoints() if c['id'] == str(cp.root))
    assert all(v['fov77_original_url'] and v['fov77_repaired_url'] for v in row['views'])
    job = studio.create({'stage':'seva_capture', 'checkpoint':str(cp.root), 'capture_set':'77'})
    assert job['config']['splatfix']['capture_set'] == '77'


@pytest.mark.parametrize('capture_set', ['saved', 'seva', '77'])
def test_capture_set_survives_validation(capture_set):
    assert validate_job({'stage':'seva_capture', 'checkpoint':'/saved', 'capture_set':capture_set})['capture_set'] == capture_set


def test_capture_set_rejects_unknown_value():
    with pytest.raises(ValueError, match='capture_set'):
        validate_job({'stage':'seva_capture', 'checkpoint':'/saved', 'capture_set':'90'})
