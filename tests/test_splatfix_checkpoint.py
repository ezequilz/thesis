"""Offline contracts for reusable selection and resumable paid image edits."""
import io
import json

import numpy as np
import pytest
from PIL import Image

from splat_explorer.image_edit import ImageEditResult
from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint, camera_from_record
from splat_explorer.splatfix.image_repair import repair_images


def make_checkpoint(tmp_path, count=2):
    checkpoint = Checkpoint.create(tmp_path, tmp_path / 'scene.ply', target_views=count,
                                   metadata={'renderer': {'backend': 'viser'}})
    for index in range(count):
        camera = Camera(np.array([index, 1., 2.]), np.eye(3), width=16, height=12)
        checkpoint.add_view(np.full((12, 16, 3), index, np.uint8), camera,
                            metadata={'step': index})
    return checkpoint


def png_bytes(size=(16, 12)):
    stream = io.BytesIO()
    Image.new('RGB', size, (100, 120, 140)).save(stream, format='PNG')
    return stream.getvalue()


class Backend:
    name = 'fake-gpt'

    def __init__(self, fail_on=None, size=(16, 12)):
        self.calls = []
        self.fail_on = fail_on
        self.size = size

    def edit(self, image_path, prompt):
        self.calls.append(image_path)
        if len(self.calls) == self.fail_on:
            return ImageEditResult(error='temporary failure')
        return ImageEditResult(images=[png_bytes(self.size)], payload={'b64_json': 'do not persist'})


def test_dynamic_views_roundtrip_and_isolated_runs(tmp_path):
    checkpoint = make_checkpoint(tmp_path, count=8)
    other = Checkpoint.create(tmp_path, tmp_path / 'scene.ply')
    assert other.root != checkpoint.root
    assert other.target_views == 6
    restored = Checkpoint.load(checkpoint.root)
    assert restored.complete
    assert len(restored.views) == 8
    assert len(list(restored.root.rglob('*.png'))) == 8
    camera = camera_from_record(restored.views[-1])
    np.testing.assert_array_equal(camera.position, [7, 1, 2])
    np.testing.assert_array_equal(camera.rotation, np.eye(3))
    assert camera.width == 16 and camera.height == 12


def test_retry_skips_success_and_originals_remain_baseline(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    originals = [checkpoint.image_path(v).read_bytes() for v in checkpoint.views]
    backend = Backend(fail_on=2)
    with pytest.raises(RuntimeError, match='temporary failure'):
        repair_images(checkpoint, backend=backend)
    restored = Checkpoint.load(checkpoint.root)
    assert restored.views[0]['image_repair']['status'] == 'complete'
    assert restored.views[1]['image_repair']['status'] == 'failed'
    successful_mtime = restored.image_path(restored.views[0], repaired=True).stat().st_mtime_ns
    retry = Backend()
    repair_images(restored, backend=retry)
    assert len(retry.calls) == 1
    assert restored.image_path(restored.views[0], repaired=True).stat().st_mtime_ns == successful_mtime
    assert [restored.image_path(v).read_bytes() for v in restored.views] == originals
    assert len(list(restored.root.rglob('*.png'))) == 4
    assert 'b64_json' not in (restored.root / 'checkpoint.json').read_text()
    # Completed checkpoints need neither credentials nor another image request.
    repair_images(restored)


@pytest.mark.parametrize('response_size,crop_box', [
    ((16, 16), (0, 2, 16, 14)),
    ((32, 24), (0, 0, 32, 24)),
    ((32, 12), (8, 0, 24, 12)),
    ((8, 6), (0, 0, 8, 6)),
])
def test_response_saved_and_resized_without_stretching(tmp_path, response_size, crop_box):
    checkpoint = make_checkpoint(tmp_path, count=1)
    original_camera = json.loads(json.dumps(checkpoint.views[0]['camera']))
    repair_images(checkpoint, backend=Backend(size=response_size))
    checkpoint = Checkpoint.load(checkpoint.root)
    metadata = checkpoint.views[0]['image_repair']
    assert metadata['status'] == 'complete'
    assert metadata['response_size'] == list(response_size)
    assert (checkpoint.root / metadata['response_image']).read_bytes() == png_bytes(response_size)
    assert checkpoint.views[0]['camera'] == original_camera
    with Image.open(checkpoint.image_path(checkpoint.views[0], repaired=True)) as image:
        assert image.size == (16, 12)
        expected = np.empty((12, 16, 3), dtype=np.uint8)
        expected[:] = [100, 120, 140]
        np.testing.assert_array_equal(np.asarray(image), expected)
    assert metadata['resize']['method'] == 'cover_center_crop'
    assert metadata['resize']['source_crop_box'] == list(crop_box)


def test_center_crop_preserves_pixels_instead_of_stretching(tmp_path):
    checkpoint = make_checkpoint(tmp_path, count=1)
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    backend = Backend()
    backend.edit = lambda *_: ImageEditResult(images=[stream.getvalue()])
    repair_images(checkpoint, backend=backend)
    with Image.open(checkpoint.image_path(checkpoint.views[0], repaired=True)) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels[2:14])


@pytest.mark.parametrize('model', ['gpt-image-2.5-sunburst', 'gpt-image-2.5-flare', 'gpt-image-2'])
def test_queued_edit_routes_selected_model_through_worker_config(tmp_path, monkeypatch, model):
    from types import SimpleNamespace
    from splat_explorer.config import Config, load_config
    from splat_explorer.splatfix import executor, cli
    from splat_explorer.image_edit_gpt import GptImageEditBackend

    checkpoint = make_checkpoint(tmp_path, count=1)
    root = tmp_path / 'job'
    root.mkdir()
    cfg = Config({'agent': {'vlm_backend': 'cli_relay'},
                  'image_edit': {'backend': 'gpt-image-2', 'model': 'gpt-image-2',
                                 'timeout_s': 123}})
    run = SimpleNamespace(run_id='job', config=SimpleNamespace(
        image_edit_backend=model,
        splatfix={'stage': 'edit', 'checkpoint': str(checkpoint.root),
                  'resolution_profile': 'training'}))
    seen = []

    def edit(self, image_path, prompt):
        seen.append((self.model, self.timeout_s, image_path, prompt))
        return ImageEditResult(images=[png_bytes()])

    monkeypatch.setattr('splat_explorer.image_edit_gpt._cli_relay_client', lambda cfg: object())
    monkeypatch.setattr(GptImageEditBackend, 'edit', edit)

    def run_process(argv, log_path, stop, progress):
        worker_cfg = load_config(argv[argv.index('--config') + 1])
        cli.edit_views(worker_cfg, SimpleNamespace(checkpoint=argv[-1]))

    monkeypatch.setattr(executor, 'run_process', run_process)
    store = SimpleNamespace(get_run=lambda _: SimpleNamespace(state=SimpleNamespace(details={})))
    executor.SplatfixExecutor(cfg, store)._local(run, root, lambda: False, lambda **kw: None)
    assert len(seen) == 1
    assert seen[0][:2] == (model, 123)
    assert seen[0][2] == checkpoint.image_path(checkpoint.views[0])
    from splat_explorer.splatfix.image_repair import DEFAULT_PROMPT
    assert seen[0][3] == DEFAULT_PROMPT
    assert cfg['image_edit']['model'] == 'gpt-image-2'
    assert Checkpoint.load(checkpoint.root).views[0]['image_repair']['model'] == model


def test_checkpoint_rejects_escaped_paths_and_invalid_counts(tmp_path):
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Checkpoint.create(tmp_path, 'scene.ply', target_views=count)
    checkpoint = make_checkpoint(tmp_path, count=1)
    checkpoint.views[0]['original_rgb'] = '../elsewhere.png'
    checkpoint.save()
    with pytest.raises(ValueError, match='inside the run'):
        Checkpoint.load(checkpoint.root)


def test_repair_requires_complete_selection_without_api_calls(tmp_path):
    checkpoint = Checkpoint.create(tmp_path, 'scene.ply')
    backend = Backend()
    with pytest.raises(ValueError, match='Finish selecting'):
        repair_images(checkpoint, backend=backend)
    assert not backend.calls


@pytest.mark.parametrize('rotation', [np.zeros((3, 3)), np.diag([1., 1., -1.]),
                                       np.diag([1., 1., 2.])])
def test_invalid_saved_rotations_rejected(tmp_path, rotation):
    checkpoint = make_checkpoint(tmp_path, count=1)
    checkpoint.views[0]['camera']['rotation'] = rotation.tolist()
    checkpoint.save()
    with pytest.raises(ValueError, match='rotation'):
        Checkpoint.load(checkpoint.root)


@pytest.mark.parametrize('key,value', [('width', 16.9), ('width', True),
                                      ('c2w', np.eye(4).tolist()),
                                      ('intrinsics', np.eye(3).tolist())])
def test_inconsistent_saved_calibration_rejected(tmp_path, key, value):
    checkpoint = make_checkpoint(tmp_path, count=1)
    checkpoint.views[0]['camera'][key] = value
    checkpoint.save()
    with pytest.raises(ValueError):
        Checkpoint.load(checkpoint.root)


def test_source_file_fingerprint_detects_changed_scene(tmp_path):
    from splat_explorer.splatfix.checkpoint import source_fingerprint, validate_source
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'original scene')
    checkpoint = Checkpoint.create(tmp_path / 'runs', source)
    assert checkpoint.manifest['source_fingerprint'] == source_fingerprint(source)
    validate_source(checkpoint)
    source.write_bytes(b'modified scene')
    with pytest.raises(ValueError, match='changed since view selection'):
        validate_source(checkpoint)
    source.unlink()
    # Loading saved cameras/images does not need the source for cached replay.
    Checkpoint.load(checkpoint.root)
    with pytest.raises(FileNotFoundError, match='Source scene'):
        validate_source(checkpoint)


def test_directory_fingerprint_covers_nested_assets_and_ignores_mount_path(tmp_path):
    from splat_explorer.splatfix.checkpoint import source_fingerprint
    import shutil
    root = tmp_path / 'assets'
    (root / 'chunk').mkdir(parents=True)
    (root / 'meta.json').write_text('{}')
    (root / 'lod-meta.json').write_text('{}')
    (root / 'chunk' / 'means.webp').write_bytes(b'pixels')
    fingerprint = source_fingerprint(root)
    assert fingerprint['file_count'] == 3
    assert source_fingerprint(root / 'meta.json') == fingerprint
    assert source_fingerprint(root / 'lod-meta.json') == fingerprint
    (root / '.DS_Store').write_bytes(b'irrelevant')
    assert source_fingerprint(root) == fingerprint
    copied = tmp_path / 'different_mount'
    shutil.copytree(root, copied)
    assert source_fingerprint(copied) == fingerprint
    (copied / 'chunk' / 'means.webp').write_bytes(b'changed pixels')
    assert source_fingerprint(copied) != fingerprint
    (root / 'chunk' / 'means.webp').rename(root / 'chunk' / 'renamed.webp')
    assert source_fingerprint(root) != fingerprint
