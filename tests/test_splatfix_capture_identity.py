"""Desktop-to-LRZ capture replay must survive libm roundoff, not changed inputs."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.artifixer import preflight, staging_cache, source_plan
from splat_explorer.splatfix.artifixer.source_staging import lean_checkpoint
from splat_explorer.splatfix.checkpoint import Checkpoint, atomic_json
from splat_explorer.splatfix.resolution import prepare_repair_checkpoint
from test_splatfix_capture_preflight import source, make_plan


@pytest.mark.parametrize('bound', [False, True])
def test_cross_platform_lean_replay_preserves_captured_cameras_and_plan(source, monkeypatch, bound):
    cp, root, runtime, captures, _ = source
    initial = preflight.prepare_source_captures(cp, runtime, root)
    cache = Path(initial['input_cache'])
    saved = json.loads((cache / 'captures.json').read_text())
    plan = make_plan(cp, runtime, root / 'desktop', count=5)
    lean = Checkpoint.load(lean_checkpoint(cp, initial, root / 'lean'))
    runtime = {**runtime, 'source_rgb_prepared': True}
    if bound:
        runtime['source_anchor_capture_signature'] = cache.name
    # The actual LRZ failure differed by one float64 ULP, while the renderer's
    # float32 intrinsics and all camera poses were byte-for-byte identical.
    fov = saved['recipe']['cameras'][0]['fov_deg']
    monkeypatch.setattr(np, 'degrees', lambda _: np.nextafter(fov, np.inf))
    monkeypatch.setattr('splat_explorer.splatfix.viser_capture.capture_rgb',
                        lambda *a, **kw: pytest.fail('Worker must not recapture'))
    gpu = prepare_repair_checkpoint(lean, root / 'gpu', runtime['resolution_profile'],
        artifixer_runtime=runtime, reuse_artifixer_inputs=True)
    assert [v['camera'] for v in gpu.views] == saved['recipe']['cameras']
    assert Path(gpu.manifest['metadata']['artifixer_input_cache']).name == cache.name
    assert [staging_cache.digest(gpu.image_path(v)) for v in gpu.views] == saved['sha256']
    assert len(captures) == 2
    source_plan.validate(plan, gpu, runtime)
    gpu_plan = source_plan.build(gpu, runtime, plan['orbit'], plan)
    assert gpu_plan == plan
    # The returned plan must also pass desktop validation and warm the exact
    # trajectory cache consumed by reconstruction, without another anchor pass.
    completed = preflight.prepare_source_captures(cp, runtime, root, source_plan=gpu_plan)
    assert completed['captured_frames'] == 3 and len(captures) == 5
    full = prepare_repair_checkpoint(cp, root / 'full', runtime['resolution_profile'],
        artifixer_runtime=runtime, reuse_artifixer_inputs=True)
    source_plan.validate(gpu_plan, full, runtime)


def test_explicit_binding_wins_over_another_equivalent_capture(source, monkeypatch):
    cp, root, runtime, _, _ = source
    initial = preflight.prepare_source_captures(cp, runtime, root)
    cache = Path(initial['input_cache'])
    original = json.loads((cache / 'captures.json').read_text())
    alternative = copy.deepcopy(original)
    for camera in alternative['recipe']['cameras']:
        camera['fov_deg'] = float(np.nextafter(camera['fov_deg'], np.inf))
    signature = hashlib.sha256(json.dumps(alternative['recipe'], sort_keys=True).encode()).hexdigest()
    alternate_cache = cache.parent / signature; alternate_cache.mkdir()
    for index, camera in enumerate(alternative['recipe']['cameras']):
        target = alternate_cache / f'{index:05d}.png'
        Image.new('RGB', (camera['width'], camera['height']), 'blue').save(target)
        alternative['sha256'][index] = staging_cache.digest(target)
    atomic_json(alternate_cache / 'captures.json', alternative)
    monkeypatch.setattr(np, 'degrees', lambda _: alternative['recipe']['cameras'][0]['fov_deg'])
    prepared = prepare_repair_checkpoint(cp, root / 'gpu', runtime['resolution_profile'], reuse_artifixer_inputs=True,
        artifixer_runtime={**runtime, 'source_rgb_prepared': True, 'source_anchor_capture_signature': cache.name})
    assert [staging_cache.digest(prepared.image_path(v)) for v in prepared.views] == original['sha256']
    assert [v['camera'] for v in prepared.views] == original['recipe']['cameras']


@pytest.mark.parametrize('changed', ['source', 'profile', 'renderer', 'scene_load', 'up_axis',
                                   'position', 'rotation', 'c2w', 'intrinsics', 'width', 'fov_deg'])
def test_roundoff_match_rejects_changed_source_or_camera(source, changed):
    cp, root, runtime, _, _ = source
    initial = preflight.prepare_source_captures(cp, runtime, root)
    saved = json.loads((Path(initial['input_cache']) / 'captures.json').read_text())['recipe']
    requested = copy.deepcopy(saved)
    requested['cameras'][0]['fov_deg'] = float(np.nextafter(requested['cameras'][0]['fov_deg'], np.inf))
    assert staging_cache.compatible_capture_recipe(saved, requested)
    if changed in ('source', 'profile', 'renderer', 'scene_load', 'up_axis'):
        requested[changed] = 'changed'
    elif changed in ('position', 'rotation', 'c2w', 'intrinsics'):
        values = np.asarray(requested['cameras'][0][changed], dtype=np.float64)
        values.flat[0] = np.nextafter(values.flat[0], np.inf)
        requested['cameras'][0][changed] = values.tolist()
    else:
        requested['cameras'][0][changed] += .001 if changed == 'fov_deg' else 1
    assert not staging_cache.compatible_capture_recipe(saved, requested)


def test_compatible_capture_still_requires_pixel_checksums(source, monkeypatch):
    cp, root, runtime, _, _ = source
    initial = preflight.prepare_source_captures(cp, runtime, root)
    cache = Path(initial['input_cache'])
    camera = json.loads((cache / 'captures.json').read_text())['recipe']['cameras'][0]
    monkeypatch.setattr(np, 'degrees', lambda _: np.nextafter(camera['fov_deg'], np.inf))
    (cache / '00000.png').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare_repair_checkpoint(cp, root / 'gpu', runtime['resolution_profile'], reuse_artifixer_inputs=True,
                                 artifixer_runtime={**runtime, 'source_rgb_prepared': True})


def test_missing_bound_cache_does_not_use_an_unbound_capture(source):
    cp, root, runtime, _, _ = source
    preflight.prepare_source_captures(cp, runtime, root)
    with pytest.raises(ValueError, match='capture cache is missing:'):
        prepare_repair_checkpoint(cp, root / 'gpu', runtime['resolution_profile'], reuse_artifixer_inputs=True,
            artifixer_runtime={**runtime, 'source_anchor_capture_signature': '0' * 64})
