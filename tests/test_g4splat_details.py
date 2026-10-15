import json
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.web.g4splat_details import build_details, _previews
from test_splatfix_results import setup


def fixture(tmp_path):
    studio, run, root = setup(tmp_path)
    result = {'reconstruction_method': 'g4splat', 'splat_path': 'artifixer3d.ply'}
    (root / 'result.json').write_text(json.dumps(result))
    sfm = root / 'official/mast3r_sfm'
    (sfm / 'images').mkdir(parents=True)
    Image.new('RGB', (8, 6)).save(sfm / 'images/000000.png')
    (root / 'recipe.json').write_text(json.dumps({'inputs': {'views': [{'image': 'images/000000.png', 'view_id': '000'}]}}))
    return studio, root, sfm, result


def test_historical_details_not_passed_and_calibration_mismatch(tmp_path):
    studio, root, sfm, _ = fixture(tmp_path)
    (sfm / 'sparse/0').mkdir(parents=True)
    (sfm / 'sparse/0/cameras.txt').write_text('1 PINHOLE 8 6 4 4 4 2\n')
    row = studio.results.run_detail(studio.results.entries()[0]['id'])
    d = row['g4splat']
    assert d['gate_status'] == 'not_available'
    assert d['views'][0]['calibration_warning']
    assert d['views'][0]['staged_wh'] == [8, 6]
    assert d['views'][0]['panels']['input']['url']
    assert d['views'][0]['panels']['chart']['url'] is None
    assert all(s['status'] == 'Not available' for s in d['generation'])


def test_shared_depth_scale_invalid_mask_and_cached_previews(tmp_path):
    studio, root, sfm, result = fixture(tmp_path)
    folder = sfm / 'plane-refine-depths'
    folder.mkdir()
    a = np.array([[0, 2], [np.nan, 4]], dtype='float32')
    b = np.array([[2, 8], [4, 6]], dtype='float32')
    Image.fromarray(a).save(folder / 'depth_frame000000.tiff')
    Image.fromarray(b).save(folder / 'refine_depth_frame000000.tiff')
    d = build_details(studio, root / 'result.json', result)
    assert d['depth_scale']['max'] > 7
    assert d['views'][0]['panels']['chart']['stats']['valid_fraction'] == .5
    previews = list((root / 'debug-previews').glob('*/*.png'))
    before = [p.stat().st_mtime_ns for p in previews]
    build_details(studio, root / 'result.json', result)
    assert before == [p.stat().st_mtime_ns for p in previews]
    assert any(np.any(np.all(np.asarray(Image.open(p)) == [187, 58, 150], axis=2)) for p in previews)


def test_unsafe_artifacts_not_read_or_exposed(tmp_path):
    studio, root, sfm, result = fixture(tmp_path)
    secret = tmp_path / 'private.json'
    secret.write_text('{"passed":true}')
    (root / 'reprojection-check.json').symlink_to(secret)
    (sfm / 'images/000000.png').unlink()
    (sfm / 'images/000000.png').symlink_to(secret)
    d = build_details(studio, root / 'result.json', result)
    assert d['gate_status'] == 'not_available'
    assert d['views'][0]['panels']['input']['url'] is None
    assert 'Reprojection gate report' not in [a['label'] for a in d['artifacts']]


def test_gate_failures_native_pairing_and_generation(tmp_path):
    studio, root, sfm, result = fixture(tmp_path)
    (root / 'reprojection-check.json').write_text(json.dumps({'passed': False, 'error': 'bad rays', 'views': [{'image': '000000.png', 'passed': False}]}))
    native = root / 'official/free_gaussians'
    (native / 'train/ours_7000/renders').mkdir(parents=True)
    Image.new('RGB', (8, 6)).save(native / 'train/ours_7000/renders/00000.png')
    (native / 'cameras.json').write_text('[{"img_name":"different"}]')
    generation = sfm / 'see3d_render/stage1/cat_img'
    generation.mkdir(parents=True)
    Image.new('RGB', (8, 6)).save(generation / 'overview.png')
    d = build_details(studio, root / 'result.json', result)
    assert d['gate_status'] == 'failed'
    assert d['views'][0]['reprojection']['passed'] is False
    assert not d['native_order_known']
    assert 'native' not in d['views'][0]['panels']
    assert len(d['generation'][0]['panels']) == 1
    (native / 'cameras.json').write_text('[{"img_name":"000000"}]')
    d = build_details(studio, root / 'result.json', result)
    assert d['native_order_known'] and d['views'][0]['panels']['native']['url']


def test_source_depth_preview_uses_saved_validity(tmp_path):
    studio, root, sfm, result = fixture(tmp_path)
    inputs = root.parent.parent / 'inputs'
    inputs.mkdir()
    np.savez(inputs / 'depth.npz', depth=np.ones((2, 3), dtype='float32'), valid=np.array([[True, True, False]]*2))
    source = {'enabled': True, 'views': [{'image': '000000.png', 'path': 'depth.npz'}]}
    (root / 'recipe.json').write_text(json.dumps({'inputs': {'views': [{'image': 'images/000000.png'}], 'source_depth': source}}))
    (inputs / 'inputs.json').write_text(json.dumps({'views': [{'image': 'images/000000.png'}], 'source_depth': source}))
    d = build_details(studio, root / 'result.json', result)
    assert d['source_depth']['enabled']
    assert d['views'][0]['panels']['source']['stats']['valid_fraction'] == 2/3


def test_page_has_research_sections_and_no_fabricated_metrics():
    html = Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()
    for term in ['renderG4(r)', 'Generation overview', 'Reprojection not available', 'Calibration mismatch', 'Artifacts and logs', 'Shared across all displayed views', 'Training views are not held-out evaluation']:
        assert term in html


def local_export(root, *, suffix='abcd', pinned=True):
    import hashlib
    run = root.parent.parent.parent
    inputs = run / f'g4splat-inputs-{suffix}/inputs'
    (inputs / 'source-depth').mkdir(parents=True)
    (inputs / 'images').mkdir()
    Image.new('RGB', (3, 2)).save(inputs / 'images/000000.png')
    path = inputs / 'source-depth/000000.npz'
    valid = np.array([[True, True, False]]*2)
    np.savez_compressed(path, depth=np.where(valid, 2, 0).astype('float32'), valid=valid,
                        confidence=valid.astype('float32'), intrinsics=np.eye(3), c2w=np.eye(4))
    body = {'method': 'g4splat', 'method_variant': 'source-splat-depth-extension',
            'views': [{'image': 'images/000000.png', 'sha256': 'rgb-test', 'view_id': '000'}],
            'source_depth': {'enabled': True, 'views': [{'image': '000000.png',
                'path': 'source-depth/000000.npz', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'valid_fraction': 2/3}]}}
    manifest = inputs / 'inputs.json'
    manifest.write_text(json.dumps(body))
    if pinned:
        (run / 'g4splat-staging.json').write_text(json.dumps({'input_manifest': str(manifest.relative_to(run)),
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'status': 'transferred'}))
    return inputs, body


def test_local_staged_depth_is_shown_before_any_download_for_failed_run(tmp_path):
    studio, root, _, result = fixture(tmp_path)
    inputs, body = local_export(root)
    (root / 'recipe.json').unlink()
    (root / 'result.json').unlink()
    (root / 'artifixer3d.ply').unlink()
    from types import SimpleNamespace
    saved = studio.store.list_runs()[0]
    record = saved.to_dict()
    record['config']['splatfix'].update(stage='repair', reconstruction_method='g4splat', source_depth=True)
    record['state']['status'] = 'error'
    studio.store.list_runs = lambda: [SimpleNamespace(run_id=saved.run_id, path=saved.path, to_dict=lambda: record)]
    # No gpu/inputs and no result manifest: the catalog's pending/failed row
    # must still resolve the canonical details route and local staging export.
    row = studio.results.run_detail(studio.results.entries()[0]['id'])
    d = row['g4splat']
    assert d['input_staging']['status'] == 'transferred'
    assert d['input_staging']['manifest_checksum_verified'] is True
    v = d['views'][0]
    assert v['source_export']['checksum_status'] == 'verified'
    assert [v['source_export']['width'], v['source_export']['height']] == [3, 2]
    assert v['source_export']['intrinsics'] == np.eye(3).tolist()
    for key in ('source', 'source_valid', 'source_confidence'):
        assert v['panels'][key]['url']
    assert v['source_export']['positive_valid_fraction'] == 2/3
    assert not (root.parent.parent / 'inputs').exists()


def test_local_export_missing_changed_and_ambiguous_are_not_verified(tmp_path):
    studio, root, _, result = fixture(tmp_path)
    inputs, body = local_export(root)
    (root / 'recipe.json').write_text(json.dumps({'inputs': body}))
    depth = inputs / 'source-depth/000000.npz'
    depth.unlink()
    d = build_details(studio, root / 'result.json', result)
    assert d['views'][0]['source_export']['checksum_status'] == 'not_available'
    assert d['views'][0]['panels']['source']['url'] is None
    # Changed pinned manifest must not silently choose a different local export.
    (inputs / 'inputs.json').write_text('{}')
    d = build_details(studio, root / 'result.json', result)
    assert d['input_staging']['status'] == 'unverified'
    assert d['views'][0]['panels']['source']['url'] is None


def test_unpinned_local_export_requires_unambiguous_input_identity(tmp_path):
    studio, root, _, result = fixture(tmp_path)
    (root / 'recipe.json').unlink()
    inputs, body = local_export(root, pinned=False)
    d = build_details(studio, root / 'result.json', result)
    assert d['input_staging']['status'] == 'exported_transfer_unrecorded'
    assert d['views'][0]['panels']['source']['url']
    other, body2 = local_export(root, suffix='different', pinned=False)
    body2['views'][0]['sha256'] = 'different-rgb'
    (other / 'inputs.json').write_text(json.dumps(body2))
    d = build_details(studio, root / 'result.json', result)
    assert d['input_staging']['status'] == 'ambiguous'
    assert all(not v['panels']['source']['url'] for v in d['views'])


def test_local_source_checksum_mismatch_is_explicit(tmp_path):
    studio, root, _, result = fixture(tmp_path)
    inputs, body = local_export(root)
    (root / 'recipe.json').write_text(json.dumps({'inputs': body}))
    path = inputs / 'source-depth/000000.npz'
    np.savez(path, depth=np.ones((2, 3)), valid=np.ones((2, 3), dtype=bool))
    d = build_details(studio, root / 'result.json', result)
    assert d['views'][0]['source_export']['checksum_status'] == 'mismatch'


def test_progressive_export_discovery_before_launch_or_result(tmp_path):
    studio, root, _, result = fixture(tmp_path)
    (root / 'recipe.json').unlink()
    run = root.parent.parent.parent
    manifest = run / 'g4splat-inputs-progress/inputs/inputs.json'
    staging = {'status': 'exporting', 'source_depth_enabled': True, 'input_manifest': str(manifest.relative_to(run))}
    (run / 'g4splat-staging.json').write_text(json.dumps(staging))
    d = build_details(studio, root / 'result.json', {})
    assert d['input_staging']['status'] == 'exporting'
    assert d['source_depth']['enabled'] is True
    assert d['views'] == []
    inputs, body = local_export(root, suffix='progress', pinned=False)
    finished = body['source_depth']['views']
    body['source_depth'].update(status='exporting', views=[])
    manifest.write_text(json.dumps(body))
    d = build_details(studio, root / 'result.json', {})
    assert d['views'][0]['source_export_status'] == 'pending'
    assert d['views'][0]['panels']['source']['url'] is None
    body['source_depth']['views'] = finished
    manifest.write_text(json.dumps(body))
    d = build_details(studio, root / 'result.json', {})
    assert d['input_staging']['status'] == 'exporting'
    assert d['views'][0]['source_export_status'] == 'ready'
    assert d['views'][0]['panels']['source']['url']
    assert not d['input_staging']['manifest_checksum_verified']
