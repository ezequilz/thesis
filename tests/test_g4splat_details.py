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
    d = build_details(studio, root / 'result.json', result)
    assert d['source_depth']['enabled']
    assert d['views'][0]['panels']['source']['stats']['valid_fraction'] == 2/3


def test_page_has_research_sections_and_no_fabricated_metrics():
    html = Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()
    for term in ['renderG4(r)', 'Generation overview', 'Reprojection not available', 'Calibration mismatch', 'Artifacts and logs', 'Shared across all displayed views', 'Training views are not held-out evaluation']:
        assert term in html
