import json
from pathlib import Path
from types import SimpleNamespace

from splat_explorer.config import Config
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.web.splatfix_studio import SplatfixStudio
from splat_explorer.web.splatfix_results import ResultViser


def setup(tmp_path):
    cfg = Config({'output': {'dir': str(tmp_path)}, 'agent': {'model': 'test'}})
    studio = SplatfixStudio(SimpleNamespace(cfg=cfg), SceneRunStore(tmp_path / 'scene-runs'))
    studio.scenes = lambda: [{'id': 'room', 'label': 'Room', 'path': str(tmp_path / 'source.ply')}]
    record = studio.create({'stage': 'select', 'scene_id': 'room'})
    root = studio.root / record['run_id']
    result = root / 'gpu/results/benchmark_1.3b_abc'
    result.mkdir(parents=True)
    (result / 'artifixer3d.ply').write_bytes(b'ply')
    (result / 'original.ply').write_bytes(b'ply')
    (result / 'result.json').write_text(json.dumps({'splat_path': '/workspace/jobs/run/results/benchmark_1.3b_abc/artifixer3d.ply', 'model_variant': '1.3b'}))
    return studio, root, result


def test_catalog_maps_remote_result_and_metrics_without_conflating_runs(tmp_path):
    studio, root, result = setup(tmp_path)
    external = studio.benchmark_root / 'bicycle'
    external.mkdir(parents=True)
    (external / 'published-test-metrics-unrelated.json').write_text(json.dumps({'benchmark_provenance': {'root': '/other'}, 'aggregate': {'bad': {}}}))
    (external / 'published-test-metrics-matching.json').write_text(json.dumps({'benchmark_provenance': {'root': f'/remote/{root.name}/results/benchmark_1.3b_abc'}, 'aggregate': {'artifixer3d': {'psnr': 16.8}}, 'published_test_count': 25}))
    rows = studio.results.entries()
    assert len(rows) == 1
    row = rows[0]
    assert row['original'] and row['repaired']
    assert row['metrics']['artifixer3d']['psnr'] == 16.8
    assert row['test_count'] == 25
    assert '_paths' not in row
    assert studio.results.detail(row['id'])['_paths']['repaired'] == result / 'artifixer3d.ply'
    (root / 'scene_original.ply').write_bytes(b'ply')
    assert len(studio.results.entries()) == 2


def test_escaping_result_does_not_enter_catalog(tmp_path):
    studio, _, result = setup(tmp_path)
    (result / 'artifixer3d.ply').unlink()
    outside = tmp_path / 'secret.ply'
    outside.write_bytes(b'ply')
    (result / 'artifixer3d.ply').symlink_to(outside)
    assert studio.results.entries() == []


def test_fresh_fit_disables_highlight(tmp_path):
    studio, _, _ = setup(tmp_path)
    key = studio.results.entries()[0]['id']
    viewer = ResultViser(studio.results, background=False)
    assert viewer.snapshot(key)['highlight_available'] is False
    assert viewer.show(key, highlight=True)['error'] == 'highlight unavailable'
    assert viewer.ply_paths('unknown') == {'original': None, 'repaired': None}


def test_result_page_has_review_and_real_metric_fields():
    html = Path('src/splat_explorer/web/static/splatfix_results.html').read_text()
    assert '/api/splatfix/results' in html
    assert 'r.viser_url' in html
    assert 'm.psnr' in html and 'm.ssim' in html and 'm.lpips' in html
    review = Path('src/splat_explorer/web/static/scene_run_viser.html').read_text()
    assert 'resultReview' in review and 'highlight.hidden=true' in review


def test_saved_checkpoint_original_uses_registered_source(tmp_path):
    studio, root, result = setup(tmp_path)
    (result / 'original.ply').unlink()
    cp = studio.checkpoint_root / 'saved'
    cp.mkdir(parents=True)
    source = tmp_path / 'source.ply'
    source.write_bytes(b'ply')
    (cp / 'checkpoint.json').write_text(json.dumps({'scene_path': str(source)}))
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text())
    config['splatfix'].update(stage='repair', checkpoint=str(cp), mode='baseline')
    config_path.write_text(json.dumps(config))
    row = studio.results.entries()[0]
    assert row['original']
    assert studio.results.detail(row['id'])['_paths']['original'] == source
    unregistered = tmp_path / 'unregistered.ply'
    unregistered.write_bytes(b'ply')
    (cp / 'checkpoint.json').write_text(json.dumps({'scene_path': str(unregistered)}))
    assert not studio.results.entries()[0]['original']


def test_catalog_handles_relative_output_root(tmp_path, monkeypatch):
    studio, _, _ = setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    studio.root = Path('scene-runs')
    assert len(studio.results.entries()) == 1


def test_detail_excludes_generated_anchor_predictions_and_preserves_indices(tmp_path):
    studio, _, root = setup(tmp_path)
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    for i in (0, 2, 10):
        (pred / f'{i:05}.png').write_bytes(b'png')
    (root / 'supervision.json').write_text(json.dumps({'groups': [
        {'source_index': 0, 'reference': '/saved/original.png'},
        {'source_index': 2, 'reference': None}, {'source_index': 10, 'reference': None}]}))
    row = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert [f['index'] for f in row['gallery']['frames']] == [2, 10]
    assert row['gallery']['reference_count'] == 1
    assert row['gallery']['verified_inputs'] is True
    assert row['gallery']['expected_count'] == 2
    assert '_manifest' not in row and '_paths' not in row
    assert row['detail_url'].startswith('/splatfix/results/run?id=')


def test_detail_maps_benchmark_predictions_and_split(tmp_path):
    studio, _, root = setup(tmp_path)
    pred = root / 'inference/model/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    for i in (0, 1, 2):
        (pred / f'{i:05}.png').write_bytes(b'png')
    prep = root / 'prepared/bicycle'
    prep.mkdir(parents=True)
    (prep / 'refs.json').write_text('[1]')
    (prep / 'targets.json').write_text('[0,2]')
    (prep / 'split_trajectory.json').write_text(json.dumps({'test': {'bicycle': {'selected_indices_path': 'refs.json', 'target_indices_path': 'targets.json'}}}))
    p = root / 'result.json'
    data = json.loads(p.read_text())
    data.update(prediction_frames=f'/remote/{root.name}/inference/model/frames/batch_0000/pred', inference_split='prepared/bicycle/split_trajectory.json')
    p.write_text(json.dumps(data))
    row = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert [f['index'] for f in row['gallery']['frames']] == [0, 2]
    assert row['gallery']['reference_count'] == 1
    assert studio.results.run_detail('unknown') is None


def test_detail_rejects_escaping_images_and_does_not_use_plus(tmp_path):
    studio, _, root = setup(tmp_path)
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    secret = tmp_path / 'secret.png'
    secret.write_bytes(b'png')
    (pred / '00001.png').symlink_to(secret)
    plus = root / 'plus'
    plus.mkdir()
    (plus / '00001.png').write_bytes(b'png')
    p = root / 'result.json'
    data = json.loads(p.read_text())
    data.update(prediction_frames='../escape', plus_frames='plus')
    p.write_text(json.dumps(data))
    assert studio.results.run_detail(studio.results.entries()[0]['id'])['gallery']['frames'] == []


def test_detail_page_has_accessible_fullsize_navigation():
    html = Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()
    assert 'loading="lazy"' in html and 'showModal()' in html
    assert "e.key==='ArrowLeft'" in html and "e.key==='ArrowRight'" in html
    assert 'requestFullscreen' in html and 'aria-label="Close image"' in html
    assert 'g.verified_inputs' in html
