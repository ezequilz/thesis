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
