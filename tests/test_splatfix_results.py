import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_context_selection_logs_are_exposed_in_result_details(tmp_path):
    studio, _, result = setup(tmp_path)
    evidence = {'complete': True, 'sequences': [{'reference_indices': [1, 2]}]}
    (result / 'inference-context.json').write_text(json.dumps(evidence))
    row = studio.results.run_detail(studio.results.entries()[0]['id'])
    artifact, = [a for a in row['json_artifacts'] if a['name'] == 'inference-context.json']
    assert artifact['snapshot'] == evidence and artifact['url']


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
    assert 'formatBytes(r.output_size_bytes)' in html
    assert 'dim run-size' in html
    review = Path('src/splat_explorer/web/static/scene_run_viser.html').read_text()
    assert 'resultReview' in review and 'highlight.hidden=true' in review


def test_catalog_size_uses_job_status_not_completed_manifest(tmp_path):
    studio, root, _ = setup(tmp_path)
    studio.store.set_status(root.name, 'running')
    row = studio.results.entries()[0]
    assert row['status'] == 'completed'
    assert row['output_size_bytes'] is None
    studio.store.set_status(root.name, 'stopped')
    row = studio.results.entries()[0]
    assert row['output_size_bytes'] > 0
    assert row['output_size_bytes'] == json.loads((root / 'output-size.json').read_text())['bytes']


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


@pytest.mark.parametrize('location', ['manifest', 'preview', 'fallback'])
def test_detail_plus_gallery_is_separate_and_numerically_ordered(tmp_path, location):
    studio, run, root = setup(tmp_path)
    folder = root / ('plus/splatfix/frames/batch_0000/pred' if location == 'fallback' else 'custom-plus')
    folder.mkdir(parents=True)
    for name in ('10.png', '2.jpg', '0.png'):
        (folder / name).write_bytes(b'image')
    (folder / '3.png').touch()
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    (folder / '4.png').symlink_to(outside)
    if location != 'fallback':
        manifest = root / ('result.json' if location == 'manifest' else 'inference-preview.json')
        data = json.loads(manifest.read_text()) if manifest.exists() else {}
        data['plus_frames'] = f'/remote/results/{root.name}/custom-plus'
        manifest.write_text(json.dumps(data))
    detail = studio.results.run_detail(run.name)
    assert detail['gallery']['frames'] == []
    assert [f['index'] for f in detail['plus_gallery']['frames']] == [0, 2, 10]
    assert all(f['url'] == studio.file_url(folder / f['name']) for f in detail['plus_gallery']['frames'])


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


def test_detail_shows_run_comparison_and_exact_photographic_references(tmp_path):
    studio, _, root = setup(tmp_path)
    prep = root / 'prepared'
    (prep / 'images').mkdir(parents=True)
    (prep / 'images/photo.jpg').write_bytes(b'photo')
    (prep / 'selected.json').write_text('[1]')
    (prep / 'transforms.json').write_text(json.dumps({'frames': [{}, {'file_path': 'images/photo.jpg'}]}))
    (prep / 'split.json').write_text(json.dumps({'test': {'scene': {
        'selected_indices_path': 'selected.json', 'transforms_path': 'transforms.json', 'image_root': '.'}}}))
    manifest = root / 'result.json'
    data = json.loads(manifest.read_text())
    data['inference_split'] = 'prepared/split.json'
    manifest.write_text(json.dumps(data))
    (root / 'stage-comparison.jpg').write_bytes(b'comparison')
    (root / 'stage-comparison.json').write_text(json.dumps({'caption': 'Separate control included'}))
    (root / 'trajectory-quality.png').write_bytes(b'chart')
    (root / 'trajectory-quality.json').write_text(json.dumps({'caption': 'Actual run trajectory'}))
    key = studio.results.entries()[0]['id']
    row = studio.results.run_detail(key)
    assert row['stage_comparison']['caption'] == 'Separate control included'
    assert [item['title'] for item in row['comparison_gallery']] == ['Stage comparison', 'Trajectory and image quality']
    assert row['comparison_gallery'][1]['caption'] == 'Actual run trajectory'
    assert row['reference_images'] == [{'index': 1, 'name': 'photo.jpg', 'url': studio.file_url(prep / 'images/photo.jpg')}]
    (prep / 'images/photo.jpg').unlink()
    (prep / 'images/photo.jpg').symlink_to(tmp_path / 'secret.jpg')
    (tmp_path / 'secret.jpg').write_bytes(b'private')
    assert studio.results.run_detail(key)['reference_images'] == []


def test_saved_references_recover_remote_links_only_when_run_hash_matches(tmp_path):
    import hashlib
    studio, run, root = setup(tmp_path)
    cp = studio.checkpoint_root / 'saved'
    (cp / 'views/000').mkdir(parents=True)
    image = cp / 'views/000/repaired.png'
    image.write_bytes(b'actual edited reference')
    config = json.loads((run / 'config.json').read_text())
    config['splatfix'].update(checkpoint=str(cp), stage='repair', mode='baseline')
    (run / 'config.json').write_text(json.dumps(config))
    remote = '/workspace/job/checkpoint/views/000/repaired.png'
    (root / 'request.json').write_text(json.dumps({'checkpoint_root': '/workspace/job/checkpoint',
        'references': [remote], 'reference_sha256': [hashlib.sha256(image.read_bytes()).hexdigest()]}))
    (root / 'supervision.json').write_text(json.dumps({'groups': [{'source_index': 4, 'reference': remote}]}))
    key = studio.results.entries()[0]['id']
    assert studio.results.run_detail(key)['reference_images'][0]['url'] == studio.file_url(image)
    image.write_bytes(b'a newer edit must not be shown for the historical run')
    assert studio.results.run_detail(key)['reference_images'] == []


def test_interrupted_result_visible_without_reconstructed_ply(tmp_path):
    studio, root, result = setup(tmp_path)
    (result / 'result.json').unlink()
    (result / 'artifixer3d.ply').unlink()
    (result / 'original.ply').unlink()
    (result / 'partial-result.json').write_text(json.dumps({'incomplete': True, 'interruption': {'reason': 'Stopped before expiry'}, 'frame_count': 346}))
    (result / 'benchmark-run.json').write_text('{}')
    rows = studio.results.entries()
    assert len(rows) == 1
    assert rows[0]['incomplete'] is True
    assert rows[0]['repaired'] is False
    assert rows[0]['ply_url'] is None
    assert rows[0]['diagnostics_url']
    detail = studio.results.run_detail(rows[0]['id'])
    assert detail['interruption']['reason'] == 'Stopped before expiry'
    assert detail['gallery']['count'] == 0


def test_completed_result_supersedes_partial_record(tmp_path):
    studio, root, result = setup(tmp_path)
    (result / 'partial-result.json').write_text(json.dumps({'incomplete': True}))
    rows = studio.results.entries()
    assert len(rows) == 1
    assert not rows[0]['incomplete']


def test_running_inference_preview_becomes_completed_result_at_same_url(tmp_path):
    from splat_explorer.splatfix.inference_preview import publish_preview
    studio, run, root = setup(tmp_path)
    (root / 'result.json').unlink()
    (root / 'artifixer3d.ply').unlink()
    (root / 'original.ply').unlink()
    studio.store.update_status(run.name, status='running', phase='artifixer3d')
    pred = root / 'inference/model/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    for i in (0, 1, 2):
        (pred / f'{i:05d}.png').write_bytes(b'png')
    publish_preview(root, pred, [0, 2], 1, frame_count=3)
    row, = studio.results.entries()
    key = row['id']
    detail = studio.results.run_detail(key)
    assert row['status'] == 'running' and not row['incomplete']
    assert row['inference_ready'] and not row['repaired'] and not row['ply_url']
    assert [f['index'] for f in detail['gallery']['frames']] == [0, 2]
    assert detail['gallery']['verified_inputs'] and detail['gallery']['reference_count'] == 1
    (root / 'artifixer3d.ply').write_bytes(b'ply')
    (root / 'result.json').write_text(json.dumps({'splat_path': 'artifixer3d.ply'}))
    studio.store.update_status(run.name, status='completed')
    finished, = studio.results.entries()
    assert finished['id'] == key and finished['repaired']
    assert not finished['inference_ready']


def test_reconstruction_job_appears_before_outputs_and_keeps_detail_link(tmp_path):
    studio, root, result = setup(tmp_path)
    import shutil
    shutil.rmtree(result)
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text())
    config['splatfix'].update(stage='benchmark', model='1.3b', source=str(tmp_path / 'benchmark'))
    config_path.write_text(json.dumps(config))
    row, = studio.results.entries()
    assert row['id'] == root.name
    assert row['can_stop'] and not row['repaired']
    assert studio.results.run_detail(row['id'])['gallery']['frames'] == []
    result.mkdir()
    (result / 'inference-preview.json').write_text(json.dumps({'inference_ready': True}))
    preview, = studio.results.entries()
    assert studio.results.run_detail(row['id'])['inference_ready']
    studio.store.update_status(root.name, status='stopping')
    assert not studio.results.run_detail(row['id'])['can_stop']
    (result / 'result.json').write_text('{}')
    (result / 'artifixer3d.ply').write_bytes(b'ply')
    assert studio.results.run_detail(row['id'])['repaired']
    assert studio.results.entries()[0]['id'] == preview['id']


def test_live_detail_exposes_graceful_stop_action():
    html = Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()
    assert 'Stop gracefully &amp; download' in html
    assert "fetch('/api/splatfix/cancel'" in html
    assert 'run_id:liveRun' in html and 'r.can_stop' in html


def test_hidden_run_persists_across_catalog_reload_and_output_changes(tmp_path):
    from splat_explorer.web.splatfix_results import ResultCatalog
    studio, root, result = setup(tmp_path)
    key = studio.results.entries()[0]['id']
    studio.results.set_hidden(root.name, True)
    studio.results = ResultCatalog(studio)
    assert studio.results.entries()[0]['hidden'] is True
    assert studio.results.run_detail(key)['hidden'] is True
    (root / 'scene_original.ply').write_bytes(b'ply')
    assert all(row['hidden'] for row in studio.results.entries())
    (result / 'result.json').rename(result / 'partial-result.json')
    assert all(row['hidden'] for row in studio.results.entries())
    studio.results.set_hidden(root.name, False)
    assert all(not row['hidden'] for row in studio.results.entries())
    saved = json.loads((studio.root / 'result-visibility.json').read_text())
    assert saved[root.name]['hidden'] is False
    assert saved[root.name]['updated_at']
    assert (result / 'artifixer3d.ply').read_bytes() == b'ply'


def test_visibility_rejects_invalid_updates_and_preserves_other_records(tmp_path):
    import pytest
    studio, root, _ = setup(tmp_path)
    second = studio.create({'stage': 'select', 'scene_id': 'room'})
    studio.results.set_hidden(root.name, True)
    studio.results.set_hidden(second['run_id'], True)
    studio.results.set_hidden(root.name, False)
    assert studio.results.visibility_records()[second['run_id']]['hidden'] is True
    for run_id, hidden in [('missing', True), ('../escape', True), (root.name, 'false'), (None, False)]:
        with pytest.raises(ValueError):
            studio.results.set_hidden(run_id, hidden)
    path = studio.root / 'result-visibility.json'
    path.write_text('{broken')
    with pytest.raises(ValueError):
        studio.results.set_hidden(root.name, True)
    assert path.read_text() == '{broken'


def test_visibility_endpoint_saves_and_reports_write_failure(tmp_path, monkeypatch):
    from splat_explorer.web.server import DashboardHandler
    studio, root, _ = setup(tmp_path)
    responses = []
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=studio), command='POST',
                              _send_json=lambda body, code=200: responses.append((body, code)))
    endpoint = '/api/splatfix/results/visibility'
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name, 'hidden': True})
    assert responses[-1][1] == 200 and responses[-1][0]['hidden'] is True
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name, 'hidden': 'false'})
    assert responses[-1][1] == 400
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr('splat_explorer.web.splatfix_results._atomic_write_json', fail)
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name, 'hidden': False})
    assert responses[-1][1] == 500
    assert studio.results.entries()[0]['hidden'] is True


def test_result_viewer_loads_partial_run_cameras_and_navigates(tmp_path):
    import numpy as np
    from test_scene_run_viser import _FakeServer, _tiny_scene
    studio, _, root = setup(tmp_path)
    (root / 'result.json').rename(root / 'partial-result.json')
    (root / 'partial-result.json').write_text(json.dumps({
        'incomplete': True, 'splat_path': 'artifixer3d.ply',
        'inference_split': 'prepared/scene/split_trajectory.json'}))
    prep = root / 'prepared/scene'
    prep.mkdir(parents=True)
    (prep / 'split_trajectory.json').write_text(json.dumps({'test': {'scene': {
        'transforms_path': 'transforms.json', 'selected_indices_path': 'selected.json'}}}))
    (prep / 'selected.json').write_text('[1, 0]')
    poses = [np.eye(4), np.eye(4)]
    poses[1][:3, 3] = [3, 4, 5]
    (prep / 'transforms.json').write_text(json.dumps({'fl_y': 700, 'h': 800, 'frames': [
        {'transform_matrix': pose.tolist()} for pose in poses]}))
    # An empty copied checkpoint must not suppress valid split cameras.
    (root / 'checkpoint').mkdir()
    (root / 'checkpoint/checkpoint.json').write_text('{}')
    key = studio.results.entries()[0]['id']
    viewer = ResultViser(studio.results, background=False, server_factory=_FakeServer,
                         load_scene=lambda *a, **k: _tiny_scene(), idle_timeout=None)
    state = viewer.show(key)
    assert state['status'] == 'ready' and state['view_count'] == 2 and state['view_index'] == 0
    client = SimpleNamespace(camera=SimpleNamespace())
    viewer._server._clients['test'] = client
    viewer._server._connect(client)
    np.testing.assert_array_equal(client.camera.position, [3, 4, 5])
    np.testing.assert_array_equal(client.camera.look_at, [3, 4, 4])
    np.testing.assert_array_equal(client.camera.up_direction, [0, 1, 0])
    assert viewer.move_view(key, 1)['view_index'] == 1
    np.testing.assert_array_equal(client.camera.position, [0, 0, 0])
    assert viewer.move_view(key, -1)['view_index'] == 0
    viewer.show(key, toggle=True)
    np.testing.assert_array_equal(client.camera.position, [3, 4, 5])


def test_input_only_detail_shows_references_and_original_captures(tmp_path):
    studio, run, root = setup(tmp_path)
    (root / 'result.json').unlink()
    (root / 'artifixer3d.ply').unlink()
    images = root / 'input-images'
    images.mkdir()
    (images / 'ref.png').write_bytes(b'edited')
    (images / 'rgb.png').write_bytes(b'original')
    (root / 'input-preview.json').write_text(json.dumps({'inputs_ready': True,
        'reference_images': [{'index': 0, 'name': 'ref.png', 'path': 'input-images/ref.png'}],
        'original_rgb_images': [{'index': 7, 'name': 'rgb.png', 'path': 'input-images/rgb.png'}]}))
    row, = studio.results.entries()
    detail = studio.results.run_detail(run.name)
    assert not detail['inference_ready'] and not detail['gallery']['frames']
    assert detail['reference_images'][0]['url'] == studio.file_url(images / 'ref.png')
    assert detail['original_rgb_images'][0]['index'] == 7
    (root / 'inference-preview.json').write_text(json.dumps({'inference_ready': True}))
    assert studio.results.entries()[0]['id'] == row['id']
    assert len(studio.results.entries()) == 1
    assert studio.results.run_detail(row['id'])['reference_images'] == detail['reference_images']
    (images / 'rgb.png').unlink()
    secret = tmp_path / 'secret.png'
    secret.write_bytes(b'private')
    (images / 'rgb.png').symlink_to(secret)
    assert studio.results.run_detail(row['id'])['original_rgb_images'] == []


def test_input_preview_gallery_grows_before_reconstruction_finishes(tmp_path):
    studio, run, root = setup(tmp_path)
    (root / 'result.json').unlink()
    (root / 'artifixer3d.ply').unlink()
    (root / 'input-preview.json').write_text(json.dumps({'inputs_ready': True}))
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    key = studio.results.entries()[0]['id']
    assert studio.results.run_detail(key, deferred=True)['gallery']['frames'] == []
    for index in (2, 10):
        (pred / f'{index:05d}.png').write_bytes(b'prediction')
        detail = studio.results.run_detail(key, deferred=True)
        assert [f['index'] for f in detail['gallery']['frames']] == ([] if index == 2 else [2])
        assert detail['gallery_pending']
        detail.update(studio.results.detail_section(key, 'gallery'))
        assert not detail['inference_ready']
        assert not detail['gallery']['verified_inputs']
        assert [f['index'] for f in detail['gallery']['frames']] == ([2] if index == 2 else [2, 10])


def test_gallery_cache_survives_restart_and_stops_discovery_after_24_hours(tmp_path, monkeypatch):
    from datetime import datetime
    from splat_explorer.web.splatfix_results import ResultCatalog
    studio, run, root = setup(tmp_path)
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    (pred / '00002.png').write_bytes(b'prediction')
    source = studio.results.detail(run.name)
    created = datetime.fromisoformat(source['created_at'].replace('Z', '+00:00')).timestamp()
    monkeypatch.setattr('splat_explorer.web.splatfix_results.time.time', lambda: created + 86399)
    first = studio.results.detail_section(run.name, 'gallery')
    assert [f['index'] for f in first['gallery']['frames']] == [2]
    (pred / '00010.png').write_bytes(b'prediction')
    second = studio.results.detail_section(run.name, 'gallery')
    assert [f['index'] for f in second['gallery']['frames']] == [2, 10]
    restarted = ResultCatalog(studio)
    monkeypatch.setattr('splat_explorer.web.splatfix_results.time.time', lambda: created + 86400)
    monkeypatch.setattr(restarted, '_scan_gallery_section', lambda *_: pytest.fail('Old run must not scan'))
    monkeypatch.setattr(restarted, '_reconstruction_gallery', lambda *a, **kw: pytest.fail('Old thumbnail must not scan'))
    (pred / '00020.png').write_bytes(b'late prediction')
    detail = restarted.run_detail(run.name, deferred=True)
    assert 'gallery' not in detail['deferred_sections']
    assert detail['gallery_lookup_expired'] and not detail['gallery_pending']
    assert detail['gallery'] == second['gallery']
    assert restarted.detail_section(run.name, 'gallery')['gallery'] == second['gallery']
    assert restarted.run_detail(run.name)['gallery'] == second['gallery']
    assert restarted.entries()[0]['preview_url'] == second['gallery']['frames'][0]['url']
    forced = studio.results.detail_section(run.name, 'gallery', force=True)
    assert [f['index'] for f in forced['gallery']['frames']] == [2, 10, 20]
    assert restarted.run_detail(run.name, deferred=True)['gallery'] == forced['gallery']



def test_old_run_without_gallery_cache_does_not_scan(tmp_path, monkeypatch):
    from datetime import datetime
    studio, run, _ = setup(tmp_path)
    source = studio.results.detail(run.name)
    created = datetime.fromisoformat(source['created_at'].replace('Z', '+00:00')).timestamp()
    monkeypatch.setattr('splat_explorer.web.splatfix_results.time.time', lambda: created + 86401)
    monkeypatch.setattr(studio.results, '_scan_gallery_section', lambda *_: pytest.fail('No cache is not permission to scan'))
    detail = studio.results.run_detail(run.name, deferred=True)
    assert detail['gallery']['frames'] == []
    assert 'gallery' not in detail['deferred_sections']
    assert studio.results.detail_section(run.name, 'gallery')['gallery_lookup_expired']


def test_preview_split_references_load_even_with_verified_prediction_indices(tmp_path):
    studio, _, root = setup(tmp_path)
    (root / 'result.json').unlink()
    prep = root / 'prepared/bicycle'
    prep.mkdir(parents=True)
    (prep / 'ref.jpg').write_bytes(b'reference')
    (prep / 'selected.json').write_text('[1]')
    (prep / 'targets.json').write_text('[0]')
    (prep / 'transforms.json').write_text(json.dumps({'frames': [{}, {'file_path': 'ref.jpg'}]}))
    (prep / 'split.json').write_text(json.dumps({'test': {'scene': {'image_root': '.',
        'transforms_path': 'transforms.json', 'selected_indices_path': 'selected.json',
        'target_indices_path': 'targets.json'}}}))
    (root / 'inference-preview.json').write_text(json.dumps({'inference_ready': True,
        'reconstruction_inputs': [0], 'reference_count': 1}))
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert detail['reference_images'][0]['url'] == studio.file_url(prep / 'ref.jpg')


def test_interrupted_gallery_handles_null_resolution_metadata(tmp_path):
    studio, _, root = setup(tmp_path)
    (root / 'result.json').unlink()
    (root / 'partial-result.json').write_text(json.dumps({
        'incomplete': True, 'resolution_policy': None, 'frame_count': None,
        'prediction_frames': 'inference/splatfix/frames/batch_0000/pred'}))
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    (pred / '00007.png').write_bytes(b'prediction')
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert [frame['index'] for frame in detail['gallery']['frames']] == [7]


def test_loading_references_preserves_verified_prediction_indices(tmp_path):
    studio, _, root = setup(tmp_path)
    (root / 'result.json').unlink()
    (root / 'inference-preview.json').write_text(json.dumps({
        'inference_ready': True, 'reconstruction_inputs': [7], 'reference_count': 1,
        'frame_count': None, 'prediction_frames': 'inference/splatfix/frames/batch_0000/pred'}))
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    (pred / '00007.png').write_bytes(b'prediction')
    prep = root / 'prepared/bicycle'
    prep.mkdir(parents=True)
    (prep / 'selected.json').write_text('[0]')
    (prep / 'split.json').write_text(json.dumps({'test': {'scene': {
        'selected_indices_path': 'selected.json'}}}))
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert [frame['index'] for frame in detail['gallery']['frames']] == [7]
    assert detail['gallery']['expected_count'] == 1


def test_chained_results_use_pass_order_and_stable_job_link(tmp_path):
    studio, root, first = setup(tmp_path)
    cfg = json.loads((root / 'config.json').read_text())
    cfg['splatfix'].update(stage='benchmark', benchmark_runs=2, source='/benchmark-input', mode='baseline')
    (root / 'config.json').write_text(json.dumps(cfg))
    second = first.parent / 'benchmark_1.3b_aaa'
    second.mkdir()
    (second / 'artifixer3d.ply').write_bytes(b'ply')
    (second / 'result.json').write_text(json.dumps({'splat_path': 'artifixer3d.ply',
        'repeat_repair': {'repair_pass': 2}}))
    rows = studio.results.entries()
    assert [row['repair_pass'] for row in rows] == [1, 2]
    assert all(row['status'] == 'completed' for row in rows)
    initial = studio.results.run_detail(root.name)
    assert initial['id'] == rows[0]['id']
    assert initial['chain_total'] == 2
    assert initial['chain_position'] == 1
    assert [item['id'] for item in initial['chain_runs']] == [row['id'] for row in rows]
    following = studio.results.run_detail(rows[1]['id'])
    assert following['chain_position'] == 2
    assert following['chain_runs'][0]['detail_url'] == initial['detail_url']
    assert following['ply_url'] != initial['ply_url']


def test_chained_preview_navigation_keeps_key_after_completion(tmp_path):
    studio, root, first = setup(tmp_path)
    cfg = json.loads((root / 'config.json').read_text())
    cfg['splatfix'].update(stage='benchmark', benchmark_runs=2, source='/benchmark-input', mode='baseline')
    (root / 'config.json').write_text(json.dumps(cfg))
    second = first.parent / 'benchmark_1.3b_aaa'
    second.mkdir()
    (second / 'input-preview.json').write_text(json.dumps({'inputs_ready': True}))
    (second / 'benchmark-run.json').write_text(json.dumps({'repeat_repair': {'repair_pass': 2}}))
    rows = studio.results.entries()
    key = rows[1]['id']
    assert rows[1]['repair_pass'] == 2
    assert studio.results.run_detail(key)['chain_position'] == 2
    (second / 'result.json').write_text(json.dumps({'repeat_repair': {'repair_pass': 2}}))
    assert studio.results.run_detail(key)['status'] == 'completed'
    assert len(studio.results.entries()) == 2


def test_detail_chain_navigation_and_polling_javascript():
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for dashboard logic')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,{addEventListener(){}});return elements.get(id)};
const location={search:'?id=job',href:''};
const context=vm.createContext({document:{getElementById:get,addEventListener(){}},window:{addEventListener(){}},location,URLSearchParams,
ResizeObserver:class{observe(){}},
fetch:()=>new Promise(()=>{}),clearTimeout(){},setTimeout(){},assert});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],context);
vm.runInContext(String.raw`
const plusTiles=[0,1].map(()=>({setAttribute(){},scrollIntoView(){}}));
$('plusGallery').querySelectorAll=()=>plusTiles;
document.querySelectorAll=()=>plusTiles;
$('viewerFrame').style={};$('full').style={};
$('lightbox').showModal=()=>{$('lightbox').open=true};
renderPlus({kind:'ArtiFixer3D',plus_gallery:{frames:[{index:2,url:'/plus/2.png'},{index:10,url:'/plus/10.png'}]}});
assert.equal($('plusImages').hidden,false);
$('plusOpen').onclick();
assert.equal($('lightbox').open,true);
assert.equal($('full').src,'/plus/2.png');
assert.equal(viewerGallery,'plusGallery');
assert.equal($('captureToggle').hidden,true);
choose(viewerIndex+1,false,viewerGallery);
assert.equal($('full').src,'/plus/10.png');
assert.equal($('next').disabled,true);
$('plusFind').value='2';$('plusGo').onclick();
assert.equal(viewerIndex,0);
assert.ok($('position').textContent.includes('ArtiFixer3D+ frame 2'));
renderPlus({kind:'ArtiFixer3D'});
assert.equal($('plusOpen').disabled,true);
assert.ok($('plusGallery').innerHTML.includes('No ArtiFixer3D+ images'));
renderPlus({kind:'OracleGS'});
assert.equal($('plusImages').hidden,true);
assert.equal(cleanLog('\x1b[31mERROR\x1b[0m\r\nnext\rprogress'),'ERROR\nnext\nprogress');
$('logSearch').value='<script>';
$('logContent').querySelectorAll=()=>[];
logText='ERROR <script>alert(1)</script> & details';
renderLog();
assert.ok($('logContent').innerHTML.includes('<mark>&lt;script&gt;</mark>'));
assert.ok(!$('logContent').innerHTML.includes('<script>'));
assert.equal($('logMatches').textContent,'1 matches');
$('logSearch').value='missing';renderLog();
assert.equal($('logMatches').textContent,'0 matches');
assert.equal($('logNext').disabled,true);
const parsed=parseTrainingLog('Training progress: 5%|x| 1090/22000 [00:27<09:35, 36.33it/s, Loss=0.0506246, Synthetic Loss=0.3610111, Depth=0.0007515]\rTraining progress: 5%|x| 1090/22000 [00:28<09:35, 36.33it/s, Loss=0.0445616, Synthetic Loss=0.3573269, Depth=0.0004918]\nTraining progress: 6%|x| 1100/22000 [1:02:03<09:35, 35.14it/s, Loss=4e-2, Synthetic Loss=0.37, Depth=1e-5]\nnoise');
assert.equal(parsed.length,2);assert.equal(parsed[0].loss,0.0445616);assert.equal(parsed[0].seconds,28);assert.equal(parsed[0].synthetic,0.3573269);assert.equal(parsed[0].speed,36.33);assert.equal(parsed[1].seconds,3723);assert.equal(parsed[1].depth,0.00001);
trainingPoints=parsed;$('trainingAxis').value='seconds';renderTraining();
assert.ok($('trainingPlots').innerHTML.includes('Elapsed training time (min)'));
assert.ok(!$('trainingPlots').innerHTML.includes('NaN'));
$('trainingAxis').value='iteration';trainingPoints=parsed.slice(0,1);renderTraining();
assert.ok($('trainingPlots').innerHTML.includes('<circle'));
assert.ok(!$('trainingPlots').innerHTML.includes('NaN'));
trainingMetrics=[['loss/l1/train','loss/l1/train'],['num_particles/train','Particles']];
trainingPoints=[{iteration:1,seconds:0,'loss/l1/train':.5,'num_particles/train':20},{iteration:2,seconds:10,'loss/l1/train':.25,'num_particles/train':30}];
$('trainingAxis').value='seconds';renderTraining();
assert.ok($('trainingPlots').innerHTML.includes('loss/l1/train'));
assert.ok($('trainingPlots').innerHTML.includes('>Count</text>'));
assert.ok(!$('trainingPlots').innerHTML.includes('NaN'));
const chain={chain_total:2,chain_position:1,chain_runs:[{detail_url:'/pass1'},{detail_url:'/pass2'}]};
showChain(chain);assert.equal($('chainPrev').disabled,true);assert.equal($('chainNext').disabled,false);
moveChain(1);assert.equal(location.href,'/pass2');
showChain({...chain,chain_position:2});assert.equal($('chainNext').disabled,true);moveChain(-1);assert.equal(location.href,'/pass1');
assert.equal(needsGalleryPolling({...chain,status:'completed',job_status:'running',inference_ready:true}),true);
assert.equal(needsGalleryPolling({...chain,status:'completed',job_status:'completed'}),false);
for(const status of ['running','stopping','stopped','completed','error','gpu_expired']){
 const run={kind:'ArtiFixer3D',status,inference_ready:true,gallery:{verified_inputs:true,expected_count:1,frames:[{}]}};
 assert.equal(needsGalleryPolling(run),['running','stopping'].includes(status));
 assert.equal(needsGalleryPolling({...run,reduced:true,status:'completed'}),false);
}

showChain({...chain,chain_runs:[{detail_url:'/pass1'}]});assert.equal($('chainNext').disabled,true);
assert.ok($('chainPosition').textContent.includes('pending'));
`,context);
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_result_detail.html'], check=True)


def test_completed_pass_keeps_preview_input_provenance_during_download(tmp_path):
    studio, root, result = setup(tmp_path)
    pred = result / 'inference/pred'
    pred.mkdir(parents=True)
    for index in (0, 1, 2):
        (pred / f'{index:05d}.png').write_bytes(b'png')
    (result / 'result.json').write_text(json.dumps({'prediction_frames': 'inference/pred'}))
    (result / 'inference-preview.json').write_text(json.dumps({'inference_ready': True,
        'reconstruction_inputs': [0, 2], 'reference_count': 1}))
    gallery = studio.results.run_detail(root.name)['gallery']
    assert gallery['verified_inputs']
    assert gallery['reference_count'] == 1
    assert [frame['index'] for frame in gallery['frames']] == [0, 2]


def test_archive_moves_entire_run_and_removes_results(tmp_path):
    studio, root, result = setup(tmp_path)
    studio.store.set_status(root.name, 'stopped')
    assert studio.results.entries()[0]['can_archive'] is True
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    archived = studio.results.archive_run(root.name)
    destination = tmp_path / 'run_history' / root.name
    assert archived == {'run_id': root.name, 'path': str(destination)}
    assert not root.exists()
    assert {str(p.relative_to(destination)): p.read_bytes() for p in destination.rglob('*') if p.is_file()} == before
    assert studio.results.entries() == []
    assert studio.store.list_runs() == []


def test_archive_rejects_active_runs_collisions_and_unsafe_ids(tmp_path):
    import pytest
    studio, root, _ = setup(tmp_path)
    for status in ('queued', 'waiting_gpu', 'starting', 'running', 'stopping'):
        studio.store.set_status(root.name, status)
        assert studio.results.entries()[0]['can_archive'] is False
        with pytest.raises(ValueError, match='inactive'):
            studio.results.archive_run(root.name)
        assert root.is_dir()
    studio.store.set_status(root.name, 'completed')
    destination = tmp_path / 'run_history' / root.name
    destination.mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        studio.results.archive_run(root.name)
    assert root.is_dir()
    for run_id in ('../escape', '', None):
        with pytest.raises((ValueError, TypeError)):
            studio.results.archive_run(run_id)


def test_archive_endpoint_handles_failure_and_success(tmp_path, monkeypatch):
    from splat_explorer.web.server import DashboardHandler
    studio, root, _ = setup(tmp_path)
    responses = []
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=studio), command='POST',
                              _send_json=lambda body, code=200: responses.append((body, code)))
    endpoint = '/api/splatfix/results/archive'
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
    assert responses[-1][1] == 400
    studio.store.set_status(root.name, 'error')
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError('permission denied')
        patch.setattr(Path, 'rename', fail)
        DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
        assert responses[-1][1] == 500
        assert root.is_dir()
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
    assert responses[-1][1] == 200
    assert Path(responses[-1][0]['path']).is_dir()
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
    assert responses[-1][1] == 404


def test_delete_removes_only_selected_run_without_following_links(tmp_path):
    studio, root, result = setup(tmp_path)
    studio.store.set_status(root.name, 'error')
    other = studio.store.create_run()
    outside = tmp_path / 'preserved'
    outside.mkdir()
    (outside / 'data').write_bytes(b'keep')
    (result / 'external').symlink_to(outside, target_is_directory=True)
    assert studio.results.delete_run(root.name) == {'run_id': root.name}
    assert not root.exists()
    assert Path(other.path).is_dir()
    assert (outside / 'data').read_bytes() == b'keep'
    assert not (tmp_path / 'run_history').exists()
    assert studio.results.entries() == []


def test_delete_rejects_active_and_unsafe_runs(tmp_path):
    import pytest
    studio, root, _ = setup(tmp_path)
    for status in ('queued', 'waiting_gpu', 'starting', 'running', 'stopping'):
        studio.store.set_status(root.name, status)
        with pytest.raises(ValueError, match='inactive'):
            studio.results.delete_run(root.name)
        assert root.is_dir()
    for run_id in ('../escape', '', None):
        with pytest.raises((ValueError, TypeError)):
            studio.results.delete_run(run_id)
    alias = studio.root / 'run_20261019_000000'
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe'):
        studio.results.delete_run(alias.name)
    assert root.is_dir()


def test_delete_endpoint_handles_failure_and_success(tmp_path, monkeypatch):
    from splat_explorer.web.server import DashboardHandler
    studio, root, _ = setup(tmp_path)
    responses = []
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=studio), command='POST',
                              _send_json=lambda body, code=200: responses.append((body, code)))
    endpoint = '/api/splatfix/results/delete'
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
    assert responses[-1][1] == 400
    studio.store.set_status(root.name, 'error')
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError('permission denied')
        patch.setattr('splat_explorer.web.splatfix_results.shutil.rmtree', fail)
        DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
        assert responses[-1][1] == 500
        assert root.is_dir()
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
    assert responses[-1][1] == 200
    assert not root.exists()
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_id': root.name})
    assert responses[-1][1] == 404


def test_delete_ui_confirmation_separation_and_failure():
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for UI checks')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
const elements={};const get=id=>elements[id]??={value:'',checked:false,getAttribute(){return null},addEventListener(){},setAttribute(){}};
let confirmed=false,fail=false,calls=0,prompts=0;
const context=vm.createContext({document:{getElementById:get,hidden:true},setInterval(){},
 window:{confirm:message=>{prompts++;assert.match(message,/Permanently delete test/);return confirmed}},
 fetch:async(url,options)=>{if(!options)return new Promise(()=>{});calls++;assert.equal(url,'/api/splatfix/results/delete');assert.deepEqual(JSON.parse(options.body),{run_id:'test'});return{ok:!fail,json:async()=>fail?{error:'Delete failed'}:{run_id:'test'}}}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
const load=()=>vm.runInContext("rows=[{run_id:'test',scene:'Room',status:'error',can_archive:true,metrics:{}},{run_id:'test',scene:'Pass 2',can_archive:true,metrics:{}}];render()",context);
const click=()=>get('results').onclick({target:{closest:()=>({dataset:{run:'test',action:'delete'}})}});
(async()=>{
 load();assert.match(get('results').innerHTML,/<div class="titleline"><h2>Room<\/h2><button[^>]+data-action="delete"[^>]*>[\s\S]*?<\/button><button[^>]+aria-label="Hide run test"/);
 assert(!get('results').innerHTML.includes('delete-area'));
 assert.match(html,/\.titleline \.icon-button\{width:24px;height:24px/);
 assert.match(get('results').innerHTML,/<div class="links"><div class="run-links"><a class="review"[\s\S]*?data-action="archive"[\s\S]*?<\/button><\/div>/);
 assert(!get('results').innerHTML.split('<div class="links">')[1].split('</div>')[0].includes('data-action="delete"'));
 await click();assert.equal(calls,0);assert.equal(prompts,1);
 confirmed=true;fail=true;await click();assert.equal(calls,1);assert.equal(vm.runInContext('rows.length',context),2);assert.equal(get('error').textContent,'Delete failed');
 fail=false;await click();assert.equal(vm.runInContext('rows.length',context),0);assert.match(get('notice').textContent,/Permanently deleted test/);
 load();vm.runInContext('rows.forEach(r=>r.can_archive=false);render()',context);await click();assert.equal(calls,2);assert.equal(prompts,3);
 load();vm.runInContext("rows[0].backed_up=true;rows[0].backup_path='/history/test';render()",context);
 assert.match(get('results').innerHTML,/class="tag backup-tag" title="Backup: \/history\/test">Backed up/);
 assert.match(get('results').innerHTML,/<article class="">/);
 get('backed-up').checked=true;get('backed-up').onchange();
 assert.equal(get('count').textContent,'1 / 2 runs');
 get('backed-up').checked=false;get('backed-up').onchange();
 assert.equal(get('count').textContent,'2 / 2 runs');
 confirmed=false;await click();assert.equal(calls,3);assert.equal(prompts,3);assert.equal(vm.runInContext('rows.length',context),0);

})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_results.html'], check=True)


def test_delete_hidden_batch_rechecks_visibility_status_and_reports_failures(tmp_path, monkeypatch):
    studio, root, _ = setup(tmp_path)
    runs = [studio.store.create_run() for _ in range(4)]
    ids = [root.name] + [r.run_id for r in runs]
    for run_id in ids:
        studio.store.set_status(run_id, 'error')
        studio.results.set_hidden(run_id, True)
    studio.results.set_hidden(ids[1], False)
    studio.store.set_status(ids[2], 'running')
    original = __import__('shutil').rmtree
    def remove(path):
        if Path(path).name == ids[3]:
            raise PermissionError('denied')
        original(path)
    monkeypatch.setattr('splat_explorer.web.splatfix_results.shutil.rmtree', remove)
    result = studio.results.delete_hidden_runs(ids + [ids[0], '../escape'])
    assert result == {'deleted': [ids[0], ids[4]], 'skipped': [ids[1], ids[2], '../escape'], 'failed': [ids[3]]}
    for run_id in ids[1:4]:
        assert studio.store.run_path(run_id).is_dir()
    assert not root.exists()


def test_delete_hidden_endpoint_validates_batch(tmp_path):
    from splat_explorer.web.server import DashboardHandler
    studio, root, _ = setup(tmp_path)
    responses = []
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=studio), command='POST',
                              _send_json=lambda body, code=200: responses.append((body, code)))
    endpoint = '/api/splatfix/results/delete-hidden'
    for body in (None, {}, {'run_ids': 'all'}, {'run_ids': [None]}):
        DashboardHandler._serve_splatfix(handler, endpoint, body)
        assert responses[-1][1] == 400
    studio.store.set_status(root.name, 'error')
    studio.results.set_hidden(root.name, True)
    DashboardHandler._serve_splatfix(handler, endpoint, {'run_ids': [root.name]})
    assert responses[-1] == ({'ok': True, 'deleted': [root.name], 'skipped': [], 'failed': []}, 200)
    assert not root.exists()


def test_bulk_delete_ui_toggle_confirmation_and_partial_failure():
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for UI checks')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
assert.match(html,/id="delete-hidden"[^>]+hidden>Delete all hidden runs/);
const elements={};const get=id=>elements[id]??={value:'',checked:false,getAttribute(){return null},addEventListener(){},setAttribute(){}};
let confirmed=false,calls=0;
const context=vm.createContext({document:{getElementById:get,hidden:true},setInterval(){},
 window:{confirm:message=>{assert.match(message,/all 2 inactive hidden runs/);assert.match(message,/search and filters/);return confirmed}},
 fetch:async(url,options)=>{if(!options)return new Promise(()=>{});calls++;assert.equal(url,'/api/splatfix/results/delete-hidden');assert.deepEqual(JSON.parse(options.body),{run_ids:['a','b']});return{ok:true,json:async()=>({deleted:['a'],skipped:[],failed:['b']})}}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
(async()=>{
 vm.runInContext("rows=[{run_id:'a',hidden:true,can_archive:true},{run_id:'a',hidden:true,can_archive:true},{run_id:'b',hidden:true,can_archive:true},{run_id:'active',hidden:true,can_archive:false},{run_id:'visible',hidden:false,can_archive:true}].map(r=>({...r,metrics:{}}));render()",context);
 assert.equal(get('delete-hidden').hidden,true);
 get('show-hidden').onclick();assert.equal(get('delete-hidden').hidden,false);assert.equal(get('delete-hidden').disabled,false);
 get('search').value='nothing matches';get('search').oninput();
 await get('delete-hidden').onclick();assert.equal(calls,0);
 confirmed=true;await get('delete-hidden').onclick();assert.equal(calls,1);
 assert.equal(vm.runInContext("rows.some(r=>r.run_id==='a')",context),false);
 assert.equal(vm.runInContext('rows.length',context),3);
 assert.match(get('notice').textContent,/Deleted 1/);assert.match(get('error').textContent,/b/);
 get('show-hidden').onclick();assert.equal(get('delete-hidden').hidden,true);await get('delete-hidden').onclick();assert.equal(calls,1);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_results.html'], check=True)


def test_results_sort_by_timestamp_and_size():
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for UI checks')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
assert.match(html,/id="sort"[^>]*><option value="newest">Newest first \(default\)/);
const elements={};const get=id=>elements[id]??={value:'',checked:false,getAttribute(){return null},addEventListener(){},setAttribute(){}};
const context=vm.createContext({document:{getElementById:get,hidden:true},setInterval(){},fetch:()=>new Promise(()=>{}),assert});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
vm.runInContext(`
const make=(id,size,scene)=>({run_id:id,output_size_bytes:size,scene,metrics:{}});
const old=make('run_20261018_120000',0,'zero'),
 base=make('run_20261019_120000',1024,'base'),
 two=make('run_20261019_120000_2',3*1024**3,'two'),
 ten=make('run_20261019_120000_10',1024,'ten'),
 unknown=make('run_20261020_120000',null,'unknown');
rows=[unknown,ten,two,base,old];
const sortedNames=order=>sortRuns(rows,order).map(r=>r.scene).join(',');
assert.equal(sortedNames('newest'),'unknown,ten,two,base,zero');
assert.equal(sortedNames('oldest'),'zero,base,two,ten,unknown');
assert.equal(sortedNames('largest'),'two,ten,base,zero,unknown');
assert.equal(sortedNames('smallest'),'zero,ten,base,two,unknown');
assert.equal(rows[0],unknown);
const world=make('we_20261020_215533_02b07c345ec5',null,'world');
const worldEarlier=make('we_20261020_203322_d8600ca1e66b',null,'worldEarlier');
assert.equal(sortRuns([...rows,worldEarlier,world],'newest').map(r=>r.scene).join(','),'world,worldEarlier,unknown,ten,two,base,zero');
assert.equal(sortRuns([...rows,world],'oldest').at(-1),world);
const passes=[{...two,scene:'pass1'},{...two,scene:'pass2'}];
assert.equal(sortRuns(passes,'oldest').map(r=>r.scene).join(','),'pass1,pass2');
$('sort').value='smallest';$('sort').onchange();
assert.ok($('results').innerHTML.indexOf('<h2>zero</h2>')<$('results').innerHTML.indexOf('<h2>two</h2>'));
$('search').value='two';$('search').oninput();assert.ok($('results').innerHTML.includes('<h2>two</h2>'));assert.ok(!$('results').innerHTML.includes('<h2>zero</h2>'));
assert.equal($('sort').value,'smallest');
`,context);
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_results.html'], check=True)


def test_detail_lists_full_logs_separately_from_cached_tail(tmp_path):
    studio, root, result = setup(tmp_path)
    (root / 'gpu/stage.log').write_text('Full worker output\n' * 10000)
    (root / 'gpu/live-stage.log').write_text('Tail only')
    (result / 'reconstruction.log').write_text('Training output')
    outside = tmp_path / 'outside.log'
    outside.write_text('Unrelated private log')
    (result / 'escape.log').symlink_to(outside)
    sibling = result.parent / 'other-result'
    sibling.mkdir()
    (sibling / 'other.log').write_text('Different result')
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    logs = detail['logs']
    assert {log['name'] for log in logs} == {
        'gpu/stage.log', 'gpu/live-stage.log',
        'gpu/results/benchmark_1.3b_abc/reconstruction.log',
    }
    full = next(log for log in logs if log['name'] == 'gpu/stage.log')
    assert full['size'] > 65536
    assert not full['partial']
    assert logs[-1]['partial']
    assert all(log['url'] and log['modified'] for log in logs)


def test_artifixer_detail_includes_nested_logs_and_tensorboard_metrics(tmp_path):
    from test_training_logs import write_events
    studio, root, result = setup(tmp_path)
    directory = result / 'artifixer3d/runs/splatfix/splatfix'
    directory.mkdir(parents=True)
    (directory / 'training.log').write_text('Nested training output')
    write_events(directory, [(1, 100, {'loss/l1/train': .25}),
                             (2, 110, {'loss/l1/train': .125})])
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert any(log['name'].endswith('/training.log') for log in detail['logs'])
    decoded = next(log for log in detail['logs'] if log['name'].endswith('(decoded scalars)'))
    from urllib.parse import parse_qs, urlparse
    log_path = parse_qs(urlparse(decoded['url']).query)['path'][0]
    assert 'loss/l1/train=0.125' in studio.allowed_file(log_path).read_text()
    assert detail['training_metrics']['format'] == 'scalars'
    chart_path = parse_qs(urlparse(detail['training_metrics']['url']).query)['path'][0]
    assert json.loads(studio.allowed_file(chart_path).read_text())['points'][-1]['seconds'] == 10


def reduction_setup(tmp_path):
    studio, root, previous = setup(tmp_path)
    import shutil
    shutil.rmtree(previous)
    result = root / 'gpu/results/oraclegs'
    result.mkdir()
    (result / 'result.json').write_text(json.dumps({'reconstruction_method': 'oraclegs', 'splat_path': str(result / 'oraclegs.ply')}))
    (result / 'oraclegs.ply').write_bytes(b'final')
    inputs = root / 'gpu/inputs'
    inputs.mkdir()
    (inputs / 'source-initialization.ply').write_bytes(b'original')
    views = [{'image': f'scene/images/ref_{i}.png', 'camera': {'pose': [1]}} for i in range(6)]
    (inputs / 'inputs.json').write_text(json.dumps({'views': views, 'trajectory': {'frames': [1, 2]}}))
    for view in views:
        p = inputs / view['image']
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'repaired')
    for relative in ('dataset/images/input.png', 'dataset/sparse/0/images.txt', 'attention-raw/grid.png', 'model/point_cloud/iteration_1/point_cloud.ply', 'seva/img2trajvid/scene/first-pass/samples-rgb.mp4', 'seva/img2trajvid/scene/samples-rgb/000.png', 'seva/img2trajvid/scene/transforms.json', 'seva/img2trajvid/scene/samples-rgb.mp4', 'train.log'):
        p = result / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'content')
    studio.store.set_status(root.name, 'completed')
    return studio, root, result


def test_reduce_keeps_only_minimal_artifacts_and_refreshes_size(tmp_path):
    studio, root, result = reduction_setup(tmp_path)
    studio.results.save_comments(root.name, '# Keep these notes')
    studio.store.output_size_bytes(root.name)
    report = studio.results.reduce_run(root.name)
    assert studio.results.read_comments(root.name) == '# Keep these notes'
    assert report['reference_count'] == 6
    assert report['backup_path'] is None
    assert len(list(root.rglob('*.png'))) == 6
    assert len(list(root.rglob('*.mp4'))) == 1
    assert sorted(p.name for p in root.rglob('*.ply')) == ['oraclegs.ply', 'original.ply']
    assert not (result / 'seva/img2trajvid/scene/transforms.json').exists()
    assert (result / 'train.log').read_bytes() == b'content'
    assert 'camera' not in json.loads((result / 'recipe.json').read_text())['inputs']['views'][0]
    assert studio.results.entries()[0]['original'] is True
    assert studio.results.run_detail(root.name)['oraclegs']['views'][0]['url']
    assert report['output_size_bytes'] == studio.store.output_size_bytes(root.name)
    before = json.loads((root / 'reduction.json').read_text())['before_size_bytes']
    assert report['reduction_saved_bytes'] == max(0, before - report['output_size_bytes'])
    assert studio.results.entries()[0]['reduction_saved_bytes'] == report['reduction_saved_bytes']


def test_backup_reduce_verifies_independent_full_copy(tmp_path):
    studio, root, result = reduction_setup(tmp_path)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report = studio.results.reduce_run(root.name, backup=True)
    backup = Path(report['backup_path'])
    assert backup == tmp_path / 'run_history' / root.name
    row = studio.results.entries()[0]
    assert row['backed_up'] is True
    assert row['backup_path'] == str(backup)
    assert all((backup / name).read_bytes() == data for name, data in before.items())
    assert (backup / 'gpu/inputs/source-initialization.ply').stat().st_ino != (result / 'original.ply').stat().st_ino


def test_reduce_rejects_active_and_backup_failure_and_keeps_available_references(tmp_path, monkeypatch):
    import pytest
    from splat_explorer.web import run_reduction
    studio, root, result = reduction_setup(tmp_path)
    studio.store.set_status(root.name, 'running')
    with pytest.raises(ValueError, match='inactive'):
        studio.results.reduce_run(root.name)
    studio.store.set_status(root.name, 'completed')
    monkeypatch.setattr(run_reduction, 'digest', lambda p: str(p).encode())
    with pytest.raises(OSError, match='checksum'):
        studio.results.reduce_run(root.name, backup=True)
    assert (result / 'attention-raw/grid.png').exists()
    assert not list((tmp_path / 'run_history').iterdir())
    (root / 'gpu/inputs/scene/images/ref_0.png').unlink()
    report = studio.results.reduce_run(root.name)
    assert report['reference_count'] == 5
    assert report['missing_references'] == ['ref_0.png']
    assert len(list(root.rglob('*.png'))) == 5


def test_reduce_ui_confirmation_backup_and_size():
    import shutil
    import subprocess
    import pytest
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for UI checks')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(process.argv[1],'utf8');
const elements={};const get=id=>elements[id]??={value:'',checked:false,getAttribute(){return null},addEventListener(){},setAttribute(){}};
let confirmed=false,calls=0,backup=false,preserve=false;
const context=vm.createContext({document:{getElementById:get,hidden:true},setInterval(){},
 window:{confirm:message=>{assert.match(message,preserve?/repaired references \(two endpoints or six views\)/:/Remove all other images, poses/);if(preserve)assert.doesNotMatch(message,/Training cannot be resumed/);return confirmed}},
 fetch:async(url,options)=>{if(!options)return new Promise(()=>{});calls++;assert.equal(url,'/api/splatfix/results/reduce');assert.deepEqual(JSON.parse(options.body),{run_id:'test',backup,...(preserve?{preserve_reconstruction:true}:{})});return{ok:true,json:async()=>({output_size_bytes:1024,reduction_saved_bytes:12992276070,backup_path:backup?'/history/test':null})}}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
const load=()=>vm.runInContext("rows=[{run_id:'test',kind:'ArtiFixer3D',mode:'edited',can_reduce:true,can_archive:true,metrics:{}}];render()",context);
const click=()=>get('results').onclick({target:{closest:()=>({dataset:{run:'test',action:preserve?'reduce-reconstruction':backup?'backup-reduce':'reduce'}})}});
(async()=>{
 load();assert.match(get('results').innerHTML,/Backup &amp; reduce/);
 await click();assert.equal(calls,0);
 confirmed=true;await click();assert.equal(calls,1);assert.equal(vm.runInContext('rows[0].reduced',context),true);assert.match(get('notice').textContent,/1.0 KiB/);assert.match(get('results').innerHTML,/Reduced -12.1 GiB/);
 load();backup=true;await click();assert.equal(calls,2);assert.match(get('notice').textContent,/Full backup: \/history\/test/);assert.match(get('results').innerHTML,/>Backed up<\/span>/);assert.equal(vm.runInContext('rows[0].backed_up',context),true);
 load();backup=false;preserve=true;
 assert.ok(get('results').innerHTML.indexOf('data-action="reduce-reconstruction"')<get('results').innerHTML.indexOf('data-action="reduce"'));
 confirmed=false;await click();assert.equal(calls,2);
 confirmed=true;await click();assert.equal(calls,3);
 load();vm.runInContext('rows[0].can_archive=false',context);await click();assert.equal(calls,3);
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_results.html'], check=True)


def test_reduce_stopped_oraclegs_without_result_or_plys(tmp_path):
    studio, root, result = reduction_setup(tmp_path)
    config = json.loads((root / 'config.json').read_text())
    config['splatfix'].update(reconstruction_method='oraclegs', stage='repair', checkpoint=str(tmp_path / 'checkpoint'))
    (root / 'config.json').write_text(json.dumps(config))
    (result / 'result.json').unlink()
    (result / 'oraclegs.ply').unlink()
    (root / 'gpu/inputs/source-initialization.ply').unlink()
    studio.store.set_status(root.name, 'stopped')
    assert studio.results.entries()[0]['can_reduce'] is True
    report = studio.results.reduce_run(root.name)
    assert report['reduced'] is True
    assert len(list(root.rglob('*.png'))) == 6
    assert (result / 'train.log').exists()
    assert studio.results.entries()[0]['reduced'] is True


def test_reduce_legacy_completed_without_original_ply(tmp_path):
    studio, root, result = reduction_setup(tmp_path)
    (root / 'gpu/inputs/source-initialization.ply').unlink()
    assert studio.results.entries()[0]['can_reduce'] is True
    studio.results.reduce_run(root.name)
    assert (result / 'oraclegs.ply').exists()


def test_reduce_rescans_even_if_dashboard_cached_intermediate_size(tmp_path, monkeypatch):
    from splat_explorer.web import run_reduction
    studio, root, _ = reduction_setup(tmp_path)
    real_reduce = run_reduction.reduce_oraclegs
    def reduce_with_stale_cache(*args, **kwargs):
        report = real_reduce(*args, **kwargs)
        studio.store.output_size_bytes(root.name)
        cache = root / 'output-size.json'
        saved = json.loads(cache.read_text())
        saved['bytes'] = 999999999
        cache.write_text(json.dumps(saved))
        return report
    monkeypatch.setattr(run_reduction, 'reduce_oraclegs', reduce_with_stale_cache)
    report = studio.results.reduce_run(root.name)
    assert report['output_size_bytes'] != 999999999
    assert studio.results.entries()[0]['output_size_bytes'] == report['output_size_bytes']


def artifixer_reduction_setup(tmp_path):
    from PIL import Image
    studio, root, result = setup(tmp_path)
    for branch in ('inference', 'plus'):
        folder = result / branch / 'splatfix/frames/batch_0000/pred'
        folder.mkdir(parents=True)
        for index in (10, 2, 0):
            Image.new('RGB', (33, 25), (index * 20, 30, 80)).save(folder / f'{index}.png')
    Image.new('RGB', (32, 24)).save(result / 'stage-comparison.jpg')
    Image.new('RGB', (32, 24)).save(result / 'trajectory-quality.png')
    (result / 'trajectory-quality.json').write_text('{"caption": "Saved quality chart"}')
    (result / 'degraded').mkdir()
    (result / 'degraded/000.png').write_bytes(b'degraded')
    (result / 'train.log').write_text('training log')
    (result / 'checkpoint.pt').write_bytes(b'weights')
    studio.store.set_status(root.name, 'completed')
    return studio, root, result


@pytest.mark.parametrize('backup', [False, True])
def test_artifixer_reduce_encodes_both_passes_and_exposes_verified_videos(tmp_path, backup):
    import shutil
    import pytest
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    studio, root, result = artifixer_reduction_setup(tmp_path)
    studio.results.save_comments(root.name, '# Keep ArtiFixer notes')
    predictions = {p: p.read_bytes() for p in result.glob('**/pred/*.png')}
    (result / 'cameras.json').write_text('{"poses": []}')
    assert studio.results.entries()[0]['can_reduce'] is True
    report = studio.results.reduce_run(root.name, backup=backup)
    assert studio.results.read_comments(root.name) == '# Keep ArtiFixer notes'
    assert len(report['inference_videos']) == 2
    assert all(v['frame_count'] == 3 and v['frame_names'] == ['0.png', '2.png', '10.png'] for v in report['inference_videos'])
    assert set(root.rglob('*.png')) == set(predictions) | {result / 'trajectory-quality.png'}
    assert all(p.read_bytes() == content for p, content in predictions.items())
    assert not (result / 'degraded').exists()
    assert not (result / 'cameras.json').exists()
    assert (result / 'stage-comparison.jpg').exists()
    assert not (result / 'checkpoint.pt').exists()
    assert (result / 'original.ply').exists() and (result / 'artifixer3d.ply').exists()
    assert (result / 'train.log').read_text() == 'training log'
    if backup:
        saved = Path(report['backup_path'])
        assert len(list(saved.rglob('*.png'))) == 8
        assert (saved / result.relative_to(root) / 'cameras.json').exists()
        assert (saved / result.relative_to(root) / 'checkpoint.pt').exists()
    else:
        assert report['backup_path'] is None
    detail = studio.results.run_detail(root.name)
    assert len(detail['inference_videos']) == 2
    assert all(v['url'] for v in detail['inference_videos'])
    assert len(detail['comparison_gallery']) == 2
    assert detail['comparison_gallery'][1]['caption'] == 'Saved quality chart'
    assert studio.results.entries()[0]['preview_label'] == 'img 1 preview'
    assert detail['original_rgb_images'] == []
    assert [frame['name'] for frame in detail['gallery']['frames']] == ['0.png', '2.png', '10.png']


@pytest.mark.parametrize('empty_batch', [False, True])
def test_artifixer_reduce_removes_empty_interrupted_frames(tmp_path, empty_batch):
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    studio, root, result = artifixer_reduction_setup(tmp_path)
    folder = result / 'inference/splatfix/frames/batch_0000/pred'
    if empty_batch:
        for path in folder.glob('*.png'):
            path.unlink()
    empty = folder / '00511.png'
    empty.touch()
    predictions = {p: p.read_bytes() for p in result.glob('**/pred/*.png') if p != empty}
    report = studio.results.reduce_run(root.name)
    assert report['incomplete_inference_frames'] == [str(empty.relative_to(root))]
    assert len(report['inference_videos']) == (1 if empty_batch else 2)
    assert all(v['frame_count'] == 3 for v in report['inference_videos'])
    assert all('00511.png' not in v['frame_names'] for v in report['inference_videos'])
    assert not empty.exists()
    assert all(p.read_bytes() == data for p, data in predictions.items())
    assert not (result / 'checkpoint.pt').exists()
    assert json.loads((root / 'reduction.json').read_text())['incomplete_inference_frames'] == report['incomplete_inference_frames']


def test_artifixer_reduce_rejects_nonempty_corrupt_frame(tmp_path):
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    studio, root, result = artifixer_reduction_setup(tmp_path)
    corrupt = result / 'inference/splatfix/frames/batch_0000/pred/00511.png'
    corrupt.write_bytes(b'incomplete PNG')
    predictions = {p: p.read_bytes() for p in result.glob('**/pred/*.png')}
    with pytest.raises(ValueError):
        studio.results.reduce_run(root.name)
    assert all(p.read_bytes() == data for p, data in predictions.items())
    assert (result / 'checkpoint.pt').exists()
    assert not (root / 'reduction.json').exists()


def test_artifixer_failed_second_video_keeps_all_source_images(tmp_path, monkeypatch):
    import pytest
    from splat_explorer.web import run_reduction
    studio, root, result = artifixer_reduction_setup(tmp_path)
    calls = []
    def encode(frames, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise ValueError('Verification failed')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'first video')
    monkeypatch.setattr(run_reduction, 'encode_inference_video', encode)
    with pytest.raises(ValueError, match='Verification failed'):
        studio.results.reduce_run(root.name)
    assert len(list(root.rglob('*.png'))) == 8
    assert (result / 'checkpoint.pt').exists()
    assert not (root / 'reduction.json').exists()


def test_save_inference_videos_endpoint_keeps_sources_and_survives_refresh(tmp_path):
    import shutil
    import pytest
    from splat_explorer.web.server import DashboardHandler
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    studio, root, result = artifixer_reduction_setup(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    responses = []
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=studio), command='POST',
                              _send_json=lambda body, code=200: responses.append((body, code)))
    DashboardHandler._serve_splatfix(handler, '/api/splatfix/results/save-inference-videos', {'run_id': root.name})
    report, status = responses[-1]
    assert status == 200 and len(report['inference_videos']) == 2
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not (root / 'reduction.json').exists()
    assert (result / 'checkpoint.pt').exists()
    detail = studio.results.run_detail(root.name)
    assert detail['reduced'] is False and len(detail['inference_videos']) == 2
    assert len(detail['gallery']['frames']) == 3
    studio.results.save_inference_videos(root.name)
    assert len(list(root.rglob('*.mp4'))) == 2
    assert len(list(root.rglob('*.png'))) == 8


def test_save_inference_videos_rejects_active_and_keeps_images_on_failure(tmp_path, monkeypatch):
    import pytest
    from splat_explorer.web import run_reduction
    studio, root, _ = artifixer_reduction_setup(tmp_path)
    studio.store.set_status(root.name, 'running')
    with pytest.raises(ValueError, match='finish'):
        studio.results.save_inference_videos(root.name)
    studio.store.set_status(root.name, 'completed')
    def fail(*args):
        raise ValueError('Encode failed')
    monkeypatch.setattr(run_reduction, 'encode_inference_video', fail)
    with pytest.raises(ValueError, match='Encode failed'):
        studio.results.save_inference_videos(root.name)
    assert len(list(root.rglob('*.png'))) == 8
    assert not (root / 'reduction.json').exists()


def test_interrupted_artifixer_reduction_exports_sog_original_and_keeps_partial(tmp_path, monkeypatch):
    import numpy as np
    from splat_explorer.scene import GaussianScene, load_ply
    from splat_explorer.web.run_reduction import reduce_artifixer
    studio, root, result = artifixer_reduction_setup(tmp_path)
    (result / 'result.json').unlink()
    (result / 'original.ply').unlink()
    partial = result / 'distill.interrupted/interrupted.ply'
    partial.parent.mkdir()
    (result / 'artifixer3d.ply').rename(partial)
    (result / 'partial-result.json').write_text(json.dumps({'incomplete': True, 'splat_path': str(partial)}))
    (partial.parent / 'stop-state.json').write_text('{"step":14311}')
    source = tmp_path / 'sog-scene'
    source.mkdir()
    (source / 'lod-meta.json').write_text('{}')
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return GaussianScene(means=np.zeros((1,3),dtype=np.float32), scales=np.ones((1,3),dtype=np.float32),
                             quats=np.array([[1,0,0,0]],dtype=np.float32), opacities=np.ones(1,dtype=np.float32)*.5,
                             colors=np.ones((1,3),dtype=np.float32)*.5)
    monkeypatch.setattr('splat_explorer.scene.load_scene', load)
    def encode(frames, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b'verified video')
    monkeypatch.setattr('splat_explorer.web.run_reduction.encode_inference_video', encode)
    reduce_artifixer(root, [{'root':result, 'ply_copies':[(source,result/'original.ply'),(partial,partial)]}])
    assert calls == [(source, {'min_opacity': 0.0, 'lod_level': 0})]
    assert len(load_ply(result / 'original.ply').means) == 1
    assert partial.read_bytes() == b'ply'
    assert (partial.parent / 'stop-state.json').exists()
    assert (result / 'partial-result.json').exists()
    assert (source / 'lod-meta.json').exists()


def test_failed_sog_export_keeps_interrupted_run_inputs(tmp_path, monkeypatch):
    import pytest
    from splat_explorer.web.run_reduction import reduce_artifixer
    _, root, result = artifixer_reduction_setup(tmp_path)
    source = tmp_path / 'source-sog'
    source.mkdir()
    def fail(*args, **kwargs):
        raise ValueError('Invalid scene')
    monkeypatch.setattr('splat_explorer.scene.load_scene', fail)
    monkeypatch.setattr('splat_explorer.web.run_reduction.encode_inference_video', lambda *args: None)
    with pytest.raises(ValueError, match='source images were kept'):
        reduce_artifixer(root, [{'root':result,'ply_copies':[(source,result/'original.ply')]}])
    assert len(list(root.rglob('*.png'))) == 8
    assert (result / 'checkpoint.pt').exists()
    assert not (root / 'reduction.json').exists()


def test_manifestless_stopped_run_retains_checkpoint_artifacts(tmp_path):
    studio, root, result = setup(tmp_path)
    (result / 'result.json').unlink()
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text())
    config['splatfix'].update(stage='repair', checkpoint=str(root / 'gpu/checkpoint'))
    config_path.write_text(json.dumps(config))
    studio.store.set_status(root.name, 'stopped')
    checkpoint = root / 'gpu/checkpoint'
    views = checkpoint / 'views/000'
    views.mkdir(parents=True)
    (views / 'repaired.png').write_bytes(b'image')
    (checkpoint / 'checkpoint.json').write_text('{}')
    (checkpoint / 'snapshot.pt').write_bytes(b'checkpoint')
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'private')
    (views / 'escaped.png').symlink_to(outside)
    detail = studio.results.run_detail(root.name)
    assert detail['status'] == 'stopped'
    assert detail['gallery']['frames'] == []
    saved = detail['saved_artifacts']
    assert len(saved['image_groups']) == 1
    assert [p['name'] for p in saved['image_groups'][0]['images']] == ['repaired.png']
    assert {p['name'] for p in saved['checkpoints']} >= {'checkpoint/checkpoint.json', 'checkpoint/snapshot.pt'}
    assert studio.allowed_file(str(checkpoint / 'snapshot.pt')) == checkpoint / 'snapshot.pt'
    assert 'showSavedArtifacts(r)' in Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()


def test_comments_roundtrip_clear_and_run_isolation(tmp_path):
    studio, root, _ = setup(tmp_path)
    other = studio.create({'stage': 'select', 'scene_id': 'room'})
    key = studio.results.entries()[0]['id']
    assert studio.results.run_detail(key)['comments'] == ''
    markdown = '# Observation\n\n- Improved geometry — café\n<script>literal text</script>\n'
    studio.results.save_comments(root.name, markdown)
    assert (root / 'comments.md').read_text(encoding='utf-8') == markdown
    assert studio.results.run_detail(key)['comments'] == markdown
    assert studio.results.read_comments(other['run_id']) == ''
    studio.results.save_comments(root.name, '')
    assert (root / 'comments.md').read_text() == ''
    assert not list(root.glob('.comments-*'))


def test_comments_reject_unsafe_paths_and_invalid_content(tmp_path):
    studio, root, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        studio.results.save_comments('../outside', 'test')
    with pytest.raises(ValueError):
        studio.results.save_comments(root.name, None)
    outside = tmp_path / 'outside.md'
    outside.write_text('untouched')
    (root / 'comments.md').unlink()
    (root / 'comments.md').symlink_to(outside)
    with pytest.raises(ValueError):
        studio.results.save_comments(root.name, 'overwrite')
    with pytest.raises(ValueError):
        studio.results.read_comments(root.name)
    assert outside.read_text() == 'untouched'


def test_comments_failed_replace_preserves_saved_notes(tmp_path, monkeypatch):
    studio, root, _ = setup(tmp_path)
    studio.results.save_comments(root.name, 'saved')
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError):
        studio.results.save_comments(root.name, 'new draft')
    assert (root / 'comments.md').read_text() == 'saved'
    assert not list(root.glob('.comments-*'))


def test_comments_autosave_serializes_edits_and_keeps_drafts():
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for dashboard logic')
    script = r'''const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map(),listeners={},timers=[];
const get=id=>{if(!elements.has(id))elements.set(id,{value:'',setAttribute(){},addEventListener(event,fn){listeners[id+':'+event]=fn}});return elements.get(id)};
const pending=[];
const context=vm.createContext({document:{getElementById:get,addEventListener(){}},window:{addEventListener(){}},
 clearTimeout(){},setTimeout(fn){timers.push(fn)},
 fetch(url,options){return new Promise(resolve=>pending.push({resolve,body:JSON.parse(options.body)}))},assert});
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('// WorldExplorer')[0];
vm.runInContext(source,context);
(async()=>{
 await vm.runInContext(`(async()=>{
 showComments({run_id:'run_one',comments:'saved'});
 $('comments').value='first';
 const first=saveComments();
 $('comments').value='second';
 showComments({run_id:'run_one',comments:'stale refresh'});
 assert.equal($('comments').value,'second');
 await saveComments();
 })()`,context);
 assert.equal(pending.length,1);assert.equal(pending[0].body.comments,'first');
 pending[0].resolve({ok:true,json:async()=>({ok:true})});
 await new Promise(setImmediate);
 timers.pop()();
 assert.equal(pending.length,2);assert.equal(pending[1].body.comments,'second');
 pending[1].resolve({ok:false,json:async()=>({error:'Disk full'})});
 await new Promise(setImmediate);
 assert.equal(get('comments').value,'second');assert.match(get('commentsStatus').textContent,/retrying/);
 timers.pop()();
 pending[2].resolve({ok:true,json:async()=>({ok:true})});
 await new Promise(setImmediate);
 assert.equal(get('commentsStatus').textContent,'Saved to comments.md');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_result_detail.html'], check=True)


def test_run_color_persists_and_is_shared_by_results_and_details(tmp_path):
    studio, root, _ = setup(tmp_path)
    assert studio.results.entries()[0]['color'] == ''
    for color in ('blue', 'orange', ''):
        studio.results.save_color(root.name, color)
        assert json.loads((root / 'color-label.json').read_text()) == {'color': color}
        assert studio.results.entries()[0]['color'] == color
        assert studio.results.run_detail(root.name)['color'] == color
    with pytest.raises(ValueError):
        studio.results.save_color(root.name, 'invalid')
    with pytest.raises(ValueError):
        studio.results.save_color('../outside', 'blue')


def test_run_color_survives_reduction(tmp_path):
    studio, root, _ = reduction_setup(tmp_path)
    studio.results.save_color(root.name, 'teal')
    studio.results.reduce_run(root.name)
    assert studio.results.read_color(root.name) == 'teal'


def test_custom_run_name_is_separate_searchable_metadata(tmp_path):
    studio, root, _ = setup(tmp_path)
    original_config = (root / 'config.json').read_bytes()
    other = studio.create({'stage': 'select', 'scene_id': 'room'})
    assert studio.results.entries()[0]['custom_name'] == ''
    studio.results.save_name(root.name, '  Jellyfish — best geometry  ')
    assert json.loads((root / 'run-name.json').read_text()) == {'name': 'Jellyfish — best geometry'}
    row = next(r for r in studio.results.entries() if r['run_id'] == root.name)
    assert row['custom_name'] == 'Jellyfish — best geometry'
    assert row['scene'] == 'room'
    assert studio.results.run_detail(root.name)['custom_name'] == row['custom_name']
    assert studio.results.read_name(other['run_id']) == ''
    assert (root / 'config.json').read_bytes() == original_config
    studio.results.save_name(root.name, '')
    assert studio.results.read_name(root.name) == ''
    for invalid in (None, 123, 'a' * 201):
        with pytest.raises(ValueError):
            studio.results.save_name(root.name, invalid)
    with pytest.raises(ValueError):
        studio.results.save_name('../outside', 'name')


def test_custom_run_name_survives_reduction(tmp_path):
    studio, root, _ = reduction_setup(tmp_path)
    studio.results.save_name(root.name, 'Favorite result')
    studio.results.reduce_run(root.name)
    assert studio.results.read_name(root.name) == 'Favorite result'


def test_review_lookup_skips_catalog_bookkeeping(tmp_path, monkeypatch):
    studio, _, root = setup(tmp_path)
    key = studio.results.entries()[0]['id']
    def unexpected(*args, **kwargs):
        raise AssertionError("Viewer must not build cards or count output sizes")
    monkeypatch.setattr(studio.store, 'output_size_bytes', unexpected)
    monkeypatch.setattr(studio.results, '_row', unexpected)
    row = studio.results.detail(key, review=True)
    assert row['_paths']['repaired'] == root / 'artifixer3d.ply'


@pytest.mark.parametrize('backup_path', [None, '', '   ', False, 123, []])
def test_runs_without_recorded_backup_are_not_backed_up(tmp_path, backup_path):
    studio, root, _ = setup(tmp_path)
    assert studio.results.entries()[0]['backed_up'] is False
    (root / 'reduction.json').write_text(json.dumps({'reduced': True, 'backup_path': backup_path}))
    row = studio.results.entries()[0]
    assert row['backed_up'] is False
    assert row['backup_path'] is None


def test_saved_gallery_and_urls_reuse_deduplicated_artifacts(tmp_path):
    from splat_explorer.splatfix.deduplication import deduplicate_artifacts
    studio, root, result = setup(tmp_path)
    original = root / 'gpu/checkpoint/trajectory/00000.png'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'original RGB')
    copy = result / 'input-images/original_rgb_images/00000.png'
    copy.parent.mkdir(parents=True)
    copy.write_bytes(original.read_bytes())
    checkpoint = root / 'gpu/checkpoint/model.ckpt'
    checkpoint.write_bytes(b'model weights')
    (result / 'model.ckpt').write_bytes(checkpoint.read_bytes())
    deduplicate_artifacts(root)
    assert studio.file_url(copy) == studio.file_url(original)
    assert studio.file_url(result / 'model.ckpt') == studio.file_url(checkpoint)
    saved = studio.results.run_detail(studio.results.entries()[0]['id'])['saved_artifacts']
    images = [image for group in saved['image_groups'] for image in group['images']]
    assert len(images) == 1
    assert images[0]['url'] == studio.file_url(original)
    assert len([item for item in saved['checkpoints'] if item['name'].endswith('.ckpt')]) == 1


def test_fast_detail_skips_expensive_sections_and_unrelated_run_sizes(tmp_path, monkeypatch):
    studio, run, _ = setup(tmp_path)
    calls = []
    def section(row, name):
        calls.append(name)
        return {name + '_loaded': True}
    monkeypatch.setattr(studio.results, '_detail_section', section)
    monkeypatch.setattr(studio.store, 'output_size_bytes', lambda *_: pytest.fail('Detail must not calculate run sizes'))
    monkeypatch.setattr(studio.results, '_reconstruction_gallery', lambda *a, **kw: pytest.fail('Initial detail must not scan images'))
    fast = studio.results.run_detail(run.name, deferred=True)
    assert fast['run_id'] == run.name and fast['gallery']['frames'] == []
    assert set(fast['deferred_sections']) == {'training', 'artifacts', 'previews', 'anchor', 'gallery'}
    assert calls == []
    assert studio.results.detail_section(run.name, 'anchor') == {'anchor_loaded': True}
    assert calls == ['anchor']
    assert studio.results.detail_section('missing', 'training') is None
    with pytest.raises(ValueError, match='Unknown detail section'):
        studio.results.detail_section(run.name, 'invalid')


def test_detail_sections_are_independent(tmp_path, monkeypatch):
    import splat_explorer.web.artifixer_anchor_chart as charts
    import splat_explorer.web.training_logs as logs
    studio, run, _ = setup(tmp_path)
    monkeypatch.setattr(charts, 'anchor_chart', lambda *_: (_ for _ in ()).throw(RuntimeError('slow chart failed')))
    monkeypatch.setattr(logs, 'export_training_logs', lambda *_: ([], None, None))
    assert studio.results.run_detail(run.name, deferred=True)['run_id'] == run.name
    assert studio.results.detail_section(run.name, 'training')['logs'] == []
    assert studio.results.detail_section(run.name, 'artifacts')['saved_artifacts']['checkpoints']
    with pytest.raises(RuntimeError, match='slow chart failed'):
        studio.results.detail_section(run.name, 'anchor')


def test_background_detail_sections_render_independently_and_retry():
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for dashboard logic')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map(), requests=[];
const get=id=>{if(!elements.has(id))elements.set(id,{addEventListener(){}});return elements.get(id)};
const context=vm.createContext({document:{getElementById:get,addEventListener(){}},window:{addEventListener(){}},
location:{search:'?id=job'},URLSearchParams,ResizeObserver:class{observe(){}},
fetch:url=>new Promise((resolve,reject)=>requests.push({url,resolve,reject})),clearTimeout(){},setTimeout(){},assert});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],context);
(async()=>{
vm.runInContext(`
let rendered=[];renderImageGallery=r=>rendered.push('gallery');showAnchor=r=>rendered.push('anchor');showLogs=r=>rendered.push('logs');showTraining=r=>rendered.push('training');
currentDetail={deferred_sections:['anchor','training','gallery']};loadDetailSections('job',currentDetail);
loadDetailSections('job',currentDetail);
`,context);
assert.equal(requests.length,4); // initial metadata plus exactly one request per section
requests.find(r=>r.url.includes('section=training')).resolve({ok:true,json:async()=>({logs:[]})});
await new Promise(setImmediate);
vm.runInContext(`assert.equal(rendered.join(','),'logs,training');assert.ok($('backgroundStatus').textContent.includes('anchor'));`,context);
requests.find(r=>r.url.includes('section=gallery')).resolve({ok:true,json:async()=>({gallery:{frames:[{index:2}]},gallery_pending:false})});
await new Promise(setImmediate);
vm.runInContext(`assert.ok(rendered.includes('gallery'));assert.equal(currentDetail.gallery.frames[0].index,2);`,context);
requests.find(r=>r.url.includes('section=anchor')).reject(Error('failed'));
await new Promise(setImmediate);
vm.runInContext(`assert.ok($('backgroundStatus').textContent.includes('Refresh to retry'));currentDetail.kind='ArtiFixer3D';currentDetail.deferred_sections=['anchor','training'];loadDetailSections('job',currentDetail,true);`,context);
requests.filter(r=>r.url.includes('section=anchor')).at(-1).resolve({ok:true,json:async()=>({anchor_chart:{url:'/chart'}})});
await new Promise(setImmediate);
vm.runInContext(`assert.ok(rendered.includes('anchor'));assert.equal(currentDetail.anchor_chart.url,'/chart');`,context);
assert.ok(requests.filter(r=>r.url.includes('section=gallery')).at(-1).url.includes('&force=1'));
requests.filter(r=>r.url.includes('section=gallery')).at(-1).resolve({ok:true,json:async()=>({gallery:{frames:[{index:2},{index:10}]},gallery_pending:false})});
await new Promise(setImmediate);
vm.runInContext(`assert.equal(currentDetail.gallery.frames.length,2);assert.equal(rendered.filter(x=>x==='gallery').length,2);`,context);

})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/splatfix_result_detail.html'], check=True)


def test_catalog_cache_returns_cards_without_waiting_for_size_scan(tmp_path, monkeypatch):
    import threading
    studio, _, _ = setup(tmp_path)
    catalog = studio.results
    started, finish = threading.Event(), threading.Event()
    calls = []
    def entries(*, cached_sizes_only=False):
        calls.append(cached_sizes_only)
        if cached_sizes_only:
            return [{'id': 'run', 'output_size_bytes': None}]
        started.set()
        assert finish.wait(5)
        return [{'id': 'run', 'output_size_bytes': 42}]
    monkeypatch.setattr(catalog, 'entries', entries)
    try:
        first = catalog.catalog_snapshot()
        assert started.wait(5)
        assert first['results'][0]['output_size_bytes'] is None
        assert first['refreshing']
        first['results'].clear()  # callers cannot mutate the cached snapshot
        again = catalog.catalog_snapshot(refresh=True)
        assert again['results'][0]['id'] == 'run'
        assert calls == [True, False]  # concurrent requests share one refresh
    finally:
        finish.set()
        # Join only the worker belonging to this catalog test.
        for thread in threading.enumerate():
            if thread.name == 'results-catalog':
                thread.join(5)
    assert catalog.catalog_snapshot()['results'][0]['output_size_bytes'] == 42
    assert calls == [True, False]


def test_catalog_invalidated_during_refresh_cannot_publish_old_rows(tmp_path, monkeypatch):
    studio, _, _ = setup(tmp_path)
    catalog = studio.results
    catalog._catalog_rows = [{'id': 'old'}]
    revision = catalog._catalog_revision
    def entries():
        catalog.invalidate_catalog()
        return [{'id': 'deleted'}]
    monkeypatch.setattr(catalog, 'entries', entries)
    catalog._refresh_catalog(revision)
    assert catalog._catalog_rows is None
    assert not catalog._catalog_refreshing


def test_catalog_refresh_failure_keeps_last_good_snapshot(tmp_path, monkeypatch):
    studio, _, _ = setup(tmp_path)
    catalog = studio.results
    catalog._catalog_rows = [{'id': 'saved'}]
    monkeypatch.setattr(catalog, 'entries', lambda: (_ for _ in ()).throw(OSError('transfer in progress')))
    catalog._refresh_catalog(catalog._catalog_revision)
    assert catalog._catalog_rows == [{'id': 'saved'}]
    assert catalog._catalog_error and not catalog._catalog_refreshing


def test_cached_size_lookup_does_not_walk_run_files(tmp_path, monkeypatch):
    import os
    studio, run, _ = setup(tmp_path)
    studio.store.set_status(run.name, 'stopped')
    monkeypatch.setattr(os, 'scandir', lambda *_: pytest.fail('Cold list must not scan output trees'))
    assert studio.store.output_size_bytes(run.name, cached_only=True) is None


def test_card_preview_resolves_only_first_verified_input(tmp_path, monkeypatch):
    studio, _, root = setup(tmp_path)
    pred = root / 'inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    for index in range(100):
        (pred / f'{index}.png').write_bytes(b'image')
    (root / 'supervision.json').write_text(json.dumps({'groups': [
        {'source_index': 0, 'reference': True}, {'source_index': 2}, {'source_index': 10}]}))
    urls = []
    monkeypatch.setattr(studio, 'file_url', lambda path: urls.append(path.name) or '/' + path.name)
    gallery, allowed = studio.results._reconstruction_gallery(root / 'result.json', {}, limit=1)
    assert allowed == {2, 10}
    assert [frame['index'] for frame in gallery['frames']] == [2]
    assert urls == ['2.png']


def test_teacher_gallery_uses_model_input_order_not_camera_storage_ids(tmp_path):
    studio, run, root = setup(tmp_path)
    outputs = [*range(266, 258, -1), *range(1, 17)]
    for phase in ('inference', 'plus'):
        pred = root / phase / 'splatfix/frames/batch_0000/pred'
        pred.mkdir(parents=True)
        # Files and even the output ownership list may be numerically sorted.
        for i in [0, *sorted(outputs), 999]: (pred / f'{i:05d}.png').write_bytes(b'image')
        (root / f'{phase}-teacher.json').write_text(json.dumps({
            'inference_model': '14b-teacher', 'clips': [{'output': phase,
                'input_frame_indices': [*range(266,258,-1), 0, *range(1,17)],
                'frame_indices': sorted(outputs), 'warmup_frame_indices': list(range(259,267))}]}))
    detail = studio.results.run_detail(run.name)
    frames = detail['gallery']['frames']
    assert [f['index'] for f in frames] == outputs
    assert frames[0]['warmup'] and frames[0]['input_position'] == 1
    assert frames[8]['input_position'] == 10 and not frames[8]['warmup']
    assert 'Model input order' in detail['gallery']['order_note']
    assert [f['index'] for f in detail['plus_gallery']['frames']] == outputs
    card, _ = studio.results._reconstruction_gallery(root / 'result.json', {}, limit=1)
    assert card['frames'][0]['index'] == 266
