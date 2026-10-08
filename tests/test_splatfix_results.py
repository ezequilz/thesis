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
