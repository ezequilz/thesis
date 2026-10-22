import json
from types import SimpleNamespace

from splat_explorer.config import Config
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.web.splatfix_studio import SplatfixStudio


def test_snapshot_preserves_request_and_future_controls(tmp_path):
    store = SceneRunStore(tmp_path / 'scene-runs')
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})), store)
    studio.scenes = lambda: [{'id': 'room', 'label': 'Room'}]
    body = {'stage': 'select', 'scene_id': 'room', 'views': 7,
            'dashboard_snapshot': {'controls': [{'id': 'future', 'value': 'x', 'label': 'Future setting'}]}}
    run = studio.create(body)
    path = store.run_path(run['run_id']) / 'settings.json'
    saved = json.loads(path.read_text())
    assert saved['dashboard'] == body['dashboard_snapshot']
    assert saved['submitted'] == {'stage': 'select', 'scene_id': 'room', 'views': 7}
    assert saved['effective_config']['splatfix']['views'] == 7
    assert saved['effective_config']['splatfix']['renderer'] == 'viser'
    assert saved['run_id'] == run['run_id']
    body['dashboard_snapshot']['controls'][0]['value'] = 'changed'
    assert json.loads(path.read_text()) == saved


def test_detail_snapshot_and_historical_fallback(tmp_path):
    store = SceneRunStore(tmp_path / 'scene-runs')
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})), store)
    run = store.create_run({'pipeline': 'splatfix', 'splatfix': {'stage': 'repair', 'checkpoint': '/saved'}})
    path = store.run_path(run.run_id) / 'settings.json'
    detail = studio.results.run_detail(run.run_id)
    assert detail['settings_snapshot'] == json.loads(path.read_text())
    assert detail['settings_snapshot']['dashboard'] is None
    assert detail['settings_url'].endswith('settings.json')
    path.unlink()
    detail = studio.results.run_detail(run.run_id)
    assert not detail['settings_snapshot']
    assert detail['settings_url'] is None
    assert detail['config']['pipeline'] == 'splatfix'
