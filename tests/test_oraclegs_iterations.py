"""Iteration testing through queue normalization, saved PLYs and review navigation."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.oraclegs.backend import validate_options
from splat_explorer.splatfix.oraclegs.execution import job_runtime
from splat_explorer.splatfix.oraclegs.model_worker import install_snapshot_marker
from splat_explorer.splatfix.oraclegs.repair import DEFAULT_RUNTIME, training_args
from splat_explorer.web.oraclegs_checkpoints import iteration_checkpoints
from splat_explorer.web.splatfix_results import ResultViser


@pytest.mark.parametrize('enabled,final,expected', [
    (False, 22000, [3000, 7000, 15000, 22000]),
    (True, 22000, list(range(1000, 22001, 1000))),
    (True, 2500, [1000, 2000, 2500]),
    (True, 750, [750]),
])
def test_training_retains_full_budget_with_optional_snapshot_schedule(enabled, final, expected):
    runtime = {**DEFAULT_RUNTIME, 'oracle_iterations_testing':enabled, 'iterations':final}
    args = training_args('dataset', 'model', 6, runtime)
    assert args[args.index('--iterations') + 1] == str(final)
    assert list(map(int, args[args.index('--save_iterations') + 1:])) == expected


@pytest.mark.parametrize('enabled', [False, True])
def test_setting_survives_artifixer_queue_settings_and_worker_runtime(tmp_path, monkeypatch, enabled):
    from test_oraclegs_artifixer_inputs import fixture
    from test_splatfix_results import setup
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    monkeypatch.setattr(CpuSplatRenderer, 'render_depth',
                        lambda self, camera:np.full((camera.height, camera.width), 4, np.float32))
    studio, _, _ = setup(tmp_path/'dashboard')
    _, row = fixture(tmp_path)
    row.update(config={'splatfix':{}}, scene='room')
    studio.results.detail = lambda key: row
    queued = studio.results.reconstruct_artifixer('result', oracle_iterations_testing=enabled)
    options = queued['config']['splatfix']
    assert options['oracle_iterations_testing'] is enabled
    settings = json.loads((studio.root/queued['run_id']/'settings.json').read_text())
    assert settings['effective_config']['splatfix']['oracle_iterations_testing'] is enabled
    runtime = job_runtime({'splatfix':{'oraclegs_runtime':{'iterations':22000}}}, options)
    assert runtime['oracle_iterations_testing'] is enabled
    assert runtime['iterations'] == 22000


@pytest.mark.parametrize('value', ['true', 'false', 1, 0, None, []])
def test_iteration_setting_rejects_non_booleans(value):
    with pytest.raises(ValueError, match='oracle_iterations_testing'):
        validate_options({'oracle_iterations_testing':value})


@pytest.mark.parametrize('enabled', [None, False, True])
def test_reconstruction_endpoint_forwards_iteration_setting(enabled):
    from splat_explorer.web.server import DashboardHandler
    calls, replies = [], []
    results = SimpleNamespace(invalidate_catalog=lambda:None,
        reconstruct_artifixer=lambda *a, **kw: calls.append(kw) or {'run_id':'queued'})
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=SimpleNamespace(results=results)),
        command='POST', _send_json=lambda body, *args:replies.append(body))
    body = {'id':'run'}
    if enabled is not None:
        body['oracle_iterations_testing'] = enabled
    DashboardHandler._serve_splatfix(handler, '/api/splatfix/results/reconstruct-artifixer', body)
    assert calls[0]['oracle_iterations_testing'] is (enabled is True)
    assert replies[0]['ok']


def snapshot(root, iteration):
    path = root/f'model/point_cloud/iteration_{iteration}/point_cloud.ply'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'ply\ncomplete snapshot')
    atomic_json(path.parent/'saved-snapshot.json', {'iteration':iteration, 'size_bytes':path.stat().st_size})
    return path


def result_row(root):
    root.mkdir(exist_ok=True)
    atomic_json(root/'result.json', {'reconstruction_method':'oraclegs'})
    original = root/'original.ply'
    original.write_bytes(b'ply')
    return {'_manifest':root/'result.json', '_paths':{'original':original, 'repaired':None},
            'config':{'splatfix':{'oracle_iterations_testing':True}}, 'scene':'room', 'run_id':'test'}


def test_snapshot_marker_is_written_only_after_successful_ply_save(tmp_path, monkeypatch):
    class Scene:
        model_path = str(tmp_path/'model')

        def save(self, iteration):
            path = Path(self.model_path)/f'point_cloud/iteration_{iteration}/point_cloud.ply'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'ply')
            if iteration == 2000:
                raise OSError('disk full')

    monkeypatch.setitem(sys.modules, 'scene', SimpleNamespace(Scene=Scene))
    install_snapshot_marker()
    Scene().save(1000)
    marker = tmp_path/'model/point_cloud/iteration_1000/saved-snapshot.json'
    assert json.loads(marker.read_text()) == {'iteration':1000, 'size_bytes':3}
    with pytest.raises(OSError):
        Scene().save(2000)
    assert not (tmp_path/'model/point_cloud/iteration_2000/saved-snapshot.json').exists()
    assert json.loads((tmp_path/'model/saved-snapshot.json').read_text())['iteration'] == 1000


def test_checkpoint_discovery_sorts_numerically_and_ignores_unconfirmed_or_unsafe_files(tmp_path):
    row = result_row(tmp_path/'result')
    root = row['_manifest'].parent
    for iteration in (10000, 2000, 1000):
        snapshot(root, iteration)
    snapshot(root, 3000).write_bytes(b'truncated')
    snapshot(root, 4000).with_name('saved-snapshot.json').unlink()
    outside = snapshot(tmp_path/'outside', 5000)
    (root/'model/point_cloud/iteration_5000').symlink_to(outside.parent)
    snapshot(root, 6000).unlink()
    (root/'model/point_cloud/iteration_6000/point_cloud.ply').symlink_to(outside)
    assert [c['iteration'] for c in iteration_checkpoints(row)] == [1000, 2000, 10000]
    row['config']['splatfix']['oracle_iterations_testing'] = False
    assert iteration_checkpoints(row) == []
    # The downloaded recipe also preserves the option when queue metadata is absent.
    atomic_json(root/'recipe.json', {'runtime':{'oracle_iterations_testing':True}})
    assert len(iteration_checkpoints(row)) == 3


def test_graceful_stop_includes_off_interval_snapshot_and_deduplicates_final(tmp_path):
    row = result_row(tmp_path)
    snapshot(tmp_path, 1000)
    snapshot(tmp_path, 2000)
    stopped = tmp_path/'oraclegs-partial.ply'
    stopped.write_bytes(b'ply\nstop')
    row['_paths']['repaired'] = stopped
    row['_manifest'] = tmp_path/'partial-result.json'
    for iteration, expected in [(2371, [1000, 2000, 2371]), (2000, [1000, 2000])]:
        atomic_json(row['_manifest'], {'incomplete':True, 'saved_iteration':iteration})
        checkpoints = iteration_checkpoints(row)
        assert [c['iteration'] for c in checkpoints] == expected
        assert checkpoints[-1]['path'] == stopped


def test_result_details_publish_download_urls_for_saved_iterations(tmp_path):
    from test_splatfix_results import setup
    studio, _, root = setup(tmp_path)
    atomic_json(root/'recipe.json', {'runtime':{'oracle_iterations_testing':True}})
    atomic_json(root/'result.json', {'reconstruction_method':'oraclegs',
        'splat_path':'artifixer3d.ply', 'saved_iteration':2500})
    snapshot(root, 1000)
    snapshot(root, 2000)
    key = studio.results.entries()[0]['id']
    detail = studio.results.run_detail(key, deferred=True)
    assert [c['iteration'] for c in detail['iteration_checkpoints']] == [1000, 2000, 2500]
    assert detail['iteration_checkpoints'][0]['url'] == studio.file_url(
        root/'model/point_cloud/iteration_1000/point_cloud.ply')
    assert detail['iteration_checkpoints'][-1]['url'] == detail['ply_url']


def test_iteration_navigation_reloads_selected_ply_and_preserves_camera(tmp_path):
    from test_scene_run_viser import _FakeServer, _tiny_scene, _studio
    studio, _ = _studio(tmp_path)
    row = result_row(tmp_path/'result')
    paths = {i:snapshot(row['_manifest'].parent, i) for i in (1000, 2000, 10000)}
    loaded = []
    catalog = SimpleNamespace(cfg=studio.cfg, detail=lambda *a, **kw:row,
                              review_views=lambda *a, **kw:[])
    def loader(path, **kwargs):
        loaded.append(path)
        return _tiny_scene()
    viewer = ResultViser(catalog, background=False, server_factory=_FakeServer,
                         load_scene=loader, idle_timeout=None)
    state = viewer.show('result')
    assert state['iteration'] == 10000 and state['iteration_index'] == 2
    assert state['iteration_count'] == 3 and loaded[0] == paths[10000]
    camera = SimpleNamespace(position=np.array([8., 9., 10.]), look_at=np.array([0., 1., 2.]),
        up_direction=np.array([0., 1., 0.]), fov=0.8, wxyz=np.array([1., 0., 0., 0.]))
    viewer._server._clients['test'] = SimpleNamespace(camera=camera)
    for delta, iteration in [(-1, 2000), (-1, 1000), (1, 2000)]:
        state = viewer.move_iteration('result', delta)
        assert state['status'] == 'ready' and state['iteration'] == iteration
        assert loaded[-1] == paths[iteration]
        assert len(viewer._scenes) == 2
        assert set(viewer._server.scene.active) == {'/review/repaired'}
        np.testing.assert_array_equal(camera.position, [8., 9., 10.])
        np.testing.assert_array_equal(camera.look_at, [0., 1., 2.])
        assert camera.fov == 0.8
    viewer.show('result', toggle=True)
    assert viewer.snapshot('result')['which'] == 'original'
    assert viewer.show('result', toggle=True)['iteration'] == 2000
    viewer.move_iteration('result', -1)
    count = len(loaded)
    assert viewer.move_iteration('result', -1)['iteration'] == 1000
    assert len(loaded) == count  # Do not wrap or reload beyond the endpoint.
    viewer._server._clients.clear()
    viewer.release('result')
    assert not viewer._checkpoints and not viewer._checkpoint_indices


@pytest.mark.parametrize('delta', [True, 1.0, '1', 0, 2])
def test_iteration_navigation_rejects_invalid_delta(tmp_path, delta):
    viewer = ResultViser(SimpleNamespace(cfg={}), background=False)
    with pytest.raises(ValueError, match='iteration_delta'):
        viewer.move_iteration('result', delta)


def test_viewer_endpoint_routes_iteration_navigation():
    from splat_explorer.web.server import DashboardHandler
    calls, replies = [], []
    visor = SimpleNamespace(move_iteration=lambda *a, **kw: calls.append((a, kw)) or {'ok':True})
    results = SimpleNamespace(visor=visor, invalidate_catalog=lambda:None)
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=SimpleNamespace(results=results)),
        command='POST', path='/api/splatfix/results/viser?id=result',
        _send_json=lambda body, *args:replies.append(body))
    DashboardHandler._serve_splatfix(handler, '/api/splatfix/results/viser', {'iteration_delta':-1, 'client':'tab'})
    assert calls == [(('result', -1), {'client':'tab'})]
    assert replies == [{'ok':True}]


def test_iteration_controls_send_deltas_and_disable_at_boundaries_and_during_loading():
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for viewer JavaScript test')
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync(process.argv[1], 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map(), requests = [];
function element(id) {
  if (!elements.has(id)) elements.set(id, {classList:{toggle(){}, remove(){}},
    setAttribute(){}, removeAttribute(){}, firstChild:{}, src:'', disabled:false});
  return elements.get(id);
}
let next = {ok:true, status:'ready', which:'repaired', viewer_url:'http://localhost:8083',
  original:true, repaired:true, view_count:2, iteration:3000, iteration_index:2, iteration_count:3};
let release;
const context = vm.createContext({
  document:{getElementById:element, querySelector:element, body:{}, addEventListener(){}},
  location:{pathname:'/splatfix/results/viser', search:'?id=result'},
  crypto:{randomUUID:()=> 'tab'}, URLSearchParams, Blob,
  navigator:{sendBeacon:()=>true}, window:{addEventListener(){}},
  setTimeout:()=>1, clearTimeout(){},
  fetch:async (url, options)=> {
    requests.push(JSON.parse(options.body));
    const body = {...next};
    if (requests.length > 1) await new Promise(resolve=> {release=resolve;});
    return {ok:true, json:async()=>body};
  }
});
const flush = ()=> new Promise(resolve=>setImmediate(resolve));
(async()=> {
  vm.runInContext(source, context); await flush();
  assert.equal(element('iteration-nav').hidden, false);
  assert.match(element('iteration-index').textContent, /3,000 iterations/);
  assert.equal(element('iteration-next').disabled, true);
  assert.equal(element('iteration-prev').disabled, false);
  element('iteration-next').onclick(); await flush();
  assert.equal(requests.length, 1);
  next.iteration=2000; next.iteration_index=1;
  element('iteration-prev').onclick(); await flush();
  assert.equal(element('iteration-prev').disabled, true);
  assert.equal(element('iteration-next').disabled, true);
  assert.equal(requests[1].iteration_delta, -1);
  assert.equal(requests[1].client, 'tab');
  release(); await flush();
  assert.equal(element('iteration-next').disabled, false);
  next.iteration=1000; next.iteration_index=0;
  element('iteration-prev').onclick(); await flush(); release(); await flush();
  assert.equal(element('iteration-prev').disabled, true);
  assert.equal(element('iteration-next').disabled, false);
  assert.equal(element('view-prev').disabled, false);
  next.status='loading'; next.iteration=2000; next.iteration_index=1;
  element('iteration-next').onclick(); await flush(); release(); await flush();
  assert.equal(requests.at(-1).iteration_delta, 1);
  assert.equal(element('iteration-prev').disabled, true);
  assert.equal(element('iteration-next').disabled, true);
  vm.runInContext('state={ok:true,status:"ready",which:"repaired"};render()', context);
  assert.equal(element('iteration-nav').hidden, true);
})().catch(error=> {console.error(error); process.exitCode=1;});
'''
    reply = subprocess.run([node, '-e', script, 'src/splat_explorer/web/static/scene_run_viser.html'],
                           capture_output=True, text=True)
    assert reply.returncode == 0, reply.stdout + reply.stderr
