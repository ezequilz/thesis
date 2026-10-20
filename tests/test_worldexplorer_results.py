import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from splat_explorer.config import Config
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.splatfix.worldexplorer.service import WorldExplorerService
from splat_explorer.web.splatfix_studio import SplatfixStudio
from splat_explorer.web.splatfix_results import ResultViser

ID = 'we_20261020_215533_02b07c345ec5'


def setup(tmp_path):
    cfg = Config({'output': {'dir': str(tmp_path)}, 'agent': {'model': 'test'}})
    studio = SplatfixStudio(SimpleNamespace(cfg=cfg), SceneRunStore(tmp_path/'scene-runs'))
    run = tmp_path/'worldexplorer'/ID
    root = run/'results'
    root.mkdir(parents=True)
    (run/'job.json').write_text(json.dumps({'status': 'completed', 'created_at': 1,
        'request': {'operation': 'resume', 'prompt': 'A room', 'num_images_for_vggt': 64}}))
    for name in ['request.json', 'worker.log', 'scenes/room/img2trajvid/000_in/input/000.png',
                 'scenes/room/img2trajvid/selected_for_vggt/000.png',
                 'scenes/room/img2trajvid/000_in/samples-rgb/000.png',
                 'scenes/room/img2trajvid/000_in/video.mp4',
                 'nerfstudio_output/room/splatfacto/date/exports/splat/splat.ply']:
        p=root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
    return studio, run, root


def test_worldexplorer_uses_shared_catalog_media_endpoints_and_viser(tmp_path):
    studio, run, root = setup(tmp_path)
    row, = studio.results.entries()
    assert row['id'] == ID and row['kind'] == 'WorldExplorer'
    assert row['repaired'] and not row['original'] and not row['can_archive']
    assert '_paths' not in row
    detail = studio.results.run_detail(ID)
    groups = detail['worldexplorer']['image_groups']
    assert len(groups) == 3 and 'selected_for_vggt' in groups[0]['name']
    assert len(detail['worldexplorer']['videos']) == 1
    assert detail['logs'][0]['name'] == 'worker.log'
    api = WorldExplorerService(tmp_path/'worldexplorer')
    for url in [row['ply_url'], row['preview_url'], detail['logs'][0]['url'],
                detail['worldexplorer']['videos'][0]['url'], groups[0]['frames'][0]['url']]:
        query = parse_qs(urlsplit(url).query)
        assert api.file(query['id'][0], query['path'][0]).is_file()
    viewer = ResultViser(studio.results, background=False)
    assert viewer.ply_paths(ID)['repaired'].name == 'splat.ply'
    assert studio.results.review_views(ID) == []
    studio.results.set_hidden(ID, True)
    assert studio.results.entries()[0]['hidden']


def test_worldexplorer_rejects_external_media_and_ply_links(tmp_path):
    studio, run, root = setup(tmp_path)
    secret = tmp_path/'secret.png'; secret.write_bytes(b'secret')
    (root/'escape.png').symlink_to(secret)
    ply = next(root.rglob('splat.ply')); ply.unlink(); ply.symlink_to(secret)
    row, = studio.results.entries()
    assert not row['repaired']
    assert all(f['name'] != 'escape.png' for g in studio.results.run_detail(ID)['worldexplorer']['image_groups'] for f in g['frames'])
    linked = tmp_path/'worldexplorer'/'we_20261020_215534_02b07c345ec5'
    linked.symlink_to(run, target_is_directory=True)
    assert len(studio.results.entries()) == 1


def test_worldexplorer_unavailable_archive_still_has_details(tmp_path):
    studio, run, root = setup(tmp_path)
    import shutil
    shutil.rmtree(root)
    detail = studio.results.run_detail(ID)
    assert detail['worldexplorer'] == {'image_groups': [], 'videos': []}
    assert detail['logs'] == [] and not detail['repaired']


def test_relative_output_directory_handles_absolute_training_exports(tmp_path, monkeypatch):
    from splat_explorer.web import worldexplorer_details
    studio, run, root = setup(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    studio.results.cfg = Config({'output': {'dir': tmp_path.name}})
    source = root/'nerfstudio_output/events.out.tfevents.test'
    source.write_bytes(b'events')
    cache = root/'.training-metrics'; cache.mkdir()
    log = cache/'events.log'; log.write_text('decoded')
    chart = cache/'scalars.json'; chart.write_text('{}')
    monkeypatch.setattr(worldexplorer_details, 'export_training_logs',
                        lambda *args, **kwargs: ([(source.resolve(), log.resolve())], chart.resolve(), None))
    detail = studio.results.run_detail(ID)
    assert 'path=.training-metrics%2Fscalars.json' in detail['training_metrics']['url']
    assert any(log['name'].endswith('(decoded scalars)') for log in detail['logs'])


def camera_fixture(root):
    import numpy as np
    pose = np.eye(4); pose[:3, 3] = [1, 0, 0]
    transform = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3]]
    parser = next(root.rglob('splat.ply')).parents[2]/'dataparser_transforms.json'
    parser.write_text(json.dumps({'transform': transform, 'scale': 2}))
    path = root/'scenes/room/img2trajvid/transforms.json'
    path.write_text(json.dumps({'fl_y': 100, 'h': 200, 'frames': [
        {'file_path': '/remote/'+name, 'transform_matrix': pose.tolist()}
        for name in ['000_left/samples-rgb/0.png', '000_in/samples-rgb/10.png',
                     '000_in/input/000.png', '000_in/samples-rgb/2.png']]}))
    return parser, path


def test_worldexplorer_spawns_at_scaffold_and_steps_through_normalized_cameras(tmp_path):
    import numpy as np
    from test_scene_run_viser import _FakeServer, _tiny_scene
    studio, run, root = setup(tmp_path)
    camera_fixture(root)
    views = studio.results.review_views(ID)
    assert [v['label'] for v in views] == ['Scaffold 000', '000_in · frame 2',
                                         '000_in · frame 10', '000_left · frame 0']
    np.testing.assert_allclose(views[0]['center'], [2, 6, 6])
    np.testing.assert_allclose(views[0]['forward'], [0, 0, -1])
    np.testing.assert_allclose(views[0]['up'], [-1, 0, 0])
    assert np.isclose(views[0]['vfov'], np.pi/2)
    viewer = ResultViser(studio.results, background=False, server_factory=_FakeServer,
                         load_scene=lambda *a, **kw: _tiny_scene(), idle_timeout=None)
    state = viewer.show(ID)
    assert state['status'] == 'ready' and state['view_count'] == 4 and state['view_index'] == 0
    client = SimpleNamespace(camera=SimpleNamespace())
    viewer._server._clients['test'] = client
    viewer._server._connect(client)
    np.testing.assert_allclose(client.camera.position, [2, 6, 6])
    np.testing.assert_allclose(client.camera.look_at, [2, 6, 5])
    assert viewer.move_view(ID, 1)['view_label'] == '000_in · frame 2'
    assert viewer.move_view(ID, -1)['view_label'] == 'Scaffold 000'
    assert viewer.move_view(ID, -1)['view_label'] == '000_left · frame 0'


def test_worldexplorer_cameras_require_valid_matching_normalization(tmp_path):
    studio, run, root = setup(tmp_path)
    parser, path = camera_fixture(root)
    parser.write_text(json.dumps({'transform': [[1, 0, 0, 0]]*3, 'scale': 1}))
    assert studio.results.review_views(ID) == []
    parser.unlink()
    assert studio.results.review_views(ID) == []
    outside = tmp_path/'parser.json'; outside.write_text('{}')
    parser.symlink_to(outside)
    assert studio.results.review_views(ID) == []


def test_worldexplorer_videos_use_range_response():
    from splat_explorer.web.server import DashboardHandler
    handler = object.__new__(DashboardHandler)
    handler.app = SimpleNamespace(worldexplorer=SimpleNamespace(file=lambda *args: Path('/tmp/video.mp4')))
    handler.command = 'GET'
    handler.path = '/api/splatfix/worldexplorer/file?id='+ID+'&path=video.mp4'
    calls = []
    handler._send_video_file = calls.append
    handler._serve_worldexplorer('/api/splatfix/worldexplorer/file')
    assert calls == [Path('/tmp/video.mp4')]


def test_worldexplorer_gallery_uses_existing_popup_and_folder_navigation():
    import shutil
    import subprocess
    import pytest
    if not shutil.which('node'):
        pytest.skip('Node required for UI test')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,{style:{},addEventListener(){},showModal(){this.open=true}});return elements.get(id)};
const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>[]},
location:{search:''},URLSearchParams,ResizeObserver:class{observe(){}},
fetch:()=>new Promise(()=>{}),clearTimeout(){},setTimeout(){},assert});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],context);
vm.runInContext(`
const r={worldexplorer:{image_groups:[{name:'selected',frames:[{name:'first.png',label:'First',url:'/first',index:0},{name:'second.png',label:'Second',url:'/second',index:1}]},{name:'trajectory',frames:[{name:'third.png',label:'Third',url:'/third',index:0}]}],videos:[{name:'clip',url:'/clip.mp4'}]}};
showWorld(r);assert.equal(frames.length,2);assert.equal($('worldMedia').hidden,false);
choose(0,true);assert.equal($('lightbox').open,true);assert.equal($('full').src,'/first');
$('next').onclick();assert.equal($('full').src,'/second');assert.equal($('next').disabled,true);
$('worldGroup').value='trajectory';$('worldGroup').onchange();assert.equal(frames.length,1);assert.equal($('full').src,'/third');
assert.ok($('worldVideoList').innerHTML.includes('preload="none"'));
assert.equal(needsGalleryPolling({kind:'WorldExplorer',status:'running'}),true);
assert.equal(needsGalleryPolling({kind:'WorldExplorer',status:'completed',repaired:true}),false);
showWorld({});assert.equal($('worldMedia').hidden,true);
`,context);
'''
    subprocess.run(['node', '-e', script, 'src/splat_explorer/web/static/splatfix_result_detail.html'], check=True)
