import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from splat_explorer.web.oraclegs_details import build_details
from splat_explorer.splatfix.oraclegs.execution import mirror_previews
from test_splatfix_results import setup


def fixture(tmp_path):
    studio, run, old = setup(tmp_path)
    config = json.loads((run/'config.json').read_text())
    config['splatfix'].update(stage='repair', reconstruction_method='oraclegs', checkpoint=str(tmp_path/'checkpoint'), mode='edited')
    (run/'config.json').write_text(json.dumps(config))
    import shutil
    shutil.rmtree(old)
    root = run/'gpu/results/oraclegs'
    root.mkdir()
    return studio, run, root


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB',(64,36),'red').save(path)


def test_pending_run_uses_oracle_details_without_result_manifest(tmp_path):
    studio, run, root = fixture(tmp_path)
    inputs = run/'oraclegs-inputs-test/inputs/inputs.json'
    body = {'views':[{'view_id':'one','image':'scene/images/ref_000000.png'}], 'resolution':[1024,576]}
    write(inputs,body)
    png(inputs.parent/'scene/images/ref_000000.png')
    write(run/'oraclegs-staging.json',{'input_manifest':str(inputs.relative_to(run)),
          'manifest_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'status':'transferred'})
    write(run/'worker-status.json',{'status':'running','phase':'setup'})
    row = studio.results.run_detail(run.name)
    assert row['kind'] == 'OracleGS'
    d = row['oraclegs']
    assert d['views'][0]['url'].endswith('ref_000000.png')
    assert d['inputs']['resolution'] == [1024,576]
    assert not d['alignment']
    assert not d['anchors']
    assert d['state']['phase'] == 'setup'


def test_live_generation_attention_pairing_and_overview_cache(tmp_path):
    studio,run,root = fixture(tmp_path)
    write(root/'recipe.json',{'inputs':{'views':[{'image':'scene/images/ref_000000.png'}]},
                             'commands':[{'phase':'seva_generation'},{'phase':'vggt_attention'}]})
    write(run/'worker-status.json',{'status':'running','phase':'vggt_attention'})
    png(root/'dataset/images/ref_000000.png')
    png(root/'dataset/synthetic_1/target_000010.png')
    png(root/'attention-raw/synth_target_000010.png')
    png(root/'seva/img2trajvid/scene/first-pass/samples-rgb/000.png')
    png(root/'seva/img2trajvid/scene/samples-rgb/010.png')
    for name in ('first-pass/samples-rgb.mp4', 'samples-rgb.mp4'):
        (root/'seva/img2trajvid/scene'/name).write_bytes(b'saved video')
    d = build_details(studio, root/'result.json', {})
    assert len(d['anchors']) == len(d['trajectory']) == 1
    assert [v['label'] for v in d['videos']] == ['First pass · sparse anchors', 'Full generated trajectory']
    assert all(v['url'].endswith('samples-rgb.mp4') for v in d['videos'])
    assert d['synthetic'][0]['attention_url'].endswith('synth_target_000010.png')
    assert d['synthetic'][0]['role'] == 'Synthetic at intermediate pose'
    assert d['overview_url']
    overview = next((root/'debug-previews').glob('*.jpg'))
    timestamp = overview.stat().st_mtime_ns
    build_details(studio,root/'result.json',{})
    assert overview.stat().st_mtime_ns == timestamp
    assert next(s for s in d['stages'] if s['name']=='seva_generation')['status']=='completed'


def test_missing_and_failed_gates_are_distinct_and_unsafe_paths_hidden(tmp_path):
    studio,run,root = fixture(tmp_path)
    outside = tmp_path/'secret.json'
    write(outside, {'passed':True})
    (root/'camera-alignment.json').symlink_to(outside)
    write(root/'geometry-check.json',{'passed':False,'triangulated_points':0})
    write(root/'recipe.json',{'inputs':{'views':[{'image':'../../secret.json'}]}})
    d = build_details(studio,root/'result.json',{})
    assert d['alignment']=={}
    assert d['geometry']['passed'] is False
    assert d['views'][0]['url'] is None
    assert not any(a['label']=='Camera alignment gate' for a in d['artifacts'])


def test_preview_cache_throttles_and_excludes_training_state(tmp_path,monkeypatch):
    calls=[]
    lrz=SimpleNamespace(rsync_ssh_cmd=lambda cfg:'ssh',_mux_run=lambda argv:calls.append(argv))
    clock=[1000]
    monkeypatch.setattr('splat_explorer.splatfix.oraclegs.execution.time.time',lambda:clock[0])
    assert mirror_previews(lrz,{},'user@host','/remote/run',tmp_path,phase='seva_generation')
    assert not mirror_previews(lrz,{},'user@host','/remote/run',tmp_path,phase='seva_generation')
    assert len(calls)==1
    assert '- model/' in calls[0] and '- combined/' in calls[0]
    assert '--inplace' not in calls[0]
    clock[0]+=61
    mirror_previews(lrz,{},'user@host','/remote/run',tmp_path,phase='reconstruction')
    clock[0]+=61
    assert not mirror_previews(lrz,{},'user@host','/remote/run',tmp_path,phase='reconstruction')
    assert len(calls)==2


def test_preview_failure_is_recorded_for_retry(tmp_path):
    lrz=SimpleNamespace(rsync_ssh_cmd=lambda cfg:'ssh',_mux_run=lambda argv:SimpleNamespace(returncode=23))
    with pytest.raises(RuntimeError):
        mirror_previews(lrz,{},'user@host','/remote/run',tmp_path,phase='vggt_attention')
    assert json.loads((tmp_path/'oraclegs-preview-sync.json').read_text())['status']=='retrying'


def test_stop_salvages_only_latest_confirmed_snapshot(tmp_path):
    from splat_explorer.splatfix.oraclegs.partial import salvage
    prior = tmp_path/'model/point_cloud/iteration_3000/point_cloud.ply'
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b'ply\ncomplete snapshot')
    incomplete = tmp_path/'model/point_cloud/iteration_7000/point_cloud.ply'
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b'ply\ninterrupted write')
    write(tmp_path/'model/saved-snapshot.json',{'iteration':3000,'size_bytes':prior.stat().st_size})
    result = salvage(tmp_path,'Stopped by user')
    assert result['saved_iteration']==3000
    assert result['incomplete'] is True
    assert (tmp_path/'oraclegs-partial.ply').read_bytes()==prior.read_bytes()
    assert not (tmp_path/'result.json').exists()
    assert json.loads((tmp_path/'partial-result.json').read_text())['incomplete'] is True


def test_stop_before_snapshot_never_fabricates_partial_ply(tmp_path):
    from splat_explorer.splatfix.oraclegs.partial import salvage
    assert salvage(tmp_path,'Stopped during SEVA') is None
    assert not (tmp_path/'partial-result.json').exists()


def test_failed_setup_stage_uses_cached_log_provenance(tmp_path):
    studio,run,root = fixture(tmp_path)
    write(run/'worker-status.json',{'status':'error','phase':'finished','message':'Setup failed'})
    write(run/'gpu/live-stage.json',{'relative_path':'results/oraclegs/setup.log'})
    d=build_details(studio,root/'result.json',{})
    assert d['state']['phase']=='setup'
    assert next(s for s in d['stages'] if s['name']=='setup')['status']=='error'


def test_partial_snapshot_is_discoverable_and_never_completed(tmp_path):
    from splat_explorer.splatfix.oraclegs.partial import salvage
    studio,run,root=fixture(tmp_path)
    native=root/'model/point_cloud/iteration_3000/point_cloud.ply'
    native.parent.mkdir(parents=True)
    native.write_bytes(b'ply\ncompleted-save')
    write(root/'model/saved-snapshot.json', {'iteration':3000,'size_bytes':native.stat().st_size})
    salvage(root,'User stop')
    row=studio.results.run_detail(run.name)
    assert row['incomplete'] is True
    assert row['status']!='completed'
    assert row['ply_url'].endswith('oraclegs-partial.ply')
    assert row['oraclegs']['saved_iteration']==3000


def test_snapshot_marker_only_published_after_successful_save(tmp_path,monkeypatch):
    from splat_explorer.splatfix.oraclegs.model_worker import install_snapshot_marker
    import sys
    class Scene:
        model_path=str(tmp_path)
        def save(self,iteration):
            native=tmp_path/f'point_cloud/iteration_{iteration}/point_cloud.ply'
            native.parent.mkdir(parents=True)
            native.write_bytes(b'ply')
            if iteration==7000:
                raise OSError('Interrupted PLY write')
    monkeypatch.setitem(sys.modules,'scene',SimpleNamespace(Scene=Scene))
    install_snapshot_marker()
    scene=Scene()
    scene.save(3000)
    assert json.loads((tmp_path/'saved-snapshot.json').read_text())['iteration']==3000
    with pytest.raises(OSError):
        scene.save(7000)
    assert json.loads((tmp_path/'saved-snapshot.json').read_text())['iteration']==3000


def test_oracle_details_exposes_stop_until_request_then_partial_download(tmp_path):
    from splat_explorer.splatfix.oraclegs.partial import salvage
    studio,run,root=fixture(tmp_path)
    studio.store.update_status(run.name,status='running',phase='reconstruction')
    assert studio.results.run_detail(run.name)['can_stop'] is True
    studio.cancel(run.name)
    row=studio.results.run_detail(run.name)
    assert row['status']=='stopping' and not row['can_stop']
    native=root/'stop-snapshot.ply'
    native.write_bytes(b'ply\nstop saved')
    write(root/'stop-state.json',{'status':'saved','iteration':42,'size_bytes':native.stat().st_size})
    salvage(root,'User stop')
    studio.store.update_status(run.name,status='stopped',remote_finished=True)
    row=studio.results.run_detail(run.name)
    assert row['incomplete'] and row['oraclegs']['saved_iteration']==42
    assert row['ply_url'].endswith('oraclegs-partial.ply')
    assert not row['can_stop']
