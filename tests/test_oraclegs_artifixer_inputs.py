import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.oraclegs.artifixer_inputs import export_inputs, source_assets, publish_generated
from splat_explorer.splatfix.oraclegs.adapter import source_camera_model, assemble
from splat_explorer.splatfix.oraclegs.attention import plan_chunks
from splat_explorer.splatfix.oraclegs.source_geometry import validate_assets
from splat_explorer.web.splatfix_studio import SplatfixStudio
from test_oraclegs_source_geometry import source_ply


def fixture(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    (root/'pred').mkdir()
    cameras = []
    for i in range(8):
        pose = np.eye(4); pose[0,3] = i
        cameras.append({'transform_matrix':pose.tolist()})
    atomic_json(root/'preview-trajectory.json', {'camera_convention':'opencv_c2w',
        'anchors':[{'view_id':str(i), 'frame_index':i} for i in range(6)],
        'transforms':{'w':32,'h':16,'fl_x':20,'fl_y':20,'cx':16,'cy':8,'frames':cameras}})
    refs=[]
    for i in range(6):
        Image.new('RGB',(32,16),(i,0,0)).save(root/f'ref{i}.png')
        refs.append({'index':i,'path':f'ref{i}.png'})
    for i in (6,7):
        Image.new('RGB',(16,8),(0,i,0)).save(root/'pred'/f'{i:05d}.png')
    atomic_json(root/'input-preview.json', {'reference_images':refs})
    atomic_json(root/'inference-preview.json', {'prediction_frames':'pred','reconstruction_inputs':[6,7]})
    atomic_json(root/'result.json', {})
    source_ply(root/'original.ply')
    studio = SimpleNamespace(result_artifact=lambda manifest, raw:SplatfixStudio.result_artifact(None,manifest,raw))
    row = {'_manifest':root/'result.json','kind':'ArtiFixer3D','mode':'edited','run_id':'parent','id':'result', '_paths':{'original':root/'original.ply'}}
    return studio,row


@pytest.mark.parametrize('initialization',['sfm','source_splat'])
def test_reuse_preserves_camera_and_reference_roles(tmp_path,initialization):
    studio,row=fixture(tmp_path)
    inputs=tmp_path/'inputs'; output=tmp_path/'output'
    manifest=export_inputs(studio,row,inputs,initialization)
    validate_assets(inputs,manifest)
    assert bool(manifest.get('source_initialization')) == (initialization=='source_splat')
    frames=json.loads((inputs/'scene/transforms.json').read_text())['frames']
    assert len(frames)==8
    assert frames[-1]['fl_x']==10
    np.testing.assert_array_equal(frames[-1]['transform_matrix'], [[1,0,0,7],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    (output/'dataset/images').mkdir(parents=True)
    source_camera_model(inputs,output)
    publish_generated(inputs,output,manifest)
    assemble(inputs,output)
    assert len(list((output/'dataset/synthetic_6').glob('*.png')))==2
    assert len(json.loads((output/'dataset/reference_names.json').read_text()))==6
    chunks=plan_chunks(manifest)
    assert sum(len(c['target_images']) for c in chunks)==2
    assert all(len(c['reference_images'])==6 for c in chunks)
    assert Image.open(output/'combined/ref_000003.png').getpixel((0,0))==(3,0,0)
    # Snapshot survives later deletion of source inference images.
    (row['_manifest'].parent/'pred/00007.png').unlink()
    assert (inputs/'scene/images/target_000007.png').is_file()
    with pytest.raises(ValueError,match='all ArtiFixer'):
        source_assets(studio,row)


def test_missing_reference_and_invalid_initialization_rejected(tmp_path):
    studio,row=fixture(tmp_path)
    with pytest.raises(ValueError,match='Initialization'):
        export_inputs(studio,row,tmp_path/'bad','invalid')
    (row['_manifest'].parent/'ref2.png').unlink()
    with pytest.raises(ValueError,match='reference images are missing'):
        source_assets(studio,row)


def test_dashboard_queues_isolated_snapshot_and_rejects_missing_inputs(tmp_path):
    from test_splatfix_results import setup
    studio, _, _ = setup(tmp_path/'dashboard')
    _, row = fixture(tmp_path)
    row.update(config={'splatfix':{}}, scene='room')
    studio.results.detail = lambda key: row
    queued = studio.results.reconstruct_artifixer('result')
    options = queued['config']['splatfix']
    assert options['reconstruction_method']=='oraclegs'
    assert options['oracle_initialization']=='source_splat'
    snapshot = Path(options['oracle_artifixer_inputs'])
    assert snapshot != row['_manifest'].parent
    assert (snapshot/'source-initialization.ply').read_bytes() == row['_paths']['original'].read_bytes()
    before = len(studio.store.list_runs())
    (row['_manifest'].parent/'pred/00007.png').unlink()
    with pytest.raises(ValueError,match='all ArtiFixer'):
        studio.results.reconstruct_artifixer('result','sfm')
    assert len(studio.store.list_runs()) == before
    assert len(list(snapshot.parent.iterdir())) == 1


def test_button_jobs_wait_for_normal_manager_gpu_lease(tmp_path, monkeypatch):
    from test_splatfix_results import setup
    from splat_explorer.scene_runs.manager import SceneRunManager
    from splat_explorer.scene_runs.store import SceneRunStore
    from splat_explorer.splatfix.jobs import requires_gpu

    studio, active_root, _ = setup(tmp_path/'dashboard')
    _, row = fixture(tmp_path)
    row.update(config={'splatfix':{}}, scene='room')
    studio.results.detail = lambda key: row
    store = studio.store
    store.update_status(active_root.name, status='running')
    active_lease = store.acquire_gpu_lease(active_root.name)
    assert active_lease is not None
    first = studio.results.reconstruct_artifixer('result')
    second = studio.results.reconstruct_artifixer('result', 'sfm')
    for run in (first, second):
        assert run['state']['status'] == 'queued'
        assert requires_gpu(run['config'])
    executed = []
    contender = SceneRunStore(store.root)

    class Executor:
        def __init__(self, cfg, passed_store):
            assert passed_store is store

        def execute(self, run_id):
            assert store.gpu_lease_owner()['run_id'] == run_id
            other = second['run_id'] if run_id == first['run_id'] else first['run_id']
            assert contender.acquire_gpu_lease(other) is None
            executed.append(run_id)
            store.update_status(run_id, status='completed')

    manager = SceneRunManager(studio.cfg, store=store, executor_factory=Executor)
    monkeypatch.setattr(manager, '_gpu_ready', lambda force=False: {'ready':True, 'job_id':'test-allocation'})
    assert manager.run_once() is False
    assert executed == []
    assert store.get_run(first['run_id']).state.status.value == 'queued'
    store.update_status(active_root.name, status='completed')
    store.release_gpu_lease(active_lease)
    # GPU availability is checked even after the previous owner releases it.
    monkeypatch.setattr(manager, '_gpu_ready', lambda force=False: {'ready':False})
    assert manager.run_once() is False
    monkeypatch.setattr(manager, '_gpu_ready', lambda force=False: {'ready':True, 'job_id':'test-allocation'})
    assert manager.run_once() is True
    assert executed == [first['run_id']]
    assert store.get_run(second['run_id']).state.status.value == 'queued'
    assert manager.run_once() is True
    assert executed == [first['run_id'], second['run_id']]
    assert store.gpu_lease_owner() is None
