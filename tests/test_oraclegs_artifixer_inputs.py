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


@pytest.fixture(autouse=True)
def rendered_source_depth(monkeypatch):
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    monkeypatch.setattr(CpuSplatRenderer, 'render_depth',
                        lambda self, camera: np.full((camera.height, camera.width), 4, np.float32))


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
        'segments':[{'indices':[6,7], 'seed_index':5}],
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


@pytest.mark.parametrize('initialization',['sfm','source_splat','point_cloud'])
def test_reuse_preserves_camera_and_reference_roles(tmp_path,initialization):
    studio,row=fixture(tmp_path)
    inputs=tmp_path/'inputs'; output=tmp_path/'output'
    manifest=export_inputs(studio,row,inputs,initialization)
    validate_assets(inputs,manifest)
    assert bool(manifest.get('source_initialization')) == (initialization in ('source_splat','point_cloud'))
    frames=json.loads((inputs/'scene/transforms.json').read_text())['frames']
    assert len(frames)==8
    assert manifest['oracle_depth_source'] == ('source_splat' if initialization in ('source_splat','point_cloud') else 'predicted')
    if initialization in ('source_splat','point_cloud'):
        assert len(manifest['source_depth']['views']) == 6
        for record in manifest['source_depth']['views']:
            with np.load(inputs/record['path']) as data:
                assert data['invdepth'].shape == (8,16)
                assert np.all(data['invdepth'][data['valid']] == .25)
                assert not data['valid'][0].any()
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


@pytest.mark.parametrize('batching', ['within_trajectory', 'global'])
@pytest.mark.parametrize('initialization', ['source_splat', 'point_cloud'])
def test_dashboard_queues_isolated_snapshot_and_rejects_missing_inputs(tmp_path, batching, initialization):
    from test_splatfix_results import setup
    studio, _, _ = setup(tmp_path/'dashboard')
    _, row = fixture(tmp_path)
    row.update(config={'splatfix':{}}, scene='room')
    studio.results.detail = lambda key: row
    queued = studio.results.reconstruct_artifixer('result', initialization, oracle_attention_batching=batching)
    options = queued['config']['splatfix']
    assert options['oracle_artifixer_input'] == 'new_only'
    assert options['reconstruction_method']=='oraclegs'
    assert options['oracle_initialization']==initialization
    assert options['oracle_depth_source']=='source_splat'
    assert options['oracle_attention_batching'] == batching
    snapshot = Path(options['oracle_artifixer_inputs'])
    assert json.loads((snapshot/'inputs.json').read_text())['oracle_attention_batching'] == batching
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


def test_source_depth_uses_staged_reference_cameras_and_rejects_empty_depth(tmp_path, monkeypatch):
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    studio, row = fixture(tmp_path)
    seen = []
    def render(self, camera):
        seen.append(camera)
        return np.full((camera.height, camera.width), 4, np.float32)
    monkeypatch.setattr(CpuSplatRenderer, 'render_depth', render)
    export_inputs(studio, row, tmp_path/'inputs', 'source_splat')
    assert len(seen) == 6
    for index, camera in enumerate(seen):
        np.testing.assert_allclose(camera.position, [index, 0, 0])
        np.testing.assert_allclose(camera.rotation, np.eye(3))
        np.testing.assert_allclose(camera.intrinsics, [[10,0,8],[0,10,4],[0,0,1]])
    monkeypatch.setattr(CpuSplatRenderer, 'render_depth',
                        lambda self, camera: np.full((camera.height, camera.width), np.inf, np.float32))
    with pytest.raises(ValueError, match='No reliable source depth'):
        export_inputs(studio, row, tmp_path/'invalid', 'source_splat')


def test_publish_generated_stops_between_files(tmp_path, monkeypatch):
    from splat_explorer.splatfix.oraclegs import artifixer_inputs
    studio, row = fixture(tmp_path)
    inputs, output = tmp_path/'inputs', tmp_path/'output'
    manifest = export_inputs(studio, row, inputs, 'sfm')
    copied = []
    original = artifixer_inputs.shutil.copy2
    def copy(source, target):
        result = original(source, target)
        copied.append(target)
        return result
    monkeypatch.setattr(artifixer_inputs.shutil, 'copy2', copy)
    with pytest.raises(InterruptedError, match='staging ArtiFixer'):
        publish_generated(inputs, output, manifest, should_stop=lambda:bool(copied))
    assert len(copied) == 1


def test_assembly_stops_before_next_synthetic_image(tmp_path):
    studio, row = fixture(tmp_path)
    inputs, output = tmp_path/'inputs', tmp_path/'output'
    manifest = export_inputs(studio, row, inputs, 'sfm')
    (output/'dataset/images').mkdir(parents=True)
    source_camera_model(inputs, output)
    publish_generated(inputs, output, manifest)
    synthetic = output/'dataset/synthetic_6'
    with pytest.raises(InterruptedError, match='assembling OracleGS'):
        assemble(inputs, output, should_stop=lambda:(synthetic/'target_000006.png').exists())
    assert (synthetic/'target_000006.png').is_file()
    assert not (synthetic/'target_000007.png').exists()
    assert not (output/'dataset/reference_names.json').exists()


@pytest.mark.parametrize('batching', ['within_trajectory', 'global'])
def test_artifixer_attention_groups_follow_selection(tmp_path, batching):
    studio, row = fixture(tmp_path)
    path = row['_manifest'].parent/'preview-trajectory.json'
    trajectory = json.loads(path.read_text())
    trajectory['transforms']['frames'][7]['transform_matrix'][0][3] = 0
    trajectory['segments'] = [{'indices':[6], 'seed_index':5}, {'indices':[7], 'seed_index':0}]
    atomic_json(path, trajectory)
    manifest = export_inputs(studio, row, tmp_path/'inputs', 'sfm', oracle_attention_batching=batching)
    chunks = plan_chunks(manifest, mode=manifest['oracle_attention_batching'])
    assert all(len(c['reference_images']) == 6 for c in chunks)
    assert sum(len(c['target_images']) for c in chunks) == 2
    assert {len(c['segment_indices']) for c in chunks} == ({1} if batching == 'within_trajectory' else {2})


def test_invalid_attention_mode_rejected_before_staging(tmp_path):
    studio, row = fixture(tmp_path)
    with pytest.raises(ValueError, match='oracle_attention_batching'):
        export_inputs(studio, row, tmp_path/'inputs', 'sfm', oracle_attention_batching='invalid')
    assert not (tmp_path/'inputs').exists()


@pytest.mark.parametrize('backup', [False, True])
def test_reduce_keeps_reconstruction_inputs_exportable(tmp_path, backup):
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    from test_splatfix_results import setup
    studio, run, _ = setup(tmp_path)
    _, row = fixture(run / 'gpu/results')
    row['run_id'] = run.name
    studio.results.entries = lambda **kwargs: [row]
    studio.store.set_status(run.name, 'completed')
    result = row['_manifest'].parent
    (result / 'checkpoint.pt').write_bytes(b'weights')
    (result / 'input-images/original_rgb_images').mkdir(parents=True)
    (result / 'input-images/original_rgb_images/0.png').write_bytes(b'input')
    poses = (result / 'preview-trajectory.json').read_bytes()
    (result / 'output-poses.json').write_bytes(poses)
    images = {p: p.read_bytes() for p in result.glob('ref*.png')}
    images.update({p: p.read_bytes() for p in (result / 'pred').glob('*.png')})
    report = studio.results.reduce_run(run.name, backup=backup, preserve_reconstruction=True)
    assert report['preserve_reconstruction'] is True
    assert report['inference_videos'][0]['frame_count'] == 2
    assert all(p.read_bytes() == content for p, content in images.items())
    assert (result / 'output-poses.json').read_bytes() == poses
    assert not (result / 'checkpoint.pt').exists()
    assert not (result / 'input-images/original_rgb_images').exists()
    assert bool(report['backup_path']) == backup
    # Exercise the actual post-inference reconstruction export after pruning.
    destination = tmp_path / 'reconstruct'
    manifest = export_inputs(studio, row, destination, 'sfm')
    validate_assets(destination, manifest)
    assert manifest['artifixer_source']['generated_count'] == 2
    assert len(json.loads((destination / 'scene/transforms.json').read_text())['frames']) == 8


def test_reduce_reconstruction_rejects_missing_references_without_pruning(tmp_path):
    from test_splatfix_results import setup
    studio, run, _ = setup(tmp_path)
    _, row = fixture(run / 'gpu/results')
    row['run_id'] = run.name
    studio.results.entries = lambda **kwargs: [row]
    studio.store.set_status(run.name, 'completed')
    (row['_manifest'].parent / 'ref2.png').unlink()
    before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file() and p.name != 'output-size.json'}
    with pytest.raises(ValueError, match='reference images are missing'):
        studio.results.reduce_run(run.name, preserve_reconstruction=True)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (run / 'reduction.json').exists()


@pytest.mark.parametrize('backup', [False, True])
@pytest.mark.parametrize('preserve_reconstruction', [False, True])
def test_endpoint_reduce_preserves_both_inference_folders(tmp_path, backup, preserve_reconstruction):
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required')
    from test_splatfix_results import setup
    studio, run, _ = setup(tmp_path)
    _, row = fixture(run / 'gpu/results')
    row['run_id'] = run.name
    studio.results.entries = lambda **kwargs: [row]
    studio.store.set_status(run.name, 'completed')
    result = row['_manifest'].parent
    trajectory = json.loads((result / 'preview-trajectory.json').read_text())
    trajectory['anchors'] = [{'view_id': 'start', 'frame_index': 0},
                             {'view_id': 'end', 'frame_index': 7}]
    trajectory['recipe'] = {'repair_policy': 'endpoints'}
    atomic_json(result / 'preview-trajectory.json', trajectory)
    refs = json.loads((result / 'input-preview.json').read_text())['reference_images'][:2]
    atomic_json(result / 'input-preview.json', {'reference_images': refs})
    for index in range(2, 6):
        (result / f'ref{index}.png').unlink()
    shutil.rmtree(result / 'pred')
    for directory, color in [('inference', 30), ('inference-back-to-front', 150)]:
        folder = result / directory / 'splatfix/frames/batch_0000/pred'
        folder.mkdir(parents=True)
        for index in range(1, 7):
            Image.new('RGB', (16, 8), (color, index, 0)).save(folder / f'{index:05d}.png')
    atomic_json(result / 'inference-preview.json', {
        'prediction_frames': 'inference/splatfix/frames/batch_0000/pred',
        'reconstruction_inputs': list(range(1, 7)), 'reference_count': 2})
    directions = [
        {'direction': 'front-to-back', 'frame_indices': list(range(8)),
         'reference_frame_indices': [0, 7], 'output': 'inference', 'used_for_reconstruction': True},
        {'direction': 'back-to-front', 'frame_indices': list(reversed(range(8))),
         'reference_frame_indices': [7, 0], 'output': 'inference-back-to-front', 'used_for_reconstruction': False}]
    atomic_json(result / 'inference-directions.json', directions)
    atomic_json(result / 'inference-insertion-audit.json', {'series': [
        {'output': item['output'], 'export_global_indices': item['frame_indices'][1:-1]}
        for item in directions]})
    (result / 'checkpoint.pt').write_bytes(b'weights')
    (result / 'raw-capture.png').write_bytes(b'input')
    plus = result / 'plus-images'
    (plus / 'additional').mkdir(parents=True)
    Image.new('RGB', (16, 8), (20, 40, 60)).save(plus / '00000.png')
    Image.new('RGB', (16, 8), (60, 40, 20)).save(plus / 'additional/view.jpg')
    plus_images = {p: p.read_bytes() for p in plus.rglob('*') if p.is_file()}
    manifest = json.loads(row['_manifest'].read_text())
    manifest['plus_frames'] = str(plus)
    atomic_json(row['_manifest'], manifest)
    predictions = {p: p.read_bytes() for p in result.glob('**/pred/*.png')}
    report = studio.results.reduce_run(run.name, backup=backup,
                                      preserve_reconstruction=preserve_reconstruction)
    assert len(report['inference_videos']) == 3
    assert sorted(video['frame_count'] for video in report['inference_videos']) == [1, 6, 6]
    assert all(path.read_bytes() == content for path, content in predictions.items())
    assert all(path.read_bytes() == content for path, content in plus_images.items())
    assert studio.plus_previews(row['_manifest'], manifest) == [studio.file_url(plus / '00000.png')]
    assert not (result / 'checkpoint.pt').exists()
    assert not (result / 'raw-capture.png').exists()
    if backup:
        saved = Path(report['backup_path']) / result.relative_to(run)
        assert (saved / 'checkpoint.pt').exists()
        assert len(list(saved.glob('**/pred/*.png'))) == 12
    if preserve_reconstruction:
        assert json.loads((result / 'preview-trajectory.json').read_text()) == trajectory
        assert json.loads((result / 'inference-directions.json').read_text()) == directions
        assert (result / 'inference-insertion-audit.json').is_file()
        for initialization in ('sfm', 'source_splat', 'point_cloud'):
            destination = tmp_path / initialization
            manifest = export_inputs(studio, row, destination, initialization)
            validate_assets(destination, manifest)
            assert len(manifest['views']) == 2
            assert manifest['artifixer_source']['generated_count'] == 6
            frames = json.loads((destination / 'scene/transforms.json').read_text())['frames']
            assert [frame['transform_matrix'][0][3] for frame in frames] == [0, 7, 1, 2, 3, 4, 5, 6]
            assert [target['reference_index'] for target in manifest['trajectory']['frames']] == [0, 0, 0, 0, 0, 0]
            # Keep the existing forward-only reconstruction contract; never merge reverse RGBs.
            assert Image.open(destination / 'scene/images/target_000003.png').getpixel((0, 0)) == (30, 3, 0)
    else:
        assert not (result / 'preview-trajectory.json').exists()
        assert not (result / 'inference-directions.json').exists()


@pytest.mark.parametrize('selection', ['new_only', 'original_and_new'])
@pytest.mark.parametrize('batching', ['within_trajectory', 'global'])
def test_input_selection_controls_training_and_attention(tmp_path, selection, batching):
    from test_worldexplorer_source import fixture as world_fixture
    from splat_explorer.splatfix.oraclegs.adapter import append_worldexplorer
    from splat_explorer.splatfix.oraclegs.attention import worldexplorer_targets
    studio, row = fixture(tmp_path)
    source = world_fixture(tmp_path)
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    atomic_json(checkpoint/'checkpoint.json', {'scene_path': str(source)})
    row['config'] = {'splatfix': {'checkpoint': str(checkpoint)}}
    inputs, output = tmp_path/'inputs', tmp_path/'output'
    manifest = export_inputs(studio, row, inputs, 'sfm',
                             oracle_artifixer_input=selection,
                             oracle_attention_batching=batching)
    assert manifest['oracle_artifixer_input'] == selection
    assert bool(manifest.get('worldexplorer_source')) == (selection == 'original_and_new')
    (output/'dataset/images').mkdir(parents=True)
    source_camera_model(inputs, output)
    publish_generated(inputs, output, manifest)
    assemble(inputs, output)
    n, names = append_worldexplorer(inputs, output, len(manifest['views']))
    assert len(names) == (2 if selection == 'original_and_new' else 0)
    extra = worldexplorer_targets(inputs, output, manifest) if names else []
    chunks = plan_chunks(manifest, extra_targets=extra, mode=batching)
    if extra:
        assert len({f['segment_index'] for f in extra}) == 2
        assert {f['source_trajectory'] for f in extra} == {'000', '001'}
        if batching == 'within_trajectory':
            assert all(len(chunk['segment_indices']) == 1 for chunk in chunks)
    targets = [name for chunk in chunks for name in chunk['target_images']]
    synthetic = {p.name for p in (output/f'dataset/synthetic_{n}').glob('*.png')}
    assert set(targets) == {'synth_'+name for name in synthetic}
    assert len(targets) == len(synthetic) == 2 + len(names)
    assert all(len(chunk['reference_images']) == 6 for chunk in chunks)
    if names:
        assert min(f['segment_index'] for f in extra) > max(
            f['segment_index'] for f in manifest['trajectory']['frames'])


def test_new_only_default_does_not_require_worldexplorer_archive(tmp_path, monkeypatch):
    studio, row = fixture(tmp_path)
    def unexpected(*args):
        pytest.fail('new images only must not read the WorldExplorer archive')
    monkeypatch.setattr('splat_explorer.splatfix.worldexplorer.source.stage_source', unexpected)
    manifest = export_inputs(studio, row, tmp_path/'inputs', 'sfm')
    assert manifest['oracle_artifixer_input'] == 'new_only'
    assert 'worldexplorer_source' not in manifest


def test_invalid_input_selection_and_missing_originals_rejected(tmp_path):
    studio, row = fixture(tmp_path)
    with pytest.raises(ValueError, match='oracle_artifixer_input'):
        export_inputs(studio, row, tmp_path/'invalid', 'sfm', oracle_artifixer_input='invalid')
    assert not (tmp_path/'invalid').exists()
    with pytest.raises(ValueError, match='Original WorldExplorer images are unavailable'):
        export_inputs(studio, row, tmp_path/'missing', 'sfm', oracle_artifixer_input='original_and_new')


@pytest.mark.parametrize('selection', [None, 'new_only', 'original_and_new'])
@pytest.mark.parametrize('assignment', [None, 'trajectory', 'camera_proximity'])
def test_reconstruction_endpoint_forwards_input_selection(selection, assignment):
    from splat_explorer.web.server import DashboardHandler
    calls, replies = [], []
    results = SimpleNamespace(reconstruct_artifixer=lambda *args, **kwargs: calls.append((args, kwargs)) or {'run_id': 'queued'})
    handler = SimpleNamespace(app=SimpleNamespace(splatfix=SimpleNamespace(results=results)),
                              command='POST', _send_json=lambda body, *args: replies.append(body))
    body = {'id': 'run'}
    if selection is not None:
        body['oracle_artifixer_input'] = selection
    if assignment is not None:
        body['oracle_artifixer_reference_assignment'] = assignment
    DashboardHandler._serve_splatfix(handler, '/api/splatfix/results/reconstruct-artifixer', body)
    assert calls[0][1]['oracle_artifixer_reference_assignment'] == (assignment or 'trajectory')
    assert calls[0][1]['oracle_artifixer_input'] == (selection or 'new_only')
    assert replies == [{'ok': True, 'run': {'run_id': 'queued'}}]


def test_combined_selection_survives_queue_normalization(tmp_path):
    from test_splatfix_results import setup
    from test_worldexplorer_source import fixture as world_fixture
    studio, _, _ = setup(tmp_path/'dashboard')
    _, row = fixture(tmp_path)
    row.update(config={'splatfix': {}}, scene='room')
    row['_paths']['original'] = world_fixture(tmp_path)
    studio.results.detail = lambda key: row
    run = studio.results.reconstruct_artifixer('result', 'sfm', oracle_artifixer_input='original_and_new')
    options = run['config']['splatfix']
    assert options['oracle_artifixer_input'] == 'original_and_new'
    manifest = json.loads((Path(options['oracle_artifixer_inputs'])/'inputs.json').read_text())
    assert manifest['oracle_artifixer_input'] == options['oracle_artifixer_input']
    assert manifest['worldexplorer_source'] == 'worldexplorer-source'


@pytest.mark.parametrize('selection,expected', [('trajectory', [1, 1]), ('camera_proximity', [5, 5])])
@pytest.mark.parametrize('batching', ['within_trajectory', 'global'])
def test_attention_assignment_uses_generation_seed_or_explicit_proximity(tmp_path, selection, expected, batching):
    studio, row = fixture(tmp_path)
    path = row['_manifest'].parent/'preview-trajectory.json'
    trajectory = json.loads(path.read_text())
    # Both predictions are spatially nearest reference 5, but generated from 1.
    trajectory['segments'] = [{'indices':[6], 'seed_index':1}, {'indices':[7], 'cache_seed_index':1}]
    atomic_json(path, trajectory)
    manifest = export_inputs(studio, row, tmp_path/'inputs', 'sfm',
                             oracle_artifixer_reference_assignment=selection)
    assert [f['reference_index'] for f in manifest['trajectory']['frames']] == expected
    chunks = plan_chunks(manifest, mode=batching)
    assert {c['source_image'] for c in chunks} == {f'ref_{expected[0]:06d}.png'}
    if selection == 'trajectory':
        assert len(manifest['trajectory']['segments']) == 2
        assert len(chunks) == (2 if batching == 'within_trajectory' else 1)
    else:
        assert len(manifest['trajectory']['segments']) == 1


def test_missing_trajectory_mapping_requires_explicit_proximity(tmp_path):
    studio, row = fixture(tmp_path)
    path = row['_manifest'].parent/'preview-trajectory.json'
    trajectory = json.loads(path.read_text())
    trajectory.pop('segments')
    atomic_json(path, trajectory)
    with pytest.raises(ValueError, match='mapping is incomplete'):
        export_inputs(studio, row, tmp_path/'invalid', 'sfm')
    assert not (tmp_path/'invalid').exists()
    manifest = export_inputs(studio, row, tmp_path/'legacy', 'sfm', oracle_artifixer_reference_assignment='camera_proximity')
    assert manifest['oracle_artifixer_reference_assignment'] == 'camera_proximity'


@pytest.mark.parametrize('segment', [{'indices':[6,7], 'seed_index':99},
                                    {'indices':[6], 'seed_index':0}])
def test_invalid_trajectory_seed_or_coverage_rejected(tmp_path, segment):
    studio, row = fixture(tmp_path)
    path = row['_manifest'].parent/'preview-trajectory.json'
    trajectory = json.loads(path.read_text())
    trajectory['segments'] = [segment]
    atomic_json(path, trajectory)
    with pytest.raises(ValueError, match='trajectory'):
        export_inputs(studio, row, tmp_path/'inputs', 'sfm')


def test_no_split_assignment_uses_first_repaired_reference(tmp_path):
    studio, row = fixture(tmp_path)
    atomic_json(row['_manifest'].parent/'request.json', {
        'trajectory_mode':'authors_orbit', 'runtime':{'split_mode':'no-split'}})
    manifest = export_inputs(studio, row, tmp_path/'inputs', 'sfm')
    assert [f['reference_index'] for f in manifest['trajectory']['frames']] == [0,0]


@pytest.mark.parametrize('assignment', ['trajectory', 'camera_proximity'])
def test_assignment_survives_queue_normalization(tmp_path, assignment):
    from test_splatfix_results import setup
    studio, _, _ = setup(tmp_path/'dashboard')
    _, row = fixture(tmp_path)
    row.update(config={'splatfix': {}}, scene='room')
    studio.results.detail = lambda key: row
    run = studio.results.reconstruct_artifixer('result', 'sfm', oracle_artifixer_reference_assignment=assignment)
    options = run['config']['splatfix']
    assert options['oracle_artifixer_reference_assignment'] == assignment
    manifest = json.loads((Path(options['oracle_artifixer_inputs'])/'inputs.json').read_text())
    assert manifest['oracle_artifixer_reference_assignment'] == assignment



def test_invalid_reference_assignment_rejected_before_staging(tmp_path):
    studio, row = fixture(tmp_path)
    with pytest.raises(ValueError, match='oracle_artifixer_reference_assignment'):
        export_inputs(studio, row, tmp_path/'inputs', 'sfm', oracle_artifixer_reference_assignment='invalid')
    assert not (tmp_path/'inputs').exists()


def handoff_fixture(tmp_path):
    from splat_explorer.splatfix.oraclegs.source_geometry import digest
    studio, row = fixture(tmp_path)
    root = row['_manifest'].parent
    trajectory = json.loads((root/'preview-trajectory.json').read_text())
    original = [3, 0, 6, 9, 12, 15]  # Storage order need not be temporal order.
    final = [2, 0, 5, 8, 11, 15]
    trajectory['anchors'] = [{'view_id':str(i), 'frame_index':frame} for i,frame in enumerate(original)]
    cameras = []
    for i in range(16):
        pose = np.eye(4); pose[0,3] = i
        cameras.append({'transform_matrix':pose.tolist()})
    trajectory['transforms']['frames'] = cameras
    legs = [[1,2], [4,5], [7,8], [10,11], [13,14]]
    trajectory['segments'] = [{'indices':leg, 'cache_seed_index':seed}
                              for leg,seed in zip(legs, [0,3,6,9,12])]
    atomic_json(root/'preview-trajectory.json', trajectory)
    for p in (root/'pred').glob('*.png'):
        p.unlink()
    for leg in legs:
        for i in leg:
            Image.new('RGB', (16,8), (0,i,0)).save(root/'pred'/f'{i:05d}.png')
    paths, hashes = [], []
    for slot, (old,new) in enumerate(zip(original,final)):
        source = root/f'ref{slot}.png'
        if new != old:
            source = root/'handoffs/inference'/f'{new:05d}'/'repaired.png'
            source.parent.mkdir(parents=True)
            Image.new('RGB', (16,8), (100+new,0,0)).save(source)
        # Exercise rebasing LRZ paths and copied original reference fallback.
        paths.append('/workspace/results/source/'+str(source.relative_to(root)) if new != old
                     else '/workspace/checkpoint/views/'+str(slot)+'/repaired.png')
        hashes.append(digest(source))
    atomic_json(root/'request.json', {'trajectory_mode':'authors_orbit',
        'runtime':{'handoff_generation':True, 'split_mode':'single-split'}})
    atomic_json(root/'inference-references.json', {'original_frame_indices':original,
        'frame_indices':final, 'paths':paths, 'sha256':hashes})
    atomic_json(root/'inference-seeding.json', {'series':[
        {'output':'inference', 'enabled':True, 'seed_global_index':i} for i in [0,2,5,8,11]]})
    atomic_json(root/'inference-preview.json', {'prediction_frames':'pred',
        'reconstruction_inputs':[i for leg in legs for i in leg]})
    return studio, row


@pytest.mark.parametrize('batching', ['within_trajectory', 'global'])
@pytest.mark.parametrize('initialization', ['sfm', 'point_cloud'])
def test_handoff_export_uses_actual_repaired_seeds_and_single_split_legs(tmp_path, batching, initialization):
    studio,row = handoff_fixture(tmp_path)
    destination = tmp_path/'export'
    manifest = export_inputs(studio,row,destination,initialization,oracle_attention_batching=batching)
    refs = manifest['views']
    assert len(refs) == 5
    assert [Image.open(destination/v['image']).getpixel((0,0))[0] for v in refs] == [102,1,105,108,111]
    frames = json.loads((destination/'scene/transforms.json').read_text())['frames']
    assert [f['transform_matrix'][0][3] for f in frames[:5]] == [2,0,5,8,11]
    targets = manifest['trajectory']['frames']
    assert [t['target_index'] for t in targets] == [1,4,7,10,13,14]
    assert [t['reference_index'] for t in targets] == [1,0,2,3,4,4]
    assert [t['segment_index'] for t in targets] == [0,1,2,3,4,4]
    assert manifest['artifixer_source']['handoff']['reference_frame_indices'] == [2,0,5,8,11]
    chunks = plan_chunks(manifest,mode=batching)
    for chunk in chunks:
        assert chunk['source_frame'] == 0
        assert chunk['image_names'][0] == chunk['source_image']
        for index in chunk['target_indices']:
            target = next(t for t in targets if t['target_index'] == index)
            assert chunk['source_image'] == Path(refs[target['reference_index']]['image']).name
        if batching == 'within_trajectory':
            assert len(chunk['segment_indices']) == 1
    if initialization == 'point_cloud':
        assert len(manifest['source_depth']['views']) == 5
    validate_assets(destination,manifest)


@pytest.mark.parametrize('corruption', ['missing_state','missing_series','wrong_seed','wrong_hash','missing_image','unchanged_anchor','wrong_camera'])
def test_handoff_export_rejects_incomplete_or_inconsistent_artifacts(tmp_path,corruption):
    studio,row = handoff_fixture(tmp_path)
    root = row['_manifest'].parent
    path = root/'inference-references.json'
    state = json.loads(path.read_text())
    if corruption == 'missing_state':
        path.unlink()
    elif corruption == 'missing_series':
        (root/'inference-seeding.json').unlink()
    elif corruption == 'wrong_seed':
        series = json.loads((root/'inference-seeding.json').read_text())
        series['series'][1]['seed_global_index'] = 3
        atomic_json(root/'inference-seeding.json',series)
    elif corruption == 'missing_image':
        (root/'handoffs/inference/00002/repaired.png').unlink()
    else:
        if corruption == 'wrong_hash': state['sha256'][0] = 'bad'
        if corruption == 'unchanged_anchor': state['frame_indices'][0] = 3
        if corruption == 'wrong_camera': state['frame_indices'][0] = 99
        atomic_json(path,state)
    with pytest.raises(ValueError,match='[Hh]andoff|Interior'):
        export_inputs(studio,row,tmp_path/'export','sfm')
    assert not (tmp_path/'export').exists()


def test_handoff_reduction_plan_retains_metadata_and_original_trajectory(tmp_path):
    from splat_explorer.web.splatfix_results import ResultCatalog
    studio,row = handoff_fixture(tmp_path)
    root = row['_manifest'].parent
    original = json.loads((root/'preview-trajectory.json').read_text())
    plans = ResultCatalog._artifixer_plans(SimpleNamespace(studio=studio), [row], preserve_reconstruction=True)
    plan = plans[0]
    assert plan['trajectory'] == original
    kept = set(plan['reconstruction_files'])
    assert {root/'inference-references.json', root/'inference-seeding.json'} <= kept
    assert {root/f'ref{i}.png' for i in range(6)} <= kept
    assert root/'handoffs/inference/00002/repaired.png' in kept
