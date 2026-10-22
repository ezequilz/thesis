"""OracleGS calibrated data boundaries and stage contracts, without GPU weights."""
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation
from splat_explorer.rendering.base import Camera
from splat_explorer.splatfix.checkpoint import Checkpoint
from splat_explorer.splatfix.jobs import validate_job
from splat_explorer.splatfix.oraclegs import backend, repair, worker
from splat_explorer.splatfix.oraclegs.inputs import export_inputs, AXES
from splat_explorer.splatfix.oraclegs.adapter import assemble, colmap_lines, known_model, pad_seva_inputs


def checkpoint(tmp_path, count=3):
    cp = Checkpoint.create(tmp_path,tmp_path/'missing.ply',target_views=count,
                           metadata={'renderer':{'backend':'viser'}})
    for i in range(count):
        camera = Camera(np.array([i,1.,-3.]),Rotation.from_euler('xyz',[10,20+i*5,30],degrees=True).as_matrix(),
                        width=1920,height=1080,fov_deg=67.)
        view = cp.add_view(Image.new('RGB',(1920,1080),'red'),camera)
        path = cp.root/'views'/view['id']/'repaired.png'
        Image.new('RGB',(1920,1080),'blue').save(path)
        view['repaired_rgb'] = str(path.relative_to(cp.root))
    cp.save()
    return cp


@pytest.mark.parametrize('resolution,size',[('576p',(1024,576)),('720p',(1280,720))])
def test_repaired_rgb_projection_and_exact_saved_path_poses(tmp_path,resolution,size):
    cp = checkpoint(tmp_path)
    root = tmp_path/'staged'
    before = [hashlib.sha256(cp.image_path(v,repaired=True).read_bytes()).hexdigest() for v in cp.views]
    manifest = export_inputs(cp,root,runtime={'seva_resolution':resolution,'oracle_target_frames':11,'oracle_path_order':[2,0,1]})
    frames = json.loads((root/'scene/transforms.json').read_text())['frames']
    assert len(frames) == 25
    split = json.loads((root/'scene/train_test_split_3.json').read_text())
    assert split == {'train_ids':[0,1,2],'test_ids':list(range(3,25))}
    for i, v in enumerate(cp.views):
        f = frames[i]
        rgb = Image.open(root/'scene'/f['file_path'])
        assert rgb.size == size and rgb.getpixel((0,0)) == (0,0,255)
        original = np.asarray(v['camera']['c2w'])
        np.testing.assert_allclose(np.asarray(f['transform_matrix'])@AXES,original,atol=1e-7)
        camline,imline = colmap_lines(f,i+1,i+1,'x.png')
        nums = imline.split()
        q = np.asarray(nums[1:5],float)
        world = original @ np.array([.3,-.2,2,1])
        xyz = Rotation.from_quat(q[[1,2,3,0]]).as_matrix() @ world[:3] + np.array(nums[5:8],float)
        fx,fy,cx,cy = map(float,camline.split()[4:])
        projected = np.array([fx*xyz[0]/xyz[2]+cx,fy*xyz[1]/xyz[2]+cy])
        k = np.asarray(v['camera']['intrinsics'])
        old_pixel = k @ np.array([.3,-.2,2])
        expected = old_pixel[:2]/old_pixel[2]*(size[0]/1920)
        expected[1] -= (1080*size[0]/1920-size[1])/2
        np.testing.assert_allclose(projected,expected,atol=1e-5)
    assert all(0 < t['alpha'] < 1 for t in manifest['trajectory']['frames'])
    native = json.loads((root/'segments/000/scene/transforms.json').read_text())['frames']
    assert native == [frames[i] for i in manifest['seva_segments'][0]['frame_order']]
    assert before == [hashlib.sha256(cp.image_path(v,repaired=True).read_bytes()).hexdigest() for v in cp.views]
    request = repair.seva_request(root,tmp_path/'out',manifest,repair.DEFAULT_RUNTIME)
    assert request['cfg'] == [3.,2.]
    assert request['task'] == 'img2vid' and request['use_traj_prior'] is True
    assert request['chunk_strategy'] == 'interp-gt'
    assert request['replace_or_include_input'] is True
    assert request['chunk_strategy_first_pass'] == 'gt-nearest'
    assert ('L_short' in request) == (resolution=='576p')
    assert 'T' not in request  # SEVA infers the procedural anchor window.


def test_720_padding_preserves_calibration_and_pixels(tmp_path):
    cp = checkpoint(tmp_path,2)
    root = tmp_path/'inputs'
    export_inputs(cp,root,runtime={'seva_resolution':'720p','seva_fit':'pad','oracle_target_frames':3})
    _,w,h = pad_seva_inputs(root,tmp_path/'padded')
    assert (w,h) == (1280,768)
    old = json.loads((root/'scene/transforms.json').read_text())['frames'][0]
    new = json.loads((tmp_path/'padded/scene/transforms.json').read_text())['frames'][0]
    assert new['cy'] == old['cy']+24 and new['fl_y'] == old['fl_y']
    image = Image.open(tmp_path/'padded/scene'/new['file_path'])
    assert image.getpixel((0,0)) == (0,0,0)
    assert image.getpixel((0,24)) == (0,0,255)
    assert image.crop((0,24,1280,744)).tobytes() == Image.open(root/'scene'/old['file_path']).tobytes()


def staged_generation(tmp_path, adapter='trajectory', resolution='576p'):
    cp = checkpoint(tmp_path,2)
    source,output = tmp_path/'inputs',tmp_path/'out'
    manifest = export_inputs(cp,source,runtime={'oracle_target_frames':5,'oracle_adapter':adapter,'seva_resolution':resolution})
    dataset = output/'dataset'
    (dataset/'images').mkdir(parents=True)
    frames = json.loads((source/'scene/transforms.json').read_text())['frames']
    sparse = dataset/'sparse/0'
    sparse.mkdir(parents=True)
    cameras,images = [],[]
    for i,f in enumerate(frames[:2]):
        name = Path(f['file_path']).name
        Image.new('RGB',tuple(manifest['resolution']),'blue').save(dataset/'images'/name)
        c,img = colmap_lines(f,i+1,i+1,name)
        cameras.append(c); images.append(img)
    (sparse/'cameras.txt').write_text('\n'.join(cameras)+'\n')
    (sparse/'images.txt').write_text('\n'.join(images)+'\n')
    (sparse/'points3D.txt').write_text('1 0 0 1 0 0 255 0 1 0 2 0\n')
    (sparse/'depth_params.json').write_text('{"ref_000000":{"scale":1,"offset":0}}')
    generation = output/'seva'/manifest['seva_task']/'scene'
    (generation/'samples-rgb').mkdir(parents=True)
    exported = []
    for i,index in enumerate(manifest['seva_frame_order']):
        f = frames[index]
        relative = f'samples-rgb/{i:03d}.png'
        Image.new('RGB',tuple(manifest['seva_internal_resolution']),'green').save(generation/relative)
        exported.append({**f,'file_path':relative,'w':manifest['seva_internal_resolution'][0],
                         'h':manifest['seva_internal_resolution'][1],
                         'cy':f['cy']+(manifest['seva_internal_resolution'][1]-manifest['resolution'][1])//2})
    (generation/'transforms.json').write_text(json.dumps({'frames':exported}))
    return source,output,manifest


@pytest.mark.parametrize('adapter',['trajectory','saved_poses'])
@pytest.mark.parametrize('resolution',['576p','720p'])
def test_generated_reference_poses_never_replace_ground_truth(tmp_path,adapter,resolution):
    source,output,manifest = staged_generation(tmp_path,adapter,resolution)
    assemble(source,output)
    dataset = output/'dataset'
    for name in ('ref_000000.png','ref_000001.png'):
        assert Image.open(dataset/'images'/name).getpixel((0,0)) == (0,0,255)
        if adapter == 'saved_poses':
            assert Image.open(dataset/'synthetic_2'/name).getpixel((0,0)) == (0,128,0)
        else:
            assert not (dataset/'synthetic_2'/name).exists()
        assert Image.open(output/'combined'/name).getpixel((0,0)) == (0,0,255)
        if adapter == 'saved_poses':
            assert Image.open(output/'combined'/('synth_'+name)).getpixel((0,0)) == (0,128,0)
    assert json.loads((dataset/'reference_names.json').read_text()) == ['ref_000000.png','ref_000001.png']
    names = (dataset/'sparse/2/images.txt').read_text().splitlines()[::2]
    assert len(names) == (7 if adapter=='trajectory' else 2)
    for image in (dataset/'synthetic_2').glob('*.png'):
        assert Image.open(image).size == tuple(manifest['resolution'])


def test_generation_pose_mismatch_fails_closed(tmp_path):
    source,output,manifest = staged_generation(tmp_path)
    path = output/'seva'/manifest['seva_task']/'scene/transforms.json'
    body = json.loads(path.read_text())
    body['frames'][-1]['transform_matrix'][0][3] += 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='identity or order'):
        assemble(source,output)


def test_colmap_database_ids_are_not_assumed(tmp_path):
    cp = checkpoint(tmp_path,2)
    source,out = tmp_path/'inputs',tmp_path/'out'
    export_inputs(cp,source,runtime={'oracle_target_frames':4})
    out.mkdir()
    with sqlite3.connect(out/'database.db') as db:
        db.execute('CREATE TABLE images (image_id INTEGER, camera_id INTEGER, name TEXT)')
        db.execute('CREATE TABLE cameras (camera_id INTEGER, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)')
        for index in range(2):
            db.execute('INSERT INTO images VALUES (?,?,?)',(20+index,50+index,f'ref_{index:06d}.png'))
            db.execute('INSERT INTO cameras VALUES (?,0,0,0,NULL,0)',(50+index,))
    known_model(source,out)
    text = (out/'known/images.txt').read_text().splitlines()
    assert text[0].startswith('20 ') and ' 50 ref_' in text[0]
    assert text[2].startswith('21 ') and ' 51 ref_' in text[2]
    with sqlite3.connect(out/'database.db') as db:
        row = db.execute('SELECT model,width,height,params FROM cameras WHERE camera_id=50').fetchone()
        assert row[:3] == (1,1024,576)
        assert np.frombuffer(row[3],np.float64)[2:].tolist() == [512,288]


@pytest.mark.parametrize('bad',[{'seva_resolution':'1080p'},{'oracle_adapter':'unknown'},{'oracle_target_frames':True},{'oracle_path_order':[0,0]}])
def test_invalid_options(bad):
    with pytest.raises(ValueError):
        validate_job({'stage':'repair','checkpoint':'saved','reconstruction_method':'oraclegs',**bad})


def test_reference_validation_and_stop(tmp_path):
    cp = checkpoint(tmp_path,2)
    assert backend.readiness(cp)['ready']
    with pytest.raises(InterruptedError):
        export_inputs(cp,tmp_path/'cancel',should_stop=lambda:True)
    Image.new('RGB',(1920,1000)).save(cp.image_path(cp.views[0],repaired=True))
    with pytest.raises(ValueError,match='aspect ratio'):
        export_inputs(cp,tmp_path/'bad')


def test_official_vendored_sources_unchanged():
    lock = json.loads((repair.HERE/'upstream-lock.json').read_text())
    for repo,entry in lock.items():
        for name,digest in entry['files'].items():
            assert hashlib.sha256((repair.HERE/'upstream'/repo/name).read_bytes()).hexdigest() == digest


def test_worker_cancellation_and_failure(tmp_path,monkeypatch):
    request = {'inputs':'unused','output':'unused','runtime':{},'stop_at_epoch':1}
    (tmp_path/'worker-request.json').write_text(json.dumps(request))
    monkeypatch.setattr(worker,'run_prepared',lambda *a,**k:pytest.fail('expired job launched'))
    worker.execute(tmp_path)
    assert json.loads((tmp_path/'worker-status.json').read_text())['status'] == 'stopped'


@pytest.mark.parametrize('worldexplorer', [False, True])
def test_real_upstream_colmap_loader_split_and_attention(tmp_path, worldexplorer):
    # Execute the actual upstream CPU reader in isolation from its CUDA-only
    # module imports. This tests its filename/camera/attention contracts.
    import ast
    import collections
    import importlib.util
    import os
    import sys
    from splat_explorer.splatfix.oraclegs.adapter import reference_only_ground_truth
    source,output,_ = staged_generation(tmp_path)
    assemble(source,output)
    root = output/'dataset'
    attn = root/'synthetic_2_attention_maps'
    attn.mkdir()
    for p in (root/'synthetic_2').glob('*.png'):
        Image.new('L',(37,37),128).save(attn/p.name)
    originals = []
    if worldexplorer:
        from test_worldexplorer_source import fixture
        from splat_explorer.splatfix.worldexplorer.source import stage_source
        from splat_explorer.splatfix.oraclegs.adapter import append_worldexplorer
        stage_source(fixture(tmp_path), source/'worldexplorer-source')
        manifest = json.loads((source/'inputs.json').read_text())
        manifest['worldexplorer_source'] = 'worldexplorer-source'
        (source/'inputs.json').write_text(json.dumps(manifest))
        n, originals = append_worldexplorer(source, output, 2)
        assert n == 2
    spec = importlib.util.spec_from_file_location('oracle_colmap_test',repair.HERE/'upstream/oraclegs/scene/colmap_loader.py')
    colmap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colmap)
    namespace = {'os':os,'sys':sys,'np':np,'Image':Image,'qvec2rotmat':colmap.qvec2rotmat,
                 'focal2fov':lambda focal,pixels:2*np.arctan(pixels/(2*focal)),
                 'CameraInfo':collections.namedtuple('CameraInfo','uid R T FovY FovX image_path image_name depth_path depth_params attention_map_path width height is_test is_synthetic')}
    tree = ast.parse((repair.HERE/'upstream/oraclegs/scene/dataset_readers.py').read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='readColmapCameras')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'upstream-readColmapCameras','exec'),namespace)
    raw = namespace['readColmapCameras'](
        colmap.read_extrinsics_text(root/'sparse/2/images.txt'),
        colmap.read_intrinsics_text(root/'sparse/2/cameras.txt'),str(root/'images'),'',None,
        str(root/'synthetic_2'),str(attn),[])
    references = json.loads((root/'reference_names.json').read_text())
    cameras = reference_only_ground_truth(raw, references, originals)
    assert len([c for c in cameras if c.is_synthetic]) == 5 + len(originals)
    assert len([c for c in cameras if not c.is_synthetic]) == 2
    assert all(c.attention_map_path for c in cameras if c.is_synthetic and c.image_name not in originals)
    assert all(c.is_synthetic and not c.depth_path and not c.attention_map_path
               for c in cameras if c.image_name in originals)
    if originals:
        with pytest.raises(ValueError, match='attention map'):
            reference_only_ground_truth(raw, references)
        with pytest.raises(ValueError, match='Missing WorldExplorer'):
            reference_only_ground_truth([c for c in raw if c.image_name != originals[0]], references, originals)
    (attn/'target_000001.png').unlink()
    bad = [c._replace(attention_map_path='') if c.image_name=='target_000001.png' else c for c in raw]
    with pytest.raises(ValueError,match='attention map'):
        reference_only_ground_truth(bad,references,originals)


@pytest.mark.parametrize('adapter',['trajectory','saved_poses','per_view_preset'])
@pytest.mark.parametrize('stop_training',[False,True])
@pytest.mark.parametrize('count',[2,6])
def test_full_pipeline_stage_contract(tmp_path,monkeypatch,adapter,stop_training,count,depth="predicted",initialization="sfm",world_attention=None,artifixer=False):
    cp = checkpoint(tmp_path,count)
    inputs,output = tmp_path/'inputs',tmp_path/'run'
    runtime = {**repair.DEFAULT_RUNTIME, 'oracle_reference_mode':'all', 'oracle_adapter':adapter, 'oracle_depth_source':depth, 'oracle_initialization':initialization}
    manifest = export_inputs(cp,inputs,runtime={**runtime,'oracle_target_frames':5})
    if world_attention:
        from test_worldexplorer_source import fixture
        from splat_explorer.splatfix.worldexplorer.source import stage_source
        stage_source(fixture(tmp_path), inputs/'worldexplorer-source')
        manifest.update(worldexplorer_source='worldexplorer-source', oracle_worldexplorer_attention=world_attention)
        (inputs/'inputs.json').write_text(json.dumps(manifest))
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames']
    if artifixer:
        manifest['artifixer_source'] = {'run_id':'source'}
        manifest['seva_internal_resolution'] = manifest['resolution']
        (inputs/'inputs.json').write_text(json.dumps(manifest))
        for frame in frames:
            path = inputs/'scene'/frame['file_path']
            if not path.exists():
                Image.new('RGB',tuple(manifest['resolution']),'green').save(path)
    phases = []
    requested = [False]
    monkeypatch.setattr(repair,'validate_runtime',lambda _:None)
    monkeypatch.setattr(repair,'known_model',lambda *a:None)
    def fake_process(argv,log,stop):
        phase = log.stem
        phases.append(phase)
        log.write_text('test double, no GPU model executed')
        dataset = output/'dataset'
        if phase == 'sfm_export':
            cameras,images = [],[]
            for i,f in enumerate(frames[:count]):
                c,img = colmap_lines(f,i+1,i+1,Path(f['file_path']).name)
                cameras.append(c);images.append(img)
            sparse = dataset/'sparse/0'
            (sparse/'cameras.txt').write_text('\n'.join(cameras)+'\n')
            (sparse/'images.txt').write_text('\n'.join(images)+'\n')
            (sparse/'points3D.txt').write_text('1 0 0 1 0 0 255 0 1 0 2 0\n')
        elif phase == 'depth_alignment':
            (dataset/'sparse/0/depth_params.json').write_text('{"ref_000000":{"scale":1,"offset":0}}')
        elif phase.startswith('seva_generation'):
            from splat_explorer.splatfix.oraclegs.provision import huggingface_environment
            for key,value in huggingface_environment().items():
                assert key+'='+value in argv
            body = json.loads(Path(argv[-1]).read_text())
            assert body['use_traj_prior'] and body['chunk_strategy']=='interp-gt'
            generated = Path(body['work_dir'])/body['task']/'scene'
            (generated/'samples-rgb').mkdir(parents=True)
            exports=[]
            segment_frames = json.loads((Path(body['data_path'])/'scene/transforms.json').read_text())['frames']
            refs = {f['file_path'] for f in segment_frames if 'ref_' in f['file_path']}
            if adapter == 'per_view_preset':
                assert len(refs) == body['num_inputs'] == count
                assert refs.issubset({f['file_path'] for f in frames[:count]})
                assert body['replace_or_include_input'] and len(segment_frames) == count+5
            else:
                assert refs == {f['file_path'] for f in frames[:count]}
                assert body['num_inputs'] == count
            if adapter == 'trajectory':
                assert len(segment_frames) == count+5
            for i,f in enumerate(segment_frames):
                relative=f'samples-rgb/{i:03d}.png'
                Image.new('RGB',(1024,576),'green').save(generated/relative)
                exports.append({**f,'file_path':relative})
            (generated/'transforms.json').write_text(json.dumps({'frames':exports}))
        elif phase == 'vggt_attention':
            body = json.loads(Path(argv[-1]).read_text())
            assert body['chunks']
            assert all(len(c['reference_images']) == count for c in body['chunks'])
            assert all(c['image_names'][0].startswith('ref_') for c in body['chunks'])
            assert (output/'attention-chunks.json').exists()
            (output/'attention-raw').mkdir()
            for p in (output/'combined').glob('*.png'):
                Image.new('L',(37,37),128).save(output/'attention-raw'/p.name)
        elif phase == 'reconstruction':
            body = json.loads(Path(argv[-1]).read_text())
            assert argv[-3] == ('train' if artifixer or world_attention or adapter in ('trajectory','per_view_preset') else 'script')
            assert '--synth_attention_dir' in body['argv']
            assert body['argv'][body['argv'].index('--lambda_lpips')+1]=='0.3'
            assert '--gt_synth_schedule' in body['argv'] and '--depths' in body['argv']
            assert len(list((dataset/f'synthetic_{count}_attention_maps').glob('*.png'))) == len(frames)-count+(2 if world_attention == 'weighted' else 0)
            assert len(body['unweighted_synthetic']) == (2 if world_attention == 'unweighted' else 0)
            assert bool(body['source_depth']) == (depth == 'source_splat')
            assert bool(body['source_initialization']) == (initialization in ('source_splat', 'point_cloud'))
            assert body['argv'][body['argv'].index('--depths')+1] == ('' if depth == 'source_splat' else 'depths')
            assert body['oracle_initialization'] == initialization
            if initialization == 'point_cloud':
                assert '--sh_degree' not in body['argv']
            if initialization == 'source_splat':
                assert body['argv'][body['argv'].index('--sh_degree')+1] == '0'
            assert body['stop_control'] == str(output/'SAVE_STOP')
            if stop_training:
                requested[0] = True
                assert stop() is False  # Cooperative save, not immediate SIGTERM.
                assert Path(body['stop_control']).exists()
                (output/'stop-snapshot.ply').write_bytes(b'ply current state')
                (output/'stop-state.json').write_text(json.dumps({'status':'saved','iteration':42,
                    'size_bytes':(output/'stop-snapshot.ply').stat().st_size}))
                return
            ply = output/'model/point_cloud/iteration_22000/point_cloud.ply'
            ply.parent.mkdir(parents=True)
            ply.write_text('ply\nformat ascii 1.0\ncomment TEST DOUBLE\nend_header\n')
    monkeypatch.setattr(repair,'run_process',fake_process)
    if stop_training:
        with pytest.raises(InterruptedError,match='partial reconstruction saved'):
            repair.run_prepared(inputs,output,runtime=runtime,should_stop=lambda:requested[0])
        from splat_explorer.splatfix.oraclegs.partial import salvage
        result = salvage(output,'User stop')
        assert result['saved_iteration'] == 42
        assert not (output/'result.json').exists()
    else:
        result = repair.run_prepared(inputs,output,runtime=runtime)
    generations = ([f'seva_generation_{i+1:03d}_of_{count-1:03d}' for i in range(count-1)]
                   if adapter == 'trajectory' else ['seva_generation'])
    if adapter == 'per_view_preset':
        generations = [f'seva_generation_{i+1:03d}_of_{count:03d}' for i in range(count)]
    depth_phases = ['reference_depth','depth_alignment'] if depth == 'predicted' else []
    sfm_phases = [] if depth == 'source_splat' and initialization in ('source_splat', 'point_cloud') else ['sfm_features','sfm_matching','sfm_triangulation','sfm_export']
    if artifixer:
        generations = []
    assert phases == [*([] if artifixer else ['model_access']),*sfm_phases,
                      *depth_phases,*generations,'vggt_attention','reconstruction']
    assert Path(result['splat_path']).exists()
    recipe = json.loads((output/'recipe.json').read_text())
    assert len(recipe['commands'])==(2 if artifixer else 3)+len(sfm_phases)+len(depth_phases)+len(generations)
    assert ('Six GPT repaired' if artifixer else 'Only saved reference') in recipe['gt_policy']


@pytest.mark.parametrize('mode',['crop','pad'])
def test_non_widescreen_projection_after_fitting(tmp_path,mode):
    from splat_explorer.splatfix.oraclegs.inputs import fit_reference
    camera = Camera(np.array([2.,-1.,4.]),Rotation.from_euler('xyz',[12,23,34],degrees=True).as_matrix(),
                    width=1600,height=1200,fov_deg=65.)
    image = Image.new('RGB',(3200,2400),'blue')
    fitted,k,affine = fit_reference(image,camera,(1024,576),mode)
    assert fitted.size == (1024,576)
    np.testing.assert_allclose(k[:2,2],[512,288],atol=1e-8)
    points = np.array([[.2,.1,2.],[-.3,.4,3.],[0,0,4.]])
    source_pixels = (camera.intrinsics@points.T).T
    source_pixels = source_pixels/source_pixels[:,2:]
    source_pixels[:,:2] *= 2  # Saved repaired image has twice the camera raster size.
    transformed = (affine@source_pixels.T).T
    projected = (k@points.T).T
    np.testing.assert_allclose(projected/projected[:,2:],transformed,atol=1e-8)
    if mode=='crop':
        assert affine[1,2] == -96  # 3200×2400 → 1024×768, then crop 96 each side.
        assert fitted.getpixel((0,0)) == (0,0,255)
    else:
        assert fitted.getpixel((0,0)) == (0,0,0)
        assert fitted.getpixel((512,288)) == (0,0,255)


def test_720_preserves_requested_resolution_for_both_fitting_modes(tmp_path):
    cp = checkpoint(tmp_path,2)
    cropped = export_inputs(cp,tmp_path/'crop',runtime={'seva_resolution':'720p','oracle_target_frames':3})
    padded = export_inputs(cp,tmp_path/'pad',runtime={'seva_resolution':'720p','seva_fit':'pad','oracle_target_frames':3})
    assert cropped['resolution']==[1280,720] and cropped['seva_internal_resolution']==[1280,768]
    assert padded['resolution']==[1280,720] and padded['seva_internal_resolution']==[1280,768]
    for name,cy in [('crop',360),('pad',360)]:
        f=json.loads((tmp_path/name/'scene/transforms.json').read_text())['frames'][0]
        assert f['cy']==cy
        np.testing.assert_allclose(np.asarray(f['transform_matrix'])@AXES,cp.views[0]['camera']['c2w'])
    assert repair.seva_request(tmp_path/'crop',tmp_path/'out',cropped,repair.DEFAULT_RUNTIME)['pad_to_stride']==64
    assert repair.seva_request(tmp_path/'pad',tmp_path/'out',padded,repair.DEFAULT_RUNTIME)['pad_to_stride']==64


def test_generation_intrinsics_mismatch_fails_closed(tmp_path):
    source,output,manifest = staged_generation(tmp_path)
    path = output/'seva'/manifest['seva_task']/'scene/transforms.json'
    body = json.loads(path.read_text())
    body['frames'][-1]['cx'] += 10
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='calibration'):
        assemble(source,output)


def test_sfm_camera_alignment_guard(tmp_path):
    from splat_explorer.splatfix.oraclegs.adapter import verify_sfm_cameras
    source,output,_=staged_generation(tmp_path)
    verify_sfm_cameras(source,output)
    report=json.loads((output/'camera-alignment.json').read_text())
    assert report['passed'] and max(r['max_pixel_error'] for r in report['references'])<1e-4
    path=output/'dataset/sparse/0/images.txt'
    lines=path.read_text().splitlines()
    fields=lines[0].split(); fields[5]=str(float(fields[5])+0.01)
    lines[0]=' '.join(fields)
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='pose drift'):
        verify_sfm_cameras(source,output)


def test_full_response_reuses_recorded_camera_crop_without_upscaling(tmp_path):
    cp=Checkpoint.create(tmp_path,tmp_path/'scene.ply',target_views=2,metadata={'renderer':{'backend':'viser'}})
    for i in range(2):
        camera=Camera(np.array([float(i),0.,0.]),np.eye(3),width=960,height=544,fov_deg=65.)
        v=cp.add_view(Image.new('RGB',(960,544),'red'),camera)
        directory=cp.root/'views'/v['id']
        Image.new('RGB',(960,544),'blue').save(directory/'repaired.png')
        Image.new('RGB',(1667,944),'green').save(directory/'response.image',format='PNG')
        scale=544/944
        left=(1667-960/scale)/2
        v['repaired_rgb']=f"views/{v['id']}/repaired.png"
        v['image_repair']={'response_image':f"views/{v['id']}/response.image",'response_size':[1667,944],
                           'target_size':[960,544],'resize':{'method':'cover_center_crop','scale':scale,
                           'source_crop_box':[left,0,1667-left,944]}}
    cp.save()
    m=export_inputs(cp,tmp_path/'staged')
    for v in m['views']:
        assert v['source_kind']=='full_repair_response'
        image=Image.open(tmp_path/'staged'/v['image'])
        assert image.size==(1024,576) and image.getpixel((0,0))==(0,128,0)
        expected=np.array(v['camera']['intrinsics'])* (1024/960)
        expected[1,2]-=(544*1024/960-576)/2
        expected[2]=[0,0,1]
        np.testing.assert_allclose(v['staged_intrinsics'],expected,rtol=0,atol=1e-8)
    cp.views[0]['image_repair']['resize']['source_crop_box'][0]+=5
    with pytest.raises(ValueError,match='crop disagrees'):
        export_inputs(cp,tmp_path/'invalid')


def test_each_pair_has_240_distinct_outputs_and_exact_endpoints(tmp_path):
    from splat_explorer.splatfix.oraclegs.trajectory import keyframe_path
    from splat_explorer.splatfix.checkpoint import camera_from_record
    cp = checkpoint(tmp_path, 6)
    cameras = [camera_from_record(v) for v in cp.views]
    # Exercise curved translation and quaternion sign continuity, not only a line.
    for i,c in enumerate(cameras):
        c.position = np.array([np.cos(i), np.sin(i), i/10])
        c.rotation = Rotation.from_euler('y', i*65, degrees=True).as_matrix()
    path = list(keyframe_path(cameras, list(range(6)), 240))
    assert len(path) == 1200
    for i in range(5):
        segment = path[i*240:(i+1)*240]
        np.testing.assert_array_equal(segment[0][-1],cameras[i].c2w)
        np.testing.assert_array_equal(segment[-1][-1],cameras[i+1].c2w)
        poses = np.array([x[-1] for x in segment])
        assert np.linalg.norm(np.ptp(poses[:,:3,3],axis=0)) > .1
        np.testing.assert_allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-6)
    root = tmp_path/'staged'
    manifest = export_inputs(cp, root)
    frames = json.loads((root/'scene/transforms.json').read_text())['frames'][6:]
    assert len(frames) == len({f['file_path'] for f in frames}) == 1200
    assert manifest['trajectory']['target_count'] == 1200
    for i in range(5):
        native = json.loads((root/f'segments/{i:03d}/scene/transforms.json').read_text())['frames']
        split = json.loads((root/f'segments/{i:03d}/scene/train_test_split_6.json').read_text())
        assert split['train_ids'] == [0,1,2,3,4,245] and split['test_ids'] == list(range(5,245))
        assert native[5:245] == frames[i*240:(i+1)*240]
        for j,view in enumerate(cp.views):
            np.testing.assert_allclose(np.asarray(native[manifest['seva_segments'][i]['frame_order'].index(j)]['transform_matrix'])@AXES,view['camera']['c2w'])


def test_seva_parser_audit_detects_translation_and_image_pairing_errors(tmp_path):
    from splat_explorer.splatfix.oraclegs.camera_audit import verify_parsed_cameras
    cp = checkpoint(tmp_path,3)
    root = tmp_path/'inputs'
    export_inputs(cp,root,runtime={'oracle_target_frames':4})
    frames = json.loads((root/'scene/transforms.json').read_text())['frames']
    poses = (np.array([f['transform_matrix'] for f in frames])@AXES)[:,:3]
    ks = np.array([[[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1]] for f in frames])
    paths = [None if f.get('camera_only') else str(root/'scene'/f['file_path']) for f in frames]
    indices = [3,7,10]
    parsed = (paths,3,8,[0,1,2],[3.,6.5,10.],poses,ks,poses[indices],ks[indices])
    assert verify_parsed_cameras(root,parsed)['passed']
    poses[4,:,3] = poses[3,:,3]
    with pytest.raises(ValueError,match='camera poses'):
        verify_parsed_cameras(root,parsed)
    poses[4] = (np.array(frames[4]['transform_matrix'])@AXES)[:3]
    paths[0],paths[1] = paths[1],paths[0]
    with pytest.raises(ValueError,match='pairing'):
        verify_parsed_cameras(root,parsed)


def test_pinned_seva_parser_preserves_reference_pairing_and_translated_path(tmp_path):
    # Execute the actual CPU parser classes without importing SEVA's GPU stack.
    import ast
    import os
    from glob import glob
    from typing import Dict, List, Optional, Tuple
    from splat_explorer.splatfix.oraclegs import inputs as inputs_module
    source = Path(inputs_module.__file__).parent/'upstream/seva/seva/data_io.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node,ast.ClassDef)
                 and node.name in ('BaseParser','ReconfusionParser')]
    ns = dict(np=np, json=json, os=os, osp=os.path, glob=glob,
              Dict=Dict, List=List, Optional=Optional, Tuple=Tuple)
    exec(compile(tree,str(source),'exec'),ns)
    cp = checkpoint(tmp_path,3)
    root = tmp_path/'inputs'
    export_inputs(cp,root,runtime={'oracle_target_frames':5,'oracle_path_order':[2,0,1]})
    from types import SimpleNamespace
    from splat_explorer.splatfix.oraclegs.camera_audit import install_camera_only_inputs
    demo = SimpleNamespace(get_parser=lambda **kw: ns['ReconfusionParser'](kw['data_dir'], normalize=kw['normalize']))
    install_camera_only_inputs(demo)
    parser = demo.get_parser(data_dir=str(root/'segments/000/scene'), normalize=False)
    assert parser.image_paths[2:7] == [None]*5
    assert all(Path(parser.image_paths[i]).is_file() for i in [0,1,7])
    assert not list(root.rglob('target_*.png'))
    assert parser.splits_per_num_input_frames[3]['train_ids'] == [0,1,7]
    assert len(parser.splits_per_num_input_frames[3]['test_ids']) == 5
    for idx, reference in [(0,1),(1,2),(7,0)]:
        np.testing.assert_allclose(parser.camtoworlds[idx],cp.views[reference]['camera']['c2w'],atol=1e-6)
    assert np.linalg.norm(parser.camtoworlds[4,:3,3]-parser.camtoworlds[3,:3,3]) > .1


def test_diffusion_audit_checks_translated_rays_not_just_exported_poses():
    from splat_explorer.splatfix.oraclegs.camera_audit import verify_conditioning
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, :3] = Rotation.from_euler('y', np.array([30, 60, 110, 170])[:,None], degrees=True).as_matrix()
    poses[:, :3, 3] = [[10, 2, 3], [11, 3, 4], [13, 1, 4], [14, 2, 6]]
    normalized = poses.copy()
    normalized[:, :3, 3] = (poses[:, :3, 3]-[12, 2, 4])*0.3
    k = np.repeat(np.array([[.9, 0, .5], [0, 1.6, .5], [0, 0, 1.]])[None], 4, axis=0)
    # Independent inverse projection of two pixels into the first-camera frame.
    rays = []
    for pose, calibration in zip(normalized, k):
        relative = np.linalg.inv(normalized[0]) @ pose
        directions = (np.linalg.inv(calibration) @ np.array([[.1,.8],[.2,.7],[1,1]])).T
        directions = directions @ relative[:3,:3].T
        directions /= np.linalg.norm(directions,axis=-1,keepdims=True)
        moments = np.cross(relative[:3,3], directions)
        rays.append(np.concatenate([directions,moments],axis=-1).T[:,None,:])
    value = {'c2w':normalized, 'K':k, 'plucker_coordinate':np.array(rays)}
    assert verify_conditioning(poses[:,:3], k, value)['translation_scale'] == pytest.approx(.3)
    broken = {**value, 'plucker_coordinate':value['plucker_coordinate'].copy()}
    broken['plucker_coordinate'][:,3:] = 0
    with pytest.raises(ValueError, match='ray moments'):
        verify_conditioning(poses[:,:3], k, broken)
    broken = {**value, 'c2w':normalized.copy()}
    broken['c2w'][:,:3,3] = normalized[0,:3,3]
    with pytest.raises(ValueError, match='relative camera translation'):
        verify_conditioning(poses[:,:3], k, broken)
    broken = {**value, 'c2w':normalized.copy()}
    broken['c2w'][1,:3,:3] = normalized[2,:3,:3]
    with pytest.raises(ValueError, match='rotations'):
        verify_conditioning(poses[:,:3], k, broken)


def test_stale_total_budget_trajectory_requires_restaging(tmp_path):
    root = tmp_path/'staged'
    manifest = export_inputs(checkpoint(tmp_path, 3), root, runtime={'oracle_target_frames':4})
    repair.validate_staged_trajectory(root, manifest)
    old = {k:v for k,v in manifest.items() if k != 'trajectory'}
    with pytest.raises(ValueError, match='restage a fresh run'):
        repair.validate_staged_trajectory(root, old)
    split_path = root/'scene/train_test_split_3.json'
    split = json.loads(split_path.read_text())
    split['test_ids'] = split['test_ids'][:4]
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='count/split'):
        repair.validate_staged_trajectory(root, manifest)


def test_native_staging_mapping_is_validated_and_off_path_references_remain_unordered(tmp_path):
    cp = checkpoint(tmp_path,3)
    root = tmp_path/'ordered'
    m = export_inputs(cp,root,runtime={'oracle_target_frames':4,'oracle_path_order':[2,0,1]})
    repair.validate_staged_trajectory(root,m)
    split_path = root/'segments/000/scene/train_test_split_3.json'
    split = json.loads(split_path.read_text())
    split['train_ids'][1] += 1
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError,match='SEVA segment'):
        repair.validate_staged_trajectory(root,m)
    other = tmp_path/'off-path'
    m = export_inputs(cp,other,runtime={'oracle_target_frames':4,'oracle_path_order':[2,0]})
    assert m['seva_task']=='img2vid'
    request = repair.seva_request(other,tmp_path/'out',m,repair.DEFAULT_RUNTIME)
    assert request['chunk_strategy']=='interp-gt' and request['replace_or_include_input']
    assert 'traj_prior' not in request and request['task'] != 'img2trajvid_s-prob'


def test_pair_interpolation_cannot_be_bent_by_other_reference_cameras():
    from types import SimpleNamespace
    from splat_explorer.splatfix.oraclegs.trajectory import pairwise_path
    def camera(x,y):
        pose = np.eye(4)
        pose[:3,3] = [x,y,0]
        return SimpleNamespace(position=pose[:3,3],rotation=pose[:3,:3],c2w=pose)
    cameras = [camera(0,0),camera(4,0),camera(6,20)]
    first = list(pairwise_path(cameras,[0,1,2],80))[:80]
    cameras[2] = camera(-500,-700)
    changed = list(pairwise_path(cameras,[0,1,2],80))[:80]
    for a,b in zip(first,changed):
        np.testing.assert_array_equal(a[-1],b[-1])
        np.testing.assert_allclose(a[-1][:3,3], [4*a[2],0,0], atol=1e-12)


@pytest.mark.parametrize('world_attention', ['weighted', 'unweighted'])
@pytest.mark.parametrize('adapter', ['trajectory', 'saved_poses'])
def test_worldexplorer_attention_pipeline(tmp_path, monkeypatch, world_attention, adapter):
    test_full_pipeline_stage_contract(tmp_path, monkeypatch, adapter, False, 2,
                                      world_attention=world_attention)


@pytest.mark.parametrize('initialization', ['sfm', 'source_splat'])
def test_artifixer_skips_generation_but_runs_attention_and_training(tmp_path, monkeypatch, initialization):
    from test_oraclegs_source_geometry import source_ply
    source_ply(tmp_path/'missing.ply')
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    monkeypatch.setattr(CpuSplatRenderer, 'render_depth', lambda self, camera: np.full((camera.height, camera.width), 4, np.float32))
    test_full_pipeline_stage_contract(tmp_path, monkeypatch, 'saved_poses', False, 6,
                                      depth='source_splat' if initialization == 'source_splat' else 'predicted',
                                      initialization=initialization, artifixer=True)
