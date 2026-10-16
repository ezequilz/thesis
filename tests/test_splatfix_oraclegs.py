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
    assert len(frames) == 14
    split = json.loads((root/'scene/train_test_split_3.json').read_text())
    assert split == {'train_ids':[0,1,2],'test_ids':list(range(3,14))}
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
        assert any(np.allclose(f['transform_matrix'],target['transform_matrix']) for target in frames[3:])
    assert frames[3]['file_path'] == frames[2]['file_path']
    assert frames[-1]['file_path'] == frames[1]['file_path']
    assert before == [hashlib.sha256(cp.image_path(v,repaired=True).read_bytes()).hexdigest() for v in cp.views]
    request = repair.seva_request(root,tmp_path/'out',manifest,repair.DEFAULT_RUNTIME)
    assert request['cfg'] == [3.,2.]
    assert request['task'] == 'img2trajvid' and request['use_traj_prior'] is True
    assert request['chunk_strategy'] == 'interp-gt'
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
    generation = output/'seva/img2trajvid/scene'
    (generation/'samples-rgb').mkdir(parents=True)
    exported = []
    for i,f in enumerate(frames):
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
        assert Image.open(dataset/'synthetic_2'/name).getpixel((0,0)) == (0,128,0)
        assert Image.open(output/'combined'/name).getpixel((0,0)) == (0,0,255)
        assert Image.open(output/'combined'/('synth_'+name)).getpixel((0,0)) == (0,128,0)
    assert json.loads((dataset/'reference_names.json').read_text()) == ['ref_000000.png','ref_000001.png']
    names = (dataset/'sparse/2/images.txt').read_text().splitlines()[::2]
    assert len(names) == (5 if adapter=='trajectory' else 2)
    for image in (dataset/'synthetic_2').glob('*.png'):
        assert Image.open(image).size == tuple(manifest['resolution'])


def test_generation_pose_mismatch_fails_closed(tmp_path):
    source,output,_ = staged_generation(tmp_path)
    path = output/'seva/img2trajvid/scene/transforms.json'
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


def test_real_upstream_colmap_loader_split_and_attention(tmp_path):
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
    cameras = reference_only_ground_truth(raw,['ref_000000.png','ref_000001.png'])
    assert len([c for c in cameras if c.is_synthetic]) == 5
    assert len([c for c in cameras if not c.is_synthetic]) == 2
    assert all(c.attention_map_path for c in cameras if c.is_synthetic)
    (attn/'target_000001.png').unlink()
    bad = [c._replace(attention_map_path='') if c.image_name=='target_000001.png' else c for c in raw]
    with pytest.raises(ValueError,match='attention map'):
        reference_only_ground_truth(bad,['ref_000000.png','ref_000001.png'])


@pytest.mark.parametrize('adapter',['trajectory','saved_poses'])
def test_full_pipeline_stage_contract(tmp_path,monkeypatch,adapter):
    cp = checkpoint(tmp_path,2)
    inputs,output = tmp_path/'inputs',tmp_path/'run'
    manifest = export_inputs(cp,inputs,runtime={'oracle_adapter':adapter,'oracle_target_frames':5})
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames']
    phases = []
    monkeypatch.setattr(repair,'validate_runtime',lambda _:None)
    monkeypatch.setattr(repair,'known_model',lambda *a:None)
    def fake_process(argv,log,stop):
        phase = log.stem
        phases.append(phase)
        log.write_text('test double, no GPU model executed')
        dataset = output/'dataset'
        if phase == 'sfm_export':
            cameras,images = [],[]
            for i,f in enumerate(frames[:2]):
                c,img = colmap_lines(f,i+1,i+1,Path(f['file_path']).name)
                cameras.append(c);images.append(img)
            sparse = dataset/'sparse/0'
            (sparse/'cameras.txt').write_text('\n'.join(cameras)+'\n')
            (sparse/'images.txt').write_text('\n'.join(images)+'\n')
            (sparse/'points3D.txt').write_text('1 0 0 1 0 0 255 0 1 0 2 0\n')
        elif phase == 'depth_alignment':
            (dataset/'sparse/0/depth_params.json').write_text('{"ref_000000":{"scale":1,"offset":0}}')
        elif phase == 'seva_generation':
            from splat_explorer.splatfix.oraclegs.provision import huggingface_environment
            for key,value in huggingface_environment().items():
                assert key+'='+value in argv
            body = json.loads(Path(argv[-1]).read_text())
            assert body['use_traj_prior'] and body['chunk_strategy']=='interp-gt'
            generated = output/'seva/img2trajvid/scene'
            (generated/'samples-rgb').mkdir(parents=True)
            exports=[]
            for i,f in enumerate(frames):
                relative=f'samples-rgb/{i:03d}.png'
                Image.new('RGB',(1024,576),'green').save(generated/relative)
                exports.append({**f,'file_path':relative})
            (generated/'transforms.json').write_text(json.dumps({'frames':exports}))
        elif phase == 'vggt_attention':
            (output/'attention-raw').mkdir()
            for p in (output/'combined').glob('*.png'):
                Image.new('L',(37,37),128).save(output/'attention-raw'/p.name)
        elif phase == 'reconstruction':
            body = json.loads(Path(argv[-1]).read_text())
            assert argv[-3] == ('train' if adapter=='trajectory' else 'script')
            assert '--synth_attention_dir' in body['argv']
            assert body['argv'][body['argv'].index('--lambda_lpips')+1]=='0.3'
            assert '--gt_synth_schedule' in body['argv'] and '--depths' in body['argv']
            assert len(list((dataset/'synthetic_2_attention_maps').glob('*.png'))) == len(frames)-2
            ply = output/'model/point_cloud/iteration_22000/point_cloud.ply'
            ply.parent.mkdir(parents=True)
            ply.write_text('ply\nformat ascii 1.0\ncomment TEST DOUBLE\nend_header\n')
    monkeypatch.setattr(repair,'run_process',fake_process)
    result = repair.run_prepared(inputs,output)
    assert phases == ['model_access','sfm_features','sfm_matching','sfm_triangulation','sfm_export',
                      'reference_depth','depth_alignment','seva_generation','vggt_attention','reconstruction']
    assert Path(result['splat_path']).exists()
    recipe = json.loads((output/'recipe.json').read_text())
    assert len(recipe['commands'])==10
    assert 'Only saved reference' in recipe['gt_policy']


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
    source,output,_ = staged_generation(tmp_path)
    path = output/'seva/img2trajvid/scene/transforms.json'
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
