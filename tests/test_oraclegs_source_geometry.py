import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from PIL import Image
from splat_explorer.scene import GaussianScene, save_ply
from splat_explorer.splatfix.oraclegs.source_geometry import inverse_depth, fit_depth, inspect_ply, validate_assets
from splat_explorer.splatfix.oraclegs.source_geometry_runtime import load_depth, initialize_gaussians, install_source_geometry
from splat_explorer.splatfix.oraclegs import backend
from splat_explorer.splatfix.oraclegs.execution import job_runtime
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from test_splatfix_oraclegs import checkpoint


def source_ply(path):
    scene = GaussianScene(np.array([[0,0,2],[1,0,3]],np.float32),
        np.full((2,3),.1,np.float32),np.array([[1,0,0,0]]*2,np.float32),
        np.full(2,.8,np.float32),np.full((2,3),.5,np.float32))
    save_ply(scene,path)
    return scene


def test_inverse_depth_keeps_scene_scale_and_masks_invalid():
    inv,mask = inverse_depth(np.array([[2,4,0],[np.inf,np.nan,-1]]),np.ones((2,3),bool))
    np.testing.assert_array_equal(inv,[[.5,.25,0],[0,0,0]])
    np.testing.assert_array_equal(mask,[[True,True,False],[False,False,False]])


def test_calibrated_crop_and_padding_do_not_blend_depth_edges():
    depth = np.tile(np.array([2,2,4,4],np.float32),(4,1))
    k = np.array([[4,0,2],[0,4,2],[0,0,1.]])
    frame = dict(w=2,h=4,fl_x=4,fl_y=4,cx=1,cy=2)
    inv,valid = fit_depth(depth,np.ones_like(depth,bool),k,frame)
    np.testing.assert_array_equal(inv,np.tile([.5,.25],(4,1)))
    assert valid.all()
    frame.update(w=6,cx=3)
    inv,valid = fit_depth(depth,np.ones_like(depth,bool),k,frame)
    assert not valid[:,[0,-1]].any()
    assert (inv[:,[0,-1]] == 0).all()


@pytest.mark.parametrize('depth',['predicted','source_splat'])
@pytest.mark.parametrize('initialization',['sfm','source_splat'])
def test_source_options_independent_staged_and_verified(tmp_path,monkeypatch,depth,initialization):
    cp = checkpoint(tmp_path,count=2)
    source = Path(cp.manifest['scene_path'])
    source_ply(source)
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    monkeypatch.setattr(CpuSplatRenderer,'render_depth',lambda self,cam:np.full((cam.height,cam.width),4,np.float32))
    options = {'oracle_depth_source':depth,'oracle_initialization':initialization,'oracle_target_frames':2}
    root = tmp_path/'inputs'
    manifest = export_inputs(cp,root,'edited',runtime=options)
    validate_assets(root,manifest)
    assert bool(manifest.get('source_depth')) == (depth=='source_splat')
    assert manifest['source_splat_used_for_initialization'] == (initialization=='source_splat')
    assert job_runtime({},options)['oracle_initialization']==initialization
    if depth=='source_splat':
        record=manifest['source_depth']['views'][0]
        inv,valid=load_depth(root/record['path'],(576,1024))
        assert np.all(inv[valid]==.25)
        with pytest.raises(ValueError,match='dimensions'):
            load_depth(root/record['path'],(720,1280))
        (root/record['path']).write_bytes(b'corrupt')
        with pytest.raises(ValueError,match='checksum'):
            validate_assets(root,manifest)
    if initialization=='source_splat':
        assert (root/'source-initialization.ply').read_bytes()==source.read_bytes()
        assert inspect_ply(source)==(2,0)


def test_defaults_and_invalid_options():
    options=backend.validate_options({})
    assert options['oracle_depth_source']=='predicted'
    assert options['oracle_initialization']=='sfm'
    for key in ('oracle_depth_source','oracle_initialization'):
        with pytest.raises(ValueError,match=key):
            backend.validate_options({key:'invalid'})


def test_source_initialization_sets_training_buffers_and_scene_lr(monkeypatch):
    # Exercise adapter state transitions without substituting a point cloud for Gaussians.
    calls=[]
    model=SimpleNamespace(load_ply=lambda path:calls.append(path),get_xyz=SimpleNamespace(shape=(7,3),device='cuda'))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(zeros=lambda shape,device:np.zeros(shape)))
    initialize_gaussians(model,'source.ply',2.5)
    assert calls==['source.ply']
    assert model.spatial_lr_scale==2.5 and model.max_radii2D.shape==(7,)


def test_source_depth_runtime_preserves_mask_and_excludes_synthetic(tmp_path,monkeypatch):
    import collections
    info_type=collections.namedtuple('Info','image_name is_synthetic depth_path depth_params')
    np.savez(tmp_path/'ref.npz',invdepth=np.array([[.5,0]],np.float32),valid=np.array([[True,False]]))
    originals=[]
    def load(*args):
        originals.append(args[2])
        return SimpleNamespace(image_height=1,image_width=2,original_image=SimpleNamespace(device='cpu',dtype=np.float32))
    module=SimpleNamespace(loadCam=load)
    monkeypatch.setitem(sys.modules,'utils',SimpleNamespace(camera_utils=module))
    monkeypatch.setitem(sys.modules,'utils.camera_utils',module)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(as_tensor=lambda value,**kw:np.asarray(value)))
    install_source_geometry({'source_depth':str(tmp_path)})
    camera=module.loadCam(None,0,info_type('ref.png',False,'prediction.png',{'scale':999}),1,False)
    assert camera.depth_reliable
    np.testing.assert_array_equal(camera.invdepthmap,[[[.5,0]]])
    np.testing.assert_array_equal(camera.depth_mask,[[[True,False]]])
    assert originals[0].depth_path=='' and originals[0].depth_params is None
    synthetic=module.loadCam(None,1,info_type('target.png',True,'',None),1,False)
    assert not hasattr(synthetic,'depth_reliable')


@pytest.mark.parametrize('adapter',['trajectory','saved_poses'])
@pytest.mark.parametrize('depth,initialization',[
    ('source_splat','sfm'), ('predicted','source_splat'), ('source_splat','source_splat')])
def test_full_pipeline_source_variants(tmp_path,monkeypatch,adapter,depth,initialization):
    import test_splatfix_oraclegs as contracts
    original=contracts.checkpoint
    def with_source(*args,**kwargs):
        cp=original(*args,**kwargs)
        source_ply(Path(cp.manifest['scene_path']))
        return cp
    monkeypatch.setattr(contracts,'checkpoint',with_source)
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    monkeypatch.setattr(CpuSplatRenderer,'render_depth',lambda self,cam:np.full((cam.height,cam.width),4,np.float32))
    contracts.test_full_pipeline_stage_contract(tmp_path,monkeypatch,adapter,False,2,depth,initialization)
