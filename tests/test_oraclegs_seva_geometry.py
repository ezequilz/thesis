import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from splat_explorer.splatfix.oraclegs import backend
from splat_explorer.splatfix.oraclegs.seva_geometry import project_reference, blend_prediction, ReferenceGuidance
from splat_explorer.splatfix.oraclegs.repair import seva_request, DEFAULT_RUNTIME
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from splat_explorer.splatfix.oraclegs.source_geometry import validate_assets
from splat_explorer.splatfix.oraclegs.execution import job_runtime
from test_splatfix_oraclegs import checkpoint
from test_oraclegs_source_geometry import source_ply


def scene():
    rgb = np.zeros((32,32,3),np.float32)
    rgb[8:24,8:24] = [1, .5, .25]
    return rgb, np.full((32,32),.5,np.float32), np.ones((32,32),bool), np.eye(4), np.array([[16,0,16],[0,16,16],[0,0,1.]])


def test_identity_translation_and_padding_use_camera_z_and_pixel_intrinsics():
    rgb,inv,valid,pose,k = scene()
    rgb += np.arange(32)[None,:,None]/100
    guide,mask = project_reference(rgb,inv,valid,pose,k,pose,k,(32,32))
    np.testing.assert_array_equal(guide[12:20,12:20],rgb[12:20,12:20])
    assert mask[12:20,12:20].all()
    target = pose.copy(); target[0,3] = .5
    shifted,_ = project_reference(rgb,inv,valid,pose,k,target,k,(32,32))
    np.testing.assert_array_equal(shifted[12:20,8:16],rgb[12:20,12:20])
    padded_k=k.copy(); padded_k[1,2]+=24
    padded,padded_mask=project_reference(rgb,inv,valid,pose,k,pose,padded_k,(32,80))
    np.testing.assert_array_equal(padded[36:44,12:20],rgb[12:20,12:20])
    assert padded_mask[36:44,12:20].all() and not padded_mask[:24].any()


def test_invalid_depth_behind_camera_and_disocclusion_are_unconstrained():
    rgb,inv,valid,pose,k=scene()
    valid[:]=False
    _,mask=project_reference(rgb,inv,valid,pose,k,pose,k,(32,32))
    assert not mask.any()
    valid[:]=True; inv[:]=np.nan
    _,mask=project_reference(rgb,inv,valid,pose,k,pose,k,(32,32))
    assert not mask.any()
    inv[:]=.5; target=pose.copy(); target[2,3]=3
    _,mask=project_reference(rgb,inv,valid,pose,k,target,k,(32,32))
    assert not mask.any()


def test_z_buffer_prefers_foreground():
    rgb,inv,valid,pose,k=scene()
    valid[:]=False
    valid[16,12]=valid[16,16]=True
    rgb[16,12]=[1,0,0]; rgb[16,16]=[0,1,0]
    inv[16,12]=.25; inv[16,16]=.5
    target=pose.copy(); target[0,3]=1
    guide,_=project_reference(rgb,inv,valid,pose,k,target,k,(32,32))
    np.testing.assert_array_equal(guide[16,8],[0,1,0])


def test_guidance_preserves_uncovered_and_conditioned_pixels_and_cfg_algebra():
    u,c,g=np.array([1.,2.,3.]),np.array([5.,6.,7.]),np.array([0.,0.,0.])
    mask=np.array([1.,0.,0.]); strength=.35; cfg=3
    bu=blend_prediction(u,g,mask,strength); bc=blend_prediction(c,g,mask,strength)
    np.testing.assert_allclose(bu+cfg*(bc-bu),blend_prediction(u+cfg*(c-u),g,mask,strength))
    np.testing.assert_array_equal(bc[1:],c[1:])


def test_setting_stages_depth_independently_and_routes_each_segment(tmp_path,monkeypatch):
    cp=checkpoint(tmp_path,count=2); source_ply(Path(cp.manifest['scene_path']))
    from splat_explorer.rendering.cpu_splat_renderer import CpuSplatRenderer
    monkeypatch.setattr(CpuSplatRenderer,'render_depth',lambda self,cam:np.full((cam.height,cam.width),4,np.float32))
    options={'seva_geometry_guidance':'source_splat','oracle_target_frames':2}
    root=tmp_path/'inputs'
    manifest=export_inputs(cp,root,'edited',runtime=options)
    assert manifest['oracle_depth_source']=='predicted' and manifest['oracle_initialization']=='sfm'
    assert manifest['source_depth'] and not manifest.get('source_initialization')
    validate_assets(root,manifest)
    runtime=job_runtime({},options)
    assert runtime['seva_geometry_guidance']=='source_splat'
    request=seva_request(root/'segments/000',tmp_path/'out',manifest,runtime)
    assert request['source_geometry_guidance']['inputs']==str(root)
    assert request['source_geometry_guidance']['strength']==.35
    references=ReferenceGuidance(root)
    rgb,inv,valid,pose,k=references.references[0]
    _,mask,index=references.target(pose,k,(1024,576))
    assert index==0 and mask.mean()>.9
    # Corruption cannot be silently bypassed when only SEVA needs source depth.
    (root/manifest['source_depth']['views'][0]['path']).write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):
        ReferenceGuidance(root)


def test_option_default_validation_and_saved_pose_request(tmp_path):
    assert backend.validate_options({})['seva_geometry_guidance']=='off'
    with pytest.raises(ValueError,match='seva_geometry_guidance'):
        backend.validate_options({'seva_geometry_guidance':'predicted'})
    manifest=dict(resolution=[1024,576],views=[{},{}],seva_resolution='576p',seva_geometry_guidance='source_splat')
    request=seva_request(tmp_path,tmp_path/'out',manifest,DEFAULT_RUNTIME)
    assert request['source_geometry_guidance']['inputs']==str(tmp_path)
    manifest['seva_geometry_guidance']='off'
    assert 'source_geometry_guidance' not in seva_request(tmp_path,tmp_path/'out',manifest,DEFAULT_RUNTIME)


def test_conditioning_audit_preserves_native_signature(tmp_path):
    from splat_explorer.splatfix.oraclegs.camera_audit import install_conditioning_audit
    def native(curr_imgs,curr_input_frame_indices,curr_c2ws,curr_Ks,curr_input_camera_indices,all_c2ws,camera_scale):
        pass
    evaluation=SimpleNamespace(get_value_dict=native)
    install_conditioning_audit(evaluation,tmp_path/'audit.json')
    assert inspect.signature(evaluation.get_value_dict)==inspect.signature(native)
