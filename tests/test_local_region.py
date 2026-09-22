import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from splat_explorer.scene_runs_ext.local_region import crop_transforms, choose_reference_frames
from splat_explorer.image_edit_gpt import GptImageEditBackend


def test_crop_preserves_pixel_rays_and_camera_poses():
    c = {'w':1408,'h':1056,'fl_x':920.,'fl_y':920.,'cx':704.,'cy':528.,
         'frames':[{'transform_matrix':np.eye(4).tolist()}]}
    out = crop_transforms(c,(896,512,1408,1024),1)
    assert (out['w'],out['h']) == (512,512)
    assert out['cx'] == -192 and out['cy'] == 16
    for u,v in [(0,0),(256,256),(511,511)]:
        assert (u-out['cx'])/out['fl_x'] == (u+896-c['cx'])/c['fl_x']
        assert (v-out['cy'])/out['fl_y'] == (v+512-c['cy'])/c['fl_y']
    assert out['frames'][0]['transform_matrix'] == c['frames'][0]['transform_matrix']
    assert c['cx'] == 704  # original metadata is untouched
    with pytest.raises(ValueError):crop_transforms(c,(896,512,1409,1024),1)
    with pytest.raises(ValueError):crop_transforms(c,(897,512,1408,1024),1)


def test_reference_selection_rejects_duplicate_or_rotation_only_poses():
    frames=[]
    for x in [0,1,2,3,4,0]:
        pose=np.eye(4);pose[0,3]=x
        frames.append({'transform_matrix':pose.tolist()})
    assert choose_reference_frames({'frames':frames},5) == [0,1,2,3,4]
    with pytest.raises(ValueError,match='distinct'):
        choose_reference_frames({'frames':[frames[0]]*5},5)


def test_multi_image_edit_keeps_target_first_and_never_drops_references(tmp_path):
    paths=[]
    for i in range(3):
        p=tmp_path/f'{i}.png';Image.new('RGB',(16,16),(i,0,0)).save(p);paths.append(p)
    calls=[]
    def edit(**kwargs):
        calls.append(kwargs)
        assert [Path(f.name) for f in kwargs['image']] == paths
        return {'data':[]}
    backend=GptImageEditBackend(SimpleNamespace(images=SimpleNamespace(edit=edit)))
    result=backend.edit_with_references(paths[0],paths[1:],'Only edit image one')
    assert result.error is None
    assert all(f.closed for f in calls[0]['image'])
    def unsupported(**kwargs):
        calls.append(kwargs)
        raise TypeError('multiple images unavailable')
    backend.client.images.edit=unsupported
    result=backend.edit_with_references(paths[0],paths[1:],'test')
    assert 'multiple images unavailable' in result.error
    assert len(calls)==2  # no fallback that silently discards context


def test_off_center_conditioning_preserves_full_frame_rays_and_normalized_k():
    from splat_explorer.scene_runs_ext.artifixer_bridge import crop_camera_conditioning
    class Tensor(np.ndarray):
        def clone(self): return self.copy()
        def contiguous(self): return self.copy()
    c={'w':16,'h':16,'cx':-8.,'cy':8.,'frames':[{'cx':-8.,'cy':8.}]}
    rays=np.arange(1*32*64*6).reshape(1,32,64,6).view(Tensor)
    k=np.array([[[1.,0.,0.],[0.,2.,0.],[0.,0.,1.]]]).view(Tensor)
    def compute(camera,*args,**kwargs):
        assert camera['cx']==32 and camera['cy']==16 and camera['w']==64
        return {'camera_rays':rays,'Ks':k,'neighbor_Ks':k.copy()}
    out=crop_camera_conditioning(compute,c,[0],[0],box=(40,8,56,24),source_size=(64,32),scale=1)
    np.testing.assert_array_equal(out['camera_rays'],rays[:,8:24,40:56])
    assert out['Ks'][0,0,0]==4 and out['Ks'][0,1,1]==4
    assert out['Ks'][0,0,2]==-1 and out['Ks'][0,1,2]==0
    assert c['cx']==-8
