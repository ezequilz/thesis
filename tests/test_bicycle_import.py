"""The importer must preserve camera geometry and observations used by MoGe."""
import importlib.util
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location('bicycle_import', Path(__file__).parents[1] / 'scripts/benchmarks/prepare_bicycle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_source(root, overlap=False):
    released = root / 'reconfusion/mipnerf360/bicycle'
    (released/'images_4').mkdir(parents=True)
    original = root/'original/bicycle/sparse/0'
    original.mkdir(parents=True)
    (original/'cameras.bin').write_bytes(struct.pack('<QiiQQdddd',1,1,1,40,32,20,20,20,16))
    frames=[]
    with (original/'images.bin').open('wb') as f:
        f.write(struct.pack('<Q',4))
        for i in range(4):
            name=f'{i}.JPG'
            f.write(struct.pack('<idddddddi',i+1,1,0,0,0,-i,0,0,1))
            f.write(name.encode()+b'\0'+struct.pack('<Qddq',1,20,16,1))
            pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[2*i+1,2,3]
            frames.append({'file_path':'./images_4/'+name,'transform_matrix':pose.tolist()})
            Image.new('RGB',(10,8)).save(released/'images_4'/name)
    (original/'points3D.bin').write_bytes(struct.pack('<QQdddBBBdQii',1,1,0,0,1,1,2,3,.1,1,1,0))
    (released/'transforms.json').write_text(json.dumps(dict(w=10,h=8,fl_x=5,fl_y=5,cx=5,cy=4,frames=frames)))
    (released/'train_test_split_3.json').write_text(json.dumps({'train_ids':[0,1,2],'test_ids':[0 if overlap else 3]}))
    return root


def test_preserves_geometry_and_measured_observations(tmp_path):
    source=fixture_source(tmp_path/'source')
    result=module.prepare(source,tmp_path/'input')
    assert result['similarity']['scale']==2
    assert result['similarity']['translation']==[1,2,3]
    sparse=tmp_path/'input/colmap/sparse/0'
    with (sparse/'images.bin').open('rb') as f:
        assert struct.unpack('<Q',f.read(8))[0]==4
        record=struct.unpack('<idddddddi',f.read(64))
        assert record[5:8]==(-1,-2,-3)
        assert f.read(6)==b'0.JPG\0'
        assert struct.unpack('<Qddq',f.read(32))==(1,5.,4.,1)
    with (sparse/'points3D.bin').open('rb') as f:
        record=struct.unpack('<QQdddBBBdQii',f.read())
        assert record[2:5]==(1.,2.,5.)
        assert record[-3:]==(1,1,0)
    assert result['website_exact_match_verified'] is False
    assert 'colmap/sparse/0/points3D.bin' in result['sha256']
    for relative,digest in result['sha256'].items():
        assert module.digest_file(tmp_path/'input'/relative)==digest


def test_rejects_split_leakage_before_writing(tmp_path):
    with pytest.raises(ValueError,match='overlap'):
        module.prepare(fixture_source(tmp_path/'source',overlap=True),tmp_path/'input')
    assert not (tmp_path/'input').exists()
