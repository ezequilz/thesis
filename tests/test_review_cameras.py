import json
import numpy as np
from splat_explorer.web.review_cameras import camera_view, benchmark_views, historical_views


def test_cv_saved_camera_orientation_and_vertical_calibration():
    view = camera_view({'position':[1,2,3], 'rotation':np.eye(3).tolist(),
                        'width':960, 'height':720, 'fov_deg':75}, 'anchor')
    np.testing.assert_array_equal(view['forward'], [0,0,1])
    np.testing.assert_array_equal(view['up'], [0,-1,0])
    assert np.isclose(view['vfov'], 2*np.arctan(np.tan(np.deg2rad(75)/2)*720/960))


def test_benchmark_selected_photo_order_opengl_and_no_scale(tmp_path):
    manifest = tmp_path/'result.json'
    manifest.write_text(json.dumps({'selected_images':['b.jpg','a.jpg'],'metric_scale':1000}))
    root=tmp_path/'prepared/scene';root.mkdir(parents=True)
    (root/'split.json').write_text(json.dumps({'test':{'scene':{'transforms_path':'transforms.json'}}}))
    pose=np.eye(4);pose[:3,3]=[4,5,6]
    (root/'transforms.json').write_text(json.dumps({'fl_y':700,'h':800,'frames':[
      {'file_path':f'images/{name}','transform_matrix':pose.tolist()} for name in ['a.jpg','b.jpg','ignored.jpg']]}))
    views=benchmark_views(manifest)
    assert [v['label'] for v in views]==['b.jpg','a.jpg']
    np.testing.assert_array_equal(views[0]['center'],[4,5,6])
    np.testing.assert_array_equal(views[0]['forward'],[0,0,-1])
    np.testing.assert_array_equal(views[0]['up'],[0,1,0])
    assert np.isclose(views[0]['vfov'],2*np.arctan(800/1400))


def test_missing_or_invalid_saved_views_are_not_synthesized(tmp_path):
    assert historical_views(tmp_path)==[]
    p=tmp_path/'requests/render-00001-repair/request.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'step':1,'camera':{'position':[1,2,3]}}))
    assert historical_views(tmp_path)==[]
