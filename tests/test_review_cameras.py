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


def test_interrupted_trajectory_uses_split_indices_references_first(tmp_path):
    manifest = tmp_path / 'partial-result.json'
    manifest.write_text(json.dumps({'inference_split': 'prepared/scene/split_trajectory.json'}))
    root = tmp_path / 'prepared/scene'
    root.mkdir(parents=True)
    (root / 'split_trajectory.json').write_text(json.dumps({'test': {'scene': {
        'transforms_path': 'transforms.json', 'selected_indices_path': 'selected.json',
        'target_indices_path': 'targets.json'}}}))
    (root / 'selected.json').write_text('[2, 1]')
    (root / 'targets.json').write_text('[0, 1, -1, 99, true]')
    frames = []
    for i in range(3):
        pose = np.eye(4)
        pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        pose[0, 3] = i
        frames.append({'transform_matrix': pose.tolist(), 'fl_y': 700, 'h': 800})
    # Intrinsics may be per frame; trajectory frames need not have file_path.
    (root / 'transforms.json').write_text(json.dumps({'frames': frames}))
    views = benchmark_views(manifest)
    assert [v['label'] for v in views] == ['Frame 3', 'Frame 2', 'Frame 1']
    np.testing.assert_array_equal(views[0]['center'], [2, 0, 0])
    np.testing.assert_array_equal(views[0]['forward'], [0, 1, 0])
    np.testing.assert_array_equal(views[0]['up'], [0, 0, 1])
    # Reject a split escaping the result directory, including via symlink.
    (root / 'split_trajectory.json').unlink()
    external = tmp_path.parent / 'outside-split.json'
    external.write_text(json.dumps({'test': {'scene': {}}}))
    (root / 'split_trajectory.json').symlink_to(external)
    assert benchmark_views(manifest) == []


def test_checkpoint_world_up_keeps_pitched_views_on_same_walking_plane(tmp_path):
    from splat_explorer.web.review_cameras import checkpoint_views
    from splat_explorer.rendering.viser_viewer import _apply_view
    from types import SimpleNamespace
    views = []
    for pitch in (0.25, -0.6):
        c, s = np.cos(pitch), np.sin(pitch)
        views.append({'camera': {'position': [1, 2, 3],
            'rotation': [[1, 0, 0], [0, c, -s], [0, s, c]],
            'width': 832, 'height': 480, 'fov_deg': 75}})
    body = {'camera_convention': 'opencv_c2w', 'metadata': {'up_axis': '-y'}, 'views': views}
    path = tmp_path / 'checkpoint.json'
    path.write_text(json.dumps(body))
    camera = SimpleNamespace()
    server = SimpleNamespace(get_clients=lambda: {0: SimpleNamespace(camera=camera)})
    for saved, loaded in zip(views, checkpoint_views(tmp_path)):
        original = camera_view(saved['camera'], '')
        _apply_view(server, loaded)
        np.testing.assert_array_equal(camera.up_direction, [0, -1, 0])
        np.testing.assert_array_equal(camera.position, original['center'])
        np.testing.assert_allclose(camera.look_at - camera.position, original['forward'])
        # Gravity projected perpendicular to the viewing direction still gives
        # the original image-up, preserving the saved image framing.
        forward = original['forward']
        image_up = camera.up_direction - np.dot(camera.up_direction, forward) * forward
        np.testing.assert_allclose(image_up / np.linalg.norm(image_up), original['up'])
        # Camera-controls forward() moves in the plane defined by world-up.
        right = np.cross(forward, camera.up_direction)
        walking_forward = np.cross(camera.up_direction, right)
        assert walking_forward[1] == 0
    # Older checkpoints without scene metadata retain their known camera pose.
    body.pop('metadata')
    path.write_text(json.dumps(body))
    np.testing.assert_allclose(checkpoint_views(tmp_path)[0]['up'],
                               camera_view(views[0]['camera'], '')['up'])
