"""Snapshot ArtiFixer RGBs and exact cameras for post-generation OracleGS."""
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from ..checkpoint import atomic_json
from ..jobs import read_json
from .backend import validate_options
from .source_geometry import digest, inspect_ply, stage_saved_pose_depth

AXES = np.diag([1., -1., -1., 1.])


def source_assets(studio, row):
    manifest = row['_manifest']
    if row['kind'] != 'ArtiFixer3D' or manifest is None or row['mode'] != 'edited':
        raise ValueError('Requires an edited ArtiFixer run with saved inference images')
    result = read_json(manifest, {})
    local = lambda raw: studio.result_artifact(manifest, raw)
    def data(raw):
        path = local(raw)
        return read_json(path, {}) if path else {}
    trajectory = data('preview-trajectory.json')
    if not trajectory:
        trajectory = data(data('request.json').get('trajectory'))
    if trajectory.get('camera_convention') != 'opencv_c2w':
        raise ValueError('Saved ArtiFixer camera trajectory is unavailable')
    anchors = trajectory.get('anchors', [])
    refs = data('input-preview.json').get('reference_images', [])
    if len(anchors) != 6 or len(refs) != 6:
        raise ValueError('Requires all six saved GPT repaired reference images')
    refs = sorted(refs, key=lambda item: item['index'])
    if [item['index'] for item in refs] != list(range(6)):
        raise ValueError('Reference image mapping is invalid')
    references = [local(item['path']) for item in refs]
    if any(p is None or not p.is_file() for p in references):
        raise ValueError('Repaired reference images are missing')
    preview = data('inference-preview.json')
    prediction = local(result.get('prediction_frames') or preview.get('prediction_frames') or 'inference/splatfix/frames/batch_0000/pred')
    if prediction is None:
        raise ValueError('ArtiFixer inference images are unavailable')
    cameras = trajectory['transforms']['frames']
    predictions = {int(p.stem): p for p in prediction.glob('*.png') if p.stem.isdigit() and p.resolve().is_relative_to(manifest.parent.resolve())}
    expected = preview.get('reconstruction_inputs')
    if not isinstance(expected, list) or not expected:
        raise ValueError('Inference input index manifest is unavailable')
    if not set(expected) <= set(predictions) or not set(predictions) <= set(range(len(cameras))):
        raise ValueError('Wait for all ArtiFixer inference images to download')
    return trajectory, references, predictions


def export_inputs(studio, row, destination, initialization):
    if initialization not in ('source_splat', 'sfm'):
        raise ValueError('Initialization must be source_splat or sfm')
    trajectory, references, predictions = source_assets(studio, row)
    root = Path(destination)
    (root/'scene/images').mkdir(parents=True)
    calibration = trajectory['transforms']
    anchors = trajectory['anchors']
    frames, views, targets = [], [], []
    with Image.open(next(iter(predictions.values()))) as image:
        size = image.size
    def stage(source, camera_index, name):
        with Image.open(source) as image:
            # Both sources cover the recorded full camera FOV; only resolution changes.
            if abs(image.width/image.height-calibration['w']/calibration['h']) > .01:
                raise ValueError('ArtiFixer RGB aspect ratio disagrees with saved cameras')
            image.convert('RGB').resize(size, Image.Resampling.LANCZOS).save(root/'scene/images'/name)
        camera = calibration['frames'][camera_index]
        pose = np.asarray(camera['transform_matrix'], dtype=float)
        if pose.shape != (4,4) or not np.isfinite(pose).all():
            raise ValueError('Invalid ArtiFixer camera pose')
        return {'file_path':'images/'+name, 'w':size[0], 'h':size[1],
                **{k:float(camera.get(k, calibration[k]))*size[axis]/calibration['wh'[axis]]
                   for k,axis in [('fl_x',0),('fl_y',1),('cx',0),('cy',1)]},
                'transform_matrix':(pose@AXES).tolist()}
    for i, source in enumerate(references):
        name = f'ref_{i:06d}.png'
        frames.append(stage(source, anchors[i]['frame_index'], name))
        views.append({'image':'scene/images/'+name, 'view_id':anchors[i]['view_id'], 'source_sha256':digest(source)})
    for index, source in sorted(predictions.items()):
        name = f'target_{index:06d}.png'
        frames.append(stage(source, index, name))
        reference = min(range(6), key=lambda i: np.linalg.norm(np.asarray(frames[i]['transform_matrix'])[:3,3]-np.asarray(frames[-1]['transform_matrix'])[:3,3]))
        targets.append({'target_index':index, 'image_name':name, 'reference_index':reference,
                        'segment_index':reference, 'alpha':0.0, 'source_sha256':digest(source)})
    options = validate_options({'oracle_reference_mode':'all', 'oracle_adapter':'saved_poses',
                                'oracle_initialization':initialization,
                                'oracle_depth_source':'source_splat' if initialization == 'source_splat' else 'predicted'})
    manifest = {**options, 'mode':'edited', 'views':views, 'resolution':list(size),
                'seva_internal_resolution':list(size), 'seva_task':'img2trajvid',
                'artifixer_source':{'run_id':row['run_id'], 'result_id':row['id'], 'generated_count':len(targets)},
                'trajectory':{'frames':targets, 'segments':[]}, 'camera_convention':'opengl_c2w'}
    if initialization == 'source_splat':
        source = row['_paths']['original']
        if source is None:
            raise ValueError('Original starting PLY is unavailable; select from scratch')
        count, degree = inspect_ply(source)
        shutil.copyfile(source, root/'source-initialization.ply')
        manifest['source_initialization'] = {'path':'source-initialization.ply', 'sha256':digest(root/'source-initialization.ply'), 'gaussian_count':count, 'sh_degree':degree}
        manifest['source_splat_used_for_initialization'] = True
        manifest['source_depth'] = stage_saved_pose_depth(root, frames[:len(views)])
    atomic_json(root/'scene/transforms.json', {'frames':frames})
    atomic_json(root/'inputs.json', manifest)
    return manifest


def publish_generated(inputs, output, manifest):
    """Supply the existing assembly boundary without invoking SEVA."""
    destination = Path(output)/'seva'/manifest['seva_task']/'scene'
    shutil.copytree(Path(inputs)/'scene', destination)
