"""Stage saved images and calibrated cameras without rendering or image blending."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from ..checkpoint import atomic_json, camera_from_record
from .backend import readiness, validate_options

SIZES = {'576p': (1024, 576), '720p': (1280, 720)}
AXES = np.diag([1., -1., -1., 1.])


def fit_reference(image, camera, size, mode):
    """Image-space affine applied identically to pixels and intrinsics."""
    width, height = size
    k = camera.intrinsics.astype(float).copy()
    k[0] *= image.width / camera.width
    k[1] *= image.height / camera.height
    if mode == 'crop':
        scale = max(width/image.width, height/image.height)
        left = (image.width-width/scale)/2
        top = (image.height-height/scale)/2
        output = image.resize(size, Image.Resampling.LANCZOS,
                              box=(left,top,left+width/scale,top+height/scale))
        affine = np.array([[scale,0,-left*scale],[0,scale,-top*scale],[0,0,1.]])
    else:
        scale = min(width/image.width, height/image.height)
        resized = (max(2,round(image.width*scale/2)*2),max(2,round(image.height*scale/2)*2))
        left,top = (width-resized[0])//2,(height-resized[1])//2
        output = Image.new('RGB',size)
        output.paste(image.resize(resized,Image.Resampling.LANCZOS),(left,top))
        affine = np.array([[resized[0]/image.width,0,left],[0,resized[1]/image.height,top],[0,0,1.]])
    return output, affine@k, affine


def export_inputs(checkpoint, destination, mode='edited', *, runtime=None, should_stop=lambda: False):
    options = validate_options(runtime or {})
    state = readiness(checkpoint)
    if not state['ready']:
        raise ValueError(state['reason'])
    if mode not in ('edited', 'baseline'):
        raise ValueError('mode must be edited or baseline')
    if mode == 'edited':
        checkpoint.require_viser_images()
    cameras = [camera_from_record(v) for v in checkpoint.views]
    order = options['oracle_path_order']
    if order is None:
        order = list(range(len(cameras)))
    if any(i >= len(cameras) for i in order):
        raise ValueError('oracle_path_order references a missing checkpoint view')
    requested = SIZES[options['seva_resolution']]
    width, height = requested
    if options['seva_fit'] == 'crop':
        width, height = width//64*64, height//64*64
    root = Path(destination)
    scene = root / 'scene'
    (scene / 'images').mkdir(parents=True, exist_ok=False)
    frames, records = [], []
    for i, (view, camera) in enumerate(zip(checkpoint.views, cameras)):
        if should_stop():
            raise InterruptedError('Cancelled during OracleGS staging')
        source = checkpoint.image_path(view, repaired=mode == 'edited')
        name = f'ref_{i:06d}.png'
        with Image.open(source) as image:
            if abs(image.width/image.height-camera.width/camera.height) > 1/max(image.size):
                raise ValueError('Repaired RGB aspect ratio differs from saved camera; refusing to distort calibration')
            fit_scale = (max if options['seva_fit'] == 'crop' else min)(width/image.width, height/image.height)
            if fit_scale > 1:
                raise ValueError('OracleGS requires saved reference RGBs at least as large as the generation preset')
            fitted, k, affine = fit_reference(image.convert('RGB'),camera,(width,height),options['seva_fit'])
            fitted.save(scene/'images'/name)
        frames.append({'file_path':'images/'+name,'w':width,'h':height,
                       'fl_x':float(k[0,0]),'fl_y':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),
                       'transform_matrix':(camera.c2w.astype(float)@AXES).tolist()})
        records.append({'view_id':view['id'], 'image': 'scene/images/'+name,
                        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'sha256':hashlib.sha256((scene/'images'/name).read_bytes()).hexdigest(),
                        'camera':view['camera'], 'source_image':str(source),
                        'image_affine':affine.tolist(), 'staged_intrinsics':k.tolist()})
    # Only camera poses are interpolated here. SEVA owns both generative passes.
    keys = np.arange(len(order), dtype=float)
    rotations = Slerp(keys, Rotation.from_matrix(np.stack([cameras[i].rotation for i in order])))
    if options['oracle_adapter'] == 'saved_poses':
        samples = keys
    else:
        if options['oracle_target_frames'] < len(order):
            raise ValueError('Target frame count must include every saved path pose')
        # Allocate samples to segments while including every exact saved pose.
        counts = np.full(len(order)-1, (options['oracle_target_frames']-1)//(len(order)-1))
        counts[:(options['oracle_target_frames']-1) % (len(order)-1)] += 1
        samples = np.concatenate([np.linspace(i,i+1,int(count)+1)[:-1] for i,count in enumerate(counts)] + [keys[-1:]])
    for index, t in enumerate(samples):
        a = min(int(t), len(order)-2)
        alpha = t-a
        left, right = frames[order[a]], frames[order[a+1]]
        pose = np.eye(4)
        pose[:3,:3] = rotations(t).as_matrix()
        pose[:3,3] = (1-alpha)*cameras[order[a]].position + alpha*cameras[order[a+1]].position
        name = f'ref_{order[int(t)]:06d}.png' if float(t).is_integer() else f'target_{index:06d}.png'
        # Placeholders satisfy ReconfusionParser; never used as training labels.
        if not (scene/'images'/name).exists():
            Image.new('RGB',(width,height)).save(scene/'images'/name)
        frames.append({'file_path':'images/'+name, 'w':width, 'h':height,
                       **{key:(1-alpha)*left[key]+alpha*right[key] for key in ('fl_x','fl_y','cx','cy')},
                       'transform_matrix':(pose@AXES).tolist()})
    n = len(records)
    atomic_json(scene/'transforms.json', {'frames':frames})
    atomic_json(scene/f'train_test_split_{n}.json', {'train_ids':list(range(n)), 'test_ids':list(range(n,len(frames)))})
    manifest = {'reconstruction_method':'oraclegs', 'mode':mode, 'views':records,
                'source_checkpoint':str(checkpoint.root), 'camera_convention':'opengl_c2w',
                'source_camera_convention':'opencv_c2w', 'resolution':[width,height], 'requested_resolution':list(requested),
                **options, 'oracle_path_order':order, 'path_order_source':'explicit' if options['oracle_path_order'] else 'checkpoint-selection-order',
                'seva_internal_resolution':[width, ((height+63)//64)*64],
                'resolution_adapter':('vertical-padding-then-crop' if height % 64 else 'calibrated-center-'+options['seva_fit']),
                'reference_role':'ground_truth', 'source_splat_used_for_initialization':False}
    atomic_json(root/'inputs.json',manifest)
    return manifest
