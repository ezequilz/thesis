"""Stage saved images and calibrated cameras without rendering or image blending."""
import hashlib
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .trajectory import keyframe_path
from ..checkpoint import atomic_json, camera_from_record
from .backend import readiness, validate_options

SIZES = {'576p': (1024, 576), '720p': (1280, 720)}
AXES = np.diag([1., -1., -1., 1.])


def fit_reference(image, camera, size, mode, source_box=None):
    """Image-space affine applied identically to pixels and intrinsics."""
    width, height = size
    x0,y0,x1,y1 = source_box or (0.,0.,float(image.width),float(image.height))
    iw,ih = x1-x0,y1-y0
    k = camera.intrinsics.astype(float).copy()
    k[0] *= iw / camera.width
    k[1] *= ih / camera.height
    k[0,2] += x0
    k[1,2] += y0
    if mode == 'crop':
        scale = max(width/iw, height/ih)
        left = x0+(iw-width/scale)/2
        top = y0+(ih-height/scale)/2
        output = image.resize(size, Image.Resampling.LANCZOS,
                              box=(left,top,left+width/scale,top+height/scale))
        affine = np.array([[scale,0,-left*scale],[0,scale,-top*scale],[0,0,1.]])
    else:
        scale = min(width/iw, height/ih)
        resized = (max(2,round(iw*scale/2)*2),max(2,round(ih*scale/2)*2))
        left,top = (width-resized[0])//2,(height-resized[1])//2
        output = Image.new('RGB',size)
        output.paste(image.resize(resized,Image.Resampling.LANCZOS,box=(x0,y0,x1,y1)),(left,top))
        affine = np.array([[resized[0]/iw,0,left-x0*resized[0]/iw],[0,resized[1]/ih,top-y0*resized[1]/ih],[0,0,1.]])
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
    ordered_path = options['oracle_adapter'] == 'trajectory' and set(order) == set(range(len(cameras)))
    requested = SIZES[options['seva_resolution']]
    width, height = requested
    # Keep the requested framing/resolution. The diffusion stride is handled by
    # temporary padding, not by cropping 720p references down to 704 pixels.
    root = Path(destination)
    scene = root / 'scene'
    (scene / 'images').mkdir(parents=True, exist_ok=False)
    frames, records = [], []
    for i, (view, camera) in enumerate(zip(checkpoint.views, cameras)):
        if should_stop():
            raise InterruptedError('Cancelled during OracleGS staging')
        source = checkpoint.image_path(view, repaired=mode == 'edited')
        source_kind, source_box = 'saved_rgb', None
        repair = view.get('image_repair') or {}
        if mode == 'edited' and repair.get('response_image'):
            source = checkpoint.image_path({'original_rgb':repair['response_image']})
            source_kind = 'full_repair_response'
            resize = repair.get('resize') or {}
            source_box = resize.get('source_crop_box')
            if (resize.get('method') != 'cover_center_crop' or
                    repair.get('target_size') != [camera.width,camera.height] or
                    not isinstance(source_box,list) or len(source_box) != 4):
                raise ValueError('Saved full repair response lacks a calibrated crop mapping')
        name = f'ref_{i:06d}.png'
        with Image.open(source) as image:
            if source_box is not None:
                box = np.asarray(source_box,dtype=float)
                scale = float((repair.get('resize') or {}).get('scale',0))
                if (not np.isfinite(box).all() or scale <= 0 or not np.isfinite(scale) or
                        repair.get('response_size') != list(image.size) or
                        not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height) or
                        not np.allclose((box[2:]-box[:2])*scale,[camera.width,camera.height],rtol=0,atol=1e-4)):
                    raise ValueError('Saved repair response crop disagrees with camera calibration')
            elif abs(image.width/image.height-camera.width/camera.height) > 1/max(image.size):
                raise ValueError('Repaired RGB aspect ratio differs from saved camera; refusing to distort calibration')
            iw,ih = (source_box[2]-source_box[0],source_box[3]-source_box[1]) if source_box else image.size
            fit_scale = (max if options['seva_fit'] == 'crop' else min)(width/iw, height/ih)
            if fit_scale > 1:
                raise ValueError('OracleGS requires saved reference RGBs at least as large as the generation preset')
            fitted, k, affine = fit_reference(image.convert('RGB'),camera,(width,height),options['seva_fit'],source_box=source_box)
            fitted.save(scene/'images'/name)
        frames.append({'file_path':'images/'+name,'w':width,'h':height,
                       'fl_x':float(k[0,0]),'fl_y':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),
                       'transform_matrix':(camera.c2w.astype(float)@AXES).tolist()})
        records.append({'view_id':view['id'], 'image': 'scene/images/'+name,
                        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'sha256':hashlib.sha256((scene/'images'/name).read_bytes()).hexdigest(),
                        'camera':view['camera'], 'source_image':str(source),
                        'source_kind':source_kind, 'source_crop_box':source_box,
                        'image_affine':affine.tolist(), 'staged_intrinsics':k.tolist()})
    # Only camera poses are interpolated here. SEVA owns both generative passes.
    if options['oracle_adapter'] == 'saved_poses':
        samples = [(min(j,len(order)-2),j,float(j == len(order)-1),i,
                    cameras[i].c2w.astype(float)) for j,i in enumerate(order)]
    else:
        samples = keyframe_path(cameras, order, options['oracle_target_frames'],
                                include_endpoints=not ordered_path)
    target_records, used_names = [], set()
    for index, (a, local_index, alpha, reference, pose) in enumerate(samples):
        if should_stop():
            raise InterruptedError('Cancelled during OracleGS trajectory staging')
        left, right = frames[order[a]], frames[order[a+1]]
        name = f'ref_{reference:06d}.png' if reference is not None else f'target_{index:06d}.png'
        # A shared endpoint is generated once for each adjacent segment. Never
        # overwrite either sample during OracleGS dataset assembly.
        if name in used_names:
            name = f'target_{index:06d}.png'
        used_names.add(name)
        target_records.append({'target_index':index, 'segment_index':a,
            'segment_frame_index':local_index, 'alpha':alpha,
            'reference_index':reference, 'image_name':name,
            'position':pose[:3,3].tolist()})
        # Placeholders satisfy ReconfusionParser; never used as training labels.
        if not (scene/'images'/name).exists():
            Image.new('RGB',(width,height)).save(scene/'images'/name)
        frames.append({'file_path':'images/'+name, 'w':width, 'h':height,
                       **{key:(1-alpha)*left[key]+alpha*right[key] for key in ('fl_x','fl_y','cx','cy')},
                       'transform_matrix':(pose@AXES).tolist()})
    n = len(records)
    atomic_json(scene/'transforms.json', {'frames':frames})
    atomic_json(scene/f'train_test_split_{n}.json', {'train_ids':list(range(n)), 'test_ids':list(range(n,len(frames)))})
    native_order = list(range(len(frames)))
    if ordered_path:
        native_order = []
        for segment, reference in enumerate(order[:-1]):
            native_order.append(reference)
            native_order.extend(n+t['target_index'] for t in target_records if t['segment_index'] == segment)
        native_order.append(order[-1])
        native = root/'ordered/scene'
        shutil.copytree(scene, native)
        atomic_json(native/'transforms.json', {'frames':[frames[i] for i in native_order]})
        atomic_json(native/f'train_test_split_{n}.json', {
            'train_ids':[i for i, original in enumerate(native_order) if original < n],
            'test_ids':[i for i, original in enumerate(native_order) if original >= n]})
    manifest = {'seva_task':'img2vid' if ordered_path else 'img2trajvid',
                'seva_frame_order':native_order, 'reconstruction_method':'oraclegs', 'mode':mode, 'views':records,
                'source_checkpoint':str(checkpoint.root), 'camera_convention':'opengl_c2w',
                'source_camera_convention':'opencv_c2w', 'resolution':[width,height], 'requested_resolution':list(requested),
                **options, 'oracle_path_order':order, 'path_order_source':'explicit' if options['oracle_path_order'] else 'checkpoint-selection-order',
                'seva_internal_resolution':[width, ((height+63)//64)*64],
                'resolution_adapter':('vertical-padding-then-crop' if height % 64 else 'calibrated-center-'+options['seva_fit']),
                'reference_role':'ground_truth', 'source_splat_used_for_initialization':False}
    manifest['trajectory'] = {
        'version':3, 'interpolation':('seva-gui-kochanek-bartels-natural-tension-0'
            if options['oracle_adapter'] == 'trajectory' else 'saved-poses'),
        'frame_count_semantics':('per-segment-interior-targets' if ordered_path else
                                 'per-segment-including-both-endpoints'),
        'reference_conditioning':('ordered-native-keyframes' if ordered_path else 'unordered-reference-set'),
        'frames_per_segment':options['oracle_target_frames'] if options['oracle_adapter'] == 'trajectory' else None,
        'target_count':len(target_records), 'closed':False,
        'segments':[{'index':j, 'from_reference':a, 'to_reference':b,
                     'translation_distance':float(np.linalg.norm(cameras[b].position-cameras[a].position))}
                    for j,(a,b) in enumerate(zip(order[:-1],order[1:]))],
        'frames':target_records}
    atomic_json(root/'inputs.json',manifest)
    return manifest
