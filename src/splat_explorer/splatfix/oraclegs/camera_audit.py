"""Verify cameras actually parsed by SEVA before either diffusion pass."""
import json
from pathlib import Path
import numpy as np

AXES = np.diag([1., -1., -1., 1.])


def install_camera_only_inputs(demo):
    """Use SEVA's native None-image support for explicitly camera-only frames.

    Keep output names in our metadata for reconstruction assembly. Legacy
    staging with actual target PNGs continues to use the upstream parser.
    """
    original = demo.get_parser
    def get_parser(*args, **kwargs):
        parser = original(*args, **kwargs)
        scene = Path(kwargs['data_dir'])
        frames = json.loads((scene/'transforms.json').read_text())['frames']
        references = {i for split in parser.splits_per_num_input_frames.values()
                      for i in split['train_ids']}
        for i, frame in enumerate(frames):
            if frame.get('camera_only', False):
                if i in references:
                    raise ValueError('SEVA reference cannot be camera-only')
                parser.image_paths[i] = None
                parser.image_names[i] = None
            elif not Path(parser.image_paths[i]).is_file():
                raise FileNotFoundError(f'Missing SEVA image: {parser.image_paths[i]}')
        return parser
    demo.get_parser = get_parser


def verify_parsed_cameras(data_path, parsed, task='img2trajvid'):
    paths, n, count, input_ids, anchor_ids, poses, intrinsics, anchors, anchor_k = parsed
    scene = Path(data_path)/'scene'
    frames = json.loads((scene/'transforms.json').read_text())['frames']
    split = json.loads((scene/f'train_test_split_{n}.json').read_text())
    ordered = frames if task == 'img2vid' else [frames[i] for i in split['train_ids']+split['test_ids']]
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    expected = np.asarray([f['transform_matrix'] for f in ordered]) @ AXES
    expected_k = np.asarray([[[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1]] for f in ordered])
    expected_inputs = split['train_ids'] if task == 'img2vid' else list(range(n))
    if count != len(split['test_ids']) or list(input_ids) != expected_inputs:
        raise ValueError('SEVA reference/target split changed before inference')
    if len(paths) != len(ordered) or any((p is not None if f.get('camera_only', False) else
                                        p is None or Path(p).resolve() != (scene/f['file_path']).resolve())
                                       for p,f in zip(paths,ordered)):
        raise ValueError('SEVA reference image/camera pairing changed before inference')
    def check(actual, wanted, label):
        actual = array(actual)
        if actual.shape != wanted.shape or not np.allclose(actual,wanted,rtol=0,atol=2e-5):
            raise ValueError(f'SEVA {label} changed before inference')
    check(poses, expected[:,:3], 'camera poses')
    check(intrinsics, expected_k, 'camera intrinsics')
    indices = np.linspace(0,count-1,len(anchor_ids)).round().astype(int)
    native_indices = np.asarray(anchor_ids, dtype=int) if task == 'img2vid' else n+indices
    check(anchors,expected[native_indices,:3], 'anchor poses')
    check(anchor_k,expected_k[native_indices], 'anchor intrinsics')
    return {'passed':True, 'reference_count':n, 'target_count':count,
            'task':task, 'reference_frame_indices':expected_inputs,
            'anchor_frame_indices':native_indices.tolist(),
            'anchor_target_indices':indices.tolist() if task != 'img2vid' else None,
            'target_center_extent':np.ptp(expected[split['test_ids'] if task == 'img2vid' else list(range(n,len(expected))),:3,3],axis=0).tolist(),
            'scope':'Actual SEVA parser outputs before inference; not generated-image geometric accuracy.'}


def install_camera_audit(demo, data_path, output, task='img2trajvid'):
    original = demo.parse_task
    def parse(*args, **kwargs):
        parsed = original(*args, **kwargs)
        report = verify_parsed_cameras(data_path, parsed, task)
        Path(output).write_text(json.dumps(report,indent=2))
        return parsed
    demo.parse_task = parse


def verify_conditioning(before, intrinsics, value):
    """Check the actual diffusion cameras, allowing SEVA's shared similarity.

    Centering and uniform positive scaling are legitimate; losing relative
    translation, rotating a camera, or changing calibration is not.
    """
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    before = array(before).astype(float)
    after = array(value['c2w']).astype(float)
    if (after.shape != (len(before), 4, 4) or not np.isfinite(after).all()
            or not np.allclose(after[:, :3, :3], before[:, :3, :3], atol=2e-5, rtol=0)):
        raise ValueError('SEVA diffusion camera rotations changed')
    source = before[:, :3, 3] - before[0, :3, 3]
    actual = after[:, :3, 3] - after[0, :3, 3]
    denom = float(np.sum(source**2))
    scale = float(np.sum(source*actual)/denom) if denom > 1e-16 else None
    if scale is not None:
        if scale <= 0 or not np.allclose(actual, source*scale, atol=2e-5, rtol=2e-5):
            raise ValueError('SEVA diffusion lost or changed relative camera translation')
    elif not np.allclose(actual, 0, atol=2e-5):
        raise ValueError('SEVA diffusion introduced relative camera translation')
    k = array(intrinsics)
    if array(value['K']).shape != k.shape or not np.allclose(array(value['K']), k, atol=1e-6, rtol=0):
        raise ValueError('SEVA diffusion intrinsics changed')
    rays = array(value['plucker_coordinate'])
    if rays.ndim != 4 or rays.shape[:2] != (len(before), 6) or not np.isfinite(rays).all():
        raise ValueError('SEVA diffusion ray encoding is invalid')
    # Rays are expressed in the first camera's coordinate frame. Their moment
    # must be origin cross direction; zeroing moments erases translation even
    # if the separately exported pose metadata still looks correct.
    centers = actual @ after[0, :3, :3]
    directions = np.moveaxis(rays[:, :3], 1, -1)
    if not np.allclose(np.linalg.norm(directions, axis=-1), 1, atol=2e-4, rtol=0):
        raise ValueError('SEVA diffusion ray directions are not unit vectors')
    moments = np.moveaxis(rays[:, 3:], 1, -1)
    expected = np.cross(centers[:, None, None, :], directions)
    if not np.allclose(moments, expected, atol=2e-4, rtol=2e-4):
        raise ValueError('SEVA diffusion ray moments disagree with camera translation')
    return {'camera_count':len(before), 'translation_scale':scale,
            'source_relative_extent':np.ptp(source,axis=0).tolist(),
            'normalized_relative_extent':np.ptp(actual,axis=0).tolist()}


def install_conditioning_audit(evaluation, output):
    """Observe both upstream passes without changing camera or model behavior."""
    import inspect
    from functools import wraps
    original = evaluation.get_value_dict
    signature = inspect.signature(original)
    report = {'passed':True, 'chunks':[],
              'scope':'Diffusion conditioning camera similarity and Plucker translation; not pixel geometry.'}
    @wraps(original)
    def checked(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        # Clone because the original may use tensor views during normalization.
        poses = arguments['curr_c2ws'].detach().cpu().numpy().copy()
        intrinsics = arguments['curr_Ks'].detach().cpu().numpy().copy()
        value = original(*args, **kwargs)
        try:
            report['chunks'].append(verify_conditioning(poses, intrinsics, value))
        except ValueError as exc:
            report.update(passed=False, error=str(exc))
            raise
        finally:
            path = Path(output)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(path)
        return value
    evaluation.get_value_dict = checked
