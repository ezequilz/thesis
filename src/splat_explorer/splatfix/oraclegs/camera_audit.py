"""Verify cameras actually parsed by SEVA before either diffusion pass."""
import json
from pathlib import Path
import numpy as np

AXES = np.diag([1., -1., -1., 1.])


def verify_parsed_cameras(data_path, parsed):
    paths, n, count, input_ids, anchor_ids, poses, intrinsics, anchors, anchor_k = parsed
    scene = Path(data_path)/'scene'
    frames = json.loads((scene/'transforms.json').read_text())['frames']
    split = json.loads((scene/f'train_test_split_{n}.json').read_text())
    ordered = [frames[i] for i in split['train_ids']+split['test_ids']]
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    expected = np.asarray([f['transform_matrix'] for f in ordered]) @ AXES
    expected_k = np.asarray([[[f['fl_x'],0,f['cx']],[0,f['fl_y'],f['cy']],[0,0,1]] for f in ordered])
    if count != len(split['test_ids']) or list(input_ids) != list(range(n)):
        raise ValueError('SEVA reference/target split changed before inference')
    if len(paths) != len(ordered) or any(Path(p).resolve() != (scene/f['file_path']).resolve()
                                       for p,f in zip(paths,ordered)):
        raise ValueError('SEVA reference image/camera pairing changed before inference')
    def check(actual, wanted, label):
        actual = array(actual)
        if actual.shape != wanted.shape or not np.allclose(actual,wanted,rtol=0,atol=2e-5):
            raise ValueError(f'SEVA {label} changed before inference')
    check(poses, expected[:,:3], 'camera poses')
    check(intrinsics, expected_k, 'camera intrinsics')
    indices = np.linspace(0,count-1,len(anchor_ids)).round().astype(int)
    check(anchors,expected[n+indices,:3], 'anchor poses')
    check(anchor_k,expected_k[n+indices], 'anchor intrinsics')
    return {'passed':True, 'reference_count':n, 'target_count':count,
            'anchor_target_indices':indices.tolist(),
            'target_center_extent':np.ptp(expected[n:,:3,3],axis=0).tolist(),
            'scope':'Actual SEVA parser outputs before inference; not generated-image geometric accuracy.'}


def install_camera_audit(demo, data_path, output):
    original = demo.parse_task
    def parse(*args, **kwargs):
        parsed = original(*args, **kwargs)
        report = verify_parsed_cameras(data_path, parsed)
        Path(output).write_text(json.dumps(report,indent=2))
        return parsed
    demo.parse_task = parse
