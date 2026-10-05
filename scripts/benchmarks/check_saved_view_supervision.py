#!/usr/bin/env python3
"""CPU contract check using real pinned authors' COLMAP materialization.

Example (inside the already provisioned authors environment):
  python check_saved_view_supervision.py --repo /workspace/third_party/ArtiFixer \
    --worker /workspace/diagnostics/official_worker.py --output /workspace/diagnostics/supervision-new

The output directory must not exist. No training, rendering, model loading,
network requests, or changes to existing runs are performed. The only intercepted
function is threedgrut_training.train_3dgrut: its caller and materialization run
unchanged, then this boundary records the command and stops before training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

PINNED_REVISION = 'a392c4dfe17459ef9952407accdb9fcdcdddba98'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(root, loops):
    import struct
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    root.mkdir()
    predictions = root / 'inference/splatfix/frames/batch_0000/pred'
    predictions.mkdir(parents=True)
    frames, anchors, references = [], [], []
    rotation = Rotation.from_euler('xyz', [17., -29., 11.], degrees=True).as_matrix()
    for loop in range(loops):
        segment = []
        for angle in np.linspace(0., 2. * np.pi, 9):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [1. + 3. * loop + .1 * np.sin(angle), 2. + .05 * (1. - np.cos(angle)), 3.]
            segment.append({'transform_matrix': pose.tolist()})
        segment[-1] = json.loads(json.dumps(segment[0]))
        anchors.append({'frame_index': len(frames), 'view_id': str(loop)})
        frames.extend(segment)
        reference = root / f'reference_{loop}.png'
        Image.new('RGB', (32, 32), (240, 10 + loop, 5)).save(reference)
        references.append(str(reference))
    for index in range(len(frames)):
        # Every prediction is distinguishable from every saved anchor.
        Image.new('RGB', (32, 32), (index + 1, 80, 120)).save(predictions / f'{index:05d}.png')
    points = root / 'points3D.bin'
    with points.open('wb') as stream:
        stream.write(struct.pack('<Q', 1))
        stream.write(struct.pack('<QdddBBBdQ', 1, 1., 2., 4., 128, 64, 32, 0., 0))
    trajectory = {'camera_convention': 'opencv_c2w', 'anchors': anchors,
                  'transforms': {'camera_model': 'OPENCV', 'w': 32, 'h': 32,
                                 'fl_x': 28., 'fl_y': 28., 'cx': 16., 'cy': 16., 'frames': frames}}
    request = {'seed':42, 'references':references, 'source_points3d':str(points), 'camera_scale':1.}
    return request, trajectory


def check_case(root, loops, worker, official):
    import numpy as np
    request, trajectory = fixture(root, loops)
    calls = []

    class StopBeforeTraining(Exception):
        pass

    def stop_training(config_name, overrides, config_dir):
        calls.append({'config_name':config_name, 'overrides':overrides, 'config_dir':str(config_dir)})
        raise StopBeforeTraining()

    original_train = official.threedgrut_training.train_3dgrut
    official.threedgrut_training.train_3dgrut = stop_training
    try:
        try:
            worker.distill(root, request, trajectory)
        except StopBeforeTraining:
            pass
        else:
            raise AssertionError('Expected the real authors materialization to reach the training boundary')
    finally:
        official.threedgrut_training.train_3dgrut = original_train
    assert len(calls) == 1
    call = calls[0]
    assert call['config_name'] == 'apps/colmap_3dgut_sparse_mcmc_lpips'
    assert 'n_iterations=30000' in call['overrides']
    assert not any(value.startswith('resume=') for value in call['overrides'])
    dataset = root / 'artifixer3d/distillation_input/splatfix'
    images, cameras = official.source_colmap_lookup(dataset)
    assert len(images) == len(cameras) == 8 * loops
    selected = json.loads((root / 'artifixer3d/distillation_input/splatfix_selected_indices.json').read_text())
    assert selected == list(range(0, 8 * loops, 8))
    mapping = json.loads((root / 'supervision.json').read_text())
    assert mapping['original_to_distillation'] == [8 * loop + i for loop in range(loops) for i in (0,1,2,3,4,5,6,7,0)]
    errors, reconstructed_c2ws = [], []
    for unique_index, original_index in enumerate(mapping['distillation_to_original']):
        matches = [image for image in images.values() if image.id == unique_index + 1]
        assert len(matches) == 1
        image = matches[0]
        w2c = np.eye(4)
        w2c[:3, :3] = official.colmap_rotation_from_qvec(image.qvec).as_matrix()
        w2c[:3, 3] = image.tvec
        expected = np.asarray(trajectory['transforms']['frames'][original_index]['transform_matrix'])
        actual = np.linalg.inv(w2c)
        error = float(np.max(np.abs(actual - expected)))
        errors.append(error)
        assert error < 1e-10
        reconstructed_c2ws.append(actual)
        camera = cameras[image.camera_id]
        assert (camera.width, camera.height) == (32,32)
        np.testing.assert_allclose(camera.params, [28.,28.,16.,16.,0.,0.,0.,0.], atol=0, rtol=0)
        target = dataset / 'images' / image.name
        if unique_index in selected:
            reference_index = selected.index(unique_index)
            expected_image = Path(request['references'][reference_index])
        else:
            expected_image = root / 'inference/splatfix/frames/batch_0000/pred' / f'{original_index:05d}.png'
            assert (dataset / 'artifixer_predictions' / f'{unique_index:05d}.png').resolve() == expected_image.resolve()
        assert target.resolve() == expected_image.resolve()
        assert digest(target) == digest(expected_image)
    duplicate_pairs = [(i,j) for i in range(len(reconstructed_c2ws)) for j in range(i)
                       if np.allclose(reconstructed_c2ws[i], reconstructed_c2ws[j], atol=1e-10, rtol=0)]
    assert duplicate_pairs == []
    render = json.loads((root / 'render_transforms_opengl.json').read_text())
    assert len(render['frames']) == 9 * loops
    for index, frame in enumerate(render['frames']):
        actual_w2c = official.opengl_c2w_to_opencv_w2c(np.asarray(frame['transform_matrix']))
        expected_w2c = np.linalg.inv(trajectory['transforms']['frames'][index]['transform_matrix'])
        np.testing.assert_allclose(actual_w2c, expected_w2c, atol=1e-10, rtol=0)
    assert (dataset / 'sparse/0/points3D.bin').read_bytes() == Path(request['source_points3d']).read_bytes()
    assert not (root / 'artifixer3d.ply').exists()
    return {'loops':loops, 'input_frames':9*loops, 'unique_training_cameras':len(images),
            'saved_reference_targets':len(selected), 'generated_targets':len(images)-len(selected),
            'full_render_trajectory_frames':len(render['frames']), 'duplicate_training_pose_pairs':duplicate_pairs,
            'maximum_pose_error':max(errors), 'training_boundary':call,
            'supervision_manifest':str(root / 'supervision.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # Must precede all real authors/torch imports. No device is visible here.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    repo, worker_path, output = args.repo.resolve(), args.worker.resolve(), args.output.resolve()
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if revision != PINNED_REVISION:
        raise ValueError(f'Expected pinned authors revision {PINNED_REVISION}, got {revision}')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(repo / 'thirdparty/3DGRUT-ArtiFixer'))
    spec = importlib.util.spec_from_file_location('diagnostic_saved_view_worker', worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    from data_processing import artifixer3d as official
    import torch
    assert Path(official.__file__).resolve() == repo / 'data_processing/artifixer3d.py'
    assert not torch.cuda.is_initialized()
    report = {'upstream_revision':revision, 'worker_path':str(worker_path), 'worker_sha256':digest(worker_path),
              'intercepted_function':'data_processing.threedgrut_training.train_3dgrut',
              'authors_materialization_sha256':digest(official.__file__),
              'cases':[check_case(output / f'loops-{loops}', loops, worker, official) for loops in (1,2)],
              'training_executed':False, 'rendering_executed':False, 'models_loaded':False}
    assert not torch.cuda.is_initialized()
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
