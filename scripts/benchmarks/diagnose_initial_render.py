"""Compare native COLMAP and source-trajectory renders of an existing checkpoint.

Read-only inputs; writes only a newly allocated diagnostic directory. No training.
Run in the authors' GPU environment, passing --repo to its pinned checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'checkpoint', 'dataset', 'trajectory', 'baseline-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 2])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Diagnostic output must be a new directory')
    for source in (args.dataset, args.baseline_dir, args.checkpoint.parent):
        if args.output.is_relative_to(source.resolve()):
            raise ValueError('Diagnostic output must be outside preserved input directories')
    sys.path[:0] = [str(args.repo), str(args.repo / 'thirdparty/3DGRUT-ArtiFixer')]
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data._utils.collate import default_collate
    from threedgrut.render import Renderer

    trajectory = json.loads(args.trajectory.read_text())
    args.output.mkdir(parents=True)
    report = {'purpose': 'Existing initial fit: native PINHOLE versus trajectory OPENCV; no training',
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest(args.checkpoint),
              'dataset': str(args.dataset.resolve()), 'trajectory': str(args.trajectory.resolve()),
              'trajectory_sha256': digest(args.trajectory), 'script_sha256': digest(__file__),
              'upstream_revision': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'native_expectation': 'COLMAP binary camera/pose, PINHOLE center=(width/2,height/2), native pixel dimensions',
              'trajectory_expectation': 'Official transforms_pose_to_c2w and add_opencv_intrinsics_from_mapping, explicit stored principal point',
              'frames': []}
    renderer = Renderer.from_checkpoint(
        checkpoint_path=args.checkpoint, path=str(args.dataset), out_dir=str(args.output / 'official'),
        save_gt=False, computes_extra_metrics=False,
        config_overrides={'path': str(args.dataset), 'experiment_name': 'initial_render_diagnostic',
                          'selected_indices_file': None, 'train_test_split_file': None,
                          'image_path_override': None, 'dataset.test_split_interval': 0,
                          'use_wandb': False})
    renderer.writer = None
    report['global_step'] = int(renderer.global_step)
    if report['global_step'] != 10000:
        raise ValueError('Expected the initial 10,000-step checkpoint')
    dataset = renderer.dataset

    def score(a, b):
        if a.shape != b.shape:
            raise ValueError(f'No implicit resize allowed: {a.shape} != {b.shape}')
        difference = a.astype(np.float64) - b.astype(np.float64)
        mse = float(np.mean(difference ** 2))
        return {'psnr_db': None if mse == 0 else float(-10 * np.log10(mse)),
                'mse': mse, 'mean_absolute_error': float(np.mean(np.abs(difference))),
                'max_absolute_error': float(np.max(np.abs(difference)))}

    def render(batch, output):
        with torch.no_grad():
            rgb = renderer.model(dataset.get_gpu_batch_with_intrinsics(batch))['pred_rgb'][0]
        rgb = rgb.detach().clamp(0, 1).cpu().numpy()
        Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(output)
        return rgb

    for index in args.frames:
        frame = trajectory['frames'][index]
        native = default_collate([dataset[index]])
        image_path = Path(dataset.image_paths[index])
        if image_path.name != Path(frame['file_path']).name:
            raise ValueError(f'Frame ordering mismatch at {index}')
        intrinsics = renderer._frame_intrinsics(trajectory, frame)
        camera_id = -(index + 1)
        dataset.add_opencv_intrinsics_from_mapping(camera_id, intrinsics)
        pose = dataset.transforms_pose_to_c2w(frame['transform_matrix'])
        converted = renderer._make_template_batch(camera_only=True)
        converted.update(pose=torch.FloatTensor(pose).view(converted['pose'].shape),
                         intr=torch.IntTensor([camera_id]), is_override=[False])
        a = render(native, args.output / f'native-{index:05d}.png')
        b = render(converted, args.output / f'trajectory-{index:05d}.png')
        # Isolate the principal-point/model distinction from pose conversion.
        # This control uses the native camera's calibration and pose through
        # the OPENCV rendering path; it does not alter the benchmark inputs.
        native_id = int(native['intr'][0])
        native_camera = dataset.intrinsics[native_id][0]
        matched_intrinsics = dict(intrinsics)
        matched_intrinsics.update(
            fl_x=float(native_camera['focal_length'][0]),
            fl_y=float(native_camera['focal_length'][1]),
            cx=float(native_camera['principal_point'][0]),
            cy=float(native_camera['principal_point'][1]))
        matched_id = -(len(trajectory['frames']) + index + 1)
        dataset.add_opencv_intrinsics_from_mapping(matched_id, matched_intrinsics)
        matched_batch = dict(converted, pose=native['pose'].clone(),
                             intr=torch.IntTensor([matched_id]))
        c = render(matched_batch, args.output / f'matched-calibration-{index:05d}.png')
        gt = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float64) / 255
        prior_path = args.baseline_dir / f'{index:05d}.png'
        prior = np.asarray(Image.open(prior_path).convert('RGB'), dtype=np.float64) / 255
        def serial(value):
            return value.tolist() if hasattr(value, 'tolist') else str(value)
        report['frames'].append({'index': index, 'name': image_path.name,
            'source_photo_sha256': digest(image_path), 'stored_baseline': str(prior_path.resolve()),
            'stored_baseline_sha256': digest(prior_path), 'native_camera_id': native_id,
            'native_camera': json.loads(json.dumps(dataset.intrinsics[native_id][0], default=serial)),
            'trajectory_camera': {key: value for key, value in intrinsics.items()
                                  if key != 'frames'},
            'native_pose': native['pose'].reshape(4, 4).numpy().tolist(), 'trajectory_pose': pose.tolist(),
            'pose_max_absolute_difference': float(np.max(np.abs(native['pose'].reshape(4, 4).numpy() - pose))),
            'native_vs_trajectory': score(a, b), 'native_vs_ground_truth': score(a, gt),
            'native_vs_matched_calibration': score(a, c),
            'trajectory_vs_ground_truth': score(b, gt), 'trajectory_vs_stored_baseline': score(b, prior),
            'native_vs_stored_baseline': score(a, prior)})
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, default=serial))
        print(json.dumps(report['frames'][-1], default=serial), flush=True)


if __name__ == '__main__':
    main()
