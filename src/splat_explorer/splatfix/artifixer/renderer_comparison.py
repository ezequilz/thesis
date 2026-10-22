"""Isolated renderer ablation on a prepared ArtiFixer request; no reconstruction.

Run with --help in the project environment. GPU rendering and optional inference
use the request's pinned authors environment. Existing inputs are never replaced.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from splat_explorer.splatfix.checkpoint import atomic_json
from splat_explorer.splatfix.artifixer.official_worker import (
    ARTIFIXER3D_CONFIG, opengl_transforms, validate_render_sources,
)
from splat_explorer.splatfix.artifixer.repair import (
    digest_file, prepare_splat_initialization, run_worker, runtime_environment,
    validate_runtime,
)

ARMS = ('current', 'gsplat', '3dgut', 'opacity_only')


def pixel_metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Comparison requires finite, identically sized images; no resizing')
    d = a - b
    mse = float(np.mean(d * d))
    return {'mae': float(np.mean(np.abs(d))), 'mse': mse,
            'max_error': float(np.max(np.abs(d))),
            'psnr_db': float(-10 * np.log10(mse)) if mse else None}


def opacity_png_values(opacity):
    opacity = np.asarray(opacity, dtype=np.float32)
    if not np.isfinite(opacity).all() or (opacity < 0).any() or (opacity > 1).any():
        raise ValueError('Rendered opacity must be finite and in [0, 1]')
    return np.round(opacity * 255).astype(np.uint8)


def calibrated_frames(trajectory):
    """Validate pinhole calibrations shared by the two GPU backends."""
    transforms = opengl_transforms(trajectory)
    for frame in transforms['frames']:
        intr = {**transforms, **frame}
        if intr.get('camera_model', 'OPENCV') not in ('OPENCV', 'PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError('Unsupported camera model for pinhole comparison')
        intr['camera_model'] = 'OPENCV'
        if any(float(intr.get(k, 0)) != 0 for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2')):
            raise ValueError('This renderer comparison requires undistorted pinhole cameras')
        for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'):
            if not np.isfinite(intr[key]):
                raise ValueError(f'Nonfinite camera parameter: {key}')
        if min(intr[k] for k in ('w', 'h', 'fl_x', 'fl_y')) <= 0:
            raise ValueError('Image dimensions and focal lengths must be positive')
        yield intr, np.asarray(frame['transform_matrix'], dtype=np.float64)


def render_pairs(root, request, trajectory, source_ply):
    """Import identical raw PLY parameters into 3DGUT and gsplat; never train."""
    import torch
    import gsplat
    from data_processing import artifixer3d as official
    from data_processing.threedgrut_training import compose_3dgrut_config
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.render import Renderer
    from omegaconf import OmegaConf

    frames = list(calibrated_frames(trajectory))
    # A single reference camera supplies the official dataset's camera adapter.
    # Every rendered camera below is registered from its exact saved calibration.
    source = root / 'camera-adapter'
    sparse = source / 'sparse/0'
    sparse.mkdir(parents=True)
    (source / 'images').mkdir()
    anchor = trajectory['anchors'][0]['frame_index']
    intr, pose = frames[anchor]
    shutil.copyfile(request['references'][0], source / 'images/reference.png')
    camera = official.opencv_camera_from_mapping(1, intr)
    qvec, tvec = official.colmap_pose_from_transforms_frame(
        {'transform_matrix': pose.tolist()}, np.eye(4))
    official.write_colmap_cameras(sparse / 'cameras.bin', [camera])
    official.write_colmap_images(sparse / 'images.bin', [(1, qvec, tvec, 1, 'reference.png')])
    shutil.copyfile(request['source_points3d'], sparse / 'points3D.bin')
    config_dir = Path(request['runtime']['repo']) / 'thirdparty/3DGRUT-ArtiFixer/configs'
    conf = compose_3dgrut_config(ARTIFIXER3D_CONFIG, [], config_dir)
    conf.path = str(source)
    conf.selected_indices_file = None
    conf.train_test_split_file = None
    conf.dataset.downsample_factor = 1
    conf.dataset.test_split_interval = 0
    conf.model.background.color = 'black'
    if conf.render.method != '3dgut':
        raise ValueError('Pinned authors configuration did not select 3DGUT')
    model = MixtureOfGaussians(conf).cuda()
    model.init_from_ply(str(source_ply), init_model=False)
    model.validate_fields()
    model.build_acc()
    renderer = Renderer(model, conf, 0, str(root / 'native'), save_gt=False,
                        compute_extra_metrics=False, save_selected_indices=False)
    atomic_json(root / 'renderer-config.json', OmegaConf.to_container(conf, resolve=False))
    coefficients = torch.cat((model.features_albedo[:, None, :],
                              model.features_specular.reshape(model.num_gaussians, -1, 3)), dim=1)
    for arm in ('gsplat', '3dgut'):
        (root / arm).mkdir()
    last_intrinsics = None
    for index, (intr, pose) in enumerate(frames):
        with torch.no_grad():
            camera_id = -1
            calibration = tuple(intr[k] for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'))
            if calibration != last_intrinsics:
                # Reuse one ray grid rather than retaining a grid per frame.
                renderer.dataset.add_opencv_intrinsics_from_mapping(camera_id, intr)
                last_intrinsics = calibration
            batch = renderer._make_template_batch(camera_only=True)
            native_pose = renderer.dataset.transforms_pose_to_c2w(pose)
            batch.update(pose=torch.FloatTensor(native_pose).view(batch['pose'].shape),
                         intr=torch.IntTensor([camera_id]), is_override=[False])
            out = model(renderer.dataset.get_gpu_batch_with_intrinsics(batch))
            gut_rgb = out['pred_rgb'][0].clamp(0, 1).cpu().numpy()
            gut_alpha = out['pred_opacity'][0].cpu().numpy().squeeze(-1)
            cv_pose = pose @ np.diag([1., -1., -1., 1.])
            K = [[intr['fl_x'], 0, intr['cx']], [0, intr['fl_y'], intr['cy']], [0, 0, 1]]
            rgb, alpha, _ = gsplat.rasterization(
                means=model.positions, quats=model.get_rotation(), scales=model.get_scale(),
                opacities=model.get_density().flatten(), colors=coefficients,
                sh_degree=model.max_n_features,
                viewmats=torch.as_tensor(np.linalg.inv(cv_pose), dtype=torch.float32, device='cuda')[None],
                Ks=torch.tensor(K, dtype=torch.float32, device='cuda')[None],
                width=int(intr['w']), height=int(intr['h']),
                backgrounds=torch.zeros((1, 3), device='cuda'), packed=False)
            splat_rgb = rgb[0].clamp(0, 1).cpu().numpy()
            splat_alpha = alpha[0, :, :, 0].cpu().numpy()
        for arm, color, opacity in (('3dgut', gut_rgb, gut_alpha), ('gsplat', splat_rgb, splat_alpha)):
            if opacity.shape != (intr['h'], intr['w']) or not np.isfinite(opacity).all():
                raise ValueError('Invalid rendered opacity shape or values')
            Image.fromarray(np.round(color * 255).astype(np.uint8)).save(root / arm / f'{index:05d}.png')
            # Both paired arms use the authors' 8-bit export/read convention.
            # Preserve raw alpha separately for quantization diagnostics.
            encoded = opacity_png_values(opacity)
            np.save(root / arm / f'{index:05d}-float.npy', opacity.astype(np.float32))
            np.save(root / arm / f'{index:05d}.npy', encoded.astype(np.float32) / 255)
            Image.fromarray(encoded).save(
                root / arm / f'{index:05d}-opacity.png')
        print(f'Rendered paired frame {index + 1}/{len(frames)}', flush=True)


def arm_trajectory(trajectory, root, arm):
    """Change only target conditioning; retain anchors, scheduling and cameras."""
    result = copy.deepcopy(trajectory)
    if arm not in ARMS:
        raise ValueError(f'Unknown comparison arm: {arm}')
    if arm != 'current':
        for i, frame in enumerate(result['frames']):
            if arm != 'opacity_only':
                frame['rgb'] = str((root / arm / f'{i:05d}.png').resolve())
            backend = '3dgut' if arm == 'opacity_only' else arm
            frame['opacity'] = str((root / backend / f'{i:05d}.npy').resolve())
    result['renderer_comparison_arm'] = arm
    return result


def compare_inputs(root, request, trajectory):
    cp = Path(request['checkpoint_root'])
    report = []
    def rgb(path):
        return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    for i, frame in enumerate(trajectory['frames']):
        baseline_rgb, baseline_alpha = rgb(cp / frame['rgb']), np.load(cp / frame['opacity'])
        gut_rgb, gut_alpha = rgb(root / '3dgut' / f'{i:05d}.png'), np.load(root / '3dgut' / f'{i:05d}.npy')
        splat_rgb, splat_alpha = rgb(root / 'gsplat' / f'{i:05d}.png'), np.load(root / 'gsplat' / f'{i:05d}.npy')
        report.append({'frame': i, 'current_vs_3dgut_rgb': pixel_metrics(baseline_rgb, gut_rgb),
                       'current_vs_3dgut_opacity': pixel_metrics(baseline_alpha, gut_alpha),
                       'matched_gsplat_vs_3dgut_rgb': pixel_metrics(splat_rgb, gut_rgb),
                       'matched_gsplat_vs_3dgut_opacity': pixel_metrics(splat_alpha, gut_alpha)})
    atomic_json(root / 'input-differences.json', {'frames': report,
        'interpretation': 'Disagreement metrics, not reconstruction quality or accuracy.'})


def compare_outputs(root, arms):
    if 'current' not in arms:
        return
    base = root / 'inference-current'
    paths = sorted(base.glob('inference*/splatfix/frames/batch_*/pred/*.png'))
    if not paths:
        raise ValueError('Current arm produced no prediction frames to compare')
    report = {}
    for arm in arms:
        if arm == 'current':
            continue
        directory = root / f'inference-{arm}'
        expected = {p.relative_to(base) for p in paths}
        actual = {p.relative_to(directory) for p in directory.glob('inference*/splatfix/frames/batch_*/pred/*.png')}
        if expected != actual:
            raise ValueError(f'Inference output frame sets differ for {arm}')
        report[arm] = []
        for path in paths:
            relative = path.relative_to(base)
            with Image.open(path) as a, Image.open(directory / relative) as b:
                metrics = pixel_metrics(np.asarray(a.convert('RGB')) / 255,
                                        np.asarray(b.convert('RGB')) / 255)
            report[arm].append({'frame': str(relative), **metrics})
    atomic_json(root / 'output-differences.json', {
        'interpretation': 'Difference from current seeded output, not quality against ground truth.',
        'arms': report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True, help='Completed prepared run request, with valid GPU-host paths')
    parser.add_argument('--source-ply', type=Path, required=True, help='The same original source PLY used by that run')
    parser.add_argument('--output', type=Path, required=True, help='New isolated directory')
    parser.add_argument('--infer', action='store_true', help='Also run identical seeded inference for each arm; no training')
    parser.add_argument('--arms', choices=ARMS, nargs='+', default=list(ARMS))
    parser.add_argument('--render-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    trajectory = json.loads(Path(request['trajectory']).read_text())
    root = args.output.resolve()
    if args.render_worker:
        repo = Path(request['runtime']['repo'])
        sys.path[:0] = [str(repo), str(repo / 'thirdparty/3DGRUT-ArtiFixer')]
        render_pairs(root, request, trajectory, args.source_ply)
        return
    validate_render_sources(request, trajectory)
    cfg = validate_runtime(request['runtime'])
    list(calibrated_frames(trajectory))
    root.mkdir(parents=True, exist_ok=False)
    snapshot = prepare_splat_initialization(args.source_ply, root / 'source.ply')
    if snapshot is None:
        raise ValueError('Renderer comparison requires a standard Gaussian PLY')
    cp = Path(request['checkpoint_root'])
    metadata = json.loads((cp / 'checkpoint.json').read_text())
    expected = metadata.get('source_fingerprint', {})
    if expected.get('kind') != 'file' or expected.get('sha256') != snapshot['source_ply_sha256']:
        raise ValueError('Source PLY must match the checkpoint source fingerprint')
    env = runtime_environment(cfg)
    atomic_json(root / 'experiment.json', {
        'source_request': str(args.request.resolve()), 'request_sha256': digest_file(args.request),
        'source': snapshot, 'arms': args.arms, 'seed': request['seed'],
        'upstream_revision': request['upstream_revision'],
        'reference_sha256': [digest_file(p) for p in request['references']],
        'controls': 'Same raw PLY, cameras, references, prompt, scale, seed and inference scheduling. No training.',
        'render_controls': 'Matched gsplat and 3DGUT use full PLY SH, black background and authors 8-bit opacity export/read convention. Raw float alpha also retained.',
        'limitations': 'Current captures may differ in SH, filtering and background. Imported PLY is not a fresh authors reconstruction. Opacity-only is an intentionally mixed diagnostic.'})
    run_worker([cfg['python'], str(Path(__file__).resolve()), '--render-worker',
                '--request', str(args.request.resolve()), '--source-ply', str(root / 'source.ply'),
                '--output', str(root)], cwd=cfg['repo'], env=env,
               log_path=root / 'render.log', should_stop=lambda: False)
    compare_inputs(root, request, trajectory)
    for arm in dict.fromkeys(args.arms):
        directory = root / f'inference-{arm}'
        directory.mkdir()
        target = arm_trajectory(trajectory, root, arm)
        atomic_json(directory / 'trajectory.json', target)
        replay = {**request, 'trajectory': str(directory / 'trajectory.json'),
                  'renderer_comparison_arm': arm}
        atomic_json(directory / 'request.json', replay)
        if args.infer:
            run_worker([cfg['python'], str(Path(__file__).with_name('official_worker.py')),
                        '--request', str(directory / 'request.json'), '--phase', 'infer'],
                       cwd=cfg['repo'], env=env, log_path=directory / 'infer.log', should_stop=lambda: False)
    if args.infer:
        compare_outputs(root, args.arms)
    print(f'Comparison saved to {root}', flush=True)


if __name__ == '__main__':
    main()
