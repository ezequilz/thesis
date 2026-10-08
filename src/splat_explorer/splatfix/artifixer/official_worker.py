"""GPU adapter with exact ArtiFixer defaults and optional MCMC regularization.

Executed in a clean pinned ArtiFixer environment, independently of splatfix's
Python environment. This file intentionally imports no scene_runs_ext patches.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import shutil
import sys


# One reconstruction recipe for saved-view repairs and photographic benchmarks.
# Keep this module standalone: it also runs inside the authors' GPU environment.
# Allocation and schedules stay upstream. The optional regularization profile
# inherits only the author base_mcmc loss penalties, not its initialization.
ARTIFIXER3D_CONFIG = 'apps/colmap_3dgut_sparse_mcmc_lpips'
ARTIFIXER3D_STEPS = 30000


REGULARIZATION_PROFILES = ('artifixer', 'base_mcmc')


def validate_regularization(profile):
    if profile not in REGULARIZATION_PROFILES:
        raise ValueError('regularization_profile must be artifixer or base_mcmc')
    return profile


def reconstruction_recipe(profile='artifixer'):
    """Record recipe identity without duplicating upstream hyperparameters."""
    validate_regularization(profile)
    return {'version': 'artifixer3d-regularization-choice-v1',
            'config_name': ARTIFIXER3D_CONFIG, 'steps': ARTIFIXER3D_STEPS,
            'initialization': 'fresh', 'parameter_source': 'pinned_upstream',
            'regularization_profile': profile,
            'custom_reconstruction_overrides': profile != 'artifixer',
            'regularization_source': 'base_mcmc' if profile == 'base_mcmc' else 'base_gs_sparse -> base_gs',
            'loss_adaptation': 'include regularizers in generated-view total' if profile == 'base_mcmc' else None}


def reconstruction_cli_args():
    return ['--config_name', ARTIFIXER3D_CONFIG,
            '--artifixer3d_steps', str(ARTIFIXER3D_STEPS)]


def reconstruction_command(python, repo, profile='artifixer'):
    validate_regularization(profile)
    if profile == 'artifixer':
        return [python, '-m', 'data_processing.run_artifixer3d']
    return [python, str(Path(__file__).resolve()), '--reconstruct', '--repo', str(repo)]


def regularized_trainer_class(base):
    class RegularizedTrainer(base):
        def get_losses(self, gpu_batch, outputs):
            losses = super().get_losses(gpu_batch, outputs)
            if self.conf.loss.get('use_lpips_override', False) and getattr(gpu_batch, 'is_override', False):
                # Upstream computes these weighted tensors but omits them from
                # its LPIPS total. Anchor losses already include them.
                losses['total_loss'] = losses['total_loss'] + losses['opacity_loss'] + losses['scale_loss']
            return losses
    return RegularizedTrainer


@contextmanager
def reconstruction_settings(profile='artifixer'):
    validate_regularization(profile)
    if profile == 'artifixer':
        yield
        return
    from data_processing import threedgrut_training as training
    original_compose, original_trainer = training.compose_3dgrut_config, training.Trainer3DGRUT

    def compose(config_name, overrides, config_dir):
        config = original_compose(config_name, overrides, config_dir)
        regularized = original_compose('base_mcmc', [], config_dir)
        # Inherit only regularization; keep the same initialization and schedules.
        for key in ('use_opacity', 'lambda_opacity', 'use_scale', 'lambda_scale'):
            config.loss[key] = regularized.loss[key]
        return config

    training.compose_3dgrut_config = compose
    training.Trainer3DGRUT = regularized_trainer_class(original_trainer)
    try:
        yield
    finally:
        training.compose_3dgrut_config, training.Trainer3DGRUT = original_compose, original_trainer


def opengl_transforms(trajectory):
    """Match the authors reconstructed_colmap loader's OpenGL C2W input.

    Saved splatfix trajectories use OpenCV C2W (including legacy schema-v1
    caches without an explicit convention). The official preparation writes
    inverse(COLMAP OpenCV W2C) @ diag(1,-1,-1,1), and its loader passes those
    OpenGL matrices unchanged to compute_camera_rays. Use this same transform
    for inference, the plus pass, and reconstruction; the ray primitive's
    OpenCV name does not change the dataset contract.
    """
    import copy
    import numpy as np
    transforms = copy.deepcopy(trajectory['transforms'])
    convention = trajectory.get('camera_convention', transforms.get('camera_convention', 'opencv_c2w'))
    if transforms.get('camera_convention', convention) != convention:
        raise ValueError('Conflicting trajectory camera conventions')
    if convention not in ('opencv_c2w', 'opengl_c2w'):
        raise ValueError(f'Unsupported trajectory camera convention: {convention}')
    if convention == 'opencv_c2w':
        flip = np.diag([1., -1., -1., 1.])
        for frame in transforms['frames']:
            frame['transform_matrix'] = (np.asarray(frame['transform_matrix']) @ flip).tolist()
    transforms['camera_convention'] = 'opengl_c2w'
    return transforms



SCALE_SAMPLING = {'version': 1, 'max_samples_per_anchor': 4096,
                  'min_samples_per_anchor': 32, 'opacity_minimum': .8,
                  'depth': 'expected_camera_z', 'pixel_center_offset': .5,
                  'sampling': 'one nearest-center valid pixel per regular grid cell'}


def sample_depth_points(depth, opacity, camera, *, max_samples=4096, min_samples=32):
    """Backproject gsplat RGB+ED camera-Z at pixel centers, not ray distance.

    gsplat ProjectionEWA3DGSFused stores mean_c.z; RGB+ED averages that
    projection depth. RasterizeToPixels evaluates pixel centers at (x+.5,y+.5).
    These points are measurement observations, never the splat initializer.
    """
    import numpy as np
    depth, opacity = np.asarray(depth), np.asarray(opacity)
    h, w = int(camera['h']), int(camera['w'])
    if depth.shape != (h, w) or opacity.shape != (h, w):
        raise ValueError('Anchor depth/opacity dimensions disagree with calibrated camera')
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(opacity) & (opacity >= SCALE_SAMPLING['opacity_minimum']) & (opacity <= 1)
    rows = min(h, max(1, int(np.sqrt(max_samples * h / w))))
    cols = min(w, max(1, max_samples // rows))
    ys, xs = np.linspace(0, h, rows + 1, dtype=int), np.linspace(0, w, cols + 1, dtype=int)
    pixels = []
    for top, bottom in zip(ys[:-1], ys[1:]):
        for left, right in zip(xs[:-1], xs[1:]):
            yy, xx = np.nonzero(valid[top:bottom, left:right])
            if len(yy):
                yy, xx = yy + top, xx + left
                nearest = np.argmin((yy + .5 - (top + bottom) / 2)**2 + (xx + .5 - (left + right) / 2)**2)
                pixels.append((int(xx[nearest]), int(yy[nearest])))
    if len(pixels) < min_samples:
        raise ValueError(f'Insufficient finite opaque anchor depth for metric alignment: {len(pixels)} < {min_samples}')
    pixels = np.asarray(pixels)
    xys = pixels.astype(np.float64) + .5
    z = depth[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    camera_xyz = np.column_stack(((xys[:, 0] - camera['cx']) / camera['fl_x'] * z,
                                 (xys[:, 1] - camera['cy']) / camera['fl_y'] * z, z))
    c2w = np.asarray(camera['transform_matrix'], dtype=np.float64)
    world = camera_xyz @ c2w[:3, :3].T + c2w[:3, 3]
    return xys, world, pixels


def measurement_colmap(directory, request, trajectory):
    """Materialize observations only from original RGB and original scene depth."""
    import numpy as np
    import struct
    from PIL import Image
    from scipy.spatial.transform import Rotation
    if trajectory.get('depth_convention') != 'expected_camera_z' or trajectory.get('camera_convention') != 'opencv_c2w':
        raise ValueError('Automatic scale requires a fresh camera-Z trajectory cache')
    if not trajectory.get('anchors'):
        raise ValueError('Automatic scale requires nonempty original anchor observations')
    cp = Path(request['checkpoint_root'])
    image_dir, sparse = directory / 'images', directory / 'sparse/0'
    image_dir.mkdir(parents=True)
    sparse.mkdir(parents=True)
    image_rows, camera_rows, points = [], [], []
    counts = []
    for image_id, anchor in enumerate(trajectory['anchors'], 1):
        frame = trajectory['transforms']['frames'][anchor['frame_index']]
        camera = {**trajectory['transforms'], **frame}
        depth = np.load(cp / anchor['depth'], allow_pickle=False)
        opacity = np.load(cp / anchor['opacity'], allow_pickle=False)
        xys, world, pixels = sample_depth_points(depth, opacity, camera,
            max_samples=SCALE_SAMPLING['max_samples_per_anchor'], min_samples=SCALE_SAMPLING['min_samples_per_anchor'])
        original = (cp / anchor['original_rgb']).resolve()
        with Image.open(original) as image:
            rgb = np.asarray(image.convert('RGB'))
        if rgb.shape[:2] != depth.shape:
            raise ValueError('Original anchor RGB dimensions disagree with measurement geometry')
        name = f'anchor_{image_id:05d}.png'
        (image_dir / name).symlink_to(original)
        w2c = np.linalg.inv(np.asarray(camera['transform_matrix']))
        quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        qvec = [quat[3], *quat[:3]]
        ids = list(range(len(points) + 1, len(points) + len(world) + 1))
        for point_id, xyz, pixel, observation in zip(ids, world, pixels, range(len(world))):
            points.append((point_id, xyz, rgb[pixel[1], pixel[0]], image_id, observation))
        image_rows.append((image_id, qvec, w2c[:3, 3], name, xys, ids))
        camera_rows.append(camera)
        counts.append(len(world))
    with (sparse / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(camera_rows)))
        for camera_id, camera in enumerate(camera_rows, 1):
            stream.write(struct.pack('<iiQQ', camera_id, 4, camera['w'], camera['h']))  # COLMAP OPENCV.
            stream.write(struct.pack('<8d', *[camera[key] for key in ('fl_x', 'fl_y', 'cx', 'cy')], 0., 0., 0., 0.))
    with (sparse / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(image_rows)))
        for image_id, qvec, tvec, name, xys, ids in image_rows:
            stream.write(struct.pack('<idddddddi', image_id, *qvec, *tvec, image_id))
            stream.write(name.encode() + b'\0')
            stream.write(struct.pack('<Q', len(ids)))
            for xy, point_id in zip(xys, ids):
                stream.write(struct.pack('<ddq', *xy, point_id))
    with (sparse / 'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(points)))
        for point_id, xyz, rgb, image_id, observation in points:
            stream.write(struct.pack('<QdddBBBdQii', point_id, *xyz, *rgb, 0., 1, image_id, observation))
    return counts


def measure_scale(root, request, trajectory):
    """Cache unchanged official MoGe alignment, shared by baseline and edited."""
    import fcntl
    import hashlib
    import math
    import os
    import uuid
    import numpy as np

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(block)
        return value.hexdigest()
    def write(path, value):
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
        try:
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    cp = Path(request['checkpoint_root'])
    inputs = {anchor[key] for anchor in trajectory['anchors'] for key in ('original_rgb', 'depth', 'opacity')}
    hashes = {path: digest(cp / path) for path in sorted(inputs)}
    for path, value in hashes.items():
        if trajectory['sha256'].get(path) != value:
            raise ValueError(f'Metric alignment input changed: {path}')
    local_model = request['runtime'].get('moge_model_path') or os.environ.get('MOGE_MODEL_PATH')
    model = {'model_id': 'Ruicheng/moge-2-vitl-normal', 'local_path': local_model}
    if local_model:
        model_path = Path(local_model)
    else:
        from huggingface_hub import hf_hub_download
        model_path = Path(hf_hub_download(repo_id=model['model_id'], filename='model.pt', local_files_only=True))
    if not model_path.is_file():
        raise FileNotFoundError(f'MoGe requires a provisioned checkpoint file: {model_path}')
    model['weights_sha256'] = digest(model_path)
    recipe = {'sampling': SCALE_SAMPLING, 'upstream_revision': request['upstream_revision'],
              'original_inputs_sha256': hashes, 'model': model,
              'transforms': trajectory['transforms'], 'anchors': trajectory['anchors']}
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    cache_root = Path(request['trajectory']).parent / 'metric_alignment'
    cache_root.mkdir(exist_ok=True)
    cached = cache_root / f'{signature}.json'
    # A cancelled worker releases the OS lock; incomplete attempts are never reused.
    with (cache_root / f'{signature}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cached.is_file():
            result = json.loads(cached.read_text())
            if result.get('signature') != signature or not math.isfinite(result['metric_scale']) or result['metric_scale'] <= 0:
                raise ValueError('Invalid cached metric alignment')
        else:
            measurement = cache_root / ('measurement_' + uuid.uuid4().hex[:12])
            counts = measurement_colmap(measurement, request, trajectory)
            from data_processing.sparse_recon.metric_alignment import align_colmap_to_metric_scale
            # Bind the unchanged authors loader to precisely the weights hashed
            # above, including direct worker invocation outside our launcher.
            previous_model = os.environ.get('MOGE_MODEL_PATH')
            os.environ['MOGE_MODEL_PATH'] = str(model_path.resolve())
            try:
                metric, stats, correspondences = align_colmap_to_metric_scale(
                    colmap_dir=measurement / 'sparse/0', image_dir=measurement / 'images',
                    output_dir=measurement, debug=False, downsample_factor=1)
            finally:
                if previous_model is None:
                    os.environ.pop('MOGE_MODEL_PATH', None)
                else:
                    os.environ['MOGE_MODEL_PATH'] = previous_model
            if not np.isfinite(metric) or metric <= 0:
                raise ValueError('Official MoGe alignment returned invalid metric scale')
            nonfinite_statistics = [name for name, value in stats.items() if not np.isfinite(value)]
            stats = {name: (None if name in nonfinite_statistics else value) for name, value in stats.items()}
            valid = 0
            for item in correspondences:
                if item.get('depths_metric') is not None and item.get('mask') is not None:
                    depth, mask = np.asarray(item['depths_metric']), np.asarray(item['mask'])
                    valid += int(np.count_nonzero(np.isfinite(depth) & (depth > 0) & np.isfinite(mask) & (mask > 0)))
            if valid < SCALE_SAMPLING['min_samples_per_anchor']:
                raise ValueError('Insufficient valid MoGe depth support; no default scale substituted')
            result = {'signature': signature, 'metric_scale': float(metric), 'camera_scale': float(metric) * .01,
                      'stats': stats, 'nonfinite_statistics': nonfinite_statistics,
                      'statistics_encoding': 'Nonfinite ancillary diagnostics stored as null; official finite scale retained',
                      'valid_metric_samples': valid, 'geometry_samples_per_anchor': counts,
                      'recipe': recipe, 'measurement_colmap': str(measurement.relative_to(Path(request['trajectory']).parent)),
                      'measurement_path_base': 'trajectory directory',
                      'provenance': 'unchanged official MoGe alignment on original rendered RGB and expected camera-Z observations; synthetic measurement geometry only',
                      'limitation': 'Scale inferred from rendered scene geometry, not calibrated photographic ground truth'}
            write(cached, result)
    write(root / 'scale-result.json', {**result, 'cache_path': str(cached)})


CAPTION_MODEL_ID = 'Qwen/Qwen3-VL-30B-A3B-Instruct'


def caption_digest(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_caption(path, expected_hash=None):
    """Validate the authors uint16 representation of finite bfloat16 tokens."""
    import h5py
    import numpy as np
    path = Path(path)
    digest = caption_digest(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError('Cached caption HDF5 checksum mismatch')
    with h5py.File(path, 'r') as source:
        if len(source) != 1:
            raise ValueError('Expected exactly one scene caption embedding')
        dataset = source[next(iter(source))]
        data = dataset[:]
        caption = dataset.attrs.get('caption', '')
        if isinstance(caption, bytes):
            caption = caption.decode('utf-8')
        if (data.ndim != 2 or not 0 < data.shape[0] <= 512 or data.shape[1] != 4096
                or data.dtype != np.dtype('uint16') or not str(caption).strip() or not np.any(data)
                or not np.isfinite((data.astype(np.uint32) << 16).view(np.float32)).all()):
            raise ValueError('Invalid authors caption embedding or empty caption text')
        indices = np.asarray(dataset.attrs.get('image_indices', []))
        return {'sha256': digest, 'caption': str(caption), 'shape': list(data.shape),
                'image_indices': indices.tolist()}


def hub_blob_address(snapshot, blob, model_id):
    """Recognize content addresses only inside this resolved model's HF cache."""
    def hexadecimal(value):
        return len(value) in (40, 64) and all(c in '0123456789abcdef' for c in value)
    snapshot, blob = Path(snapshot).resolve(), Path(blob).resolve()
    repository_name = 'models--' + model_id.replace('/', '--')
    if (not hexadecimal(snapshot.name) or snapshot.parent.name != 'snapshots'
            or snapshot.parent.parent.name != repository_name or not hexadecimal(blob.name)):
        return None
    cache_root = snapshot.parent.parent.parent
    try:
        relative = blob.relative_to(cache_root).parts
    except ValueError:
        return None
    standard = (repository_name, 'blobs', blob.name)
    sharded = ('blobs', blob.name[:2], blob.name)
    return blob.name if relative in (standard, sharded) else None


def caption_model_snapshot(model_id, *, text_only=False):
    """Resolve provisioned immutable HF snapshots without any metadata requests.

    The helper receives this exact snapshot path. Blob identities retain HF's
    content addresses (SHA256 for LFS weights), avoiding rehashing 60GB on replay.
    Copied files are identified by size/mtime locally and hashed on generation;
    their recorded hashes are not recomputed during replay.
    """
    from fnmatch import fnmatch
    from huggingface_hub import snapshot_download
    patterns = (['tokenizer/*', 'text_encoder/*'] if text_only else
                ['config.json', 'generation_config.json', 'model*.safetensors*',
                 'tokenizer*', 'vocab.json', 'merges.txt', 'preprocessor_config.json',
                 'video_preprocessor_config.json', 'processor_config.json', 'chat_template*'])
    snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True, allow_patterns=patterns))
    files = []
    for path in sorted(snapshot.rglob('*')):
        if not path.is_file() or any(part.startswith('.') for part in path.relative_to(snapshot).parts):
            continue
        relative = path.relative_to(snapshot).as_posix()
        if not any(fnmatch(relative, pattern) for pattern in patterns):
            continue
        blob = path.resolve()
        address = hub_blob_address(snapshot, blob, model_id)
        stat = path.stat()
        files.append({'path': relative, 'size': stat.st_size,
                      'content_address': address,
                      'copied_file_mtime_ns': None if address else stat.st_mtime_ns})
    if not files or not any(item['path'].endswith(('.safetensors', '.bin')) for item in files):
        raise FileNotFoundError(f'Provision the complete caption model snapshot first: {model_id}')
    return snapshot, {'model_id': model_id, 'revision': snapshot.name, 'files': files}


def prepare_caption(root, request, trajectory):
    """Caption original anchors once locally; both modes reuse exactly this HDF5."""
    import fcntl
    import hashlib
    import os
    import uuid
    cp = Path(request['checkpoint_root'])
    anchors = trajectory.get('anchors', [])
    if not anchors:
        raise ValueError('Caption preparation requires original saved anchors')
    originals = []
    for anchor in anchors:
        relative = anchor['original_rgb']
        digest = caption_digest(cp / relative)
        if trajectory['sha256'].get(relative) != digest:
            raise ValueError(f'Original caption image changed: {relative}')
        originals.append({'view_id': anchor['view_id'], 'rgb': relative, 'sha256': digest})
    qwen_path, qwen = caption_model_snapshot(CAPTION_MODEL_ID)
    wan_path, wan = caption_model_snapshot(request['runtime']['model_id'], text_only=True)
    recipe = {'version': 1, 'originals': originals, 'upstream_revision': request['upstream_revision'],
              'caption_model': qwen, 'text_encoder': wan,
              'parameters': {'num_frames': None, 'frame_stride': 1, 'dataset_fps': 60,
                             'dataset_downsample_factor': 1, 'text_encoder_max_sequence_length': 512}}
    signature = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    cache = cp / 'captions'
    cache.mkdir(exist_ok=True)
    target = cache / signature
    with (cache / (signature + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            result = json.loads((target / 'manifest.json').read_text())
            if result['recipe'] != recipe:
                raise ValueError('Cached caption recipe mismatch')
            validate_caption(target / 'caption.h5', result['sha256'])
        else:
            temporary = cache / ('.' + signature + '.' + uuid.uuid4().hex)
            source = temporary / 'source'
            (source / 'images').mkdir(parents=True)
            try:
                frames = []
                for index, original in enumerate(originals):
                    name = f'images/anchor_{index:05d}.png'
                    (source / name).symlink_to((cp / original['rgb']).resolve())
                    frames.append({'file_path': name})
                (source / 'transforms.json').write_text(json.dumps({'frames': frames}))
                from data_processing.captioning.generate_captions import generate_caption_hdf5
                generate_caption_hdf5(input_path=source, output_path=temporary / 'caption.h5',
                    dataset_downsample_factor=1, captioning_model_id=str(qwen_path),
                    text_encoder_model_id=str(wan_path))
                result = {**validate_caption(temporary / 'caption.h5'), 'recipe': recipe,
                          'signature': signature,
                          'resolved_model_paths': {'caption_model': str(qwen_path), 'text_encoder': str(wan_path)},
                          'copied_model_files_sha256': {
                              label: {item['path']: caption_digest(path / item['path'])
                                      for item in identity['files'] if item['content_address'] is None}
                              for label, path, identity in [('caption_model', qwen_path, qwen), ('text_encoder', wan_path, wan)]},
                          'model_identity_validation': 'Immutable HF snapshot revision and blob content addresses; copied files hashed only on generation, size/mtime checked on replay',
                          'provenance': 'unchanged authors local Qwen caption and UMT5 encoding of original saved anchors',
                          'input_semantics': ('single original image; one caption' if len(originals) == 1 else
                                              'ordered saved anchors supplied as one video at 60fps; processor controls frame sampling; one caption')}
                if result['image_indices'] != list(range(len(originals))):
                    raise ValueError('Caption image_indices disagree with original anchors')
                (temporary / 'manifest.json').write_text(json.dumps(result, indent=2))
                os.rename(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    result = {**result, 'caption_path': str(target / 'caption.h5')}
    (root / 'caption-result.json').write_text(json.dumps(result, indent=2))


def inference(root, request, trajectory, plus=False):
    import numpy as np
    import torch
    from PIL import Image
    from model_eval.run_inference import build_parser, get_eval_pipe, process_item
    from model_eval.checkpoint_loading import load_transformer_checkpoint
    from model_training.data.utils import compute_camera_rays, load_encoded_prompt

    validate_caption(request['caption_path'], request['caption_sha256'])
    encoded_prompt, _ = load_encoded_prompt([Path(request['caption_path'])])
    cfg = request['runtime']
    output = root / ('plus' if plus else 'inference')
    args = build_parser().parse_args([
        '--checkpoint_pt', cfg['checkpoint'], '--model_id', cfg['model_id'],
        '--save_dir', str(output), '--save_frame_outputs_only',
        '--evalset', 'reconstructed_colmap', '--render_trajectory', 'trajectory',
    ])  # All inference algorithm/scheduler/cache settings are authors defaults.
    torch.manual_seed(request['seed'])
    device = torch.device('cuda:0')
    pipe = get_eval_pipe(args, device)
    load_transformer_checkpoint(pipe.transformer, args)
    pipe.transformer.eval().requires_grad_(False)
    cp = Path(request['checkpoint_root'])
    cameras = opengl_transforms(trajectory)
    def rgb(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert('RGB'))).permute(2, 0, 1).float() / 255
    references = torch.stack([rgb(path) for path in request['references']])
    neighbors = [a['frame_index'] for a in trajectory['anchors']]
    render_dir = Path(json.loads((root / 'distillation.json').read_text())['render_dir']) if plus else None
    with torch.inference_mode():
        for segment in trajectory['segments']:
            indices = (list(segment['indices']) if 'indices' in segment else
                       list(range(segment['start'], segment['start'] + segment['count'])))
            if request.get('trajectory_mode') == 'authors_orbit':
                # Match ReconstructedColmapDataset's explicit-trajectory contract:
                # saved anchors condition inference and supervise fitting directly.
                indices = [index for index in indices if index not in neighbors]
            if not indices:
                continue
            seed_index = segment.get('seed_index')
            if seed_index is not None:
                if seed_index not in neighbors:
                    raise ValueError('Series seed must be a trusted reference camera')
                indices = [seed_index, *indices]
            if plus:
                renders = torch.stack([rgb(render_dir / 'renders' / f'{i:05d}.png') for i in indices])
                opacity = torch.stack([torch.from_numpy(np.array(Image.open(render_dir / 'opacity' / f'{i:05d}.png').convert('L'), dtype=np.float32) / 255) for i in indices])
            else:
                renders = torch.stack([rgb(cp / trajectory['frames'][i]['rgb']) for i in indices])
                opacity = torch.stack([torch.from_numpy(np.load(cp / trajectory['frames'][i]['opacity'], allow_pickle=False)) for i in indices])
            valid = torch.ones(len(indices), dtype=torch.bool)
            if seed_index is not None:
                renders[0] = references[neighbors.index(seed_index)]
                opacity[0] = 1
                valid[0] = False  # Trusted anchors supervise reconstruction directly.
            item = {'scene_id': 'splatfix', 'rgb_rendered': renders, 'rgb_neighbors': references,
                    'opacity': opacity, 'encoded_prompt': encoded_prompt,
                    'frame_indices': torch.tensor(indices),
                    'valid_frames_mask': valid}
            item.update(compute_camera_rays(cameras, indices, neighbors,
                        scale=request['camera_scale'], image_shape=renders.shape[-2:], skip_vae_check=True))
            # Independent smooth camera trajectories must not become video cuts.
            pipe.clear_inference_caches()
            process_item(pipe, item, args, output, 0, device, pipe.vae.config.scale_factor_temporal)
    if plus:
        result = json.loads((root / 'distillation.json').read_text())
        result.update(reconstruction_method='artifixer', plus_frames=str(output / 'splatfix/frames/batch_0000/pred'),
                      rgb_renderer=request.get('anchor_rgb_policy'),
                      plus_rgb_renderer=request.get('plus_rgb_renderer'),
                      mode=request['mode'], upstream_revision=request['upstream_revision'],
                      inference_target_policy=request.get('inference_target_policy', 'all_segment_frames'),
                      camera_scale=request['camera_scale'],
                      camera_scale_provenance=request.get('camera_scale_provenance'),
                      scale_estimate=request.get('scale_estimate'),
                      caption_path=request['caption_path'], caption_sha256=request['caption_sha256'],
                      initialization=request['initialization'], input_adaptation=request['input_adaptation'],
                      reconstruction='fresh ArtiFixer3D; 30000 steps; sparse MCMC LPIPS; regularization recorded in reconstruction_recipe',
                      reconstruction_recipe=reconstruction_recipe(request.get('runtime', {}).get('regularization_profile', 'artifixer')),
                      ply_stage='ArtiFixer3D; the + pass produces images, not another splat', merged=False)
        (root / 'result.json').write_text(json.dumps(result, indent=2))


def unique_supervision(trajectory, references):
    """One RGB target per exact calibrated pose; saved anchors take priority.

    Diffusion retains closed temporal loops. This catalogue is used only for
    distillation, with an explicit map back to the unchanged diffusion indices.
    No tolerance or spatial clustering can discard a distinct nearby camera.
    """
    import copy
    import numpy as np
    if len(trajectory['anchors']) != len(references):
        raise ValueError('Every saved anchor must have exactly one reference image')
    transforms = opengl_transforms(trajectory)
    frames = transforms['frames']
    groups, keys, original_to_unique = [], {}, []
    for index, frame in enumerate(frames):
        pose = np.asarray(frame['transform_matrix'], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Invalid distillation camera pose')
        # Intrinsics are included for completeness, although saved checkpoints
        # currently require shared intrinsics across all anchors.
        camera = {**transforms, **frame}
        key = tuple(pose.ravel()) + tuple(camera[name] for name in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy'))
        if key not in keys:
            keys[key] = len(groups)
            groups.append({'original_indices': [], 'reference': None})
        unique_index = keys[key]
        groups[unique_index]['original_indices'].append(index)
        original_to_unique.append(unique_index)
    for anchor, reference in zip(trajectory['anchors'], references):
        original_index = anchor['frame_index']
        if type(original_index) is not int or not 0 <= original_index < len(frames):
            raise ValueError('Anchor frame index is outside the saved trajectory')
        group = groups[original_to_unique[original_index]]
        reference = str(Path(reference).resolve())
        if group['reference'] is not None and caption_digest(group['reference']) != caption_digest(reference):
            from PIL import Image
            with Image.open(group['reference']) as first, Image.open(reference) as second:
                if not np.array_equal(np.asarray(first.convert('RGB')), np.asarray(second.convert('RGB'))):
                    raise ValueError('Conflicting saved reference RGB at an identical camera pose; cannot create coherent supervision')
        group['reference'] = reference
        group['source_index'] = original_index
    result = copy.deepcopy(transforms)
    result['frames'] = []
    selected = []
    for unique_index, group in enumerate(groups):
        source_index = group.setdefault('source_index', group['original_indices'][0])
        result['frames'].append(copy.deepcopy(frames[source_index]))
        if group['reference'] is not None:
            selected.append(unique_index)
    mapping = {'policy': 'exact calibrated camera equality; saved reference takes priority; no proximity threshold',
               'original_frame_count': len(frames), 'distillation_frame_count': len(groups),
               'original_to_distillation': original_to_unique,
               'distillation_to_original': [group['source_index'] for group in groups],
               'groups': groups, 'selected_indices': selected}
    return result, mapping


def distill(root, request, trajectory):
    import numpy as np
    import torch
    from data_processing import artifixer3d as official
    from threedgrut.export.ply_exporter import PLYExporter
    from threedgrut.render import Renderer

    torch.manual_seed(request['seed'])
    np.random.seed(request['seed'])
    source = root / 'source_colmap'
    sparse = source / 'sparse/0'
    sparse.mkdir(parents=True)
    (source / 'images').mkdir()
    render_transforms = opengl_transforms(trajectory)
    render_transforms_path = root / 'render_transforms_opengl.json'
    render_transforms_path.write_text(json.dumps(render_transforms, indent=2))
    transforms, supervision = unique_supervision(trajectory, request['references'])
    (root / 'supervision.json').write_text(json.dumps(supervision, indent=2))
    cameras, images = [], []
    selected = supervision['selected_indices']
    predictions = root / 'distillation_predictions'
    predictions.mkdir()
    original_predictions = root / 'inference/splatfix/frames/batch_0000/pred'
    for frame_index, group in enumerate(supervision['groups']):
        reference = group['reference']
        if reference is None:
            original = original_predictions / f"{group['source_index']:05d}.png"
            if not original.is_file():
                raise FileNotFoundError(f'Missing generated supervision: {original}')
            (predictions / f'{frame_index:05d}.png').symlink_to(original.resolve())
            continue
        index = len(images)
        name = f'anchor_{index:05d}.png'
        (source / 'images' / name).symlink_to(Path(reference).resolve())
        transforms['frames'][frame_index]['file_path'] = f'images/{name}'
        cameras.append(official.opencv_camera_from_mapping(index + 1, transforms))
        qvec, tvec = official.colmap_pose_from_transforms_frame(transforms['frames'][frame_index], np.eye(4))
        images.append((index + 1, qvec, tvec, index + 1, name))
    official.write_colmap_cameras(sparse / 'cameras.bin', cameras)
    official.write_colmap_images(sparse / 'images.bin', images)
    shutil.copyfile(request['source_points3d'], sparse / 'points3D.bin')
    transforms_path = root / 'transforms_opengl.json'
    transforms_path.write_text(json.dumps(transforms, indent=2))
    scene = official.PreparedScene(
        scene_id='splatfix', scene_root=root, transforms_path=transforms_path,
        colmap_dir=source, prompt_path=root / 'request.json', camera_scale=request['camera_scale'],
        has_gt=False, selected_indices=selected, target_indices_path=None,
        reconstruction_checkpoint=None, frame_count=len(transforms['frames']))
    paths = official.artifixer3d_paths(scene, root / 'artifixer3d', None, ARTIFIXER3D_STEPS)
    # Authors entry point, with the selected loss policy shared by both arms.
    # No resume checkpoint, custom fitter, SH truncation, or compositing.
    profile = request.get('runtime', {}).get('regularization_profile', 'artifixer')
    with reconstruction_settings(profile):
        checkpoint, reused = official.train_artifixer3d(
            scene, paths, artifixer_frames_dir=predictions,
            base_checkpoint=None, config_name=ARTIFIXER3D_CONFIG,
            steps=ARTIFIXER3D_STEPS, use_wandb=False, replace=False)
    gc.collect()
    torch.cuda.empty_cache()
    render_dir = official.render_artifixer3d(
        scene, paths, checkpoint=checkpoint, checkpoint_reused=reused,
        replace=False, render_trajectory_path=render_transforms_path)
    gc.collect()
    torch.cuda.empty_cache()
    renderer = Renderer.from_checkpoint(checkpoint_path=checkpoint,
        out_dir=str(root / 'export'), path=str(paths.distillation_input_dir),
        save_gt=False, computes_extra_metrics=False)
    splat_path = root / 'artifixer3d.ply'
    PLYExporter().export(renderer.model, splat_path)
    (root / 'distillation.json').write_text(json.dumps({
        'splat_path': str(splat_path), 'reconstruction_checkpoint': str(checkpoint),
        'reconstruction_recipe': reconstruction_recipe(profile),
        'render_dir': str(render_dir), 'supervision_manifest': str(root / 'supervision.json'),
        'original_frame_count': len(render_transforms['frames']),
        'distillation_frame_count': len(transforms['frames'])}, indent=2))


def validate_render_sources(request, trajectory):
    """Do not replay pre-GPU-anchor requests through the worker entry point."""
    if trajectory.get('recipe', {}).get('anchor_rgb_policy') != 'viser':
        raise ValueError('Unverified trajectory renderer: regenerate this request with Viser captures')
    cp = Path(request['checkpoint_root'])
    if request['mode'] == 'baseline':
        expected = [(cp / anchor['original_rgb']).resolve() for anchor in trajectory['anchors']]
        if [Path(path).resolve() for path in request['references']] != expected:
            raise ValueError('Baseline references must be Viser trajectory captures, not selection previews')
    else:
        checkpoint = json.loads((cp / 'checkpoint.json').read_text())
        if checkpoint.get('metadata', {}).get('renderer', {}).get('backend') != 'viser':
            raise ValueError('Edited references must originate from Viser captures')


def main():
    if sys.argv[1:2] == ['--reconstruct']:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--reconstruct', action='store_true')
        parser.add_argument('--repo', required=True)
        adapter, remaining = parser.parse_known_args()
        sys.path.insert(0, adapter.repo)
        from data_processing import run_artifixer3d as official_cli
        args = official_cli.build_parser().parse_args(remaining)
        with reconstruction_settings('base_mcmc'):
            official_cli.artifixer3d.run_artifixer3d(args)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--phase', choices=('scale', 'caption', 'infer', 'distill', 'plus'), required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    sys.path.insert(0, request['runtime']['repo'])
    trajectory = json.loads(Path(request['trajectory']).read_text())
    validate_render_sources(request, trajectory)
    if args.phase == 'plus' and request.get('plus_rgb_renderer') != 'viser':
        raise ValueError('ArtiFixer+ requires Viser captures of the reconstructed splat')
    if args.phase == 'scale':
        measure_scale(args.request.parent, request, trajectory)
    elif args.phase == 'caption':
        prepare_caption(args.request.parent, request, trajectory)
    elif args.phase == 'distill':
        distill(args.request.parent, request, trajectory)
    else:
        inference(args.request.parent, request, trajectory, plus=args.phase == 'plus')


if __name__ == '__main__':
    main()
