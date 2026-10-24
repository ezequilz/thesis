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
if __package__:
    from .pipeline_profile import rgb_backend
else:
    from pipeline_profile import rgb_backend


ARTIFIXER3D_CONFIG = 'apps/colmap_3dgut_sparse_mcmc_lpips'
ARTIFIXER3D_STEPS = 30000


def generated_loss_options(raw):
    """Defaults preserve the pinned ArtiFixer generated-view objective."""
    result = {key: raw.get(key, default) for key, default in (
        ('generated_rgb_loss', False), ('generated_lpips_downscale', True))}
    for key, value in result.items():
        if type(value) is not bool:
            raise ValueError(f'{key} must be a boolean')
    return result


def validate_gaussian_limit(value=1_000_000):
    if type(value) is not int or value < 1:
        raise ValueError('max_gaussians must be a positive integer')
    return value


def reconstruction_overrides(raw):
    return {**generated_loss_options(raw),
            'max_gaussians': validate_gaussian_limit(raw.get('max_gaussians', 1_000_000))}


def validate_reconstruction_options(raw):
    images = raw.get('reconstruction_images', 'distilled')
    initialization = raw.get('reconstruction_initialization', 'point_cloud')
    if images not in ('all', 'distilled'):
        raise ValueError('reconstruction_images must be all or distilled')
    if initialization not in ('full_splat', 'point_cloud'):
        raise ValueError('reconstruction_initialization must be full_splat or point_cloud')
    prompt_source = raw.get('prompt_source', 'worldexplorer')
    if prompt_source not in ('original_anchors', 'original_and_repaired', 'worldexplorer'):
        raise ValueError('Unknown ArtiFixer prompt_source')
    return {'reconstruction_images': images, 'reconstruction_initialization': initialization,
            'prompt_source': prompt_source, **reconstruction_overrides(raw)}


REGULARIZATION_PROFILES = ('artifixer', 'base_mcmc')


def validate_regularization(profile):
    if profile not in REGULARIZATION_PROFILES:
        raise ValueError('regularization_profile must be artifixer or base_mcmc')
    return profile


def reconstruction_recipe(profile='artifixer', *, source_ply=False,
                          generated_rgb_loss=False, generated_lpips_downscale=True, max_gaussians=1_000_000):
    """Record recipe identity and the effective generated-image loss multipliers."""
    validate_regularization(profile)
    loss_options = reconstruction_overrides(locals())
    return {'version': 'artifixer3d-generated-loss-choice-v2',
            **loss_options,
            'lambda_reconlosses_override': 1.0 if generated_rgb_loss else 0.0,
            'lambda_lpips_override': 0.1 if generated_lpips_downscale else 1.0,
            'config_name': ARTIFIXER3D_CONFIG, 'steps': ARTIFIXER3D_STEPS,
            'initialization': 'full_splat_ply' if source_ply else 'fresh',
            'training_state': 'fresh', 'start_step': 0, 'parameter_source': 'pinned_upstream',
            'initialization_source': 'original PLY; all Gaussian parameters' if source_ply else 'COLMAP points',
            'regularization_profile': profile,
            'custom_reconstruction_overrides': source_ply or profile != 'artifixer' or generated_rgb_loss or not generated_lpips_downscale or max_gaussians != 1_000_000,
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
def shared_distillation_cameras(official):
    """Share exact calibrations before upstream allocates per-camera ray grids.

    The authors' materializer emits a camera per image. Thousands of original
    WorldExplorer frames then duplicate tens of GiB of CPU rays (and GPU rays
    per loader). Camera IDs describe intrinsics, not poses: sharing identical
    records preserves every observation, pose, and selected/override index.
    """
    write_cameras = official.write_colmap_cameras
    write_images = official.write_colmap_images
    mappings = {}

    def cameras(path, records):
        unique, keys, mapping = [], {}, {}
        for camera in records:
            key = (camera.model, camera.width, camera.height,
                   tuple(float(value) for value in camera.params))
            if key not in keys:
                keys[key] = camera.id
                unique.append(camera)
            mapping[camera.id] = keys[key]
        write_cameras(path, unique)
        mappings[Path(path).parent] = mapping
        print(f'SPLATFIX_SHARED_CAMERAS: {len(records)} records -> '
              f'{len(unique)} exact calibrations; all frames retained', flush=True)

    def images(path, records):
        mapping = mappings[Path(path).parent]
        write_images(path, [(image_id, qvec, tvec, mapping[camera_id], name)
                            for image_id, qvec, tvec, camera_id, name in records])

    official.write_colmap_cameras, official.write_colmap_images = cameras, images
    try:
        yield
    finally:
        official.write_colmap_cameras, official.write_colmap_images = write_cameras, write_images


@contextmanager
def reconstruction_settings(profile='artifixer', *, source_ply=None,
                            generated_rgb_loss=False, generated_lpips_downscale=True, max_gaussians=1_000_000):
    validate_regularization(profile)
    reconstruction_overrides(locals())
    if profile == 'artifixer' and source_ply is None and not generated_rgb_loss and generated_lpips_downscale and max_gaussians == 1_000_000:
        yield
        return
    from data_processing import threedgrut_training as training
    original_compose, original_trainer = training.compose_3dgrut_config, training.Trainer3DGRUT

    def compose(config_name, overrides, config_dir):
        config = original_compose(config_name, overrides, config_dir)
        if max_gaussians != 1_000_000:
            config.strategy.add.max_n_gaussians = max_gaussians
        if generated_rgb_loss or not generated_lpips_downscale:
            # Only the override (generated-image) branch consumes these weights.
            config.loss['lambda_reconlosses_override'] = 1.0 if generated_rgb_loss else 0.0
            config.loss['lambda_lpips_override'] = 0.1 if generated_lpips_downscale else 1.0
            if generated_rgb_loss:
                config.loss['lambda_l1_override'] = config.loss['lambda_l1']
                config.loss['lambda_ssim_override'] = config.loss['lambda_ssim']
        if source_ply is not None:
            # Use the authors' full Gaussian importer, with a fresh optimizer
            # and the complete schedule. Never resume the source training step.
            config.resume = ''
            config.import_ingp.enabled = False
            config.import_ply.enabled = True
            config.import_ply.path = str(Path(source_ply).resolve())
            config.import_ply.init_global_step = 0
        if profile == 'base_mcmc':
            regularized = original_compose('base_mcmc', [], config_dir)
            for key in ('use_opacity', 'lambda_opacity', 'use_scale', 'lambda_scale'):
                config.loss[key] = regularized.loss[key]
        return config

    training.compose_3dgrut_config = compose
    training.Trainer3DGRUT = regularized_trainer_class(original_trainer) if profile == 'base_mcmc' else original_trainer
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


def encode_scene_prompt(prompt, output, model_path, anchor_count):
    """Use the pinned authors' text encoder and exact HDF5 representation."""
    import h5py
    import numpy as np
    import torch
    from data_processing.captioning.generate_captions import (
        get_text_encoder_and_tokenizer, generate_text_embedding,
    )
    with torch.inference_mode():
        encoder, tokenizer = get_text_encoder_and_tokenizer(str(model_path))
        embedding = generate_text_embedding(prompt, encoder, tokenizer, 512)
    with h5py.File(output, 'w') as target:
        dataset = target.create_dataset('anchor_00000.png', data=embedding)
        dataset.attrs['caption'] = prompt
        dataset.attrs['image_indices'] = np.arange(anchor_count)


def prepare_caption(root, request, trajectory):
    """Prepare source-specific, content-addressed authors text conditioning."""
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
    prompt_source = validate_reconstruction_options(request.get('runtime', {}) | {
        key: request[key] for key in ('prompt_source',) if key in request})['prompt_source']
    prompt = request.get('worldexplorer_prompt')
    if prompt_source == 'worldexplorer' and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError('Original WorldExplorer prompt is unavailable for this scene')
    if prompt_source == 'original_and_repaired':
        if request.get('mode') != 'edited':
            raise ValueError('Original + repaired captioning requires edited input images')
        references = request.get('references', [])
        hashes = request.get('reference_sha256', [])
        if not references or len(references) != len(hashes):
            raise ValueError('Repaired caption references and checksums are required')
        for index, (reference, expected) in enumerate(zip(references, hashes)):
            path = Path(reference).resolve()
            relative = str(path.relative_to(cp.resolve()))
            digest = caption_digest(path)
            if digest != expected:
                raise ValueError('Repaired caption reference changed')
            originals.append({'view_id': f'repaired_{index}', 'rgb': relative, 'sha256': digest})
    qwen_path, qwen = (caption_model_snapshot(CAPTION_MODEL_ID)
                       if prompt_source != 'worldexplorer' else (None, None))
    wan_path, wan = caption_model_snapshot(request['runtime']['model_id'], text_only=True)
    recipe = {'version': 1, 'originals': originals, 'upstream_revision': request['upstream_revision'],
              'caption_model': qwen, 'text_encoder': wan,
              'parameters': {'num_frames': None, 'frame_stride': 1, 'dataset_fps': 60,
                             'dataset_downsample_factor': 1, 'text_encoder_max_sequence_length': 512}}
    if prompt_source != 'original_anchors':
        recipe.update(prompt_source=prompt_source, worldexplorer_prompt=prompt if prompt_source == 'worldexplorer' else None)
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
                if prompt_source == 'worldexplorer':
                    encode_scene_prompt(prompt, temporary / 'caption.h5', wan_path, len(originals))
                else:
                    generate_caption_hdf5(input_path=source, output_path=temporary / 'caption.h5',
                        dataset_downsample_factor=1, captioning_model_id=str(qwen_path),
                        text_encoder_model_id=str(wan_path))
                result = {**validate_caption(temporary / 'caption.h5'), 'recipe': recipe,
                          'signature': signature,
                          'resolved_model_paths': {'caption_model': str(qwen_path) if qwen_path is not None else None, 'text_encoder': str(wan_path)},
                          'copied_model_files_sha256': {
                              label: {item['path']: caption_digest(path / item['path'])
                                      for item in identity['files'] if item['content_address'] is None}
                              for label, path, identity in [('caption_model', qwen_path, qwen), ('text_encoder', wan_path, wan)] if identity is not None},
                          'model_identity_validation': 'Immutable HF snapshot revision and blob content addresses; copied files hashed only on generation, size/mtime checked on replay',
                          'prompt_source': prompt_source,
                          'provenance': {'original_anchors': 'unchanged authors local Qwen caption and UMT5 encoding of original saved anchors',
                                         'original_and_repaired': 'authors Qwen/UMT5 on original anchors plus repaired input references',
                                         'worldexplorer': 'original WorldExplorer scene prompt encoded by authors UMT5'}[prompt_source],
                          'input_semantics': ('saved scene prompt; no visual captioning' if prompt_source == 'worldexplorer' else 'single original image; one caption' if len(originals) == 1 else
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


def install_image_cache_insertion(pipe):
    """Reuse the exact starter implementation, including in standalone GPU workers."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / 'scene_runs_ext' / 'starter_inference.py'
    spec = importlib.util.spec_from_file_location('splatfix_starter_inference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_starter_inference(pipe, generated_cache='clean')


def inference_jobs(request, trajectory, *, plus=False):
    """Explicit output ownership keeps reverse predictions out of reconstruction."""
    anchors = [a['frame_index'] for a in trajectory['anchors']]
    if request.get('repair_policy') != 'endpoints':
        segments = trajectory['segments']
        if (request.get('trajectory_mode') == 'authors_orbit'
                and request.get('runtime', {}).get('split_mode') == 'no-split'):
            # One temporal sequence in authored trajectory order, with every
            # saved reference retained as conditioning. Never inherit inward
            # half-leg direction changes or cache resets from segment metadata.
            segments = [{'indices': [i for i in range(len(trajectory['frames'])) if i not in anchors],
                         **({'cache_seed_index': anchors[0]} if anchors else {})}]
        return [(s, list(range(len(anchors))), 'plus' if plus else 'inference')
                for s in segments]
    if len(anchors) != 2:
        raise ValueError('Endpoint inference requires exactly two repaired references')
    mode = request['runtime'].get('endpoint_reference_mode', 'both')
    if mode not in ('both', 'initial'):
        raise ValueError('Invalid endpoint reference mode')
    indices = list(range(len(trajectory['frames'])))
    if anchors != [indices[0], indices[-1]]:
        raise ValueError('Endpoint reference cameras must bound the ordered trajectory')
    jobs = []
    for reverse in ([False] if plus else [False, True]):
        order = [1, 0] if reverse else [0, 1]
        if mode == 'initial':
            order = order[:1]
        jobs.append(({'indices': indices[::-1] if reverse else indices,
                      'cache_seed_index': anchors[order[0]],
                      'direction': 'back-to-front' if reverse else 'front-to-back'}, order,
                     'plus' if plus else 'inference-back-to-front' if reverse else 'inference'))
    return jobs


def initial_seed_index(cfg, segment, indices, neighbors):
    """New runs explicitly select seeding; missing/None preserves old segment policy."""
    seeding = cfg.get('initial_image_seeding')
    if seeding is not None and type(seeding) is not bool:
        raise ValueError('initial_image_seeding must be a boolean')
    preserve = cfg.get('image_cache_insertion', False)
    if seeding is False and not preserve:
        return None
    seed = segment.get('seed_index')
    if (seeding or preserve) and seed is None:
        seed = segment.get('cache_seed_index')
        if seed is None and indices[0] in neighbors:
            seed = indices[0]
        if seed is None:
            raise ValueError('Initial image seeding requires a trusted endpoint for each series; rebuild the trajectory')
    return seed


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
    if __package__:
        from .handoff import enabled, repair_boundary
    else:
        from handoff import enabled, repair_boundary
    handoff = enabled(cfg)
    if handoff and (request.get('repair_policy') == 'endpoints' or
                    request.get('trajectory_mode') != 'authors_orbit' or
                    cfg.get('split_mode') != 'single-split'):
        raise ValueError('Handoff generation requires single-split checkpoint authors-orbit trajectories')
    insertion = cfg.get('image_cache_insertion', False)
    if type(insertion) is not bool:
        raise ValueError('image_cache_insertion must be a boolean')
    output = root / ('plus' if plus else 'inference')
    args = build_parser().parse_args([
        '--checkpoint_pt', cfg['checkpoint'], '--model_id', cfg['model_id'],
        '--save_dir', str(output), '--save_frame_outputs_only',
        '--evalset', 'reconstructed_colmap', '--render_trajectory', 'trajectory',
        # Configure before construction: the transformer attention layers also
        # capture these settings. Changing only pipe.sink_size is insufficient.
        # Match the tested scene-runs-ext starter window and retain latent zero
        # when later generated blocks roll through the cache.
        *(['--sink_size', '1', '--local_attn_size', '21'] if insertion else []),
    ])  # Insertion off retains the authors' inference defaults.
    torch.manual_seed(request['seed'])
    device = torch.device('cuda:0')
    pipe = get_eval_pipe(args, device)
    load_transformer_checkpoint(pipe.transformer, args)
    pipe.transformer.eval().requires_grad_(False)
    if insertion:
        install_image_cache_insertion(pipe)
    cp = Path(request['checkpoint_root'])
    cameras = opengl_transforms(trajectory)
    def rgb(path):
        with Image.open(path) as im:
            return torch.from_numpy(np.array(im.convert('RGB'))).permute(2, 0, 1).float() / 255
    all_references = torch.stack([rgb(path) for path in request['references']])
    all_neighbors = [a['frame_index'] for a in trajectory['anchors']]
    active_neighbors = list(all_neighbors)
    active_reference_paths = list(request['references'])
    if handoff and plus:
        state = load_handoff_references(root, request, trajectory)
        active_neighbors = state['frame_indices']
        active_reference_paths = state['paths']
        all_references = torch.stack([rgb(path) for path in active_reference_paths])
    jobs = inference_jobs(request, trajectory, plus=plus)
    if request.get('repair_policy') == 'endpoints':
        (root / ('plus-directions.json' if plus else 'inference-directions.json')).write_text(json.dumps([
            {'direction': s['direction'], 'frame_indices': s['indices'],
             'reference_paths': [request['references'][i] for i in order],
             'reference_frame_indices': [all_neighbors[i] for i in order],
             'output': directory, 'used_for_reconstruction': directory == 'inference'}
            for s, order, directory in jobs], indent=2))
    render_dir = Path(json.loads((root / 'distillation.json').read_text())['render_dir']) if plus else None
    insertion_audits = []
    seeding_audits = []
    previous_boundary = None
    handoff_records = []
    with torch.inference_mode():
        for segment, reference_order, directory in jobs:
            output = root / directory
            references = all_references[reference_order]
            neighbors = [all_neighbors[i] for i in reference_order]
            indices = (list(segment['indices']) if 'indices' in segment else
                       list(range(segment['start'], segment['start'] + segment['count'])))
            if request.get('trajectory_mode') == 'authors_orbit' and request.get('repair_policy') != 'endpoints':
                # Match ReconstructedColmapDataset's explicit-trajectory contract:
                # saved anchors condition inference and supervise fitting directly.
                indices = [index for index in indices if index not in (all_neighbors if handoff else neighbors)]
            if not indices:
                continue
            seed_path = None
            if handoff:
                if previous_boundary is None:
                    seed_index = initial_seed_index({**cfg, 'initial_image_seeding': True},
                                                    segment, indices, all_neighbors)
                    seed_path = Path(active_reference_paths[all_neighbors.index(seed_index)])
                else:
                    replaced_index = initial_seed_index({**cfg, 'initial_image_seeding': True},
                                                        segment, indices, all_neighbors)
                    if replaced_index not in sorted(all_neighbors)[1:-1]:
                        raise ValueError('Handoff may replace only an interior trajectory reference')
                    reference_slot = all_neighbors.index(replaced_index)
                    seed_index = previous_boundary
                    source = output / 'splatfix/frames/batch_0000/pred' / f'{seed_index:05d}.png'
                    seed_path = repair_boundary(root, source, seed_index,
                        {**{k: v for k, v in cameras.items() if k != 'frames'},
                         **cameras['frames'][seed_index]}, phase=directory)
                    active_neighbors[reference_slot] = seed_index
                    active_reference_paths[reference_slot] = str(seed_path)
                    all_references[reference_slot] = rgb(seed_path)
                    handoff_records.append({'frame_index': seed_index, 'repair': str(seed_path),
                        'replaced_reference_frame_index': replaced_index, 'reference_slot': reference_slot,
                        'reference_frame_indices': list(active_neighbors),
                        'reference_paths': list(active_reference_paths),
                        'next_sequence': len(seeding_audits), 'output': directory,
                        'policy': 'replace original boundary at identical camera and global index'})
                    (root / ('plus-handoffs.json' if plus else 'inference-handoffs.json')).write_text(
                        json.dumps(handoff_records, indent=2))
                references = all_references[reference_order]
                neighbors = [active_neighbors[i] for i in reference_order]
            else:
                seed_index = initial_seed_index(cfg, segment, indices, neighbors)
            if seed_index is not None:
                if seed_index not in neighbors:
                    raise ValueError('Series seed must be a trusted reference camera')
                indices = [seed_index, *[i for i in indices if i != seed_index]]
            if plus:
                renders = torch.stack([rgb(render_dir / 'renders' / f'{i:05d}.png') for i in indices])
                opacity = torch.stack([torch.from_numpy(np.array(Image.open(render_dir / 'opacity' / f'{i:05d}.png').convert('L'), dtype=np.float32) / 255) for i in indices])
            else:
                renders = torch.stack([rgb(cp / trajectory['frames'][i]['rgb']) for i in indices])
                opacity = torch.stack([torch.from_numpy(np.load(cp / trajectory['frames'][i]['opacity'], allow_pickle=False)) for i in indices])
            valid = torch.ones(len(indices), dtype=torch.bool)
            if request.get('repair_policy') == 'endpoints':
                valid = torch.tensor([i not in all_neighbors for i in indices], dtype=torch.bool)
            if seed_index is not None:
                # References are the run-local, resolution-prepared GPT images.
                # Upstream denoise_to_latents applies VAE preprocessing/scaling;
                # the causal first latent encodes only this endpoint image.
                renders[0] = references[neighbors.index(seed_index)]
                opacity[0] = 1
                valid[0] = False  # Trusted anchors supervise reconstruction directly.
            seeding_audits.append({'output': directory, 'enabled': seed_index is not None,
                                   'seed_global_index': seed_index, 'cache_preservation': insertion,
                                   **({'reference_frame_indices': list(neighbors),
                                       'reference_paths': [active_reference_paths[i] for i in reference_order]}
                                      if handoff else {})})
            (root / ('plus-seeding.json' if plus else 'inference-seeding.json')).write_text(
                json.dumps({'series': seeding_audits}, indent=2))
            item = {'scene_id': 'splatfix', 'rgb_rendered': renders, 'rgb_neighbors': references,
                    'opacity': opacity, 'encoded_prompt': encoded_prompt,
                    'frame_indices': torch.tensor(indices),
                    'valid_frames_mask': valid}
            item.update(compute_camera_rays(cameras, indices, neighbors,
                        scale=request['camera_scale'], image_shape=renders.shape[-2:], skip_vae_check=True))
            # Independent smooth camera trajectories must not become video cuts.
            pipe.clear_inference_caches()
            if insertion:
                pipe.starter_inference_audit = None
            process_item(pipe, item, args, output, 0, device, pipe.vae.config.scale_factor_temporal)
            if insertion:
                audit = pipe.starter_inference_audit
                if not audit or not audit.get('encoded_starter_preserved'):
                    raise RuntimeError('Image cache insertion was requested but starter preservation was not verified')
                reference = str(seed_path) if handoff else request['references'][reference_order[neighbors.index(seed_index)]]
                record = {**audit, 'output': directory, 'seed_global_index': seed_index,
                          'reference_path': reference, 'reference_sha256': caption_digest(Path(reference)),
                          'generation_global_indices': indices,
                          'export_global_indices': [i for i, keep in zip(indices, valid.tolist()) if keep],
                          'starter_exported': False}
                insertion_audits.append(record)
                (root / ('plus-insertion-audit.json' if plus else 'inference-insertion-audit.json')).write_text(
                    json.dumps({'series': insertion_audits}, indent=2))
                print('SPLATFIX_STARTER_VERIFIED: ' + json.dumps(record), flush=True)
            if handoff:
                previous_boundary = indices[-1]
    if handoff:
        reference_state = {'original_frame_indices': all_neighbors, 'frame_indices': active_neighbors,
                           'paths': active_reference_paths,
                           'sha256': [caption_digest(path) for path in active_reference_paths]}
        (root / ('plus-references.json' if plus else 'inference-references.json')).write_text(
            json.dumps(reference_state, indent=2))
    if plus:
        result = json.loads((root / 'distillation.json').read_text())
        result.update(repair_policy=request.get('repair_policy', 'all'),
                      artifixer_pipeline=request.get('artifixer_pipeline', 'legacy_gsplat'),
                      endpoint_reference_mode=cfg.get('endpoint_reference_mode', 'both'),
                      reverse_inference_frames=(str(root / 'inference-back-to-front/splatfix/frames/batch_0000/pred')
                                                if request.get('repair_policy') == 'endpoints' else None),
                      reconstruction_method='artifixer', plus_frames=str(output / 'splatfix/frames/batch_0000/pred'),
                      rgb_renderer=request.get('anchor_rgb_policy'),
                      plus_rgb_renderer=request.get('plus_rgb_renderer'),
                      mode=request['mode'], upstream_revision=request['upstream_revision'],
                      handoff_generation=handoff,
                      image_cache_insertion=insertion,
                      initial_image_seeding=any(s['enabled'] for s in seeding_audits),
                      inference_target_policy=request.get('inference_target_policy', 'all_segment_frames'),
                      camera_scale=request['camera_scale'],
                      camera_scale_provenance=request.get('camera_scale_provenance'),
                      scale_estimate=request.get('scale_estimate'),
                      caption_path=request['caption_path'], caption_sha256=request['caption_sha256'],
                      initialization=request['initialization'], input_adaptation=request['input_adaptation'],
                      reconstruction='fresh ArtiFixer3D; 30000 steps; sparse MCMC LPIPS; regularization recorded in reconstruction_recipe',
                      reconstruction_recipe=reconstruction_recipe(request.get('runtime', {}).get('regularization_profile', 'artifixer'),
                                                                source_ply=bool(request.get('source_ply')),
                                                                **reconstruction_overrides(request.get('runtime', {}))),
                      ply_stage='ArtiFixer3D; the + pass produces images, not another splat', merged=False)
        (root / 'result.json').write_text(json.dumps(result, indent=2))


def unique_supervision(trajectory, references, *, excluded_indices=()):
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
        if index in excluded_indices:
            original_to_unique.append(None)
            continue
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
        if original_to_unique[original_index] is None:
            raise ValueError('Cannot exclude a ground-truth reference camera')
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


def load_handoff_references(root, request, trajectory):
    """Require the completed inference reference set before fitting or plus generation."""
    state = json.loads((root / 'inference-references.json').read_text())
    original = [a['frame_index'] for a in trajectory['anchors']]
    indices, paths, hashes = state['frame_indices'], state['paths'], state['sha256']
    if (state['original_frame_indices'] != original or
            not len(indices) == len(paths) == len(hashes) == len(original) or
            len(set(indices)) != len(indices)):
        raise ValueError('Invalid completed handoff reference set')
    endpoints = (min(original), max(original))
    for slot, (old, new, path, expected) in enumerate(zip(original, indices, paths, hashes)):
        if type(new) is not int or not 0 <= new < len(trajectory['frames']):
            raise ValueError('Handoff reference camera is outside trajectory')
        if old in endpoints:
            if new != old or Path(path).resolve() != Path(request['references'][slot]).resolve():
                raise ValueError('Handoff must preserve first and last references')
        elif new in original:
            raise ValueError('Interior reference was not replaced by a generated handoff')
        if caption_digest(path) != expected:
            raise ValueError('Handoff reference checksum mismatch')
    return state


def reconstruction_supervision(root, request, trajectory):
    if not request.get('runtime', {}).get('handoff_generation', False):
        return unique_supervision(trajectory, request['references'])
    state = load_handoff_references(root, request, trajectory)
    # Keep the endpoint in generation context, but exclude its inconsistent RGB
    # and camera from this consistency experiment's fitting catalogue entirely.
    endpoint = max(state['original_frame_indices'])
    retained_slots = [i for i, old in enumerate(state['original_frame_indices']) if old != endpoint]
    anchors = [{**trajectory['anchors'][i], 'frame_index': state['frame_indices'][i]}
               for i in retained_slots]
    references = [state['paths'][i] for i in retained_slots]
    displaced = [old for old, new in zip(state['original_frame_indices'], state['frame_indices']) if old != new]
    excluded = sorted(set(displaced) | {endpoint})
    transforms, supervision = unique_supervision({**trajectory, 'anchors': anchors}, references,
                                                 excluded_indices=set(excluded))
    supervision.update(handoff_references=state, excluded_original_reference_indices=excluded,
                       excluded_endpoint_frame_index=endpoint,
                       reconstruction_reference_slots=retained_slots,
                       policy='first reference and repaired handoffs are ground truth at their exact cameras; displaced references and final endpoint excluded')
    return transforms, supervision


def append_worldexplorer_supervision(request, transforms, supervision):
    policy = validate_reconstruction_options(request.get('runtime', {}) | {
        key: request[key] for key in ('reconstruction_images', 'reconstruction_initialization') if key in request})
    supervision.update(policy, worldexplorer_original_count=0)
    if policy['reconstruction_images'] == 'distilled':
        return
    if request.get('worldexplorer_source'):
        # This worker runs with a deliberately isolated PYTHONPATH.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'worldexplorer_bundle', Path(__file__).parents[1] / 'worldexplorer/bundle.py')
        bundle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bundle)
        load_staged = bundle.load_staged
        originals = load_staged(request['worldexplorer_source'])
        for frame in originals['frames']:
            index = len(transforms['frames'])
            transforms['frames'].append(frame)
            supervision['selected_indices'].append(index)
            supervision['groups'].append({'reference': frame['file_path'], 'source_index': None})
        supervision['worldexplorer_original_count'] = len(originals['frames'])
        supervision['distillation_frame_count'] = len(transforms['frames'])


def distill(root, request, trajectory):
    if request.get('source_ply'):
        import hashlib
        digest = hashlib.sha256()
        with Path(request['source_ply']).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != request.get('source_ply_sha256'):
            raise ValueError('Source initialization PLY changed since preparation')
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
    transforms, supervision = reconstruction_supervision(root, request, trajectory)
    append_worldexplorer_supervision(request, transforms, supervision)
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
        cameras.append(official.opencv_camera_from_mapping(index + 1, {**transforms, **transforms['frames'][frame_index]}))
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
    with shared_distillation_cameras(official), reconstruction_settings(profile, source_ply=request.get('source_ply'),
                                                                         **reconstruction_overrides(request.get('runtime', {}))):
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
        'reconstruction_recipe': reconstruction_recipe(profile, source_ply=bool(request.get('source_ply')),
                                                      **reconstruction_overrides(request.get('runtime', {}))),
        'source_ply_sha256': request.get('source_ply_sha256'),
        'render_dir': str(render_dir), 'supervision_manifest': str(root / 'supervision.json'),
        'reconstruction_images': supervision['reconstruction_images'],
        'reconstruction_initialization': supervision['reconstruction_initialization'],
        'worldexplorer_original_count': supervision['worldexplorer_original_count'],
        'original_frame_count': len(render_transforms['frames']),
        'distillation_frame_count': len(transforms['frames'])}, indent=2))


def validate_render_sources(request, trajectory):
    """Do not replay pre-GPU-anchor requests through the worker entry point."""
    expected_backend = rgb_backend(request.get('runtime', {}))
    if trajectory.get('recipe', {}).get('anchor_rgb_policy') != expected_backend:
        raise ValueError(f'Unverified trajectory renderer: expected {expected_backend} captures')
    cp = Path(request['checkpoint_root'])
    if request['mode'] == 'baseline':
        expected = [(cp / anchor['original_rgb']).resolve() for anchor in trajectory['anchors']]
        if [Path(path).resolve() for path in request['references']] != expected:
            renderer_name = 'Viser' if expected_backend == 'viser' else '3DGUT'
            raise ValueError(f'Baseline references must be {renderer_name} trajectory captures, not selection previews')
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
    if args.phase == 'plus' and request.get('plus_rgb_renderer') != rgb_backend(request.get('runtime', {})):
        raise ValueError('ArtiFixer+ renderer does not match the selected pipeline')
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
