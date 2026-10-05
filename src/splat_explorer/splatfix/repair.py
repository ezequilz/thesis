"""Offline replay through unmodified ArtiFixer inference and fresh 3DGRUT.

The no-editor baseline uses rendered anchors, not captured photographs. This
input adaptation is recorded; the authors' algorithm and optimizer stay intact.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time
import uuid

import numpy as np
from PIL import Image
from .checkpoint import Checkpoint, atomic_json, camera_from_record

UPSTREAM_REVISION = 'a392c4dfe17459ef9952407accdb9fcdcdddba98'
DEFAULT_RUNTIME = {
    'repo': '/workspace/third_party/ArtiFixer',
    'python': '/workspace/artifixer-venv/bin/python',
    'checkpoint': '/workspace/models/artifixer/artifixer-1.3b.pt',
    'model_id': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    'hf_home': '/workspace/models/huggingface',
    'native_library_dir': '/workspace/artifixer-native/root/usr/lib/x86_64-linux-gnu',
    'slang_bin': '/workspace/artifixer-slang/bin',
    'torch_extensions_dir': f'/workspace/artifixer-extensions/{UPSTREAM_REVISION[:12]}',
}


RUNTIME_COMPATIBILITY = 'typing.Self backport via typing_extensions only when absent; upstream code unchanged'


def runtime_environment(cfg):
    """Isolate packages, pinned compiler and compiled extensions from legacy jobs."""
    offline = cfg.get('model_hub_offline', True)
    if not isinstance(offline, bool):
        raise ValueError('model_hub_offline must be a boolean')
    env = dict(os.environ, HF_HUB_OFFLINE='1' if offline else '0',
               TRANSFORMERS_OFFLINE='1' if offline else '0',
               HF_HOME=cfg['hf_home'], WANDB_MODE='disabled')
    # Each dashboard worker owns one GPU and invokes the authors' single-process
    # entry point. Container launchers can leave partial torchrun variables that
    # would incorrectly select the distributed branch of that entry point.
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'):
        env.pop(key, None)
    if cfg.get('moge_model_path'):
        env['MOGE_MODEL_PATH'] = str(cfg['moge_model_path'])
    env['PYTHONPATH'] = str(Path(__file__).with_name('python_compat').resolve())
    executable = cfg.get('python', DEFAULT_RUNTIME['python'])
    python_bin = str(Path(shutil.which(executable) or executable).parent.absolute())
    path_prefix = [python_bin]
    slang = cfg.get('slang_bin', DEFAULT_RUNTIME['slang_bin'])
    if slang and Path(slang).is_dir():
        path_prefix.insert(0, str(Path(slang).resolve()))
    env['PATH'] = os.pathsep.join(path_prefix + ([env['PATH']] if env.get('PATH') else []))
    env['TORCH_EXTENSIONS_DIR'] = cfg.get('torch_extensions_dir', DEFAULT_RUNTIME['torch_extensions_dir'])
    native = cfg.get('native_library_dir')
    if native and Path(native).is_dir():
        env['LD_LIBRARY_PATH'] = str(Path(native).resolve()) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env

def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_runtime(runtime=None):
    cfg = {**DEFAULT_RUNTIME, **(runtime or {})}
    repo = Path(cfg['repo']).resolve()
    for relative in ('model_eval/run_inference.py', 'data_processing/artifixer3d.py',
                     'thirdparty/3DGRUT-ArtiFixer/threedgrut/trainer.py',
                     'thirdparty/3DGRUT-ArtiFixer/configs/apps/colmap_3dgut_sparse_mcmc_lpips.yaml'):
        if not (repo / relative).is_file():
            raise FileNotFoundError(f'Missing official ArtiFixer source: {repo / relative}; initialize the pinned 3DGRUT submodule and GPU environment')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f'ArtiFixer must be pinned to {UPSTREAM_REVISION}; got {revision}')
    for directory in (repo, repo / 'thirdparty/3DGRUT-ArtiFixer'):
        # DSS/container staging can drop executable bits without changing source.
        # Check code contents and pinned submodule commits, not mount permissions.
        changes = subprocess.check_output(['git', '-c', 'core.fileMode=false', '-C', str(directory),
            'status', '--porcelain', '--untracked-files=all', '--ignore-submodules=all'], text=True)
        if changes.strip():
            raise ValueError(f'Authors baseline requires a clean upstream checkout: {directory}')
    expected = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', 'HEAD', 'thirdparty/3DGRUT-ArtiFixer'], text=True).split()[2]
    actual = subprocess.check_output(['git', '-C', str(repo / 'thirdparty/3DGRUT-ArtiFixer'), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError('3DGRUT submodule differs from pinned authors revision')
    submodules = subprocess.check_output(['git', '-C', str(repo), 'submodule', 'status', '--recursive'], text=True)
    if any(line and line[0] in '-+U' for line in submodules.splitlines()):
        raise ValueError('ArtiFixer recursive submodules must match their pinned commits')
    if not Path(cfg['checkpoint']).is_file() or shutil.which(cfg['python']) is None:
        raise FileNotFoundError('ArtiFixer Python environment or release checkpoint is missing')
    cfg['repo'] = str(repo)
    return cfg


def write_seed_points(path, scene):
    """COLMAP point-cloud initialization only; no opacity/scale/SH continuation."""
    points = np.asarray(scene.means)
    colors = np.clip(np.asarray(scene.colors) * 255, 0, 255).astype(np.uint8)
    if not len(points) or points.shape != colors.shape or not np.isfinite(points).all():
        raise ValueError('Initial splat must have finite positions and RGB colors')
    with Path(path).open('wb') as stream:
        stream.write(struct.pack('<Q', len(points)))
        for i, (xyz, rgb) in enumerate(zip(points, colors)):
            stream.write(struct.pack('<QdddBBBdQ', i + 1, *xyz, *rgb, 0., 0))


def prepare_trajectory(checkpoint, *, frames=25, span_fraction=.04,
                       should_stop=lambda: False, renderer_factory=None, scene_path=None):
    """Cache original RGB/alpha/cameras once for both repair comparison arms."""
    from ..scene import load_scene
    from ..scene_runs_ext.bundle import camera_bundle, transforms
    from .rendering import BundleRenderer, release_cuda
    cp = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint.load(checkpoint)
    if not cp.complete:
        raise ValueError('Finish selecting all requested views before repair')
    if type(frames) is not int or frames < 9 or (frames - 1) % 4:
        raise ValueError('frames must be 1 + 4*n and at least 9')
    if not np.isfinite(span_fraction) or not 0 < span_fraction <= .15:
        raise ValueError('span_fraction must be in (0, .15]')
    signature = hashlib.sha256(json.dumps({'trajectory_version': 2, 'views': [{k: v for k, v in view.items() if k in ('id', 'camera', 'original_rgb')} for view in cp.views],
        'source': cp.manifest.get('source_fingerprint', cp.manifest['scene_path']), 'scene_load': cp.manifest.get('metadata', {}).get('scene_load', {}),
        'rgb': [digest_file(cp.image_path(view)) for view in cp.views], 'frames': frames, 'span_fraction': span_fraction}, sort_keys=True).encode()).hexdigest()
    cache = cp.root / 'trajectories'
    completed = sorted(cache.glob(signature[:16] + '_*/trajectory.json'))
    root = completed[0].parent if completed else cache / (signature[:16] + '_' + uuid.uuid4().hex[:8])
    if (root / 'trajectory.json').is_file():
        manifest = json.loads((root / 'trajectory.json').read_text())
        for name, digest in manifest['sha256'].items():
            if digest_file(cp.root / name) != digest:
                raise ValueError(f'Saved trajectory changed: {name}')
        return root, manifest
    root.parent.mkdir(exist_ok=True)
    root.mkdir(exist_ok=False)  # Never race another renderer into the same cache.
    from .checkpoint import validate_source
    source_path = str(Path(scene_path).resolve()) if scene_path else cp.manifest['scene_path']
    validate_source(Checkpoint(cp.root, {**cp.manifest, 'scene_path': source_path}))
    scene = load_scene(source_path, **cp.manifest.get('metadata', {}).get('scene_load', {}))
    renderer = (renderer_factory or BundleRenderer)(scene)
    all_cameras, records, segments, anchors = [], [], [], []
    try:
        for view in cp.views:
            camera = camera_from_record(view)
            if camera.width % 16 or camera.height % 16:
                raise ValueError('Saved image width and height must be multiples of 16 for ArtiFixer')
            if all_cameras and ((camera.width, camera.height) != (all_cameras[0].width, all_cameras[0].height)
                                or not np.allclose(camera.intrinsics, all_cameras[0].intrinsics)):
                raise ValueError('All saved views must share resolution and intrinsics')
            start = len(all_cameras)
            anchor_render = renderer.render(camera)
            depth = anchor_render[2]
            depth_path = root / f'anchor-{view["id"]}-depth.npy'
            np.save(depth_path, np.asarray(depth, dtype=np.float32))
            cameras = camera_bundle(camera, depth, frames=frames, span_fraction=span_fraction)
            segments.append({'start': start, 'count': len(cameras)})
            anchors.append({'view_id': view['id'], 'frame_index': start, 'original_rgb': view['original_rgb'],
                            'depth': str(depth_path.relative_to(cp.root)),
                            'opacity': str((root / f'{start:05d}.npy').relative_to(cp.root))})
            for local, cam in enumerate(cameras):
                if should_stop():
                    raise InterruptedError('Trajectory rendering stopped')
                index = len(all_cameras)
                rgb, alpha, _ = anchor_render if local in (0, frames - 1) else renderer.render(cam)
                rgb_path = cp.image_path(view) if local in (0, frames - 1) else root / f'{index:05d}.png'
                if local not in (0, frames - 1):
                    Image.fromarray(rgb).save(rgb_path)
                alpha_path = root / f'{index:05d}.npy'
                np.save(alpha_path, np.asarray(alpha, dtype=np.float32))
                records.append({'rgb': str(rgb_path.relative_to(cp.root)), 'opacity': str(alpha_path.relative_to(cp.root))})
                all_cameras.append(cam)
        write_seed_points(root / 'points3D.bin', scene)
    except Exception:
        shutil.rmtree(root)  # Only this invocation's unique, unpublished cache.
        raise
    finally:
        del renderer
        release_cuda()
    manifest = {'schema_version': 2, 'depth_convention': 'expected_camera_z', 'camera_convention': 'opencv_c2w', 'signature': signature, 'transforms': {'camera_model': 'OPENCV', **transforms(all_cameras)},
                'segments': segments, 'anchors': anchors, 'frames': records,
                'points3d': str((root / 'points3D.bin').relative_to(cp.root)),
                'source_scene': source_path}
    paths = {entry[key] for entry in records for key in ('rgb', 'opacity')} | {manifest['points3d']} | {anchor['depth'] for anchor in anchors}
    manifest['sha256'] = {path: digest_file(cp.root / path) for path in sorted(paths)}
    atomic_json(root / 'trajectory.json', manifest)
    return root, manifest


def run_worker(command, *, cwd, env, log_path, should_stop):
    with Path(log_path).open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if should_stop():
                    raise InterruptedError('Splatfix repair stopped')
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError(f'ArtiFixer worker failed: {Path(log_path).read_text(errors="replace")[-3000:]}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def run_repair(checkpoint_dir, output_dir, *, mode='edited', runtime=None, frames=25,
               span_fraction=.04, seed=42, camera_scale=None, should_stop=lambda: False,
               on_progress=lambda event: None):
    """Run independently with cached authors local captioning; never merge PLYs."""
    if mode not in ('edited', 'baseline'):
        raise ValueError('mode must be edited or baseline')
    if (type(seed) is not int or not 0 <= seed < 2**31
            or (camera_scale is not None and (isinstance(camera_scale, bool)
                or not np.isfinite(camera_scale) or camera_scale <= 0))):
        raise ValueError('Invalid seed or camera_scale')
    cp = Checkpoint.load(checkpoint_dir)
    if not cp.complete:
        raise ValueError('Finish selecting all requested views before repair')
    if mode == 'edited' and any(not view.get('repaired_rgb') for view in cp.views):
        raise ValueError('Edited reconstruction requires saved GPT-image repairs; run splatfix edit first')
    references = [str(cp.image_path(view, repaired=mode == 'edited')) for view in cp.views]
    for view, path in zip(cp.views, references):
        camera = camera_from_record(view)
        with Image.open(path) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError('Reference dimensions differ from saved camera')
    cfg = validate_runtime(runtime)
    on_progress({'phase': 'trajectory'})
    trajectory_root, trajectory = prepare_trajectory(cp, frames=frames, span_fraction=span_fraction, should_stop=should_stop, scene_path=cfg.get('scene_path'))
    root = Path(output_dir).resolve() / f'{mode}_{uuid.uuid4().hex[:12]}'
    root.mkdir(parents=True, exist_ok=False)
    points = cfg.get('source_points3d') or str(cp.root / trajectory['points3d'])
    request = {'checkpoint_root': str(cp.root), 'trajectory': str(trajectory_root / 'trajectory.json'),
               'references': references, 'runtime': cfg, 'mode': mode, 'seed': seed,
               'camera_scale': float(camera_scale) if camera_scale is not None else None,
               'camera_scale_provenance': ('explicit manual scene-unit multiplier' if camera_scale is not None
                                           else 'pending official MoGe alignment on original rendered geometry'),
               'conditioning_camera_convention': 'opengl_c2w (official reconstructed_colmap loader contract)',
               'supervision_policy': 'one target per exact calibrated camera; original index mapping retained; saved anchors take priority',
               'source_points3d': str(Path(points).resolve()),
               'initialization': 'source_colmap_points' if cfg.get('source_points3d') else 'original_splat_positions_and_rgb',
               'upstream_revision': UPSTREAM_REVISION, 'fit_iterations': 30000,
               'input_adaptation': 'Saved selected renders replace photographic anchors; edited mode uses cached GPT-image RGB; cached authors local Qwen/UMT5 caption of originals; independently sampled local camera loops.',
               'runtime_compatibility': RUNTIME_COMPATIBILITY,
               'native_library_path_active': bool(cfg.get('native_library_dir') and Path(cfg['native_library_dir']).is_dir()),
               'reference_sha256': [digest_file(path) for path in references],
               'source_points3d_sha256': digest_file(points), 'trajectory_signature': trajectory['signature']}
    atomic_json(root / 'request.json', request)
    env = runtime_environment(cfg)
    try:
        phases = ('scale', 'caption', 'infer', 'distill', 'plus') if camera_scale is None else ('caption', 'infer', 'distill', 'plus')
        for phase in phases:
            if should_stop():
                raise InterruptedError('Splatfix repair stopped')
            on_progress({'phase': phase, 'output_dir': str(root)})
            run_worker([cfg['python'], str(Path(__file__).with_name('official_worker.py')), '--request', str(root / 'request.json'), '--phase', phase],
                       cwd=cfg['repo'], env=(runtime_environment({**cfg, 'model_hub_offline': False}) if phase == 'caption' else env), log_path=root / f'{phase}.log', should_stop=should_stop)
            if phase == 'caption':
                caption = json.loads((root / 'caption-result.json').read_text())
                request.update(caption_path=caption['caption_path'], caption_sha256=caption['sha256'],
                               caption_preparation=caption)
                atomic_json(root / 'request.json', request)
            if phase == 'scale':
                measured = json.loads((root / 'scale-result.json').read_text())
                metric = measured['metric_scale']
                if isinstance(metric, bool) or not np.isfinite(metric) or metric <= 0:
                    raise ValueError('Official alignment returned an invalid measured scale')
                request.update(camera_scale=float(metric) * .01,
                               camera_scale_provenance='official MoGe alignment on original rendered geometry, metric_scale * 0.01',
                               scale_estimate=measured)
                atomic_json(root / 'request.json', request)
        result = json.loads((root / 'result.json').read_text())
        if not Path(result['splat_path']).is_file():
            raise RuntimeError('Authors reconstruction did not produce its isolated PLY')
        return {**result, 'output_dir': str(root)}
    except Exception as exc:
        atomic_json(root / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        raise
