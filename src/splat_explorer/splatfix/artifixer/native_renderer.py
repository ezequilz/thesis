"""Pinned 3DGUT source renderer, isolated in the authors' Python environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def expected_camera_z(weighted_distance, alpha, ray_z):
    """3DGUT accumulates sum(T*alpha*t); our scale adapter needs E[camera Z]."""
    distance, alpha, ray_z = map(np.asarray, (weighted_distance, alpha, ray_z))
    if distance.shape != alpha.shape or ray_z.shape != alpha.shape:
        raise ValueError('3DGUT depth, opacity and camera rays must have matching shapes')
    depth = np.full(alpha.shape, np.inf, dtype=np.float32)
    valid = (alpha >= .15) & np.isfinite(distance) & (distance > 0) & (ray_z > 0)
    depth[valid] = distance[valid] * ray_z[valid] / alpha[valid]
    return depth


class NativeRenderer:
    """Renderer protocol backed by one persistent native subprocess per bundle."""
    def __init__(self, source, runtime, should_stop=lambda: False):
        from .repair import prepare_splat_initialization, runtime_environment
        self.should_stop = should_stop
        self.process = None
        self.temp = tempfile.TemporaryDirectory(prefix='splatfix-3dgut-')
        self.root = Path(self.temp.name)
        try:
            info = prepare_splat_initialization(source, self.root / 'source.ply')
            if info is None:
                raise ValueError('Authors 3DGUT requires a standard Gaussian PLY; select legacy rendering for other formats')
            self.log = (self.root / 'renderer.log').open('w')
            self.process = subprocess.Popen([
                runtime['python'], str(Path(__file__).resolve()), '--repo', runtime['repo'],
                '--work', str(self.root)], cwd=runtime['repo'], env=runtime_environment(runtime),
                stdout=self.log, stderr=subprocess.STDOUT)
            self._wait(self.root / 'ready')
        except BaseException:
            self.close()
            raise

    def _wait(self, path):
        started = time.monotonic()
        while not path.exists():
            if self.should_stop():
                raise InterruptedError('3DGUT rendering stopped')
            if self.process.poll() is not None:
                tail = (self.root / 'renderer.log').read_text(errors='replace')[-4000:]
                raise RuntimeError(f'Authors 3DGUT renderer exited: {tail}')
            if time.monotonic() - started > 900:
                raise TimeoutError('Authors 3DGUT renderer did not respond within 15 minutes')
            time.sleep(.05)

    def render(self, camera):
        from ..checkpoint import atomic_json
        output = self.root / 'frame.npz'
        output.unlink(missing_ok=True)
        atomic_json(self.root / 'camera.json', {'c2w': camera.c2w.tolist(),
                    'K': camera.intrinsics.tolist(), 'width': camera.width, 'height': camera.height})
        self._wait(output)
        with np.load(output, allow_pickle=False) as result:
            return result['rgb'].copy(), result['opacity'].copy(), result['depth'].copy()

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if getattr(self, 'log', None) is not None:
            self.log.close()
        if getattr(self, 'temp', None) is not None:
            self.temp.cleanup()

    def __del__(self):
        self.close()


def capture_native(source, cameras, destinations, runtime, should_stop=lambda: False):
    from PIL import Image
    renderer = NativeRenderer(source, runtime, should_stop)
    try:
        for camera, destination in zip(cameras, destinations):
            rgb, _, _ = renderer.render(camera)
            Image.fromarray(rgb).save(destination)
    finally:
        renderer.close()


def serve(repo, work):
    sys.path[:0] = [str(repo), str(repo / 'thirdparty/3DGRUT-ArtiFixer')]
    import torch
    from data_processing.threedgrut_training import compose_3dgrut_config
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    from threedgrut.datasets.protocols import Batch

    conf = compose_3dgrut_config('apps/colmap_3dgut_sparse_mcmc_lpips', [],
                                repo / 'thirdparty/3DGRUT-ArtiFixer/configs')
    if conf.render.method != '3dgut':
        raise ValueError('Authors configuration must select 3DGUT')
    model = MixtureOfGaussians(conf).cuda()
    model.init_from_ply(str(work / 'source.ply'), init_model=False)
    model.validate_fields()
    model.build_acc()
    (work / 'ready').touch()
    calibration = None
    while True:
        request = work / 'camera.json'
        if not request.is_file():
            time.sleep(.05)
            continue
        camera = json.loads(request.read_text())
        request.unlink()
        K = np.asarray(camera['K'], dtype=np.float32)
        key = (camera['width'], camera['height'], *K.ravel())
        if calibration != key:
            params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0], dtype=np.float32)
            info, origins, directions, camera_name = ColmapDataset._create_perspective_camera(
                params, camera['width'], camera['height'])
            origins, directions = origins.cuda(), directions.cuda()
            ray_z = torch.nn.functional.normalize(directions, dim=-1)[0, ..., 2].cpu().numpy()
            calibration = key
        batch = Batch(rays_ori=origins, rays_dir=directions,
                      T_to_world=torch.tensor(camera['c2w'], dtype=torch.float32, device='cuda')[None],
                      is_override=False, **{f'intrinsics_{camera_name}': info})
        with torch.inference_mode():
            out = model(batch)
            # torchvision.save_image (used upstream) rounds RGB by adding .5.
            rgb = out['pred_rgb'][0].mul(255).add(.5).clamp(0, 255).byte().cpu().numpy()
            alpha = out['pred_opacity'][0, ..., 0].cpu().numpy()
            distance = out['pred_dist'].reshape(camera['height'], camera['width']).cpu().numpy()
        if not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
            raise ValueError('3DGUT returned invalid opacity')
        depth = expected_camera_z(distance, alpha, ray_z)
        # Match the authors' opacity PNG export and loader, including quantization.
        opacity = np.round(alpha * 255).astype(np.uint8).astype(np.float32) / 255
        temp = work / 'pending.npz'
        np.savez(temp, rgb=rgb, opacity=opacity, depth=depth)
        temp.replace(work / 'frame.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    serve(args.repo, args.work)
