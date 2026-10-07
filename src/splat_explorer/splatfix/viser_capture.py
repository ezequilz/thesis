"""Pose-only Viser RGB capture, with a durable LRZ-to-manager mailbox.

The GPU worker requests camera records; the desktop manager loads the source in
the same capture visor as the harness, captures RGB, and uploads a verified batch.
No VLM or image editor is involved. CUDA still provides opacity and depth.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import time
import uuid

from PIL import Image

from .checkpoint import atomic_json, camera_from_record, camera_to_record


class CaptureSession:
    rgb_backend = 'viser'

    def __init__(self, scene_path, *, up_axis='-y', lod_level=0, url=None,
                 should_stop=lambda: False):
        from ..rendering.viser_renderer import ViserCaptureRenderer
        from ..scene.catalog import SceneSpec, publish_live_scene
        self.renderer = ViserCaptureRenderer(None, url=url, any_client=True)
        self.should_stop = should_stop
        self.ident = 'splatfix-capture-' + uuid.uuid4().hex
        self.generation = int(time.time() * 1000)
        publish_live_scene(SceneSpec(self.ident, 'Splatfix RGB capture', Path(scene_path),
                                    up_axis=up_axis, lod_level=lod_level),
                           self.generation, reload=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            self.check_stop()
            try:
                if self.ready():
                    return
            except RuntimeError:
                pass
            time.sleep(.5)
        raise RuntimeError('Viser did not load the requested scene; keep the harness capture visor connected')

    def check_stop(self):
        if self.should_stop():
            raise InterruptedError('Viser capture stopped')

    def ready(self):
        scene = self.renderer._health().get('scene', {})
        return (scene.get('status') == 'ready' and scene.get('id') == self.ident
                and scene.get('generation') == self.generation)

    def render(self, camera):
        self.check_stop()
        if not self.ready():
            raise RuntimeError('Viser source scene changed during Splatfix capture')
        rgb = self.renderer.render(camera)
        if not self.ready():
            raise RuntimeError('Viser source scene changed during Splatfix capture')
        if rgb.shape != (camera.height, camera.width, 3):
            raise ValueError('Viser capture dimensions differ from the calibrated camera')
        return rgb


def capture_rgb(scene_path, cameras, destinations, *, up_axis='-y', lod_level=0,
                should_stop=lambda: False):
    """Capture every requested pose or fail; never retain another renderer's RGB."""
    cameras, destinations = list(cameras), [Path(p) for p in destinations]
    if len(cameras) != len(destinations):
        raise ValueError('Capture cameras and destinations must match')
    mailbox = os.environ.get('SPLATFIX_VISER_MAILBOX')
    if not mailbox:
        session = CaptureSession(scene_path, up_axis=up_axis, lod_level=lod_level,
                                 should_stop=should_stop)
        for camera, path in zip(cameras, destinations):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(session.render(camera)).save(path)
        return
    root = Path(mailbox)
    ident = uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'request.json', {
        'id': ident, 'scene_path': str(scene_path), 'up_axis': up_axis, 'lod_level': lod_level,
        'cameras': [camera_to_record(camera) for camera in cameras]})
    response = root / ident / 'response.json'
    deadline = time.monotonic() + 7200
    while not response.is_file():
        if should_stop():
            raise InterruptedError('Stopped waiting for Viser RGB captures')
        if time.monotonic() > deadline:
            raise TimeoutError('Manager did not complete Viser RGB capture within two hours')
        time.sleep(.5)
    result = json.loads(response.read_text())
    if result.get('id') != ident or result.get('renderer') != 'viser' or result.get('error'):
        raise RuntimeError(result.get('error') or 'Invalid Viser capture response')
    hashes = result.get('sha256', [])
    if len(hashes) != len(cameras):
        raise ValueError('Incomplete Viser capture batch')
    for index, (camera, destination, digest) in enumerate(zip(cameras, destinations, hashes)):
        source = response.parent / f'{index:05d}.png'
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('Viser capture checksum mismatch')
        with Image.open(source) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError('Viser capture dimensions differ from the calibrated camera')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def capture_plus_rgb(root, trajectory, *, up_axis='-y', should_stop=lambda: False):
    """Replace reconstructed conditioning RGB with same-pose Viser captures."""
    import numpy as np
    from ..rendering.base import Camera
    root = Path(root)
    distilled = json.loads((root / 'distillation.json').read_text())
    tf = trajectory['transforms']
    cameras = [Camera(np.asarray(f['transform_matrix'])[:3, 3],
                      np.asarray(f['transform_matrix'])[:3, :3],
                      width=tf['w'], height=tf['h'],
                      fov_deg=float(np.degrees(2 * np.arctan(tf['w'] / (2 * tf['fl_x'])))))
               for f in tf['frames']]
    capture_rgb(distilled['splat_path'], cameras,
                [Path(distilled['render_dir']) / 'renders' / f'{i:05d}.png' for i in range(len(cameras))],
                up_axis=up_axis, should_stop=should_stop)


class CaptureMirror:
    """Service the GPU's one outstanding batch on the manager's normal poll loop."""
    def __init__(self, root, remote_root, container_root, checkpoint, endpoint, ssh, sync,
                 *, url=None, should_stop=lambda: False, progress=lambda **event: None):
        self.root, self.remote_root = Path(root), remote_root
        self.container_root, self.checkpoint = PurePosixPath(container_root), checkpoint
        self.endpoint, self.ssh, self.sync = endpoint, ssh, sync
        self.url, self.should_stop, self.progress = url, should_stop, progress
        self.done = set()

    def poll(self):
        remote = self.remote_root + '/viser-capture'
        request_path = shlex.quote(remote + '/request.json')
        raw = self.ssh(f'if [ -f {request_path} ]; then cat {request_path}; fi')
        if not raw.strip():
            return
        request = json.loads(raw)
        ident = request.get('id', '')
        if not isinstance(ident, str) or not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Invalid Viser capture request id')
        if ident in self.done:
            return
        complete = shlex.quote(remote + '/' + ident + '/response.json')
        if self.ssh(f'if [ -f {complete} ]; then cat {complete}; fi').strip():
            self.done.add(ident)
            return
        target = self.root / 'viser-capture' / ident
        target.mkdir(parents=True, exist_ok=True)
        result = {'id': ident, 'renderer': 'viser'}
        try:
            cameras = [camera_from_record({'camera': value}) for value in request['cameras']]
            if not cameras or len(cameras) > 10000:
                raise ValueError('Invalid Viser capture batch size')
            source = PurePosixPath(request['scene_path'])
            relative = source.relative_to(self.container_root)
            if '..' in relative.parts:
                raise ValueError('Viser scene escaped the GPU job')
            if relative.parts[0] == 'source':
                scene = Path(self.checkpoint.manifest['scene_path'])
            elif relative.parts[0] == 'results' and source.suffix == '.ply':
                scene = self.root / 'viser-scenes' / (ident + '.ply')
                scene.parent.mkdir(parents=True, exist_ok=True)
                self.sync(self.endpoint + ':' + self.remote_root + '/' + str(relative), str(scene))
            else:
                raise ValueError('Viser capture requires this job’s source or reconstructed PLY')
            session = CaptureSession(scene, up_axis=request['up_axis'],
                                     lod_level=request['lod_level'], url=self.url,
                                     should_stop=self.should_stop)
            hashes = []
            for index, camera in enumerate(cameras):
                self.progress(phase='viser_capture', message=f'Viser RGB capture {index + 1}/{len(cameras)}')
                path = target / f'{index:05d}.png'
                Image.fromarray(session.render(camera)).save(path)
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            result['sha256'] = hashes
        except InterruptedError:
            raise
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        # Upload pixels first and publish the completion marker last. This also
        # makes manager restarts safe while rsync is in progress.
        marker = target / 'response.json'
        marker.unlink(missing_ok=True)
        self.ssh('mkdir -p ' + shlex.quote(remote + '/' + ident))
        self.sync(str(target) + '/', self.endpoint + ':' + remote + '/' + ident + '/')
        atomic_json(marker, result)
        self.sync(str(marker), self.endpoint + ':' + remote + '/' + ident + '/response.json')
        self.done.add(ident)
