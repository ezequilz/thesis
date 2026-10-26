"""Temporary Viser controls for collecting calibrated, reusable manual views."""
from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .checkpoint import Checkpoint, camera_from_record, camera_to_record
from ..rendering.base import Camera
from ..scene.catalog import openable_scene_path


def camera_from_client(client, width, height):
    """Viser uses OpenCV quaternions and a vertical field of view."""
    live = client.camera
    wxyz = np.array(live.wxyz, dtype=float)
    camera = Camera(
        position=np.array(live.position, dtype=float),
        rotation=Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix(),
        width=width, height=height,
        fov_deg=float(np.degrees(2 * np.arctan(np.tan(float(live.fov) / 2) * width / height))),
    )
    return camera_from_record(camera_to_record(camera))


class ManualSelection:
    def __init__(self, server, capture, scene_state):
        self.server, self.capture, self.scene_state = server, capture, scene_state
        self.lock = threading.Lock()
        self.folder = None
        self.checkpoint = None

    def start(self, params):
        with self.lock:
            if self.folder is not None:
                raise ValueError('Finish or close the current Custom selection in Viser first')
            cp = Checkpoint.load(openable_scene_path(params['checkpoint']))
            metadata = cp.manifest['metadata']
            if metadata.get('selection') != 'manual' or cp.complete:
                raise ValueError('Expected an unfinished manual checkpoint')
            self.checkpoint = cp
            self.expected = metadata['manual_scene']
            self.width, self.height = metadata['width'], metadata['height']
            self.seva = metadata.get('capture_set') == 'seva'
            self.horizontal_fov = 54. if self.seva else metadata.get('horizontal_fov')
            self.depth_renderer = None
            self.folder = self.server.gui.add_folder('Custom view finding')
            with self.folder:
                self.status = self.server.gui.add_markdown(self.progress())
                self.save_button = self.server.gui.add_button('Save current position')
                self.finish_button = self.server.gui.add_button('Save selection')
                self.close_button = self.server.gui.add_button('Close (keep saved views)')
            self.save_button.on_click(self.save_current)
            self.finish_button.on_click(self.finish)
            self.close_button.on_click(self.close)
            for client in getattr(self.server, 'get_clients', lambda: {})().values():
                self.apply_fov(client)
            return {'ok': True}

    def apply_fov(self, client):
        if self.folder is not None and (self.seva or self.horizontal_fov):
            aspect = float(getattr(client.camera, 'aspect', 1024 / 576))
            if not np.isfinite(aspect) or aspect <= 0:
                aspect = 1024 / 576
            fov = float(2 * np.arctan(np.tan(np.radians(54. if self.seva else self.horizontal_fov) / 2) / aspect))
            # Save clicks should not queue another projection change when the
            # preview already has the requested FOV.
            if not np.isclose(float(client.camera.fov), fov, rtol=0, atol=1e-6):
                client.camera.fov = fov

    def save_seva(self, image, camera):
        from .oraclegs.captures import save_capture
        from ..scene import load_scene
        from ..rendering.cpu_splat_renderer import CpuSplatRenderer
        cp = self.checkpoint
        if self.depth_renderer is None:
            loading = cp.manifest['metadata'].get('scene_load', {})
            self.depth_renderer = CpuSplatRenderer(load_scene(openable_scene_path(cp.manifest['scene_path']),
                min_opacity=loading.get('min_opacity', 0.), lod_level=loading.get('lod_level', 0)))
        depth = self.depth_renderer.render_depth(camera)
        view = {'id':f'{len(cp.views):03d}', 'camera':camera_to_record(camera),
                'original_rgb':None, 'repaired_rgb':None, 'metadata':{'selection':'manual'}}
        view['seva_capture'] = save_capture(cp, view, camera, np.array(image.convert('RGB')), depth)
        self.require_scene()
        cp.views.append(view)
        try:
            cp.save()
        except Exception:
            cp.views.pop()
            raise

    def progress(self):
        cp = self.checkpoint
        return f'**{len(cp.views)} / {cp.target_views} views saved** · {self.width} × {self.height}' + (f' · **{54 if self.seva else self.horizontal_fov:g}° horizontal FOV**' if self.seva or self.horizontal_fov else '')

    def require_scene(self):
        state = self.scene_state
        if (state.get('status') != 'ready' or state.get('id') != self.expected['id']
                or state.get('generation') != self.expected['generation']):
            raise ValueError('The selected scene is not ready or has changed. Wait for it to load, or start a new Custom selection.')

    def save_current(self, event):
        # Viser dispatches callbacks on threads. Ignore concurrent/double clicks.
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.folder is None:
                return
            self.save_button.disabled = True
            self.require_scene()
            if event.client is None:
                raise ValueError('Save from a connected Viser tab')
            if self.checkpoint.complete:
                raise ValueError('Requested view selection is already complete')
            self.apply_fov(event.client)
            camera = camera_from_client(event.client, self.width, self.height)
            if self.seva or self.horizontal_fov:
                camera.fov_deg = 54. if self.seva else float(self.horizontal_fov)
            if any(np.allclose(camera.c2w, camera_from_record(v).c2w, atol=1e-5)
                   for v in self.checkpoint.views):
                raise ValueError('This position is already saved. Move to another view.')
            raw = self.capture.render({
                'position': camera.position.tolist(), 'wxyz': camera.rotation_wxyz().tolist(),
                'fov': camera.vertical_fov_rad(), 'width': camera.width, 'height': camera.height,
                'client_id': event.client.client_id,
            }, any_client=True)
            self.require_scene()
            with Image.open(io.BytesIO(raw)) as image:
                if self.seva:
                    self.save_seva(image, camera)
                else:
                    self.checkpoint.add_view(image, camera, {'selection': 'manual'})
            self.status.content = self.progress()
        except Exception as exc:
            self.status.content = self.progress() + '\n\n**Could not save:** ' + str(exc)
        finally:
            if self.folder is not None:
                self.save_button.disabled = self.checkpoint.complete
            self.lock.release()

    def finish(self, event):
        with self.lock:
            if self.folder is None:
                return
            if not self.checkpoint.views:
                self.status.content = self.progress() + '\n\nSave at least one position first.'
                return
            if self.checkpoint.endpoint_only and len(self.checkpoint.views) < 2:
                self.status.content = self.progress() + '\n\nEndpoint checkpoints require at least two positions.'
                return
            old_target = self.checkpoint.target_views
            self.checkpoint.manifest['target_views'] = len(self.checkpoint.views)
            try:
                self.checkpoint.save()
            except Exception as exc:
                self.checkpoint.manifest['target_views'] = old_target
                self.status.content = self.progress() + '\n\n**Could not finish:** ' + str(exc)
                return
            self._remove()

    def _remove(self):
        self.folder.remove()
        self.folder = None

    def close(self, event):
        with self.lock:
            if self.folder is not None:
                self._remove()
