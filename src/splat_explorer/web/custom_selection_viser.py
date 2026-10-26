"""An isolated, temporary Viser 1.1.0 server for manual Splatfix captures."""
from __future__ import annotations

import logging
import threading
import time
import uuid

from ..scene.catalog import openable_scene_path
from ..splatfix.manual_selection import ManualSelection
from ..rendering.viser_viewer import _VisorCapture, _install_splats, _set_client_camera, _view_pose
from .scene_run_studio import _cfg_get

logger = logging.getLogger(__name__)
CUSTOM_VISER_VERSION = '1.1.0'


class CustomSelectionViser:
    """One Custom session, independent of the main and results viewers.

    No live-scene mailbox, agent following, or shared HTTP capture API is used.
    The session's own client supplies both the saved pose and WebGL readback.
    """
    def __init__(self, cfg, *, server_factory=None, scene_loader=None,
                 background=True, idle_timeout=45.):
        self.cfg = cfg
        self._server_factory = server_factory
        self._scene_loader = scene_loader
        self._background = background
        self._idle_timeout = idle_timeout
        self._lock = threading.RLock()
        self._server = self._selection = self._checkpoint = None
        self._session = None
        self._status = 'idle'
        self._error = None
        self._port = None
        self._disconnected_since = None

    def start(self, checkpoint):
        with self._lock:
            self._maintain_locked(time.monotonic())
            if self._server is not None:
                raise ValueError('Finish or close the current Custom selection tab first')
            factory = self._server_factory
            if factory is None:
                import viser
                if viser.__version__ != CUSTOM_VISER_VERSION:
                    raise ValueError('Custom selection requires Viser 1.1.0 in the dashboard runtime; run scripts/start.sh')
                factory = viser.ViserServer
            port = int(_cfg_get(self.cfg, 'viewer.port', 8080)) + 4
            server = factory(host=_cfg_get(self.cfg, 'viewer.host', '0.0.0.0'), port=port)
            # Docker publishes this exact port; do not silently serve a different
            # viewer when another process occupies it.
            if server.get_port() != port:
                server.stop()
                raise ValueError(f'Custom viewer port {port} is already in use')
            self._server, self._checkpoint = server, checkpoint
            self._session = session = uuid.uuid4().hex
            self._status, self._error, self._port = 'loading', None, port
            self._disconnected_since = None
            if self._background:
                threading.Thread(target=self._load, args=(session,), daemon=True).start()
                threading.Thread(target=self._watchdog, args=(session,), daemon=True).start()
            return self._snapshot_locked()

    def _load(self, session):
        with self._lock:
            if session != self._session or self._server is None:
                return
            cp = self._checkpoint
        try:
            from ..scene import load_scene
            metadata = cp.manifest['metadata']
            loading = metadata.get('scene_load', {})
            scene = (self._scene_loader or load_scene)(
                openable_scene_path(cp.manifest['scene_path']),
                min_opacity=loading.get('min_opacity', 0.), lod_level=loading.get('lod_level', 0))
            view = _view_pose(scene, metadata['up_axis'],
                              float(_cfg_get(self.cfg, 'camera.fov_deg', 75.)))
            with self._lock:
                if session != self._session or self._server is None:
                    return
                server = self._server
                server.gui.configure_theme(control_layout='floating', control_width='medium',
                                           show_logo=False, show_share_button=False, dark_mode=True)
                server.gui.set_panel_label('Custom view finding')
                server.scene.world_axes.visible = False
                _install_splats(server, scene, int(_cfg_get(self.cfg, 'viewer.max_splats', 0)))
                state = {**metadata['manual_scene'], 'status': 'ready'}
                capture = _VisorCapture(server, threading.Lock(), [])
                selection = ManualSelection(server, capture, state)
                selection.start({'checkpoint': str(cp.root)})
                # Use the same checkpoint instance that save/finish mutate.
                self._selection, self._checkpoint = selection, selection.checkpoint

                def connect(client):
                    _set_client_camera(client, view['center'], view['center'] + view['forward'],
                                       up=view['up'], fov=view['vfov'])
                    selection.apply_fov(client)

                server.on_client_connect(connect)
                for client in server.get_clients().values():
                    connect(client)
                self._status = 'ready'
                self._disconnected_since = time.monotonic()
        except Exception as exc:
            logger.exception('Custom viewer scene load failed')
            with self._lock:
                if session == self._session:
                    self._error = str(exc)
                    self._stop_locked('error')

    def _snapshot_locked(self):
        cp = self._checkpoint
        return {'ok': self._status != 'error', 'session': self._session, 'status': self._status,
                'error': self._error, 'viewer_port': self._port if self._server else None,
                'viser_version': CUSTOM_VISER_VERSION, 'checkpoint': str(cp.root) if cp else None,
                'saved_views': len(cp.views) if cp else 0, 'target_views': cp.target_views if cp else 0,
                'viewer_path': '/splatfix/custom?session=' + self._session if self._session else None}

    def snapshot(self, session):
        with self._lock:
            self._require_session(session)
            self._maintain_locked(time.monotonic())
            return self._snapshot_locked()

    def _require_session(self, session):
        if not session or session != self._session:
            raise ValueError('Custom session is no longer available. Start Custom selection from Splatfix.')

    def release(self, session):
        with self._lock:
            self._require_session(session)
            self._stop_locked('closed')
            return self._snapshot_locked()

    def _stop_locked(self, status):
        selection = self._selection
        # A readback already in flight must finish writing before shutdown.
        if selection is not None:
            if not selection.lock.acquire(blocking=False):
                return False
            try:
                if selection.folder is not None:
                    selection._remove()
            finally:
                selection.lock.release()
        server, self._server = self._server, None
        self._selection = None
        self._status = status
        if server is not None:
            server.stop()
        return True

    def _maintain_locked(self, now):
        if self._server is None or self._status == 'loading':
            return
        if self._selection is not None and self._selection.folder is None:
            self._stop_locked('finished' if self._checkpoint.complete else 'closed')
        elif self._server.get_clients():
            self._disconnected_since = None
        else:
            if self._disconnected_since is None:
                self._disconnected_since = now
            if now - self._disconnected_since >= self._idle_timeout:
                self._stop_locked('closed')

    def _watchdog(self, session):
        while True:
            time.sleep(5)
            with self._lock:
                if session != self._session or self._server is None:
                    return
                self._maintain_locked(time.monotonic())
