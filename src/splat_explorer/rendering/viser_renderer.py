"""Harness RGB: visor WebGL when a tab is connected, else CPU splat layers.

Each step the renderer checks the viewer's capture API (:8081). If a visor
tab is connected, the agent camera is POSTed there and the frame comes from
viser's WebGL `get_render()` — the VLM sees exactly what the browser shows.
Any failure (no tab, throttled tab, timeout) falls back to the CPU rasterizer
for that frame only, so episodes never hang and recover as soon as a tab is
back. `last_backend` records which path produced the latest frame.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from ..scene import GaussianScene
from .base import Camera
from .cpu_splat_renderer import CpuSplatRenderer

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:8081"
_DEFAULT_VIEWER = "http://localhost:8080"


class ViserCaptureError(RuntimeError):
    """The visor render API refused or failed a capture."""


class FastDepthRenderer:
    """Min-z disc stamp of projected gaussians — fast enough to sit next to a
    WebGL RGB capture, accurate enough for move_toward unprojection."""

    def __init__(self, scene: GaussianScene, near: float = 0.05, max_radius_px: int = 8):
        self.means = scene.means
        self.max_scale = scene.scales.max(axis=1)
        self.near = float(near)
        self.max_radius = int(max_radius_px)

    def render_depth(self, camera: Camera) -> np.ndarray:
        W, H = camera.width, camera.height
        fx, fy = camera.fx, camera.fy
        w2c = camera.w2c
        pcam = self.means @ w2c[:3, :3].T + w2c[:3, 3]
        z = pcam[:, 2]
        keep = z > self.near
        if not np.any(keep):
            return np.full((H, W), np.inf, dtype=np.float32)

        pcam, z = pcam[keep], z[keep]
        u = fx * pcam[:, 0] / z + W / 2.0
        v = fy * pcam[:, 1] / z + H / 2.0
        r = np.minimum(
            np.maximum(np.ceil(2.0 * self.max_scale[keep] * max(fx, fy) / z), 1.0),
            float(self.max_radius),
        ).astype(np.int32)

        depth = np.full((H, W), np.inf, dtype=np.float32)
        for radius in np.unique(r):
            sel = r == radius
            ui = np.round(u[sel]).astype(np.int32)
            vi = np.round(v[sel]).astype(np.int32)
            zi = z[sel].astype(np.float32)
            rad = int(radius)
            for dv in range(-rad, rad + 1):
                for du in range(-rad, rad + 1):
                    if du * du + dv * dv > rad * rad:
                        continue
                    px = ui + du
                    py = vi + dv
                    on = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                    if not np.any(on):
                        continue
                    np.minimum.at(depth, (py[on], px[on]), zi[on])
        return depth


class ViserCaptureRenderer:
    """RGB from the visor WebGL view when a tab is connected; CPU splat
    rasterizer otherwise. The choice is re-evaluated every frame."""

    def __init__(
        self,
        scene: GaussianScene,
        url: str | None = None,
        viewer_url: str | None = None,
        timeout_s: float = 60.0,
        max_splat_radius_px: int = 120,
    ):
        self.url = (url or os.environ.get("VISER_RENDER_URL") or _DEFAULT_URL).rstrip("/")
        self.viewer_url = (
            viewer_url or os.environ.get("VISER_VIEWER_URL") or _DEFAULT_VIEWER
        ).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._cpu = CpuSplatRenderer(scene, max_splat_radius_px=max_splat_radius_px)
        self._depth = FastDepthRenderer(scene)
        self.last_backend: str | None = None  # "viser" | "cpu_splats", per frame
        self._warned = False
        logger.info("Viser capture renderer -> %s (open the visor at %s)", self.url, self.viewer_url)

    def render(self, camera: Camera) -> np.ndarray:
        return self.render_with_depth(camera)[0]

    def render_with_depth(self, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        try:
            if int(self._health().get("clients", 0)) < 1:
                raise ViserCaptureError(f"no visor tab connected — open {self.viewer_url}")
            rgb = self._capture(camera)
        except ViserCaptureError as exc:
            if not self._warned:
                logger.warning("Visor capture unavailable (%s); rendering with cpu_splats", exc)
                self._warned = True
            self.last_backend = "cpu_splats"
            return self._cpu.render_with_depth(camera)
        if self._warned:
            logger.info("Visor capture recovered; back to WebGL frames")
            self._warned = False
        self.last_backend = "viser"
        return rgb, self._depth.render_depth(camera)

    def _capture(self, camera: Camera) -> np.ndarray:
        body = json.dumps({
            "position": np.asarray(camera.position, dtype=np.float64).tolist(),
            "wxyz": camera.rotation_wxyz().tolist(),
            "fov": camera.vertical_fov_rad(),
            "width": int(camera.width),
            "height": int(camera.height),
        }).encode()
        t0 = time.perf_counter()
        raw = self._request("POST", "/render", body=body, content_type="application/json")
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))
        rgb, alpha = img[..., :3].astype(np.float32), img[..., 3:4].astype(np.float32) / 255.0
        # PNG captures use a transparent clear; composite over black to match
        # the live visor canvas.
        out = np.clip(rgb * alpha, 0, 255).astype(np.uint8)
        logger.info("Visor capture %dx%d in %.2fs", camera.width, camera.height, time.perf_counter() - t0)
        return out

    def _json(self, method: str, path: str, timeout: float | None = None) -> dict:
        raw = self._request(method, path, timeout=timeout)
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError as exc:
            raise ViserCaptureError(f"Invalid JSON from {self.url}{path}") from exc

    def _health(self) -> dict:
        return self._json("GET", "/health", timeout=2.0)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float | None = None,
    ) -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise ViserCaptureError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ViserCaptureError(f"{method} {self.url}{path} failed: {reason}") from exc
