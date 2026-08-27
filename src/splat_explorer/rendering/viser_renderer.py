"""Harness RGB from visor WebGL; depth from the CPU EWA rasterizer (on demand).

RGB is captured from the dashboard's always-on visor iframe, which is sized
to the selected VLM resolution (default 960x720). Depth uses
CpuSplatRenderer.render_depth() — the same anisotropic footprints as the
old CPU images — but only when the episode loop asks for it (compute_depth,
send_depth, view_depth, or a lazy move_toward). render() is RGB-only so
default runs skip the slow CPU depth pass.
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


class ViserCaptureRenderer:
    """RGB from the dashboard visor WebGL view at the selected VLM size."""

    def __init__(
        self,
        scene: GaussianScene,
        url: str | None = None,
        viewer_url: str | None = None,
        timeout_s: float = 180.0,
        client_wait_s: float = 90.0,
        max_splat_radius_px: int = 120,
    ):
        self.url = (url or os.environ.get("VISER_RENDER_URL") or _DEFAULT_URL).rstrip("/")
        self.viewer_url = (
            viewer_url or os.environ.get("VISER_VIEWER_URL") or _DEFAULT_VIEWER
        ).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.client_wait_s = float(client_wait_s)
        # Same EWA pipeline as cpu_splats, but render_depth() skips RGB.
        self._cpu = CpuSplatRenderer(scene, max_splat_radius_px=max_splat_radius_px)
        self.last_backend: str | None = None
        self._visor_settled = False
        logger.info(
            "Viser capture renderer -> %s (dashboard visor at VLM resolution; %s is HD viewing only)",
            self.url, self.viewer_url,
        )

    def render(self, camera: Camera) -> np.ndarray:
        """RGB from the visor only — no CPU depth pass."""
        self._wait_for_visor(camera.width, camera.height)
        rgb = self._capture(camera)
        self.last_backend = "viser"
        return rgb

    def render_depth(self, camera: Camera) -> np.ndarray:
        """CPU front-surface depth only (used lazily by move_toward)."""
        return self._cpu.render_depth(camera)

    def render_with_depth(self, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.render(camera)
        depth = self.render_depth(camera)
        return rgb, depth

    def _wait_for_visor(self, width: int, height: int) -> None:
        deadline = time.time() + self.client_wait_s
        last = "no visor tab"
        warned = False
        while time.time() <= deadline:
            try:
                info = self._health()
            except ViserCaptureError as exc:
                last = str(exc)
                time.sleep(0.5)
                continue
            viewports = info.get("viewports") or []
            sized = [
                v for v in viewports
                if v.get("usable")
                and int(v.get("width") or 0) >= width * 0.9
                and int(v.get("height") or 0) >= height * 0.9
            ]
            if sized:
                if not self._visor_settled:
                    time.sleep(0.8)
                    self._visor_settled = True
                return
            last = (
                f"{info.get('clients', 0)} connected, none at {width}x{height} "
                f"(viewports={viewports}). Keep the episode dashboard tab "
                "visible so its capture visor can answer."
            )
            if not warned:
                logger.warning("Waiting for the dashboard capture visor: %s", last)
                warned = True
            time.sleep(0.5)
        raise ViserCaptureError(last)

    def _capture(self, camera: Camera) -> np.ndarray:
        body = json.dumps({
            "position": np.asarray(camera.position, dtype=np.float64).tolist(),
            "wxyz": camera.rotation_wxyz().tolist(),
            "fov": camera.vertical_fov_rad(),
            "width": int(camera.width),
            "height": int(camera.height),
        }).encode()
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        raw = b""
        for attempt in range(1, 4):
            try:
                raw = self._request("POST", "/render", body=body, content_type="application/json")
                last_exc = None
                break
            except ViserCaptureError as exc:
                last_exc = exc
                msg = str(exc).lower()
                retryable = "closed connection" in msg or "timed out" in msg or "timeout" in msg
                if not retryable or attempt == 3:
                    raise
                logger.warning("Visor capture attempt %d failed (%s); retrying", attempt, exc)
                time.sleep(0.4 * attempt)
        if last_exc is not None:
            raise last_exc
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        need_w, need_h = int(camera.width), int(camera.height)
        if img.shape[1] != need_w or img.shape[0] != need_h:
            logger.warning(
                "Capture size %dx%d != VLM %dx%d",
                img.shape[1], img.shape[0], need_w, need_h,
            )
            from .viser_viewer import _center_crop_and_resize
            img = _center_crop_and_resize(img, need_w, need_h)
        logger.info("Visor capture %dx%d in %.2fs", need_w, need_h, time.perf_counter() - t0)
        return img

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
