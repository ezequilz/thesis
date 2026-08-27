"""Background image-to-image repair queued from report_artifact.

When the inspector sets regenerate=yes (or fix=1), the harness keeps
exploring. If the dashboard/config tick `image_regeneration` is on, a
worker thread sends the already-rendered RGB PNG to gpt-image-2 via
CliRelay (`/v1/images/edits`):

  "Please regenerate and fix this image. Repair artifacts and upscale to
   higher resolution."

If the tick is off (default), the decision is recorded and the pipeline
stops before the paid request.

Outputs (next to the episode's step_NNN.png):
  step_NNN_regen.png   repaired pixels, if any were extracted
  step_NNN_regen.json  status, timing, model, reply excerpt (no giant b64)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

logger = logging.getLogger(__name__)

REGENERATE_PROMPT = (
    "Please regenerate and fix this image. Repair artifacts and upscale to "
    "higher resolution."
)
# Paid image model on CliRelay (~4¢/call). The inspector VLM stays on its own model.
IMAGE_MODEL = "gpt-image-2"

# Image generation can be slower than a tool-call decide().
REQUEST_TIMEOUT_S = 180
MAX_WORKERS = 2
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
_DATA_URL = re.compile(
    r"data:image/([A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)",
)
OnDone = Callable[["RegenerateResult"], None]


@dataclass
class RegenerateResult:
    """Outcome of one background repair request."""

    step: int
    status: str  # queued | ok | no_image | error | disabled | skipped
    seconds: float = 0.0
    model: str = ""
    image_name: str | None = None
    extra_image_names: list[str] = field(default_factory=list)
    reply_text: str = ""
    error: str | None = None
    n_images: int = 0


def regenerate_frame_name(step: int) -> str:
    return f"step_{int(step):03d}_regen.png"


def regenerate_meta_name(step: int) -> str:
    return f"step_{int(step):03d}_regen.json"


def extract_images(payload: Any) -> list[bytes]:
    """Pull image bytes out of a chat-completions-like response payload.

    Walks dicts/lists/strings for data:image/...;base64 URLs, OpenAI-style
    b64_json fields, and raw PNG/JPEG blobs. Deduplicates identical buffers.
    """
    found: list[bytes] = []
    seen: set[int] = set()

    def add(data: bytes) -> None:
        if len(data) < 24:
            return
        key = hash((len(data), data[:32], data[-16:]))
        if key in seen:
            return
        seen.add(key)
        found.append(data)

    def add_b64(text: str) -> None:
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 32:
            return
        try:
            add(base64.b64decode(compact, validate=False))
        except Exception:
            return

    def walk(obj: Any) -> None:
        if obj is None or isinstance(obj, (int, float, bool)):
            return
        if isinstance(obj, (bytes, bytearray)):
            add(bytes(obj))
            return
        if isinstance(obj, str):
            matched = False
            for match in _DATA_URL.finditer(obj):
                add_b64(match.group(2))
                matched = True
            if not matched and obj.startswith(("iVBOR", "/9j/")):
                add_b64(obj)
            return
        if isinstance(obj, dict):
            b64 = obj.get("b64_json")
            if isinstance(b64, str):
                add_b64(b64)
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                walk(item)

    walk(payload)
    return found


def save_png(data: bytes, path: Path) -> None:
    """Write image bytes as RGB PNG (converts JPEG/etc. so the dashboard can serve it)."""
    if data.startswith(PNG_MAGIC):
        path.write_bytes(data)
        return
    img = Image.open(io.BytesIO(data))
    img.convert("RGB").save(path, format="PNG")


def summarize_payload(obj: Any, max_str: int = 400) -> Any:
    """JSON-safe copy of a response with giant base64 blobs replaced."""
    if isinstance(obj, str):
        if obj.startswith("data:image"):
            return f"<data-url {len(obj)} chars>"
        if len(obj) > max_str:
            return obj[:max_str] + f"... <{len(obj)} chars total>"
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes {len(obj)}>"
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in ("b64_json", "data") and isinstance(value, str) and len(value) > 80:
                out[key] = f"<omitted {len(value)} chars>"
            else:
                out[key] = summarize_payload(value, max_str)
        return out
    if isinstance(obj, (list, tuple)):
        return [summarize_payload(item, max_str) for item in obj]
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def response_to_dict(response: Any) -> dict:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    if isinstance(response, dict):
        return response
    return {"repr": repr(response)}


def _message_text(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p).strip()
    return ""


def _file_data_url(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def ask_regenerate(client, model: str, image_path: Path,
                   timeout_s: float = REQUEST_TIMEOUT_S) -> tuple[dict, str | None]:
    """Send the RGB frame + static repair prompt to gpt-image-2.

    Prefers CliRelay `/v1/images/edits` (native image I/O). Falls back to
    chat.completions with the same model if the images endpoint is missing.
    Returns (response dict, error).
    """
    edit = getattr(getattr(client, "images", None), "edit", None)
    if edit is not None:
        payload, error = _images_edit(edit, model, image_path, timeout_s)
        if error is None or "TypeError" not in (error or ""):
            return payload, error
        logger.warning("images.edit rejected args (%s); trying chat.completions", error)
    return _chat_regenerate(client, model, image_path, timeout_s)


def _images_edit(edit, model: str, image_path: Path,
                 timeout_s: float) -> tuple[dict, str | None]:
    attempts = (
        {"image": "file", "timeout": True},
        {"image": "file", "timeout": False},
        {"image": "list", "timeout": False},
    )
    last_error = None
    for spec in attempts:
        try:
            with image_path.open("rb") as fh:
                image = [fh] if spec["image"] == "list" else fh
                kwargs: dict[str, Any] = {
                    "model": model,
                    "image": image,
                    "prompt": REGENERATE_PROMPT,
                }
                if spec["timeout"]:
                    kwargs["timeout"] = timeout_s
                return response_to_dict(edit(**kwargs)), None
        except TypeError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        except Exception as exc:
            logger.exception("Regenerate CliRelay images.edit failed")
            return {}, f"{type(exc).__name__}: {exc}"
    return {}, last_error


def _chat_regenerate(client, model: str, image_path: Path,
                     timeout_s: float) -> tuple[dict, str | None]:
    content = [
        {"type": "text", "text": REGENERATE_PROMPT},
        {"type": "image_url", "image_url": {"url": _file_data_url(image_path)}},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            timeout=timeout_s,
        )
    except TypeError:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            logger.exception("Regenerate CliRelay request failed")
            return {}, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        logger.exception("Regenerate CliRelay request failed")
        return {}, f"{type(exc).__name__}: {exc}"
    return response_to_dict(response), None


def write_meta(episode_dir: Path, result: RegenerateResult, payload: dict | None = None) -> None:
    body = {
        "step": result.step,
        "status": result.status,
        "seconds": round(result.seconds, 3),
        "model": result.model,
        "image_name": result.image_name,
        "extra_image_names": result.extra_image_names,
        "n_images": result.n_images,
        "reply_text": result.reply_text,
        "error": result.error,
        "prompt": REGENERATE_PROMPT,
    }
    if payload is not None:
        body["response"] = summarize_payload(payload)
    path = episode_dir / regenerate_meta_name(result.step)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2))
    tmp.replace(path)


def record_disabled(episode_dir: Path, step: int,
                    reason: str = "image regeneration setting off") -> RegenerateResult:
    """Persist a regenerate=yes decision without calling gpt-image-2."""
    result = RegenerateResult(
        step=step, status="disabled", model=IMAGE_MODEL, error=reason,
    )
    write_meta(episode_dir, result)
    return result


def regenerator_from_policy(policy) -> "Regenerator | None":
    """Build a Regenerator from a live CliRelay client, always on gpt-image-2."""
    client = getattr(policy, "client", None)
    if client is None:
        return None
    return Regenerator(client=client, model=IMAGE_MODEL)


class Regenerator:
    """Runs repair requests on a small thread pool so the episode loop is not blocked."""

    def __init__(self, client, model: str, max_workers: int = MAX_WORKERS,
                 timeout_s: float = REQUEST_TIMEOUT_S):
        self.client = client
        self.model = model
        self.timeout_s = timeout_s
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="regen",
        )
        self._futures: list[Future] = []
        self._lock = threading.Lock()

    def submit(
        self,
        image_path: Path,
        episode_dir: Path,
        step: int,
        on_done: OnDone | None = None,
    ) -> Future:
        queued = RegenerateResult(step=step, status="queued", model=self.model)
        write_meta(episode_dir, queued)
        logger.info("Queued regenerate for step %d (%s)", step, image_path.name)
        future = self._pool.submit(
            self._run, Path(image_path), Path(episode_dir), int(step), on_done,
        )
        with self._lock:
            self._futures.append(future)
        return future

    def wait(self, timeout: float | None = 300.0) -> list[RegenerateResult]:
        """Block until queued jobs finish (called at end of the episode)."""
        with self._lock:
            futures = list(self._futures)
        if not futures:
            self._pool.shutdown(wait=False)
            return []
        logger.info("Waiting for %d regenerate job(s)", len(futures))
        results: list[RegenerateResult] = []
        remaining = timeout
        t0 = time.perf_counter()
        try:
            for future in futures:
                slice_timeout = None if remaining is None else max(0.0, remaining)
                try:
                    results.append(future.result(timeout=slice_timeout))
                except TimeoutError:
                    logger.error("Regenerate job still running after wait timeout")
                except Exception as exc:
                    logger.exception("Regenerate job failed to return: %s", exc)
                if remaining is not None:
                    remaining = timeout - (time.perf_counter() - t0)
        finally:
            self._pool.shutdown(wait=False, cancel_futures=True)
        return results

    def _run(self, image_path: Path, episode_dir: Path, step: int,
             on_done: OnDone | None) -> RegenerateResult:
        t0 = time.perf_counter()
        result = RegenerateResult(step=step, status="error", model=self.model)
        payload: dict | None = None
        try:
            if not image_path.is_file():
                result.error = f"source image missing: {image_path}"
            else:
                payload, error = ask_regenerate(
                    self.client, self.model, image_path, timeout_s=self.timeout_s,
                )
                result.seconds = time.perf_counter() - t0
                if error:
                    result.error = error
                else:
                    result.reply_text = _message_text(payload)
                    images = extract_images(payload)
                    result.n_images = len(images)
                    if images:
                        names = []
                        for i, data in enumerate(images):
                            name = regenerate_frame_name(step) if i == 0 \
                                else f"step_{step:03d}_regen_{i + 1}.png"
                            save_png(data, episode_dir / name)
                            names.append(name)
                        result.image_name = names[0]
                        result.extra_image_names = names[1:]
                        result.status = "ok"
                        logger.info(
                            "Regenerate step %d saved %s (%d image(s), %.1fs)",
                            step, result.image_name, result.n_images, result.seconds,
                        )
                    else:
                        result.status = "no_image"
                        logger.warning(
                            "Regenerate step %d returned no image (%.1fs). "
                            "See %s for the raw CliRelay payload.",
                            step, result.seconds, regenerate_meta_name(step),
                        )
        except Exception as exc:
            result.seconds = time.perf_counter() - t0
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Regenerate step %d crashed", step)
        write_meta(episode_dir, result, payload)
        if on_done is not None:
            try:
                on_done(result)
            except Exception:
                logger.exception("Regenerate on_done callback failed")
        return result
