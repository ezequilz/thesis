"""Gemini-via-browser-session VLM backend (research/testing stand-in).

Talks to gemini.google.com through the unofficial `gemini_webapi` package
(HanaokaYuzu/Gemini-API — the maintained fork; the original dsdanielpark
python-gemini-api is archived and its image protocol no longer works),
authenticating with the __Secure-1PSID/__Secure-1PSIDTS cookies from a
logged-in browser session instead of an API key. This is a stopgap so the
explore loop can be exercised against a real vision model for free; it is
NOT for production use (reverse-engineered endpoint, cookies expire, no SLA).

Key difference from the OpenAI backend: the web client has no native tool
calling and no reliable server-side conversation state. So every turn we send
a single self-contained prompt containing the task, a text rendering of the
action tools, a compact history of past actions, and the current screenshot —
and we parse one JSON action object out of the free-text reply.

Cookie sources (first match wins):
  1. agent.cookie_file config key / GEMINI_COOKIE_FILE env — path to a
     *.json or *.txt cookie export (e.g. via the ExportThisCookies extension).
     This is the only option that works inside Docker (mount the file).
  2. agent.chrome_profile — name of a Chrome profile *directory* (e.g.
     "Profile 1", not the display name) whose google.com cookies are read and
     decrypted via browser_cookie3. Use this when the logged-in session lives
     in a non-default profile, which auto_cookies cannot see. macOS/host only.
  3. agent.auto_cookies: true — pull cookies from the default local browser
     profile via browser_cookie3. Host-only, will not work in a container.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from ..tasks.artifact_hunt import SYSTEM_PROMPT
from .actions import ACTION_TOOLS, Action

logger = logging.getLogger(__name__)

# Matches a ```json ... ``` fence, else the first {...} blob in the reply.
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def render_tool_catalog() -> str:
    """Render ACTION_TOOLS as plain text for models without native tool calling."""
    lines = []
    for tool in ACTION_TOOLS:
        fn = tool["function"]
        args = []
        for name, spec in fn["parameters"]["properties"].items():
            desc = spec.get("type", "any")
            if "enum" in spec:
                desc += " (one of: " + ", ".join(map(str, spec["enum"])) + ")"
            args.append(f'"{name}": {desc}')
        lines.append(f"- {fn['name']}: {fn['description']} Args: {{{', '.join(args)}}}")
    return "\n".join(lines)


RESPONSE_FORMAT = """\
Respond with ONLY one JSON object choosing your next action, no prose:
{"action": "<tool name>", "args": {<tool args>}}

Example: {"action": "move", "args": {"direction": "forward", "distance": 1.0}}"""


def parse_action(text: str) -> Action | None:
    """Extract the first JSON action object from a free-text model reply."""
    match = _FENCED_JSON.search(text) or _BARE_JSON.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(1) if match.re is _FENCED_JSON else match.group(0))
    except json.JSONDecodeError:
        return None
    name = obj.get("action") or obj.get("name") or obj.get("tool")
    if not isinstance(name, str):
        return None
    known = {tool["function"]["name"] for tool in ACTION_TOOLS}
    if name not in known:
        return None
    args = obj.get("args") or obj.get("arguments") or {}
    return Action(name, args if isinstance(args, dict) else {})


def _session_cookies(cookies: dict[str, str], source: str) -> tuple[str, str]:
    """Pick the two Gemini session cookies out of a full cookie dict."""
    psid = cookies.get("__Secure-1PSID")
    if not psid:
        raise RuntimeError(
            f"No __Secure-1PSID cookie found in {source} — "
            f"log in at gemini.google.com there first."
        )
    return psid, cookies.get("__Secure-1PSIDTS", "")


def _cookies_from_chrome_profile(profile: str) -> tuple[str, str]:
    """Read and decrypt session cookies from a named Chrome profile.

    browser_cookie3's chrome() only looks at the default profile; sessions in
    other profiles need the profile's Cookies sqlite passed explicitly.
    """
    import browser_cookie3

    path = Path.home() / "Library/Application Support/Google/Chrome" / profile / "Cookies"
    if not path.is_file():
        raise RuntimeError(
            f"Chrome cookie store not found: {path} — chrome_profile must be the "
            f"profile directory name (e.g. 'Profile 1'), see chrome://version."
        )
    jar = browser_cookie3.chrome(cookie_file=str(path), domain_name="google.com")
    return _session_cookies({c.name: c.value for c in jar}, f"Chrome profile {profile!r}")


def _cookies_from_file(path: str) -> tuple[str, str]:
    """Load session cookies from a browser-extension export (*.json or *.txt).

    Accepts either a flat {name: value} dict or the list-of-objects format
    that extensions like ExportThisCookies produce.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        cookies = {c["name"]: c["value"] for c in data if "name" in c and "value" in c}
    else:
        cookies = dict(data)
    return _session_cookies(cookies, f"cookie file {path}")


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


UPLOAD_ENDPOINT = "https://push.clients6.google.com/upload/"


async def _upload_file_resumable(file, client, push_id, filename=None, verbose=False):
    """Authenticated resumable upload, replacing gemini_webapi's uploader.

    The current gemini.google.com web app uploads attachments with the scotty
    resumable protocol on push.clients6.google.com, sending account cookies
    (verified via browser network capture, Aug 2026). gemini_webapi 2.1.0
    still posts anonymous multipart data to content-push.googleapis.com,
    which the generation endpoint then rejects with API error 1100.

    Signature matches gemini_webapi.utils.upload_file so it can be patched in.
    """
    from gemini_webapi.constants import Headers

    if isinstance(file, (str, Path)):
        path = Path(file)
        content = path.read_bytes()
        filename = filename or path.name
    elif isinstance(file, io.BytesIO):
        content = file.getvalue()
        filename = filename or "file.png"
    else:
        content = file
        filename = filename or "file.png"

    base = {**Headers.REFERER.value, "Push-ID": push_id, "X-Tenant-Id": "bard-storage"}
    start = await client.post(
        UPLOAD_ENDPOINT,
        headers={
            **base,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(content)),
        },
        data=f"File name: {filename}",
    )
    start.raise_for_status()
    upload_url = start.headers["x-goog-upload-url"]

    finalize = await client.post(
        upload_url,
        headers={**base, "X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "0"},
        data=content,
    )
    finalize.raise_for_status()
    if verbose:
        logger.debug("Resumable upload finalized: %s", finalize.text[:100])
    return finalize.text


def _install_resumable_upload() -> None:
    import gemini_webapi.client as _gwc

    _gwc.upload_file = _upload_file_resumable


class GeminiWebPolicy:
    """Drives the explore loop via a Gemini web session (cookie auth).

    Stateless prompting: full task + tool catalog + action history is resent
    every turn, since the reverse-engineered web session cannot be trusted to
    keep multimodal conversation state.
    """

    # Retries per step for empty/unparseable replies (the web endpoint is
    # known to return an empty first payload occasionally).
    MAX_ATTEMPTS = 3
    # Cap the action history included in the prompt.
    MAX_HISTORY_LINES = 20

    def __init__(self, cookie_file: str = "", chrome_profile: str = "", auto_cookies: bool = False):
        try:
            from gemini_webapi import GeminiClient
        except ImportError as exc:
            raise RuntimeError(
                "gemini_webapi missing — pip install '.[gemini-web]'"
            ) from exc

        cookie_file = cookie_file or os.environ.get("GEMINI_COOKIE_FILE", "")
        if cookie_file:
            logger.info("Gemini web: using cookie file %s", cookie_file)
            psid, psidts = _cookies_from_file(cookie_file)
        elif chrome_profile:
            logger.info("Gemini web: reading cookies from Chrome profile %r", chrome_profile)
            psid, psidts = _cookies_from_chrome_profile(chrome_profile)
        elif auto_cookies:
            # gemini_webapi falls back to browser_cookie3 (default profile)
            # when no cookies are passed explicitly.
            logger.info("Gemini web: auto-collecting cookies from local browser")
            psid, psidts = None, None
        else:
            raise RuntimeError(
                "Gemini web backend needs cookies: set agent.cookie_file (or the "
                "GEMINI_COOKIE_FILE env var) to a cookie export, agent.chrome_profile "
                "to a Chrome profile directory, or agent.auto_cookies: true with "
                "gemini.google.com logged in locally."
            )

        _install_resumable_upload()

        # gemini_webapi is asyncio-based; the policy owns a private event loop
        # so decide() keeps its synchronous interface.
        self._loop = asyncio.new_event_loop()
        self.client = GeminiClient(psid, psidts or None)
        self._loop.run_until_complete(self.client.init(timeout=120, auto_close=False))
        self._history: list[str] = []

    def decide(self, observation: np.ndarray, pose_description: str, step: int) -> Action:
        prompt = self._build_prompt(pose_description, step)
        image = _png_bytes(observation)

        action = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            text = self._ask(prompt, image)
            if text:
                logger.debug("Gemini reply (step %d, attempt %d): %s", step, attempt, text)
                action = parse_action(text)
                if action:
                    break
            logger.warning("Gemini web: no parseable action (step %d, attempt %d)", step, attempt)

        if action is None:
            # Keep the episode alive rather than crash mid-run; the trace will
            # show the fallback so failures stay visible.
            action = Action("rotate", {"yaw_degrees": 45.0})
            logger.error("Gemini web: falling back to %s after %d attempts", action, self.MAX_ATTEMPTS)

        self._history.append(f"step {step}: {action.name} {json.dumps(action.args)}")
        logger.info("Gemini chose: %s %s", action.name, action.args)
        return action

    # Hard cap per request; gemini_webapi retries some server errors forever.
    REQUEST_TIMEOUT_S = 150

    def _ask(self, prompt: str, image: bytes) -> str:
        import tempfile

        try:
            # A real .png path (not BytesIO) so the upload carries the proper
            # filename and mime type; gemini_webapi names raw bytes "*.txt".
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "view.png"
                path.write_bytes(image)
                response = self._loop.run_until_complete(
                    asyncio.wait_for(
                        self.client.generate_content(prompt, files=[path]),
                        timeout=self.REQUEST_TIMEOUT_S,
                    )
                )
        except Exception:
            logger.exception("Gemini web request failed")
            return ""
        return (response.text or "").strip()

    def _build_prompt(self, pose_description: str, step: int) -> str:
        history = self._history[-self.MAX_HISTORY_LINES:]
        history_block = (
            "Your previous actions:\n" + "\n".join(history)
            if history
            else "This is your first step; no actions taken yet."
        )
        return (
            f"{SYSTEM_PROMPT}\n"
            f"Available tools:\n{render_tool_catalog()}\n\n"
            f"{history_block}\n\n"
            f"Step {step}. Current pose: {pose_description}. "
            f"The attached image is your current view.\n\n"
            f"{RESPONSE_FORMAT}"
        )
