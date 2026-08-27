"""Background regenerate/fix jobs queued from report_artifact."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.agent.actions import ACTION_TOOLS, Action, wants_regenerate
from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.agent.cli_relay import parse_action, render_tool_catalog
from splat_explorer.agent.loop import run_episode
from splat_explorer.agent.regenerate import (
    Regenerator,
    extract_images,
    regenerate_frame_name,
    save_png,
    summarize_payload,
)


def _tiny_png_bytes(color=(12, 34, 56), size=(6, 4)) -> bytes:
    buf = io.BytesIO()
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_wants_regenerate_accepts_yes_no_and_one_zero():
    assert wants_regenerate({"regenerate": "yes"})
    assert wants_regenerate({"regenerate": "YES"})
    assert wants_regenerate({"regenerate": 1})
    assert wants_regenerate({"regenerate": True})
    assert wants_regenerate({"fix": "yes"})
    assert wants_regenerate({"fix": 1})
    assert not wants_regenerate({"regenerate": "no"})
    assert not wants_regenerate({"regenerate": 0})
    assert not wants_regenerate({"regenerate": False})
    assert not wants_regenerate({"regenerate": "nope"})
    assert not wants_regenerate({})
    assert not wants_regenerate(None)


def test_v3_prompt_mentions_regenerate():
    from splat_explorer.tasks.registry import load_prompt

    text = load_prompt("v3").system_prompt(False)
    assert "regenerate=yes" in text

    spec = next(t for t in ACTION_TOOLS if t["function"]["name"] == "report_artifact")
    props = spec["function"]["parameters"]["properties"]
    assert "regenerate" in props
    assert "regenerate" not in spec["function"]["parameters"]["required"]
    assert props["regenerate"]["enum"] == ["yes", "no"]
    catalog = render_tool_catalog()
    assert "regenerate" in catalog
    action = parse_action(
        '{"action": "report_artifact", "args": {'
        '"description": "hole", "image_region": "center", '
        '"severity": "high", "regenerate": "yes"}}'
    )
    assert action is not None
    assert action.args["regenerate"] == "yes"
    assert wants_regenerate(action.args)


def test_extract_images_from_data_url_and_b64_json():
    png = _tiny_png_bytes()
    b64 = base64.b64encode(png).decode()
    payload = {
        "choices": [{
            "message": {
                "content": f"here you go ![fix](data:image/png;base64,{b64})",
            }
        }]
    }
    found = extract_images(payload)
    assert len(found) == 1
    assert found[0].startswith(b"\x89PNG")

    payload2 = {"data": [{"b64_json": b64}]}
    found2 = extract_images(payload2)
    assert len(found2) == 1
    assert found2[0] == png


def test_summarize_payload_strips_giant_data_urls():
    huge = "data:image/png;base64," + ("A" * 200)
    out = summarize_payload({"url": huge, "text": "ok"})
    assert out["text"] == "ok"
    assert "data-url" in out["url"]
    assert "A" * 50 not in out["url"]


def test_save_png_converts_jpeg(tmp_path: Path):
    jpeg_buf = io.BytesIO()
    Image.fromarray(np.full((4, 4, 3), 90, dtype=np.uint8)).save(jpeg_buf, format="JPEG")
    out = tmp_path / "out.png"
    save_png(jpeg_buf.getvalue(), out)
    assert out.is_file()
    img = Image.open(out)
    assert img.format == "PNG"


class _BlankRenderer:
    def render(self, camera):
        return np.full((camera.height, camera.width, 3), 30, dtype=np.uint8)


class _ReportPolicy:
    def __init__(self):
        self.last_debug = None
        self.allow_done = True

    def decide(self, observation, pose, step, depth_image=None, map_image=None,
               coverage_image=None):
        if step == 0:
            return Action("report_artifact", {
                "description": "floater",
                "image_region": "center",
                "severity": "medium",
                "regenerate": "yes",
            })
        return Action("done", {"summary": "ok"})


class _FakeRegen:
    def __init__(self):
        self.calls = []

    def submit(self, image_path, episode_dir, step, on_done=None):
        self.calls.append((Path(image_path).name, Path(episode_dir), step))
        png = episode_dir / regenerate_frame_name(step)
        Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8)).save(png)
        (episode_dir / f"step_{step:03d}_regen.json").write_text(json.dumps({
            "step": step, "status": "ok", "image_name": png.name,
        }))
        class _Fut:
            def result(self, timeout=None):
                from splat_explorer.agent.regenerate import RegenerateResult
                return RegenerateResult(
                    step=step, status="ok", image_name=png.name, n_images=1,
                )
        return _Fut()

    def wait(self, timeout=None):
        from splat_explorer.agent.regenerate import RegenerateResult
        return [
            RegenerateResult(step=step, status="ok", image_name=regenerate_frame_name(step))
            for _, _, step in self.calls
        ]


def test_loop_queues_regenerate_in_background(tmp_path: Path):
    regen = _FakeRegen()
    policy = _ReportPolicy()
    episode_dir = run_episode(
        renderer=_BlankRenderer(),
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=policy,
        output_dir=tmp_path,
        width=32, height=24, fov_deg=75.0,
        max_steps=2,
        send_map=False,
        image_regeneration=True,
        regenerator=regen,
    )
    assert len(regen.calls) == 1
    assert regen.calls[0][0] == "step_000.png"
    artifacts = json.loads((episode_dir / "artifacts.json").read_text())
    assert artifacts[0]["regenerate_requested"] is True
    assert artifacts[0]["regenerate_status"] == "ok"
    assert artifacts[0]["regenerate_frame"] == "step_000_regen.png"
    steps = [
        json.loads(line) for line in (episode_dir / "actions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report = next(s for s in steps if s["action"]["name"] == "report_artifact")
    assert report["regenerate_frame"] == "step_000_regen.png"
    assert report["regen_status"] == "queued"


def test_loop_holds_regenerate_when_setting_off(tmp_path: Path):
    regen = _FakeRegen()
    episode_dir = run_episode(
        renderer=_BlankRenderer(),
        rig=CameraRig(np.array([0.0, 1.5, 0.0]), up_axis="+y"),
        policy=_ReportPolicy(),
        output_dir=tmp_path,
        width=32, height=24, fov_deg=75.0,
        max_steps=2,
        send_map=False,
        image_regeneration=False,
        regenerator=regen,
    )
    assert regen.calls == []
    artifacts = json.loads((episode_dir / "artifacts.json").read_text())
    assert artifacts[0]["regenerate_requested"] is True
    assert artifacts[0]["regenerate_status"] == "disabled"
    meta = json.loads((episode_dir / "step_000_regen.json").read_text())
    assert meta["status"] == "disabled"
    steps = [
        json.loads(line) for line in (episode_dir / "actions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report = next(s for s in steps if s["action"]["name"] == "report_artifact")
    assert report["regen_status"] == "disabled"
    assert "regenerate_frame" not in report


class _FakeImages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.payload)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="json"):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.images = _FakeImages(payload)


def test_regenerator_uses_gpt_image_2_edits(tmp_path: Path):
    png = _tiny_png_bytes((9, 8, 7), (5, 5))
    b64 = base64.b64encode(png).decode()
    payload = {"data": [{"b64_json": b64}]}
    src = tmp_path / "step_002.png"
    Image.fromarray(np.full((5, 5, 3), 1, dtype=np.uint8)).save(src)
    client = _FakeClient(payload)
    regen = Regenerator(client=client, model="gpt-image-2", max_workers=1)
    regen.submit(src, tmp_path, 2)
    results = regen.wait()
    assert results[0].status == "ok"
    assert results[0].n_images == 1
    assert results[0].model == "gpt-image-2"
    out = tmp_path / "step_002_regen.png"
    assert out.is_file()
    meta = json.loads((tmp_path / "step_002_regen.json").read_text())
    assert meta["status"] == "ok"
    assert meta["model"] == "gpt-image-2"
    assert client.images.calls
    sent = client.images.calls[0]
    assert sent["model"] == "gpt-image-2"
    assert sent["prompt"].startswith("Please regenerate and fix this image")
    assert "image" in sent
