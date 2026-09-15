"""LRZ job packing / ingest (no live SSH)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from splat_explorer.agent.camera_rig import CameraRig
from splat_explorer.repair import make_repair_backend
from splat_explorer.repair_lrz import (
    LrzRemoteRepair,
    apply_packed_job,
    camera_from_dict,
    camera_to_dict,
    ingest_job_results,
    lrz_configured,
    lrz_session_alive,
    pack_refine_job,
    session_required_message,
    srun_worker_command,
    ssh_argv,
    wait_for_job_results,
)
from splat_explorer.scene import GaussianScene


def _scene(n=4):
    return GaussianScene(
        means=np.zeros((n, 3), np.float32),
        scales=np.full((n, 3), 0.05, np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full((n,), 0.8, np.float32),
        colors=np.full((n, 3), 0.4, np.float32),
    )


def test_camera_roundtrip():
    camera = CameraRig(np.array([0.0, 0.0, -2.0]), up_axis="+y").camera(32, 24, 75.0)
    body = camera_to_dict(camera)
    back = camera_from_dict(body)
    assert back.width == 32
    assert back.height == 24
    np.testing.assert_allclose(back.position, camera.position)
    np.testing.assert_allclose(back.rotation, camera.rotation, atol=1e-6)


def test_pack_and_fake_worker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _scene()
    camera = CameraRig(np.array([0.0, 1.0, 0.0]), up_axis="+y").camera(16, 12, 75.0)
    rendered = np.full((12, 16, 3), 40, np.uint8)
    repaired = np.full((12, 16, 3), 200, np.uint8)
    job_dir = pack_refine_job(scene, camera, rendered, repaired, params={"iters": 2})
    assert (job_dir / "scene.ply").is_file()
    assert (job_dir / "RUN.txt").is_file()
    assert "ssh-session.sh" in (job_dir / "RUN.txt").read_text()
    status = json.loads((job_dir / "status.json").read_text())
    assert status["phase"] == "packed"

    class Fake:
        def apply(self, scene, camera, rendered_rgb, repaired_rgb):
            scene.colors[:] = 0.9
            return {
                "backend": "gsfix-gsplat",
                "n_visible": scene.num_gaussians,
                "n_updated": scene.num_gaussians,
                "n_spawned": 0,
                "n_gaussians": scene.num_gaussians,
                "n_iters": 20,
                "l1_before": 0.5,
                "l1_after": 0.1,
                "render_rgb": repaired_rgb,
            }

    stats = apply_packed_job(job_dir, backend=Fake())
    assert stats["l1_after"] == 0.1
    working = _scene()
    ingested = ingest_job_results(working, job_dir)
    assert ingested["backend"] == "gsfix-gsplat"
    assert working.colors.mean() > 0.8
    assert ingested["render_rgb"].shape == (12, 16, 3)


def test_packed_job_runs_apply_until_when_max_chunks_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = _scene()
    camera = CameraRig(np.array([0.0, 1.0, 0.0]), up_axis="+y").camera(16, 12, 75.0)
    rendered = np.full((12, 16, 3), 40, np.uint8)
    repaired = np.full((12, 16, 3), 200, np.uint8)
    job_dir = pack_refine_job(
        scene, camera, rendered, repaired,
        params={"iters": 20, "max_chunks": 0},
    )
    calls = {"apply": 0, "until": 0, "ckpt": 0}

    class Fake:
        def apply(self, *_a, **_k):
            calls["apply"] += 1
            raise AssertionError("focused repairs must not re-run apply() per chunk")

        def apply_until(self, scene, camera, rendered_rgb, repaired_rgb, **kwargs):
            calls["until"] += 1
            assert kwargs.get("should_stop") is not None
            scene.colors[:] = 0.9
            stats = {
                "backend": "gsfix-gsplat",
                "n_visible": scene.num_gaussians,
                "n_updated": scene.num_gaussians,
                "n_spawned": 0,
                "n_gaussians": scene.num_gaussians,
                "n_iters": 40,
                "n_chunks": 2,
                "l1_before": 0.5,
                "l1_after": 0.2,
                "render_rgb": repaired_rgb,
                "phase": "refine",
            }
            kwargs["on_checkpoint"](stats)
            calls["ckpt"] += 1
            return stats

    stats = apply_packed_job(job_dir, backend=Fake())
    assert calls == {"apply": 0, "until": 1, "ckpt": 1}
    assert stats["n_iters"] == 40
    assert (job_dir / "scene_repaired.ply").is_file()
    status = json.loads((job_dir / "status.json").read_text())
    assert status["has_ply"] is True
    assert status["checkpoint_iters"] == 40


def test_packed_job_stop_file_is_visible_to_gpu_loop(tmp_path, monkeypatch):
    from splat_explorer.repair_lrz import STOP_NAME, job_stop_requested

    monkeypatch.chdir(tmp_path)
    scene = _scene()
    camera = CameraRig(np.array([0.0, 1.0, 0.0]), up_axis="+y").camera(16, 12, 75.0)
    job_dir = pack_refine_job(
        scene, camera,
        np.full((12, 16, 3), 40, np.uint8),
        np.full((12, 16, 3), 200, np.uint8),
        params={"max_chunks": 0},
    )
    seen = {"stop": None}

    class Fake:
        def apply_until(self, scene, camera, rendered_rgb, repaired_rgb, **kwargs):
            should_stop = kwargs["should_stop"]
            assert should_stop() is False
            (job_dir / STOP_NAME).write_text("")
            seen["stop"] = should_stop()
            scene.colors[:] = 0.7
            stats = {
                "backend": "gsfix-gsplat",
                "n_iters": 20,
                "l1_before": 0.4,
                "l1_after": 0.3,
                "render_rgb": repaired_rgb,
                "n_gaussians": scene.num_gaussians,
                "n_updated": scene.num_gaussians,
                "n_visible": scene.num_gaussians,
                "n_spawned": 0,
            }
            kwargs["on_checkpoint"](stats)
            return stats

    apply_packed_job(job_dir, backend=Fake())
    assert seen["stop"] is True
    assert job_stop_requested(job_dir) is True


def test_lrz_apply_until_is_a_single_remote_job(monkeypatch):
    calls = {"n": 0}

    def fake_apply(self, scene, camera, rendered_rgb, repaired_rgb):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("LRZ apply_until re-uploaded the scene")
        return {
            "n_iters": 60,
            "n_chunks": 3,
            "l1_before": 0.4,
            "l1_after": 0.2,
            "backend": "gsfix-gsplat",
        }

    monkeypatch.setattr(LrzRemoteRepair, "apply", fake_apply)
    camera = CameraRig(np.array([0.0, 0.0, -1.0]), up_axis="+y").camera(8, 8, 75.0)
    out = LrzRemoteRepair(max_chunks=0).apply_until(
        _scene(), camera,
        np.zeros((8, 8, 3), np.uint8),
        np.ones((8, 8, 3), np.uint8) * 200,
        should_stop=lambda: False,
    )
    assert calls["n"] == 1
    assert out["n_iters"] == 60
    assert out["l1_after"] == 0.2


def test_session_missing_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("LRZ_SSH_CONTROL_PATH", str(tmp_path / "missing-cm"))
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    assert lrz_session_alive() is False
    msg = session_required_message()
    assert "ssh-session.sh" in msg
    assert "open the LRZ SSH session first" in msg.lower() or "Open the LRZ SSH session first" in msg


def test_ssh_argv_uses_control_path(monkeypatch, tmp_path):
    sock = tmp_path / "cm-lrz"
    sock.write_text("")
    monkeypatch.setenv("LRZ_SSH_CONTROL_PATH", str(sock))
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    argv = ssh_argv({"user": "go73kaf2", "host": "login.ai.lrz.de"}, multiplex=True)
    assert any("ControlMaster=no" == a for a in argv)
    assert "BatchMode=yes" in argv
    assert any(a.startswith("ConnectTimeout=") for a in argv)
    assert "ServerAliveInterval=5" in argv
    assert any(str(sock) in a for a in argv)
    assert argv[0] in ("ssh", "/usr/bin/ssh") or argv[0].endswith("/ssh")


def test_lrz_apply_requires_session(monkeypatch, tmp_path):
    from splat_explorer.agent.camera_rig import CameraRig

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    monkeypatch.setattr("splat_explorer.repair_lrz.get_ssh_password", lambda: None)
    monkeypatch.delenv("LRZ_SSH_PASSWORD", raising=False)
    camera = CameraRig(np.array([0.0, 0.0, -1.0]), up_axis="+y").camera(8, 8, 75.0)
    with pytest.raises(RuntimeError, match="ssh-session"):
        LrzRemoteRepair(iters=1, densify=False).apply(
            _scene(), camera, np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8) * 200,
        )
    monkeypatch.chdir(tmp_path)
    job_dir = tmp_path / "outputs" / "lrz-jobs" / "abc"
    job_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Stopped"):
        wait_for_job_results(job_dir, should_stop=lambda: True, poll_s=0.01, timeout_s=2)


def test_make_repair_backend_lrz(monkeypatch):
    from splat_explorer.repair_gsfix import gsplat_refine_available

    if gsplat_refine_available():
        pytest.skip("local CUDA wins over LRZ")
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    paper = make_repair_backend("gsfix-gsplat")
    assert isinstance(paper, LrzRemoteRepair)
    assert paper.method == "gsfix-gsplat"
    assert paper.max_chunks == 1
    assert paper.densify is False
    focused = make_repair_backend("gsfix-gsplat", studio=True, focused=True)
    assert isinstance(focused, LrzRemoteRepair)
    assert focused.max_chunks == 0
    baseline = make_repair_backend("gsfix-gsplat-baseline")
    assert isinstance(baseline, LrzRemoteRepair)
    assert baseline.method == "gsfix-gsplat-baseline"
    assert baseline.max_chunks == 0
    vis = make_repair_backend("gsfix-gsplat-visprune")
    assert isinstance(vis, LrzRemoteRepair)
    assert vis.method == "gsfix-gsplat-visprune"
    assert vis.max_chunks == 1
    assert vis.densify is True
    auto = make_repair_backend("auto", studio=True)
    assert isinstance(auto, LrzRemoteRepair)


def test_lrz_visprune_params_include_experimental_flags():
    packed = LrzRemoteRepair(method="gsfix-gsplat-visprune", densify=True)._params()
    assert packed["method"] == "gsfix-gsplat-visprune"
    assert packed["densify"] is True
    assert packed["freeze_occluded"] is False
    assert packed["error_prune"] is True
    assert packed["anchor_weight"] == 0.3
    paper = LrzRemoteRepair(method="gsfix-gsplat")._params()
    assert "freeze_occluded" not in paper
    assert paper["densify"] is False


def test_srun_worker_overlaps_sleep_hold():
    cmd = srun_worker_command(
        {
            "job_id": "5777469",
            "cpus": 4,
            "workspace": "/dss/ws",
            "container": "/dss/ws/containers/pytorch.sqsh",
            "container_name": "splat-repair",
        },
        "abc",
    )
    assert "--overlap" in cmd
    assert "--jobid=5777469" in cmd
    assert "--gres=gpu:1" in cmd
    assert "--mem=62G" in cmd
    assert "/workspace/python" in cmd
    assert "--container-name=splat-repair-5777469" in cmd


def test_srun_setup_starts_named_container():
    from splat_explorer.repair_lrz import srun_setup_command

    cmd = srun_setup_command(
        {
            "job_id": "5777731",
            "cpus": 4,
            "workspace": "/dss/ws",
            "container": "/dss/ws/containers/pytorch.sqsh",
            "container_name": "splat-repair",
        },
    )
    assert "--overlap" in cmd
    assert "--jobid=5777731" in cmd
    assert "--container-image=/dss/ws/containers/pytorch.sqsh" in cmd
    assert "--container-name=splat-repair-5777731" in cmd
    assert "repair_lrz --setup" in cmd
    assert "/workspace/python" in cmd
    assert "--cpus-per-task=1" in cmd


def test_patch_gsplat_nvcc_single_thread(tmp_path):
    from splat_explorer.repair_lrz import _patch_gsplat_nvcc_single_thread

    backend = tmp_path / "gsplat" / "cuda" / "_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(
        '        extra_cuda_cflags = [opt_level]\n'
        '        if not NO_FAST_MATH:\n'
        '            extra_cuda_cflags += ["-use_fast_math"]\n'
        '        sources = ()\n'
    )
    _patch_gsplat_nvcc_single_thread(tmp_path)
    text = backend.read_text()
    assert '        extra_cuda_cflags += ["--threads", "1"]' in text
    _patch_gsplat_nvcc_single_thread(tmp_path)
    assert backend.read_text().count("--threads") == 1


def test_gsplat_ninja_is_single_job():
    from splat_explorer.repair_lrz import gsplat_ninja_command

    cmd = gsplat_ninja_command(Path("/workspace/python/torch_extensions/gsplat_cuda"))
    assert cmd[0] == "ninja"
    assert "-j1" in cmd
    assert "-C" in cmd


def test_compile_gsplat_reuses_existing_so(tmp_path, monkeypatch):
    from splat_explorer.repair_lrz import compile_gsplat_cuda_extension

    so = tmp_path / "gsplat_cuda" / "gsplat_cuda.so"
    so.parent.mkdir()
    so.write_bytes(b"elf")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    out = compile_gsplat_cuda_extension()
    assert out["ok"] is True
    assert out["reused"] is True
    assert out["so"] == str(so)


def test_compile_gsplat_spawns_child_when_torch_imported(tmp_path, monkeypatch):
    import sys
    import types

    from splat_explorer import repair_lrz as m

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    hits: list[Path | None] = [None]
    monkeypatch.setattr(m, "_find_gsplat_cuda_so", lambda: hits[0])
    added = "torch" not in sys.modules
    if added:
        sys.modules["torch"] = types.ModuleType("torch")
    calls: list[list[str]] = []

    def fake_call(argv, **kwargs):
        calls.append([str(x) for x in argv])
        so = tmp_path / "gsplat_cuda.so"
        so.write_bytes(b"elf")
        hits[0] = so
        return 0

    monkeypatch.setattr(m.subprocess, "check_call", fake_call)
    try:
        out = m.compile_gsplat_cuda_extension(site=str(tmp_path))
    finally:
        if added:
            sys.modules.pop("torch", None)
    assert out["ok"] is True
    assert out["reused"] is False
    assert any("--compile-gsplat-cuda" in c for c in calls)
    assert any("--site" in c for c in calls)


def test_run_gsplat_ninja_refuses_loaded_torch(monkeypatch, tmp_path):
    import sys
    import types

    from splat_explorer import repair_lrz as m

    ninja = tmp_path / "build.ninja"
    ninja.write_text("rule dummy\n")
    added = "torch" not in sys.modules
    if added:
        sys.modules["torch"] = types.ModuleType("torch")
    try:
        with pytest.raises(RuntimeError, match="torch"):
            m.run_gsplat_ninja_build(tmp_path)
    finally:
        if added:
            sys.modules.pop("torch", None)


def test_example_yaml_alone_is_not_configured(monkeypatch):
    monkeypatch.delenv("LRZ_JOB_ID", raising=False)
    if Path("configs/lrz.local.yaml").is_file():
        pytest.skip("local LRZ config present")
    assert lrz_configured() is False


def test_parse_squeue_and_nvidia_smi():
    from splat_explorer.repair_lrz import parse_nvidia_smi_csv, parse_probe_bundle, parse_squeue_line

    row = parse_squeue_line(
        "5777469|R|lrz-dgx-a100-80x8|lrz-dgx-a100-002|1:02|6:00:00|None"
    )
    assert row["job_id"] == "5777469"
    assert row["state"] == "R"
    assert row["node"] == "lrz-dgx-a100-002"
    assert row["reason"] is None
    assert parse_squeue_line("") is None
    pending = parse_squeue_line("5777469|PD|lrz-hgx-a100-80x4||0:00|6:00:00|Priority")
    assert pending["state"] == "PD"
    assert pending["reason"] == "Priority"
    waiting = parse_squeue_line(
        "5778174|PD|lrz-hgx-a100-80x4||0:00|24:00:00|Priority|gs-24h|2026-09-10T03:10:58"
    )
    assert waiting["name"] == "gs-24h"
    assert waiting["start_time"] == "2026-09-10T03:10:58"
    held = parse_squeue_line(
        "5786047|R|lrz-dgx-a100-80x8|lrz-dgx-a100-002|1:02|48:00:00|None|gs-48h|2026-09-13T15:00:00|32G"
    )
    assert held["mem"] == "32G"
    assert held["partition"] == "lrz-dgx-a100-80x8"

    gpus = parse_nvidia_smi_csv(
        "0, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 29, 61.00, 400.00, 8.0\n"
    )
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert gpus[0]["memory_total_mib"] == 81920
    assert gpus[0]["memory_pct"] == 0.0
    assert gpus[0]["memory_free_mib"] == 81920
    assert gpus[0]["compute_cap"] == "8.0"

    bundle = parse_probe_bundle(
        "SQUEUE\n5777469|R|p|node|1:00|6:00:00|None\n"
        "CONTAINER\nOK 123456\nNGC\nMISSING\nWORKSPACE\nOK\n"
        "SETUP\nMISSING\nSTATUS\nNONE\n"
    )
    assert bundle["squeue"].startswith("5777469")
    assert bundle["container"] == "OK 123456"
    assert bundle["ngc"] == "MISSING"
    assert bundle["setup"] == "MISSING"


def test_connection_checks_ssh_down_is_next_action():
    from splat_explorer.repair_lrz import build_connection_checks, next_action_from_checks

    checks = build_connection_checks(
        configured=True, session=False, job_id="5777469",
        slurm=None, container=None, gpu=None, gpu_error=None, probed=False,
    )
    ssh = next(c for c in checks if c["id"] == "ssh")
    assert ssh["ok"] is False
    action = next_action_from_checks(checks)
    assert action and "ssh-session.sh" in action


def test_connection_checks_missing_container():
    from splat_explorer.repair_lrz import build_connection_checks, next_action_from_checks

    checks = build_connection_checks(
        configured=True, session=True, job_id="5777469",
        slurm={"job_id": "5777469", "state": "R", "node": "lrz-dgx-a100-002",
               "elapsed": "1:00", "timelimit": "6:00:00"},
        container={"ok": False, "bytes": None},
        gpu=None, gpu_error=None, probed=True,
    )
    action = next_action_from_checks(checks)
    assert action and "enroot import" in action
    slurm = next(c for c in checks if c["id"] == "slurm")
    assert slurm["ok"] is True


def test_connection_checks_missing_setup_points_at_load_button():
    from splat_explorer.repair_lrz import build_connection_checks, next_action_from_checks

    checks = build_connection_checks(
        configured=True, session=True, job_id="5777731",
        slurm={"job_id": "5777731", "state": "R", "node": "lrz-hgx-a100-004",
               "elapsed": "1:00", "timelimit": "2:00:00"},
        container={"ok": True, "bytes": 12_000_000_000},
        gpu=None, gpu_error=None, probed=True,
        setup={"ok": False},
    )
    action = next_action_from_checks(checks)
    assert action and "Load GPU setup" in action
    setup = next(c for c in checks if c["id"] == "setup")
    assert setup["ok"] is False


def test_parse_setup_marker_and_ok_line():
    from splat_explorer.repair_lrz import parse_setup_marker_text, parse_setup_ok_output

    missing = parse_setup_marker_text("MISSING", job_id="5777731")
    assert missing["ok"] is False
    marker = parse_setup_marker_text(
        'OK\n{"ok": true, "job_id": "5777731", "gpu": "NVIDIA A100-SXM4-80GB", "torch": "2.5.1"}\n',
        job_id="5777731",
    )
    assert marker["ok"] is True
    assert marker["gpu"] == "NVIDIA A100-SXM4-80GB"
    parsed = parse_setup_ok_output(
        "loading\nSETUP_OK {\"ok\": true, \"gpu\": \"NVIDIA A100-SXM4-80GB\", \"gsplat\": \"1.5.2\"}\n"
    )
    assert parsed["gpu"].startswith("NVIDIA A100")
    assert parsed["gsplat"] == "1.5.2"


def test_request_setup_requires_session(monkeypatch, tmp_path):
    from splat_explorer.repair_lrz import request_lrz_setup, reset_setup_cache

    reset_setup_cache()
    monkeypatch.setenv("LRZ_SSH_CONTROL_PATH", str(tmp_path / "missing-cm"))
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    with pytest.raises(RuntimeError, match="ssh-session"):
        request_lrz_setup()


def test_dashboard_snapshot_ssh_down(monkeypatch, tmp_path):
    from splat_explorer.repair_lrz import lrz_dashboard_snapshot, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777469",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    body = lrz_dashboard_snapshot(
        repair_job={"status": "idle", "backend": "gsfix-gsplat", "episode": "ep1"},
    )
    assert body["ready"] is False
    assert body["repair"]["status"] == "idle"
    assert "ssh-session.sh" in (body["next_action"] or "")
    assert body["probing"] is False
    assert body["gpu"] is None
    assert body["jobs"] == []
    assert body["scripts"]["allocate_8h"] == "scripts/lrz/allocate.sh 8h"
    assert body["scripts"]["setup"] == "scripts/lrz/load-setup.sh"
    assert "setup" in body
    assert body["setup"]["ok"] is False
    assert "10 min" in body["hint"]
    assert body["history"]["jobs"] == []
    assert body["history"]["default_days"] == 14
    assert body["history"]["end"] == "now"
    assert "sacct" in body["hint"]


def test_dashboard_snapshot_lists_packed_jobs(monkeypatch, tmp_path):
    from splat_explorer.repair_lrz import lrz_dashboard_snapshot, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777469",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: False)
    scene = _scene()
    camera = CameraRig(np.array([0.0, 0.0, -1.0]), up_axis="+y").camera(8, 8, 75.0)
    pack_refine_job(
        scene, camera, np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8) * 200,
        job_id="packed1",
    )
    body = lrz_dashboard_snapshot(repair_job={"status": "running", "message": "rsync_up"})
    assert body["packed_jobs"][0]["id"] == "packed1"
    assert body["packed_jobs"][0]["phase"] == "packed"
    assert body["current_packed"]["id"] == "packed1"
    assert body["repair"]["status"] == "running"


def test_nvidia_smi_command_overlaps_hold_job():
    from splat_explorer.repair_lrz import nvidia_smi_command

    cmd = nvidia_smi_command({"job_id": "5777469"})
    assert "--overlap" in cmd
    assert "--jobid=5777469" in cmd
    assert "nvidia-smi" in cmd
    assert "--gres=gpu:1" in cmd
    assert "--mem=1G" in cmd
    assert "GPUCSV" in cmd
    assert "query-compute-apps" in cmd
    assert "CUDA_VISIBLE_DEVICES" in cmd


def test_sbatch_hold_8h_24h_and_after():
    from splat_explorer.repair_lrz import (
        parse_sbatch_output,
        parse_sinfo_lines,
        parse_squeue_lines,
        sbatch_hold_command,
        select_slurm_job,
        sinfo_command,
        widen_command,
    )

    eight = sbatch_hold_command(8)
    assert "--time=08:00:00" in eight
    assert "--wrap='sleep 28800'" in eight
    assert "--job-name=gs-8h" in eight
    assert "lrz-hgx-a100-80x4,lrz-dgx-a100-80x8" in eight
    assert "--gres=gpu:1" in eight
    assert "--mem=64G" in eight
    six = sbatch_hold_command(6, begin="2026-09-10T09:00", partition="lrz-hgx-h100-94x4")
    assert "--time=06:00:00" in six
    assert "sleep 21600" in six
    assert "--begin=2026-09-10T09:00:00" in six
    assert "--partition=lrz-hgx-h100-94x4" in six
    twenty = sbatch_hold_command("24h", after_job="5777469")
    assert "--time=24:00:00" in twenty
    assert "sleep 86400" in twenty
    assert "--dependency=afterany:5777469" in twenty
    assert parse_sbatch_output("Submitted batch job 5777470\n") == "5777470"
    jobs = parse_squeue_lines(
        "5777469|R|lrz-dgx-a100-80x8|lrz-dgx-a100-002|1:02|08:00:00|None|gs-8h\n"
        "5777470|PD|lrz-hgx-a100-80x4||0:00|24:00:00|Priority|gs-24h\n"
    )
    assert [j["job_id"] for j in jobs] == ["5777469", "5777470"]
    assert jobs[0]["name"] == "gs-8h"
    picked = select_slurm_job(jobs, "5777470")
    assert picked["state"] == "PD"
    assert picked["current"] is True
    rows = parse_sinfo_lines(
        "lrz-dgx-a100-80x8|up|14-00:00:0|4|mix|lrz-dgx-a100-[001-002,004-005]\n"
        "lrz-hgx-a100-80x4*|up|14-00:00:0|1|mix|lrz-hgx-a100-004\n"
    )
    assert rows[0]["state"] == "mix"
    assert rows[1]["partition"] == "lrz-hgx-a100-80x4"
    assert "lrz-v100x2" in sinfo_command()
    assert "Partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8" in widen_command("5777469")


def test_write_lrz_job_id(tmp_path):
    from splat_explorer.repair_lrz import write_lrz_job_id

    path = tmp_path / "lrz.local.yaml"
    path.write_text('user: go73kaf2\njob_id: ""\n')
    write_lrz_job_id("5777470", path)
    assert 'job_id: "5777470"' in path.read_text()
    write_lrz_job_id("", path)
    assert 'job_id: ""' in path.read_text()


def test_allocate_one_sbatch_no_wait(monkeypatch, tmp_path):
    from splat_explorer.repair_lrz import allocate_lrz_gpu, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    cfg_path = tmp_path / "configs" / "lrz.local.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text('user: go73kaf2\nhost: login.ai.lrz.de\njob_id: ""\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr(
        "splat_explorer.repair_lrz._ssh_run",
        lambda cfg, remote, timeout=25: type("R", (), {
            "returncode": 0, "stdout": "Submitted batch job 5777471", "stderr": "",
        })(),
    )
    body = allocate_lrz_gpu(8)
    assert body["job_id"] == "5777471"
    assert body["switched"] is True
    assert 'job_id: "5777471"' in cfg_path.read_text()
    chained = allocate_lrz_gpu(24, after="5777471")
    assert chained["after_job"] == "5777471"
    assert chained["switched"] is False
    assert "--dependency=afterany:5777471" in chained["command"]


def test_login_probe_lists_all_jobs_once():
    from splat_explorer.repair_lrz import _login_probe_script, default_history_start

    script = _login_probe_script(
        {"job_id": "5777469", "workspace": "/dss/ws",
         "container": "/dss/ws/containers/pytorch.sqsh"},
        None,
    )
    assert "squeue --me" in script
    assert "%S" in script
    assert "squeue --me --start" not in script
    assert "squeue --me --long" not in script
    assert script.count("squeue") == 1
    assert "--job=" not in script
    assert "setup-5777469.json" in script
    assert "echo SETUP" in script
    assert "date '+%F %T %Z %z'" in script
    assert "id -u" in script
    assert "sacct" in script
    assert "--allocations" in script
    assert "--duplicates" in script
    assert "--parsable2" in script
    assert "--user=go73kaf2" in script
    assert "--endtime=now" in script
    assert f"--starttime={default_history_start()}" in script
    custom = _login_probe_script(
        {"job_id": "5777469", "workspace": "/dss/ws",
         "container": "/dss/ws/containers/pytorch.sqsh", "user": "go73kaf2"},
        None,
        history_start="2026-09-08",
        history_end="now",
    )
    assert "--starttime=2026-09-08" in custom


def test_connection_checks_missing_job_points_at_allocate():
    from splat_explorer.repair_lrz import build_connection_checks, next_action_from_checks

    checks = build_connection_checks(
        configured=False, session=True, job_id="",
        slurm=None, container=None, gpu=None, gpu_error=None, probed=False,
    )
    action = next_action_from_checks(checks)
    assert action and "allocate.sh" in action


def test_parse_scontrol_counts_free_gpus():
    from splat_explorer.repair_lrz import parse_scontrol_nodes, slurm_begin_spec, summarize_gpu_availability

    assert slurm_begin_spec("now") is None
    assert slurm_begin_spec("tomorrow") == "tomorrow"
    assert slurm_begin_spec("2026-09-10T09:00") == "2026-09-10T09:00:00"
    nodes = parse_scontrol_nodes(
        "NodeName=lrz-dgx-a100-001 Arch=x86_64\n"
        "   AvailableFeatures=A100-80GB\n"
        "   State=MIXED ThreadsPerCore=2\n"
        "   Partitions=lrz-dgx-a100-80x8\n"
        "   CfgTRES=cpu=252,mem=1951G,billing=3996312,gres/gpu=8\n"
        "   AllocTRES=cpu=176,mem=676G,gres/gpu=8\n"
        "\n"
        "NodeName=lrz-dgx-a100-002 Arch=x86_64\n"
        "   AvailableFeatures=A100-80GB\n"
        "   State=MIXED\n"
        "   Partitions=lrz-dgx-a100-80x8\n"
        "   CfgTRES=cpu=252,mem=1951G,billing=3996312,gres/gpu=8\n"
        "   AllocTRES=cpu=152,mem=460G,gres/gpu=7\n"
        "\n"
        "NodeName=lrz-hgx-a100-003 Arch=x86_64\n"
        "   State=INVAL\n"
        "   Partitions=lrz-hgx-a100-80x4\n"
        "   CfgTRES=cpu=1,mem=1G,gres/gpu=4\n"
        "   AllocTRES=cpu=0,mem=0G,gres/gpu=0\n"
    )
    assert nodes[0]["gpu_free"] == 0
    assert nodes[1]["gpu_free"] == 1
    assert nodes[1]["name"] == "lrz-dgx-a100-002"
    assert nodes[2]["down"] is True
    summary = summarize_gpu_availability(nodes)
    dgx = next(r for r in summary if r["id"] == "lrz-dgx-a100-80x8")
    assert dgx["gpu_free"] == 1
    assert dgx["has_free"] is True
    hgx = next(r for r in summary if r["id"] == "lrz-hgx-a100-80x4")
    assert hgx["gpu_free"] == 0
    assert hgx["nodes_down"] == 1


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _login_stdout(squeue_lines: str, sacct: str = "") -> str:
    return (
        "SQUEUE\n" + squeue_lines + "\n"
        "CONTAINER\nOK 12\nNGC\nMISSING\nWORKSPACE\nOK\nSETUP\nMISSING\nSTATUS\nNONE\n"
        "DATE\n2026-09-10 16:16:00 CEST +0200\n"
        "UID\n12345\n"
        "SQUEUE_LONG\nJOBID PARTITION NAME\n"
        "SACCT\n" + (sacct or "JobID|JobName|Partition|State|Submit|Start|End|Elapsed|Timelimit|ExitCode|NodeList\n")
    )


def test_probe_skips_nvidia_smi_when_jobs_are_pending(monkeypatch):
    from splat_explorer.repair_lrz import probe_lrz_gpu, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    calls: list[str] = []

    def fake_ssh(cfg, remote, timeout=25):
        calls.append(remote)
        if "squeue" in remote:
            return _Proc(stdout=_login_stdout(
                "5778174|PD|lrz-hgx-a100-80x4||0:00|24:00:00|Priority|gs-24h|2026-09-10T03:10:58\n"
                "5777728|PD|lrz-hgx-h100-94x4||0:00|24:00:00|Priority|gs-h100-|2026-09-10T03:10:58\n"
            ))
        raise AssertionError(f"unexpected ssh: {remote}")

    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", fake_ssh)
    body = probe_lrz_gpu({
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    assert [j["job_id"] for j in body["jobs"]] == ["5778174", "5777728"]
    assert body["jobs"][0]["start_time"] == "2026-09-10T03:10:58"
    assert body["slurm"] is None
    assert body["gpu"] is None
    assert body["gpu_error"] is None
    assert all("nvidia-smi" not in c for c in calls)


def test_probe_keeps_jobs_when_srun_is_forbidden(monkeypatch):
    from splat_explorer.repair_lrz import probe_lrz_gpu, reset_gpu_probe_cache

    reset_gpu_probe_cache()

    def fake_ssh(cfg, remote, timeout=25):
        if "squeue" in remote:
            return _Proc(stdout=_login_stdout(
                "5777731|R|lrz-hgx-a100-80x4|lrz-hgx-a100-004|1:02|02:00:00|None|gs-meeti|2026-09-09T08:00:00\n"
            ))
        if "nvidia-smi" in remote:
            return _Proc(
                returncode=1,
                stderr="srun: error: Unable to create step for job 5777731: Access/permission denied (forbidden)",
            )
        raise AssertionError(f"unexpected ssh: {remote}")

    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", fake_ssh)
    body = probe_lrz_gpu({
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    assert body["jobs"][0]["job_id"] == "5777731"
    assert body["gpu"] is None
    assert body["gpu_error"] is None


def test_dashboard_hides_gpu_when_only_pending_jobs(monkeypatch, tmp_path):
    import time

    from splat_explorer.repair_lrz import (
        _PROBE,
        lrz_dashboard_snapshot,
        reset_gpu_probe_cache,
    )

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    with _PROBE["lock"]:
        _PROBE["body"] = {
            "slurm": None,
            "jobs": [
                {
                    "job_id": "5778174", "state": "PD", "name": "gs-24h",
                    "partition": "lrz-hgx-a100-80x4", "reason": "Priority",
                    "start_time": "2026-09-10T03:10:58", "elapsed": "0:00",
                    "timelimit": "24:00:00", "node": "", "current": False,
                },
            ],
            "container": {"ok": True, "bytes": 12},
            "gpu": {"gpus": [{"name": "NVIDIA A100-SXM4-80GB"}], "node": "stale"},
            "gpu_error": "srun: error: Access/permission denied (forbidden)",
            "ngc": False,
            "workspace_ok": True,
            "remote_status": {"phase": "refine"},
            "setup": {"ok": False},
        }
        _PROBE["at"] = time.time()
        _PROBE["error"] = None
        _PROBE["inflight"] = False
    body = lrz_dashboard_snapshot(repair_job={"status": "idle"})
    assert body["gpu"] is None
    assert body["gpu_running"] is False
    assert body["remote_status"] is None
    assert body["jobs"][0]["start_time"] == "2026-09-10T03:10:58"
    assert body["probe_error"] is None
    slurm_check = next(c for c in body["checks"] if c["id"] == "slurm")
    assert slurm_check["ok"] is False
    assert "allocation ended" in slurm_check["detail"]


def test_connection_checks_stale_job_points_at_queued_rows():
    from splat_explorer.repair_lrz import build_connection_checks, next_action_from_checks

    checks = build_connection_checks(
        configured=True, session=True, job_id="5777731",
        slurm=None, container={"ok": True, "bytes": 1},
        gpu=None, gpu_error="forbidden", probed=True,
        jobs=[{
            "job_id": "5778174", "state": "PD", "reason": "Priority",
            "start_time": "2026-09-10T03:10:58",
        }],
    )
    slurm = next(c for c in checks if c["id"] == "slurm")
    assert slurm["ok"] is False
    assert "allocation ended" in slurm["detail"]
    gpu = next(c for c in checks if c["id"] == "gpu")
    assert gpu["ok"] is None
    action = next_action_from_checks(checks)
    assert action and "Use" in action


def test_session_check_timeout_is_down(monkeypatch, tmp_path):
    import subprocess

    from splat_explorer.repair_lrz import lrz_session_alive, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.control_path",
        lambda: tmp_path / "cm-lrz",
    )
    (tmp_path / "cm-lrz").write_text("")

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 2)

    monkeypatch.setattr("splat_explorer.repair_lrz.subprocess.run", boom)
    assert lrz_session_alive({
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "",
        "workspace": "/dss/ws",
    }) is False


def test_scancel_command_and_pending_cancel(monkeypatch, tmp_path):
    from splat_explorer.repair_lrz import cancel_lrz_job, reset_gpu_probe_cache, scancel_command

    assert scancel_command("5777728") == "scancel 5777728"
    reset_gpu_probe_cache()
    cfg_path = tmp_path / "configs" / "lrz.local.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text('user: go73kaf2\nhost: login.ai.lrz.de\njob_id: "5778174"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5778174",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G",
    })
    monkeypatch.setattr(
        "splat_explorer.repair_lrz._ssh_run",
        lambda cfg, remote, timeout=25: type("R", (), {
            "returncode": 0, "stdout": "", "stderr": "",
        })(),
    )
    body = cancel_lrz_job("5778174")
    assert body["job_id"] == "5778174"
    assert body["command"] == "scancel 5778174"
    assert body["cleared"] is True
    assert 'job_id: ""' in cfg_path.read_text()


def test_cancel_running_job_requires_confirm(monkeypatch, tmp_path):
    import time

    from splat_explorer.repair_lrz import _PROBE, cancel_lrz_job, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "lrz.local.yaml").write_text(
        'user: go73kaf2\nhost: login.ai.lrz.de\njob_id: "5778400"\n'
    )
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5778400",
        "workspace": "/dss/ws",
    })
    with _PROBE["lock"]:
        _PROBE["body"] = {
            "jobs": [{"job_id": "5778400", "state": "R", "name": "gs-6h"}],
            "slurm": {"job_id": "5778400", "state": "R"},
        }
        _PROBE["at"] = time.time()
    with pytest.raises(RuntimeError, match="cancel\\?"):
        cancel_lrz_job("5778400")
    called = []

    def fake_ssh(cfg, remote, timeout=25):
        called.append(remote)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", fake_ssh)
    body = cancel_lrz_job("5778400", confirm=True)
    assert called == ["scancel 5778400"]
    assert body["was_running"] is True
    assert body["cleared"] is True


def test_parse_sacct_and_history_window():
    from splat_explorer.repair_lrz import (
        default_history_start,
        normalize_history_window,
        normalize_sacct_time,
        parse_sacct_lines,
        sacct_command,
    )

    assert normalize_sacct_time(None, default="now") == "now"
    assert normalize_sacct_time("2026-09-08", default="now") == "2026-09-08"
    assert normalize_sacct_time("2026-09-08T09:00", default="now") == "2026-09-08T09:00:00"
    start, end = normalize_history_window(None, None)
    assert start == default_history_start()
    assert end == "now"
    start, end = normalize_history_window("2026-09-08", "now")
    assert start == "2026-09-08"
    assert end == "now"
    with pytest.raises(ValueError, match="History time"):
        normalize_sacct_time("yesterday", default="now")
    cmd = sacct_command("go73kaf2", "2026-09-08", "now")
    assert "--user=go73kaf2" in cmd
    assert "--starttime=2026-09-08" in cmd
    assert "--endtime=now" in cmd
    assert "--allocations" in cmd
    assert "--duplicates" in cmd
    assert "--parsable2" in cmd
    assert "State%50" in cmd

    rows = parse_sacct_lines(
        "JobID|JobName|Partition|State|Submit|Start|End|Elapsed|Timelimit|ExitCode|NodeList\n"
        "5778400|gs-6h|lrz-hgx-a100-80x4|COMPLETED|"
        "2026-09-08T08:00:00|2026-09-08T08:00:12|2026-09-08T14:00:12|06:00:00|06:00:00|0:0|lrz-hgx-a100-004\n"
        "5778400.batch|batch||COMPLETED|"
        "2026-09-08T08:00:00|2026-09-08T08:00:12|2026-09-08T14:00:12|06:00:00|06:00:00|0:0|lrz-hgx-a100-004\n"
        "5778174|gs-24h|lrz-hgx-a100-80x4|CANCELLED by 12345|"
        "2026-09-09T01:00:00|2026-09-09T01:00:05|2026-09-09T03:10:00|02:09:55|24:00:00|0:0|lrz-hgx-a100-001\n"
        "5777469|gs-8h|lrz-dgx-a100-80x8|NODE_FAIL|"
        "2026-09-08T09:00:00|2026-09-08T09:00:05|2026-09-08T09:30:00|00:29:55|08:00:00|1:0|lrz-dgx-a100-002\n"
        "5777469|gs-8h|lrz-dgx-a100-80x8|COMPLETED|"
        "2026-09-08T10:00:00|2026-09-08T10:00:05|2026-09-08T18:00:05|08:00:00|08:00:00|0:0|lrz-dgx-a100-002\n"
    )
    ids = [r["job_id"] for r in rows]
    assert "5778400.batch" not in ids
    assert ids[0] == "5778174"
    assert rows[0]["state"] == "CANCELLED by 12345"
    assert ids.count("5777469") == 2
    assert rows[-1]["job_id"] == "5778400"
    assert rows[-1]["state"] == "COMPLETED"


def test_probe_parses_sacct_history(monkeypatch):
    from splat_explorer.repair_lrz import probe_lrz_gpu, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    sacct = (
        "JobID|JobName|Partition|State|Submit|Start|End|Elapsed|Timelimit|ExitCode|NodeList\n"
        "5778400|gs-6h|lrz-hgx-a100-80x4|COMPLETED|"
        "2026-09-08T08:00:00|2026-09-08T08:00:12|2026-09-08T14:00:12|06:00:00|06:00:00|0:0|lrz-hgx-a100-004\n"
    )

    def fake_ssh(cfg, remote, timeout=25):
        if "sacct" in remote:
            return _Proc(stdout=_login_stdout(
                "5778174|PD|lrz-hgx-a100-80x4||0:00|24:00:00|Priority|gs-24h|2026-09-10T03:10:58\n",
                sacct=sacct,
            ))
        raise AssertionError(f"unexpected ssh: {remote}")

    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", fake_ssh)
    body = probe_lrz_gpu({
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    }, history_start="2026-09-08", history_end="now")
    assert body["history_start"] == "2026-09-08"
    assert body["history_end"] == "now"
    assert body["history_jobs"][0]["job_id"] == "5778400"
    assert body["history_jobs"][0]["state"] == "COMPLETED"
    assert body["cluster_time"] == "2026-09-10 16:16:00 CEST +0200"
    assert body["uid"] == "12345"
    assert "squeue_long" not in body


def test_dashboard_snapshot_includes_history(monkeypatch, tmp_path):
    import time

    from splat_explorer.repair_lrz import (
        _PROBE,
        lrz_dashboard_snapshot,
        reset_gpu_probe_cache,
    )

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    with _PROBE["lock"]:
        _PROBE["body"] = {
            "slurm": None,
            "jobs": [],
            "history_jobs": [{
                "job_id": "5778400", "name": "gs-6h", "state": "COMPLETED",
                "partition": "lrz-hgx-a100-80x4", "submit": "2026-09-08T08:00:00",
                "start": "2026-09-08T08:00:12", "end": "2026-09-08T14:00:12",
                "elapsed": "06:00:00", "timelimit": "06:00:00",
                "exit_code": "0:0", "node": "lrz-hgx-a100-004",
            }],
            "history_start": "2026-09-08",
            "history_end": "now",
            "cluster_time": "2026-09-10 16:16:00 CEST +0200",
            "uid": "12345",
            "container": {"ok": True, "bytes": 12},
            "gpu": None,
            "gpu_error": None,
            "ngc": False,
            "workspace_ok": True,
            "setup": {"ok": False},
        }
        _PROBE["at"] = time.time()
        _PROBE["error"] = None
        _PROBE["inflight"] = False
        _PROBE["history_start"] = "2026-09-08"
        _PROBE["history_end"] = "now"
    body = lrz_dashboard_snapshot(repair_job={"status": "idle"})
    assert body["history"]["jobs"][0]["job_id"] == "5778400"
    assert body["history"]["start"] == "2026-09-08"
    assert body["history"]["cluster_time"].startswith("2026-09-10")
    assert body["history"]["uid"] == "12345"
    assert body["reviewing"] is False


def test_session_alive_is_cached(monkeypatch, tmp_path):
    from splat_explorer.repair_lrz import lrz_session_alive, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    sock = tmp_path / "cm-lrz"
    sock.write_text("")
    monkeypatch.setattr("splat_explorer.repair_lrz.control_path", lambda: sock)
    calls: list[int] = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        return _Proc(returncode=0, stdout="", stderr="Master running")

    monkeypatch.setattr("splat_explorer.repair_lrz.subprocess.run", fake_run)
    cfg = {"user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "", "workspace": "/dss/ws"}
    assert lrz_session_alive(cfg) is True
    assert lrz_session_alive(cfg) is True
    assert len(calls) == 1


def test_ssh_run_serializes_concurrent_calls(monkeypatch):
    import threading
    import time

    from splat_explorer.repair_lrz import _ssh_run, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    inflight = 0
    max_in = 0
    lock = threading.Lock()

    def fake_run(*args, **kwargs):
        nonlocal inflight, max_in
        with lock:
            inflight += 1
            max_in = max(max_in, inflight)
        time.sleep(0.12)
        with lock:
            inflight -= 1
        return _Proc(returncode=0, stdout="ok")

    monkeypatch.setattr("splat_explorer.repair_lrz.subprocess.run", fake_run)
    cfg = {"user": "go73kaf2", "host": "login.ai.lrz.de"}
    threads = [
        threading.Thread(target=lambda: _ssh_run(cfg, "true"))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max_in == 1


def test_probe_and_review_do_not_overlap_ssh(monkeypatch):
    import threading
    import time

    from splat_explorer.repair_lrz import probe_lrz_gpu, reset_gpu_probe_cache, review_lrz_partitions

    reset_gpu_probe_cache()
    inflight = 0
    max_in = 0
    lock = threading.Lock()

    def fake_run(*args, **kwargs):
        nonlocal inflight, max_in
        argv = args[0] if args else []
        remote = argv[-1] if argv else ""
        with lock:
            inflight += 1
            max_in = max(max_in, inflight)
        time.sleep(0.12)
        with lock:
            inflight -= 1
        if "sinfo" in str(remote):
            return _Proc(stdout="SINFO\nlrz-hgx-a100-80x4|up|14-00:00:0|1|mix|n1\nSCONTROL\n")
        return _Proc(stdout=_login_stdout(
            "5778174|PD|lrz-hgx-a100-80x4||0:00|24:00:00|Priority|gs-24h|2026-09-10T03:10:58\n"
        ))

    monkeypatch.setattr(
        "splat_explorer.repair_lrz.lrz_session_alive",
        lambda cfg=None, force=False: True,
    )
    monkeypatch.setattr("splat_explorer.repair_lrz.subprocess.run", fake_run)
    cfg = {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    }
    errors: list[BaseException] = []

    def run_probe():
        try:
            probe_lrz_gpu(cfg)
        except BaseException as exc:  # noqa: BLE001 — collect for the parent thread
            errors.append(exc)

    def run_review():
        try:
            review_lrz_partitions(force=True, cfg=cfg)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = threading.Thread(target=run_probe)
    second = threading.Thread(target=run_review)
    first.start()
    time.sleep(0.03)
    second.start()
    first.join()
    second.join()
    assert errors == []
    assert max_in == 1


def test_review_uses_ten_minute_cache(monkeypatch):
    from splat_explorer.repair_lrz import reset_gpu_probe_cache, review_lrz_partitions

    reset_gpu_probe_cache()
    calls: list[str] = []

    def fake_ssh(cfg, remote, timeout=25):
        calls.append(remote)
        return _Proc(stdout="SINFO\nlrz-hgx-a100-80x4|up|14-00:00:0|1|mix|n1\nSCONTROL\n")

    monkeypatch.setattr(
        "splat_explorer.repair_lrz.lrz_session_alive",
        lambda cfg=None, force=False: True,
    )
    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", fake_ssh)
    first = review_lrz_partitions(force=True, cfg={"user": "go73kaf2", "host": "login.ai.lrz.de"})
    second = review_lrz_partitions(force=False, cfg={"user": "go73kaf2", "host": "login.ai.lrz.de"})
    assert first["cached"] is False
    assert second["cached"] is True
    assert len(calls) == 1


def test_dashboard_snapshot_skips_ssh_when_probe_cached(monkeypatch, tmp_path):
    import time

    from splat_explorer.repair_lrz import (
        _PROBE,
        lrz_dashboard_snapshot,
        reset_gpu_probe_cache,
    )

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5777731",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.lrz_session_alive",
        lambda cfg=None, force=False: True,
    )

    def boom(*args, **kwargs):
        raise AssertionError("cached snapshot must not open SSH")

    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", boom)
    monkeypatch.setattr("splat_explorer.repair_lrz.subprocess.run", boom)
    with _PROBE["lock"]:
        _PROBE["body"] = {
            "slurm": None, "jobs": [], "container": {"ok": True, "bytes": 1},
            "gpu": None, "gpu_error": None, "ngc": False, "workspace_ok": True,
            "setup": {"ok": False}, "history_jobs": [],
        }
        _PROBE["at"] = time.time()
        _PROBE["error"] = None
        _PROBE["inflight"] = False
    body = lrz_dashboard_snapshot(repair_job={"status": "idle"})
    assert body["probing"] is False
    assert body["jobs"] == []


def test_container_name_is_job_scoped():
    from splat_explorer.repair_lrz import container_name_for_job

    assert container_name_for_job({
        "container_name": "splat-repair", "job_id": "5786047",
    }) == "splat-repair-5786047"
    assert container_name_for_job({"container_name": "splat-repair", "job_id": ""}) == "splat-repair"


def test_occupancy_picks_slurm_gpu_not_full_gpu0():
    from splat_explorer.repair_lrz import (
        gpu_occupancy_summary,
        occupancy_blocks_setup,
        parse_gpu_occupancy_text,
        pick_connected_gpu,
    )

    text = (
        "GPUENV\n"
        "CUDA_VISIBLE_DEVICES=0\n"
        "SLURM_JOB_GPUS=3\n"
        "SLURM_STEP_GPUS=3\n"
        "NVIDIA_VISIBLE_DEVICES=\n"
        "SLURM_JOB_ID=5786047\n"
        "USER=go73kaf2\n"
        "HOSTNAME=lrz-dgx-a100-002\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\n"
        "GPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "1, GPU-bbb, NVIDIA A100-SXM4-80GB, 100, 81920, 0, 0, 30, 55.00, 400.00, 8.0\n"
        "2, GPU-ccc, NVIDIA A100-SXM4-80GB, 200, 81920, 0, 0, 30, 55.00, 400.00, 8.0\n"
        "3, GPU-ddd, NVIDIA A100-SXM4-80GB, 12, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "4, GPU-eee, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 29, 52.00, 400.00, 8.0\n"
        "5, GPU-fff, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 29, 52.00, 400.00, 8.0\n"
        "6, GPU-ggg, NVIDIA A100-SXM4-80GB, 40000, 81920, 80, 70, 55, 250.00, 400.00, 8.0\n"
        "7, GPU-hhh, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 29, 52.00, 400.00, 8.0\n"
        "GPUAPPS\n"
        "GPU-aaa, 1111, python, 77000\n"
        "GPU-ggg, 2222, python, 39000\n"
        "GPUPROCS\n"
        " 1111 colleague /usr/bin/python train.py\n"
        " 2222 otherlab /usr/bin/python train.py\n"
    )
    occ = parse_gpu_occupancy_text(text)
    assert occ["scope"] == "allocated"
    assert occ["gpus"][0]["allocated"] is False
    assert occ["gpus"][0]["memory_pct"] == 94.9
    assert occ["gpus"][3]["allocated"] is True
    connected = pick_connected_gpu(occ["gpus"])
    assert connected["index"] == 3
    assert connected["memory_used_mib"] == 12
    summary = gpu_occupancy_summary(occ)
    assert summary["foreign_on_allocated"] == []
    assert occupancy_blocks_setup(summary) is None
    assert any(p["user"] == "colleague" and not p["allocated_gpu"] for p in summary["processes"])


def test_occupancy_warns_when_nvidia_smi_is_node_wide():
    from splat_explorer.repair_lrz import (
        gpu_occupancy_summary,
        occupancy_needs_overwrite,
        parse_gpu_occupancy_text,
        pick_connected_gpu,
    )

    text = (
        "GPUENV\n"
        "CUDA_VISIBLE_DEVICES=0\n"
        "SLURM_JOB_GPUS=\n"
        "USER=go73kaf2\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\n"
        "GPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "1, GPU-bbb, NVIDIA A100-SXM4-80GB, 12, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "2, GPU-ccc, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 30, 55.00, 400.00, 8.0\n"
        "GPUAPPS\n"
        "GPUPROCS\n"
    )
    occ = parse_gpu_occupancy_text(text)
    assert occ["scope"] == "node"
    assert pick_connected_gpu(occ["gpus"]) is None
    assert "GPU 0" in (occ.get("warning") or "")
    summary = gpu_occupancy_summary(occ)
    assert occupancy_needs_overwrite(summary) is None


def test_occupancy_requires_overwrite_for_foreign_process():
    from splat_explorer.repair_lrz import gpu_occupancy_summary, occupancy_needs_overwrite, parse_gpu_occupancy_text

    text = (
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=0\nUSER=go73kaf2\n"
        "GPUDEVS\n0\n"
        "GPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "GPUAPPS\n"
        "GPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n"
        " 1111 ge72fon2 /alphafold3_venv/bin/python3\n"
    )
    occ = parse_gpu_occupancy_text(text)
    summary = gpu_occupancy_summary(occ)
    assert occ["gpus"][0]["allocated"] is True
    assert summary["foreign_on_allocated"][0]["user"] == "ge72fon2"
    msg = occupancy_needs_overwrite(summary)
    assert msg and "ge72fon2" in msg and "alphafold3_venv" in msg
    assert "this reserved" in msg.lower() or "Load GPU setup" in msg
    assert summary["needs_overwrite"] == msg


def test_occupancy_ignores_our_repair_process():
    from splat_explorer.repair_lrz import gpu_occupancy_summary, occupancy_needs_overwrite, parse_gpu_occupancy_text

    text = (
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_STEP_GPUS=1\nUSER=go73kaf2\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\n"
        "GPUCSV\n"
        "0, GPU-ours, NVIDIA A100-SXM4-80GB, 574, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "GPUAPPS\n"
        "GPU-ours, 2839766, python, 574\n"
        "GPUPROCS\n"
        " 2839766 go73kaf2 python -m splat_explorer.repair_lrz --job-dir /workspace/inputs/abc\n"
    )
    occ = parse_gpu_occupancy_text(text)
    assert occ["scope"] == "allocated"
    assert occ["gpus"][0]["allocated"] is True
    assert occ["physical_gpus"] == [1]
    summary = gpu_occupancy_summary(occ)
    assert occupancy_needs_overwrite(summary) is None
    assert summary["ours_on_allocated"]
    assert summary["ours_on_allocated"][0]["pid"] == 2839766


def test_setup_status_ignores_marker_while_inflight():
    import time

    from splat_explorer.repair_lrz import _SETUP, lrz_setup_status

    with _SETUP["lock"]:
        _SETUP["inflight"] = True
        _SETUP["ok"] = False
        _SETUP["job_id"] = "5786047"
        _SETUP["at"] = time.time() - 125
        _SETUP["message"] = "Starting named Pyxis container"
        _SETUP["error"] = None
        _SETUP["detail"] = None
    try:
        body = lrz_setup_status(
            {
                "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
                "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
                "cpus": 4, "mem": "32G", "container_name": "splat-repair",
            },
            probe_setup={"ok": True, "job_id": "5786047", "gpu": "NVIDIA A100-SXM4-80GB"},
        )
        assert body["inflight"] is True
        assert body["ok"] is False
        assert body["elapsed_s"] >= 120
        assert "Pyxis" in body["message"]
    finally:
        with _SETUP["lock"]:
            _SETUP["inflight"] = False
            _SETUP["ok"] = False
            _SETUP["message"] = ""
            _SETUP["job_id"] = ""
            _SETUP["error"] = None
            _SETUP["detail"] = None
            _SETUP["at"] = 0.0


def test_detect_cuda_arch_respects_env(monkeypatch):
    from splat_explorer.repair_lrz import detect_cuda_arch_list

    monkeypatch.setenv("LRZ_CUDA_ARCH", "9.0")
    assert detect_cuda_arch_list() == "9.0"
    monkeypatch.delenv("LRZ_CUDA_ARCH")
    monkeypatch.setenv("LRZ_CUDA_ARCH", "")
    assert detect_cuda_arch_list() == "8.0"


def test_dashboard_snapshot_uses_allocated_gpu_not_gpu0(monkeypatch, tmp_path):
    import time

    from splat_explorer.repair_lrz import (
        _PROBE,
        lrz_dashboard_snapshot,
        parse_gpu_occupancy_text,
        reset_gpu_probe_cache,
    )

    reset_gpu_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_configured", lambda: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    occ = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=3\nUSER=go73kaf2\n"
        "GPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "3, GPU-ddd, NVIDIA A100-SXM4-80GB, 12, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "GPUAPPS\nGPUPROCS\n"
    )
    occ["node"] = "lrz-dgx-a100-002"
    with _PROBE["lock"]:
        _PROBE["body"] = {
            "slurm": {
                "job_id": "5786047", "state": "R", "node": "lrz-dgx-a100-002",
                "partition": "lrz-dgx-a100-80x8", "elapsed": "5:00:00",
                "timelimit": "2-00:00:00", "name": "gs-48h",
            },
            "jobs": [{
                "job_id": "5786047", "state": "R", "node": "lrz-dgx-a100-002",
                "partition": "lrz-dgx-a100-80x8", "elapsed": "5:00:00",
                "timelimit": "2-00:00:00", "name": "gs-48h",
            }],
            "history_jobs": [],
            "container": {"ok": True, "bytes": 17_700_000_000},
            "gpu": occ,
            "gpu_error": None,
            "ngc": False,
            "workspace_ok": True,
            "setup": {"ok": False},
        }
        _PROBE["at"] = time.time()
        _PROBE["error"] = None
        _PROBE["inflight"] = False
    body = lrz_dashboard_snapshot(repair_job={"status": "idle"})
    assert body["gpu_free_mib"] == 81908
    assert body["gpu_occupancy"]["allocated"][0]["index"] == 3
    assert body["gpu"][0]["memory_used_mib"] == 77773
    gpu_check = next(c for c in body["checks"] if c["id"] == "gpu")
    assert gpu_check["ok"] is True
    assert "GPU 3" in gpu_check["detail"]


def test_wipe_gpu_command_resets_this_jobs_cuda_device():
    from splat_explorer.repair_lrz import wipe_gpu_srun_command

    cmd = wipe_gpu_srun_command({"job_id": "5786047"}, gpu_indices=[3], pids=[1111, 2222])
    assert "--overlap" in cmd
    assert "--jobid=5786047" in cmd
    assert "fuser -v" in cmd
    assert "KILL_FOREIGN" in cmd
    assert "KEEP_OURS" in cmd
    assert "cudaDeviceReset" in cmd
    assert "gpu-reset" in cmd
    assert "ALLOWED_IDX" in cmd
    assert "Device Minor" in cmd
    assert "NVIDIA_VISIBLE_DEVICES" in cmd
    assert "REJECT_HINT_GPU0" in cmd
    assert "SKIP_SHARED_PID" in cmd
    assert "fuser -k" not in cmd
    assert "HINT=3" in cmd or "HINT='3'" in cmd
    assert "1111" not in cmd
    assert "2222" not in cmd
    empty = wipe_gpu_srun_command({"job_id": "5786047"}, gpu_indices=[], pids=[])
    assert "ALLOWED_IDX" in empty
    assert "SKIP_PHYSICAL" in empty


def test_physical_wipe_indices_never_guess_gpu0_on_a_full_node():
    from splat_explorer.repair_lrz import parse_gpu_occupancy_text, physical_indices_for_wipe

    ours = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=3\nUSER=go73kaf2\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\nGPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "3, GPU-ddd, NVIDIA A100-SXM4-80GB, 12, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n 1111 colleague /usr/bin/python train.py\n"
    )
    assert physical_indices_for_wipe(ours) == [3]
    unscoped = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=\nUSER=go73kaf2\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\nGPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "1, GPU-bbb, NVIDIA A100-SXM4-80GB, 12, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "3, GPU-ddd, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 31, 55.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n 1111 ge72fon2 /alphafold3_venv/bin/python3\n"
    )
    assert unscoped["scope"] == "node"
    assert physical_indices_for_wipe(unscoped) == []
    leftover_ours = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=0\nUSER=go73kaf2\n"
        "GPUDEVS\n0\nGPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n 1111 ge72fon2 /alphafold3_venv/bin/python3\n"
    )
    assert physical_indices_for_wipe(leftover_ours) == [0]
    remapped = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=\nSLURM_STEP_GPUS=1\n"
        "USER=go73kaf2\nGPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\nGPUCSV\n"
        "0, GPU-ours, NVIDIA A100-SXM4-80GB, 574, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-ours, 2839766, python, 574\n"
        "GPUPROCS\n 2839766 go73kaf2 python -m splat_explorer.repair_lrz\n"
    )
    assert remapped["scope"] == "allocated"
    assert remapped["gpus"][0]["index"] == 0
    assert remapped["gpus"][0]["allocated"] is True
    assert remapped["physical_gpus"] == [1]
    assert physical_indices_for_wipe(remapped) == [1]


def test_request_setup_auto_starts_when_alphafold_leftover(monkeypatch):
    from splat_explorer.repair_lrz import (
        _PROBE,
        _SETUP,
        parse_gpu_occupancy_text,
        request_lrz_setup,
        reset_gpu_probe_cache,
        reset_setup_cache,
    )

    reset_setup_cache()
    reset_gpu_probe_cache()
    occ = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=0\nUSER=go73kaf2\n"
        "GPUDEVS\n0\nGPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n 1111 ge72fon2 /alphafold3_venv/bin/python3\n"
    )
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    with _PROBE["lock"]:
        _PROBE["body"] = {
            "slurm": {"job_id": "5786047", "state": "R"},
            "jobs": [{"job_id": "5786047", "state": "R"}],
            "gpu": occ,
        }
        _PROBE["inflight"] = False
    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
            started.append(args)
        def start(self):
            return None

    monkeypatch.setattr("splat_explorer.repair_lrz.threading.Thread", FakeThread)
    try:
        body = request_lrz_setup()
        assert body["inflight"] is True
        assert started
        assert "leftover" in (body.get("message") or "").lower() or "Uploading" in (body.get("message") or "")
    finally:
        with _SETUP["lock"]:
            _SETUP["inflight"] = False
        reset_setup_cache()


def test_request_setup_probes_occupancy_when_cache_empty(monkeypatch):
    from splat_explorer.repair_lrz import (
        _SETUP,
        parse_gpu_occupancy_text,
        request_lrz_setup,
        reset_gpu_probe_cache,
        reset_setup_cache,
    )

    reset_setup_cache()
    reset_gpu_probe_cache()
    occ = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=\nUSER=go73kaf2\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\nGPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n 1111 ge72fon2 /alphafold3_venv/bin/python3\n"
    )
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_gpu_occupancy", lambda cfg=None: occ)
    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
            started.append(True)
        def start(self):
            return None

    monkeypatch.setattr("splat_explorer.repair_lrz.threading.Thread", FakeThread)
    try:
        body = request_lrz_setup()
        assert body["inflight"] is True
        assert started
    finally:
        with _SETUP["lock"]:
            _SETUP["inflight"] = False
        reset_setup_cache()


def test_setup_continues_after_wipe_if_vram_still_busy(monkeypatch):
    from splat_explorer.repair_lrz import parse_gpu_occupancy_text, setup_lrz_gpu

    busy = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=0\nUSER=go73kaf2\n"
        "GPUDEVS\n0\nGPUCSV\n"
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 77773, 81920, 0, 0, 31, 60.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-aaa, 1111, python, 77000\n"
        "GPUPROCS\n 1111 ge72fon2 /alphafold3_venv/bin/python3\n"
    )
    cfg = {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    }
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_job", lambda cfg=None: "R")
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_gpu_occupancy", lambda cfg=None: busy)
    monkeypatch.setattr("splat_explorer.repair_lrz.wipe_allocated_gpu", lambda *a, **k: "WIPE_DONE")
    monkeypatch.setattr("splat_explorer.repair_lrz.sync_code_to_dss", lambda cfg: None)
    monkeypatch.setattr("splat_explorer.repair_lrz.launch_detached_setup", lambda cfg: "99")
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.poll_detached_setup",
        lambda cfg, timeout=None: {"ok": True, "gpu": "NVIDIA A100-SXM4-80GB"},
    )
    monkeypatch.setattr("splat_explorer.repair_lrz._set_setup_message", lambda msg: None)
    out = setup_lrz_gpu(cfg, overwrite=True)
    assert out["ok"] is True
    assert out["gpu"] == "NVIDIA A100-SXM4-80GB"


def test_probe_occupancy_does_not_bump_squeue_ttl(monkeypatch):
    from splat_explorer.repair_lrz import _PROBE, probe_gpu_occupancy, reset_gpu_probe_cache

    reset_gpu_probe_cache()
    with _PROBE["lock"]:
        _PROBE["at"] = 123.0
        _PROBE["body"] = {"slurm": {"job_id": "5786047", "state": "R", "node": "lrz-dgx-a100-002"}}
    class Result:
        returncode = 0
        stdout = (
            "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_JOB_GPUS=0\nUSER=go73kaf2\n"
            "GPUDEVS\n0\nGPUCSV\n"
            "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 12, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
            "GPUAPPS\nGPUPROCS\n"
        )
        stderr = ""
    monkeypatch.setattr("splat_explorer.repair_lrz._ssh_run", lambda *a, **k: Result())
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "job_id": "5786047", "user": "go73kaf2", "host": "login.ai.lrz.de",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "32G", "container_name": "splat-repair",
    })
    occ = probe_gpu_occupancy({"job_id": "5786047"})
    assert occ["gpus"][0]["index"] == 0
    with _PROBE["lock"]:
        assert _PROBE["at"] == 123.0
        assert _PROBE["body"]["occupancy_at"]
        assert _PROBE["body"]["gpu"]["gpus"][0]["index"] == 0


def test_setup_reuses_our_worker_without_wipe(monkeypatch):
    from splat_explorer.repair_lrz import parse_gpu_occupancy_text, setup_lrz_gpu

    ours = parse_gpu_occupancy_text(
        "GPUENV\nCUDA_VISIBLE_DEVICES=0\nSLURM_STEP_GPUS=1\nUSER=go73kaf2\n"
        "GPUDEVS\n0\n1\n2\n3\n4\n5\n6\n7\nGPUCSV\n"
        "0, GPU-ours, NVIDIA A100-SXM4-80GB, 574, 81920, 0, 0, 31, 58.00, 400.00, 8.0\n"
        "GPUAPPS\nGPU-ours, 2839766, python, 574\n"
        "GPUPROCS\n 2839766 go73kaf2 python -m splat_explorer.repair_lrz --job-dir /workspace/inputs/abc\n"
    )
    cfg = {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "64G", "container_name": "splat-repair",
    }
    wiped = []
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_job", lambda cfg=None: "R")
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_gpu_occupancy", lambda cfg=None: ours)
    monkeypatch.setattr("splat_explorer.repair_lrz.wipe_allocated_gpu", lambda *a, **k: wiped.append(True) or "WIPE")
    monkeypatch.setattr("splat_explorer.repair_lrz.sync_code_to_dss", lambda cfg: None)
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.read_remote_setup_marker",
        lambda cfg=None: {"ok": True, "job_id": "5786047", "gpu": "NVIDIA A100-SXM4-80GB"},
    )
    monkeypatch.setattr("splat_explorer.repair_lrz._set_setup_message", lambda msg: None)
    out = setup_lrz_gpu(cfg)
    assert out["ok"] is True
    assert out.get("reused") is True
    assert wiped == []


def test_ensure_lrz_gpu_ready_reuses_dss_marker(monkeypatch):
    from splat_explorer.repair_lrz import ensure_lrz_gpu_ready, reset_setup_cache

    reset_setup_cache()
    started = []
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_job", lambda cfg=None: "R")
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "64G", "container_name": "splat-repair",
    })
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.read_remote_setup_marker",
        lambda cfg=None: {
            "ok": True, "job_id": "5786047", "gpu": "NVIDIA A100-SXM4-80GB",
            "torch": "2.5.1", "gsplat": "1.5.0",
        },
    )
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.request_lrz_setup",
        lambda **kwargs: started.append(kwargs) or {"inflight": True},
    )
    body = ensure_lrz_gpu_ready()
    assert body["ok"] is True
    assert started == []
    assert "A100" in (body.get("message") or "")
    reset_setup_cache()


def test_srun_mem_flag_leaves_headroom_on_32g_hold():
    from splat_explorer.repair_lrz import srun_mem_flag

    assert srun_mem_flag({"mem": "32G"}) == "--mem=24G"
    assert srun_mem_flag({"mem": "64G"}) == "--mem=62G"
    assert srun_mem_flag({"mem": "32G"}, probe=True) == "--mem=1G"


def test_tight_host_ram_32g_vs_64g():
    from splat_explorer.repair_lrz import tight_host_ram

    assert tight_host_ram({"mem": "32G"}, cgroup_mb=80 * 1024) is True
    assert tight_host_ram({"mem": "64G"}, cgroup_mb=80 * 1024) is False
    assert tight_host_ram({"mem": "64G"}, cgroup_mb=32 * 1024) is True


def test_lrz_params_enable_packed_on_32g_hold():
    from splat_explorer.repair_lrz import (
        LrzRemoteRepair, remember_live_allocation, reset_live_allocation,
    )

    reset_live_allocation()
    remember_live_allocation(
        job_id="5786047", state="R", mem="32G",
        partition="lrz-dgx-a100-80x8", node="lrz-dgx-a100-002",
    )
    try:
        params = LrzRemoteRepair()._params()
        assert params["packed"] is True
        assert params["train_max_edge"] == 512
        assert params["sparse_grad"] is False
    finally:
        reset_live_allocation()
    wide = LrzRemoteRepair()._params({"mem": "64G"})
    assert wide["packed"] is False
    assert wide["train_max_edge"] == 0
    assert wide["sparse_grad"] is False


def test_enable_packed_job_params(tmp_path):
    from splat_explorer.repair_lrz import PARAMS_JSON, enable_packed_job_params

    path = tmp_path / PARAMS_JSON
    path.write_text('{"method": "gsfix-gsplat", "packed": false}')
    assert enable_packed_job_params(tmp_path) is True
    assert json.loads(path.read_text())["packed"] is True
    assert enable_packed_job_params(tmp_path) is False


def test_enable_tighter_job_params_packs_and_shrinks(tmp_path):
    from splat_explorer.repair_lrz import PARAMS_JSON, enable_tighter_job_params

    path = tmp_path / PARAMS_JSON
    path.write_text('{"method": "gsfix-gsplat", "packed": false}')
    assert enable_tighter_job_params(tmp_path) is True
    body = json.loads(path.read_text())
    assert body["packed"] is True
    assert body.get("sparse_grad") is not True
    assert body["train_max_edge"] == 512
    assert enable_tighter_job_params(tmp_path) is True
    assert json.loads(path.read_text())["train_max_edge"] == 384


def test_overlapping_step_ids_skips_hold():
    from splat_explorer.repair_lrz import overlapping_step_ids

    text = (
        "5786047.extern PD\n"
        "5786047.105 R\n"
        "5786047.batch R\n"
        "5786047 R\n"
        "999.1 R\n"
    )
    assert overlapping_step_ids(text, "5786047") == ["5786047.105"]


def test_pythonpath_exports_single_compile_job_on_32g_hold():
    from splat_explorer.repair_lrz import _remote_pythonpath_exports

    tight = _remote_pythonpath_exports({"mem": "64G", "job_mem": "32G"})
    assert "MAX_JOBS=1" in tight
    assert "TORCH_NUM_THREADS=1" in tight
    assert "TORCH_EXTENSIONS_DIR=/workspace/python/torch_extensions" in tight
    wide = _remote_pythonpath_exports({"mem": "96G"})
    assert "MAX_JOBS=1" in wide
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in wide
    assert "FAST_COMPILE=1" in wide
    assert "TMPDIR=/workspace/tmp" in wide
    assert "NVCC_APPEND_FLAGS='--threads=1'" in wide
    assert "PATH=/workspace/python/bin:$PATH" in wide


def test_oom_error_does_not_dump_ply_loader_logs():
    import subprocess

    from splat_explorer.repair_lrz import format_remote_command_error

    result = subprocess.CompletedProcess(
        ["ssh", "go73kaf2@login.ai.lrz.de", "srun"],
        1,
        stdout="2026-09-14 INFO splat_explorer.scene.ply_loader: Loading 3DGS PLY scene.ply\n",
        stderr=(
            "slurmstepd: error: Detected 2 oom_kill events in StepId=5786047.67. "
            "Some of the step tasks have been OOM Killed.\n"
            "srun: error: lrz-dgx-a100-002: task 0: Out Of Memory\n"
        ),
    )
    msg = format_remote_command_error(["/usr/bin/ssh", "-4", "srun"], result)
    assert "host RAM" in msg
    assert "ply_loader" not in msg
    assert "oom_kill" in msg.lower() or "Out Of Memory" in msg


def test_overlay_running_job_uses_packed_status():
    from splat_explorer.repair_lrz import overlay_running_job_message

    job = overlay_running_job_message(
        {"status": "running", "started_at": 100.0, "message": "Step 13 rsync_up via LRZ ControlMaster · 0s"},
        {"phase": "cuda_ready", "message": "GPU ready: NVIDIA A100-SXM4-80GB. Starting photometric refine…"},
    )
    assert "rsync_up" not in job["message"]
    assert "GPU ready" in job["message"]
    assert job["message"].endswith("s")


def test_occupancy_probe_skips_second_srun_during_repair(monkeypatch):
    from splat_explorer.repair_lrz import probe_gpu_occupancy

    monkeypatch.setattr("splat_explorer.repair_lrz.gpu_work_owner", lambda: "repair")
    called = []
    monkeypatch.setattr(
        "splat_explorer.repair_lrz._ssh_run",
        lambda *a, **k: called.append(True) or (_ for _ in ()).throw(AssertionError("ssh")),
    )
    monkeypatch.setattr(
        "splat_explorer.repair_lrz._cached_occupancy_raw",
        lambda: ({"gpus": [{"index": 0, "name": "A100"}]}, 1.0),
    )
    occ = probe_gpu_occupancy({"job_id": "5786047"})
    assert occ["deferred"] is True
    assert called == []


def test_catalog_entry_matches_truncated_squeue_partition():
    from splat_explorer.repair_lrz import catalog_entry_for_partition

    dgx = catalog_entry_for_partition("lrz-dgx-a")
    assert dgx["id"] == "lrz-dgx-a100-80x8"
    assert dgx["family"] == "A100"
    h100 = catalog_entry_for_partition("lrz-hgx-h100-94x4")
    assert h100["family"] == "H100"
    hgx = catalog_entry_for_partition("lrz-hgx-a100-80x4")
    assert hgx["label"].startswith("HGX A100")


def test_setup_matches_allocation_rejects_h100_marker_on_a100():
    from splat_explorer.repair_lrz import setup_matches_allocation

    ok, reason = setup_matches_allocation(
        {
            "ok": True, "job_id": "1", "gpu": "NVIDIA H100 80GB HBM3",
            "cuda_arch": "9.0", "family": "H100",
        },
        job_id="1",
        slurm={"partition": "lrz-dgx-a100-80x8"},
        connected={"name": "NVIDIA A100-SXM4-80GB", "compute_cap": "8.0"},
    )
    assert ok is False
    assert "H100" in reason
    assert "A100" in reason
    same, _ = setup_matches_allocation(
        {
            "ok": True, "job_id": "5786047", "gpu": "NVIDIA A100-SXM4-80GB",
            "cuda_arch": "8.0",
        },
        job_id="5786047",
        slurm={"partition": "lrz-dgx-a100-80x8", "job_id": "5786047"},
        connected={"name": "NVIDIA A100-SXM4-80GB", "compute_cap": "8.0"},
    )
    assert same is True


def test_srun_mem_flag_uses_live_hold_not_yaml_64g():
    from splat_explorer.repair_lrz import (
        remember_live_allocation, reset_live_allocation, srun_mem_flag,
    )

    reset_live_allocation()
    remember_live_allocation(
        job_id="5786047", state="R", mem="32G",
        partition="lrz-dgx-a100-80x8", node="lrz-dgx-a100-002",
    )
    try:
        assert srun_mem_flag({"mem": "64G", "job_id": "5786047"}) == "--mem=24G"
        assert srun_mem_flag({"mem": "64G", "job_id": "999"}) == "--mem=62G"
        assert "--mem=62G" in (
            __import__("splat_explorer.repair_lrz", fromlist=["srun_worker_command"])
            .srun_worker_command(
                {
                    "job_id": "5777469", "cpus": 4, "workspace": "/dss/ws",
                    "container": "/dss/ws/containers/pytorch.sqsh",
                    "container_name": "splat-repair", "mem": "64G",
                },
                "abc",
            )
        )
    finally:
        reset_live_allocation()


def test_memory_required_error_mentions_live_hold():
    import subprocess

    from splat_explorer.repair_lrz import (
        format_remote_command_error, remember_live_allocation, reset_live_allocation,
    )

    reset_live_allocation()
    remember_live_allocation(job_id="5786047", state="R", mem="32G")
    try:
        result = subprocess.CompletedProcess(
            ["ssh", "go73kaf2@login.ai.lrz.de", "srun"],
            1,
            stdout="",
            stderr="srun: error: Unable to create step for job 5786047: Memory required by task is not available\n",
        )
        msg = format_remote_command_error(["/usr/bin/ssh", "-4", "srun"], result)
        assert "32G" in msg
        assert "Memory required" in msg or "free host RAM" in msg
        assert "yaml" in msg.lower()
    finally:
        reset_live_allocation()


def test_ensure_lrz_gpu_ready_reloads_on_family_mismatch(monkeypatch):
    from splat_explorer.repair_lrz import ensure_lrz_gpu_ready, reset_setup_cache

    reset_setup_cache()
    started = []
    monkeypatch.setattr("splat_explorer.repair_lrz.lrz_session_alive", lambda cfg=None: True)
    monkeypatch.setattr("splat_explorer.repair_lrz.probe_job", lambda cfg=None: "R")
    monkeypatch.setattr("splat_explorer.repair_lrz.load_lrz_config", lambda: {
        "user": "go73kaf2", "host": "login.ai.lrz.de", "job_id": "5786047",
        "workspace": "/dss/ws", "container": "/dss/ws/containers/pytorch.sqsh",
        "cpus": 4, "mem": "64G", "container_name": "splat-repair",
    })
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.live_allocation",
        lambda job_id=None: {
            "job_id": "5786047", "state": "R", "mem": "32G",
            "partition": "lrz-dgx-a100-80x8", "node": "lrz-dgx-a100-002",
        },
    )
    monkeypatch.setattr(
        "splat_explorer.repair_lrz._cached_occupancy_raw",
        lambda: ({
            "gpus": [{
                "index": 0, "name": "NVIDIA A100-SXM4-80GB",
                "compute_cap": "8.0", "allocated": True,
            }],
        }, 1.0),
    )
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.read_remote_setup_marker",
        lambda cfg=None: {
            "ok": True, "job_id": "5786047", "gpu": "NVIDIA H100 80GB HBM3",
            "cuda_arch": "9.0", "family": "H100",
        },
    )
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.request_lrz_setup",
        lambda **kwargs: started.append(kwargs) or {"inflight": True},
    )
    monkeypatch.setattr(
        "splat_explorer.repair_lrz.wait_for_lrz_setup",
        lambda cfg=None: {"ok": True, "reloaded": True, "job_id": "5786047"},
    )
    body = ensure_lrz_gpu_ready()
    assert started and started[0].get("force") is True
    assert body.get("reloaded") is True
    reset_setup_cache()


def test_squeue_parser_includes_expected_end_and_time_left():
    from splat_explorer.repair_lrz import SQUEUE_FORMAT, parse_squeue_line

    assert SQUEUE_FORMAT.endswith("|%e|%L")
    row = parse_squeue_line(
        "5786047|R|lrz-dgx-a100-80x8|lrz-dgx-a100-002|1:02|"
        "08:00:00|None|gs-8h|2026-09-15T18:00:00|64G|"
        "2026-09-16T02:00:00|07:58:58"
    )
    assert row["expected_end"] == "2026-09-16T02:00:00"
    assert row["time_left"] == "07:58:58"
    assert row["sched_nodes"] is None
    legacy = parse_squeue_line(
        "5786047|PD|p||0:00|08:00:00|Priority|gs-8h|"
        "2026-09-15T18:00:00|64G|node[01-02]"
    )
    assert legacy["expected_end"] is None
    assert legacy["time_left"] is None
    assert legacy["sched_nodes"] == "node[01-02]"


def test_scene_run_protocol_and_srun_commands():
    from splat_explorer.scene_runs.lrz_transport import (
        active_job_squeue_command,
        request_protocol_body,
        scene_worker_launch_command,
        scene_worker_srun_command,
    )

    cfg = {
        "job_id": "5786047",
        "user": "go73kaf2",
        "host": "login.ai.lrz.de",
        "workspace": "/dss/ws",
        "container": "/dss/ws/containers/pytorch.sqsh",
        "container_name": "splat-repair",
        "cpus": 4,
        "mem": "64G",
    }
    camera = CameraRig(
        np.array([0.0, 0.0, -1.0]), up_axis="+y",
    ).camera(32, 24, 75.0)
    body = request_protocol_body(
        request_id="step-00003",
        step=3,
        camera=camera,
        repair_seconds=180,
        deadline=2_000_000_000,
    )
    assert body["camera"]["width"] == 32
    assert body["repair_seconds"] == 180.0
    assert body["deadline_unix"] == 2_000_000_000.0
    assert "Repair visible 3D Gaussian rendering artifacts" in body["prompt"]
    queue = active_job_squeue_command("5786047")
    assert "--job=5786047" in queue
    assert "%e|%L" in queue
    srun = scene_worker_srun_command(
        cfg, "run-abc", overall_deadline=2_000_000_000,
    )
    assert "--jobid=5786047" in srun
    assert "--overlap" in srun
    assert "--container-name=splat-repair-5786047" in srun
    assert "splat_explorer.scene_runs.gpu_worker" in srun
    assert "/workspace/scene-runs/run-abc" in srun
    launch = scene_worker_launch_command(
        cfg, "run-abc", overall_deadline=2_000_000_000,
    )
    assert "nohup" in launch
    assert "worker.log" in launch
    assert "launcher.pid" in launch


def test_scene_gpu_worker_keeps_scene_and_qwen_backend_resident(tmp_path):
    import io
    from types import SimpleNamespace

    from PIL import Image

    from splat_explorer.scene_runs.gpu_worker import (
        CHECKPOINT_NAME,
        METRICS_NAME,
        REGENERATED_NAME,
        REQUEST_NAME,
        RESPONSE_NAME,
        SceneRunGpuWorker,
        atomic_write_json,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "scene.ply").write_bytes(b"ply")
    calls = {"load": 0, "editor": 0, "edit": 0, "repair": 0}
    scene = SimpleNamespace(value=0)

    def load_scene(_path):
        calls["load"] += 1
        return scene

    def save_scene(value, path):
        Path(path).write_bytes(f"scene-{value.value}".encode())

    image = Image.new("RGB", (8, 6), (20, 30, 40))
    png = io.BytesIO()
    image.save(png, format="PNG")
    png_bytes = png.getvalue()

    class Editor:
        def edit(self, _path, _prompt):
            calls["edit"] += 1
            return SimpleNamespace(
                images=[png_bytes], payload={"backend": "qwen"}, error=None,
            )

    def make_editor(_config):
        calls["editor"] += 1
        return Editor()

    class Repair:
        def apply_until(self, value, _camera, _rendered, _repaired, **kwargs):
            calls["repair"] += 1
            value.value += 1
            stats = {"backend": "gsfix-gsplat", "n_iters": 20}
            kwargs["on_checkpoint"](stats)
            return stats

    def make_repair(params):
        assert params["max_chunks"] == 0
        assert params["densify"] is True
        return Repair()

    worker = SceneRunGpuWorker(
        run_dir,
        scene_loader=load_scene,
        scene_saver=save_scene,
        image_edit_factory=make_editor,
        repair_factory=make_repair,
    )
    worker._load_scene_once()
    camera = CameraRig(
        np.array([0.0, 0.0, -1.0]), up_axis="+y",
    ).camera(8, 6, 75.0)
    from splat_explorer.repair_lrz import camera_to_dict

    for index in range(2):
        request_dir = run_dir / "requests" / f"step-{index:05d}"
        request_dir.mkdir(parents=True)
        (request_dir / "rendered.png").write_bytes(png_bytes)
        atomic_write_json(
            request_dir / REQUEST_NAME,
            {
                "request_id": request_dir.name,
                "step": index,
                "camera": camera_to_dict(camera),
                "repair_seconds": 180,
            },
        )
        response = worker.process_request(request_dir)
        assert response["status"] == "ok"
        assert (request_dir / REGENERATED_NAME).is_file()
        assert (request_dir / METRICS_NAME).is_file()
        assert (request_dir / RESPONSE_NAME).is_file()

    assert calls == {"load": 1, "editor": 1, "edit": 2, "repair": 2}
    assert scene.value == 2
    assert (run_dir / CHECKPOINT_NAME).read_bytes() == b"scene-2"






