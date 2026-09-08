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
    backend = make_repair_backend("gsfix-gsplat")
    assert isinstance(backend, LrzRemoteRepair)
    assert backend.method == "gsfix-gsplat"
    baseline = make_repair_backend("gsfix-gsplat-baseline")
    assert isinstance(baseline, LrzRemoteRepair)
    assert baseline.method == "gsfix-gsplat-baseline"
    auto = make_repair_backend("auto", studio=True)
    assert isinstance(auto, LrzRemoteRepair)


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

    gpus = parse_nvidia_smi_csv(
        "0, NVIDIA A100-SXM4-80GB, 0, 81920, 0, 0, 29, 61.00, 400.00, 8.0\n"
    )
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert gpus[0]["memory_total_mib"] == 81920
    assert gpus[0]["memory_pct"] == 0.0
    assert gpus[0]["compute_cap"] == "8.0"

    bundle = parse_probe_bundle(
        "SQUEUE\n5777469|R|p|node|1:00|6:00:00|None\n"
        "CONTAINER\nOK 123456\nNGC\nMISSING\nWORKSPACE\nOK\nSTATUS\nNONE\n"
    )
    assert bundle["squeue"].startswith("5777469")
    assert bundle["container"] == "OK 123456"
    assert bundle["ngc"] == "MISSING"


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

