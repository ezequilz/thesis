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
    assert "/workspace/python" in cmd
    assert "--container-name=splat-repair" in cmd


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
    assert "--container-name=splat-repair" in cmd
    assert "repair_lrz --setup" in cmd
    assert "/workspace/python" in cmd


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

