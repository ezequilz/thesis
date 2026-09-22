"""Extended contracts and transaction safety, without CUDA or model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from splat_explorer.agent.actions import Action, ACTION_TOOLS
from splat_explorer.rendering.base import Camera
from splat_explorer.scene import GaussianScene
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.scene_runs.models import SceneRunConfig
from splat_explorer.scene_runs_ext.config import (
    configure_policy, edit_prompt, proposal, validate_options, validate_repair_resolution,
)
from splat_explorer.scene_runs_ext.bundle import camera_bundle, transforms, repair_camera, selected_views
from splat_explorer.scene_runs_ext.pipeline import repair
from splat_explorer.web.scene_run_studio import SceneRunStudio, SceneRunValidationError


def scene():
    return GaussianScene(np.zeros((2,3),np.float32), np.ones((2,3),np.float32),
                         np.array([[1,0,0,0]]*2,np.float32), np.ones(2,np.float32),
                         np.full((2,3),.4,np.float32))


def camera():
    return Camera(np.zeros(3,np.float32), np.eye(3,dtype=np.float32), width=32, height=32)


def test_bundle_has_translation_preserves_calibration_and_returns_to_anchor():
    c = camera()
    views = camera_bundle(c, np.full((32,32), 3.), frames=9)
    assert views[0] is c and views[-1] is c
    assert max(np.linalg.norm(v.position-c.position) for v in views) > .1
    for v in views:
        np.testing.assert_allclose(v.rotation.T@v.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(v.intrinsics,c.intrinsics)
    np.testing.assert_allclose(transforms(views)["frames"][0]["transform_matrix"], c.c2w)
    with pytest.raises(ValueError,match="depth"):
        camera_bundle(c, np.full((32,32), np.inf))


def test_extended_config_policy_and_baseline_are_separate(tmp_path):
    original=json.dumps(ACTION_TOOLS)
    policy=SimpleNamespace(_tools=ACTION_TOOLS)
    configure_policy(policy)
    assert json.dumps(ACTION_TOOLS)==original
    assert "intervention" not in next(t for t in policy._tools if t["function"]["name"]=="report_artifact")["function"]["parameters"]["properties"]
    assert "cabinet" in edit_prompt(Action("report_artifact",{"description":"cabinet", "intervention":"appearance"}))
    assert "pipeline" not in SceneRunConfig().to_dict()
    with pytest.raises(ValueError): validate_options({"frames": 10})
    with pytest.raises(ValueError): validate_options({"span_fraction": float("nan")})
    with pytest.raises(ValueError): validate_options({"python": "bad"})
    app=SimpleNamespace(cfg=SimpleNamespace(output=SimpleNamespace(dir=str(tmp_path))))
    store=SceneRunStore(tmp_path/"scene-runs")
    baseline=SceneRunStudio(app,store)
    extended=SceneRunStudio(app,store,pipeline="extended")
    old=baseline.create({})
    new=extended.create({"extended":{"frames":9}})
    assert len(baseline.list_runs())==len(extended.list_runs())==1
    assert new["config"]["pipeline"]=="extended"
    assert new["config"]["extended"]["frames"]==9
    assert old["config"]["repair_backend"]=="gsfix-gsplat"
    assert new["config"]["width"] == 640 and new["config"]["repair_width"] == 1920
    assert new["config"]["repair_height"] == 1440
    with pytest.raises(SceneRunValidationError): extended.validate_config({"width":641})
    with pytest.raises(SceneRunValidationError): extended.validate_config({"image_edit_backend":"qwen-image-edit"})
    with pytest.raises(SceneRunValidationError, match="multiples of 16"):
        extended.validate_config({"repair_width":1928, "repair_height":1440})
    with pytest.raises(SceneRunValidationError, match="aspect ratio"):
        extended.validate_config({"repair_width":1920, "repair_height":1088})
    assert validate_repair_resolution(640, 480, 0, 0) == (0, 0)
    tuned = SceneRunConfig(pipeline="extended", width=640, height=480,
                            repair_width=1280, repair_height=960)
    assert tuned.to_dict()["repair_width"] == 1280
    with pytest.raises(ValueError, match="aspect ratio"):
        SceneRunConfig(pipeline="extended", width=640, height=480,
                       repair_width=1920, repair_height=1088)


def test_setup_and_both_pipelines_share_one_lease(tmp_path):
    store=SceneRunStore(tmp_path)
    run=store.create_run({"pipeline":"extended"})
    lease=store.acquire_setup_lease()
    assert lease and store.acquire_gpu_lease(run.run_id) is None
    lease.release()
    runlease=store.acquire_gpu_lease(run.run_id)
    assert runlease and store.acquire_setup_lease() is None
    runlease.release()
    assert store.gpu_lease_owner() is None


class Renderer:
    def __init__(self, scene): pass
    def render(self, c):
        return (np.full((c.height,c.width,3),60,np.uint8),
                np.ones((c.height,c.width),np.float32),
                np.full((c.height,c.width),3.,np.float32))


def propagate(root,runtime,stop):
    out=root/"pred"
    out.mkdir()
    cameras=json.loads((root/"bundle.json").read_text())["transforms"]
    count=len(cameras["frames"])
    for i in range(count): Image.new("RGB",(cameras["w"],cameras["h"]),(100,110,120)).save(out/f"{i:05d}.png")
    return out


def test_repair_fits_every_propagated_view_and_commits_only_clone(tmp_path):
    s=scene(); Image.new("RGB",(64,64)).save(tmp_path/"anchor.png")
    phases=[]
    def fitter(candidate,views,targets,**kw):
        assert len(views)==len(targets)==9
        assert "intervention" not in kw
        # The independently edited black anchor must never become a fit target.
        assert all(np.array_equal(t[0,0], [100,110,120]) for t in targets)
        candidate.colors[:]=.8
        return {"n_iters":3}
    candidate, metrics=repair(s,camera(),tmp_path/"anchor.png",tmp_path,
        options={"frames":9,"fit_iterations":3},runtime={},proposal={"intervention":"appearance"},
        should_stop=lambda:False,on_progress=lambda p:phases.append(p["phase"]),
        renderer_factory=Renderer,propagator=propagate,fitter=fitter)
    assert phases==["bundle_render","artifixer_propagate","multiview_fit","native_validation"]
    np.testing.assert_allclose(s.colors,.4)
    np.testing.assert_allclose(candidate.colors,.8)
    assert metrics["generated_frames"]==9
    assert metrics["anchor_role"] == "generation_reference_only"
    assert metrics["fitting_resolution"] == [64,64]
    assert metrics["exploration_resolution"] == [32,32]
    assert len(list((tmp_path/"extended/targets").glob("*.png")))==9
    assert len(list((tmp_path/"extended/validation").glob("*.png")))==4


def test_native_repair_resolution_preserves_frustum_and_uses_explicit_budget():
    c = Camera(np.zeros(3), np.eye(3), width=640, height=480, fov_deg=75)
    native = repair_camera(c, (1448,1086))
    assert (native.width,native.height) == (1408,1056)
    assert native.fov_deg == c.fov_deg
    np.testing.assert_allclose(native.intrinsics[:2] / 2.2, c.intrinsics[:2], rtol=1e-6)
    limited = repair_camera(c, (1448,1086), max_pixels=640*480)
    assert (limited.width,limited.height) == (640,480)
    with pytest.raises(ValueError, match="aspect ratio"):
        repair_camera(c, (1448,1024))
    with pytest.raises(ValueError, match="pixel budget"):
        repair_camera(c, (1448,1086), max_pixels=1)


def test_selected_views_are_fitted_jointly_with_calibrated_references(tmp_path):
    Image.new("RGB",(64,64)).save(tmp_path/"anchor.png")
    Image.new("RGB",(96,96)).save(tmp_path/"earlier.png")
    c = camera()
    other = Camera(np.array([1,0,0],np.float32), c.rotation, width=32,height=32)
    def fitter(candidate, views, targets, **kwargs):
        assert len(views) == len(targets) == 18
        assert all(v.width == 64 and v.height == 64 for v in views)
        np.testing.assert_allclose(views[9].position, other.position)
        assert kwargs["iterations"] == 2000
        return {}
    _, metrics = repair(scene(),c,tmp_path/"anchor.png",tmp_path,
        options={"frames":9},runtime={},proposal={"repair_scope":"scene"},
        selected_views=[{"step":0,"camera":other,"reference_path":tmp_path/"earlier.png"}],
        should_stop=lambda:False,on_progress=lambda _:None,
        renderer_factory=Renderer,propagator=propagate,fitter=fitter)
    manifest=json.loads((tmp_path/"extended/bundle.json").read_text())
    assert [s["start"] for s in manifest["segments"]] == [0,9]
    assert [r["frame_index"] for r in manifest["references"]] == [0,9]
    assert metrics["selected_steps"] == [0] and metrics["reference_views"] == 2


def test_agent_view_selection_only_resolves_completed_recorded_cameras(tmp_path):
    from splat_explorer.repair_lrz import camera_to_dict
    request = tmp_path/"repair-00003"
    observation = tmp_path/"render-00001"
    observation.mkdir()
    body = {"step":1,"operation":"render","camera":camera_to_dict(camera())}
    (observation/"request.json").write_text(json.dumps(body))
    (observation/"response.json").write_text(json.dumps({"status":"ok"}))
    views = selected_views(request,{"view_steps":[1,1,3],"repair_scope":"scene"},3)
    assert len(views) == 1 and views[0]["step"] == 1
    assert "reference_path" not in views[0]
    for steps in ([2],[4],[-1],[True],["../scene"]):
        with pytest.raises(ValueError):
            selected_views(request,{"view_steps":steps},3)
    with pytest.raises(ValueError,match="explore"):
        selected_views(request,{"repair_scope":"scene"},3)


@pytest.mark.parametrize("failure",["missing_frame","fit_error","stop"])
def test_failed_candidate_never_mutates_source(tmp_path,failure):
    s=scene(); Image.new("RGB",(32,32)).save(tmp_path/"anchor.png")
    stopped=[False]
    def generator(root,runtime,stop):
        out=propagate(root,runtime,stop)
        if failure=="missing_frame": (out/"00008.png").unlink()
        if failure=="stop": stopped[0]=True
        return out
    def fitter(candidate,*args,**kw):
        candidate.colors[:]=0
        raise RuntimeError("fit failed")
    with pytest.raises((RuntimeError,InterruptedError)):
        repair(s,camera(),tmp_path/"anchor.png",tmp_path,options={"frames":9},runtime={},
            proposal={},should_stop=lambda:stopped[0],on_progress=lambda _:None,
            renderer_factory=Renderer,propagator=generator,fitter=fitter)
    np.testing.assert_allclose(s.colors,.4)


def _venv_python(tmp_path):
    python = tmp_path / "artifixer-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    return python


def test_artifixer_pip_install_is_isolated_from_ngc():
    from splat_explorer.scene_runs_ext.setup import PUBLIC_PYPI, PYTORCH_CU128, pip_install_command
    cmd = pip_install_command("/workspace/artifixer-venv/bin/python", "--upgrade", "pip")
    assert cmd[:6] == [
        "/workspace/artifixer-venv/bin/python", "-m", "pip",
        "--isolated", "--disable-pip-version-check", "install",
    ]
    assert "--index-url" in cmd and PUBLIC_PYPI in cmd
    assert "pypi.ngc.nvidia.com" not in " ".join(cmd)
    torch = pip_install_command(
        "python", "torch==2.11.0", "torchvision",
        index=PYTORCH_CU128, extra_index=PUBLIC_PYPI,
    )
    assert "--isolated" in torch
    assert PYTORCH_CU128 in torch
    assert PUBLIC_PYPI in torch


def test_fused_ssim_is_built_against_the_venv_torch():
    from splat_explorer.scene_runs_ext.setup import (
        pip_install_command, split_torch_extension_requirements,
    )
    text = (
        "torchmetrics\n"
        "setuptools <72.1.0\n"
        "# Fused-ssim\n"
        "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157\n"
        "opencv-python<4.12.0 # because of our numpy<2.0 requirement\n"
    )
    filtered, extensions = split_torch_extension_requirements(text)
    assert "fused-ssim" not in filtered
    assert "torchmetrics" in filtered and "setuptools <72.1.0" in filtered
    assert "opencv-python<4.12.0" in filtered
    assert extensions == [
        "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157",
    ]
    cmd = pip_install_command("python", "--no-build-isolation", extensions[0])
    assert "--no-build-isolation" in cmd
    assert "--isolated" in cmd
    assert cmd.index("--isolated") < cmd.index("install")
    assert extensions[0] in cmd


def test_public_pypi_replaces_unresolvable_ngc_index():
    import os
    from splat_explorer.scene_runs_ext.setup import PUBLIC_PYPI, use_public_pypi
    env = use_public_pypi({
        "PIP_INDEX_URL": "https://pypi.ngc.nvidia.com",
        "PIP_EXTRA_INDEX_URL": "https://pypi.ngc.nvidia.com",
        "PIP_CONFIG_FILE": "/etc/pip.conf",
        "PIP_TRUSTED_HOST": "pypi.ngc.nvidia.com",
        "PATH": "/usr/bin",
    })
    assert env["PIP_INDEX_URL"] == PUBLIC_PYPI
    assert env["PIP_EXTRA_INDEX_URL"] == PUBLIC_PYPI
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert "ngc.nvidia.com" not in " ".join(env.values())
    assert env["PATH"] == "/usr/bin"


def test_existing_venv_without_pip_is_seeded_by_ensurepip(tmp_path, monkeypatch):
    from splat_explorer.scene_runs_ext import setup
    python = _venv_python(tmp_path)
    checks = {"n": 0}
    commands = []

    def imports(py, module, env):
        assert module == "pip" and str(py) == str(python)
        checks["n"] += 1
        return checks["n"] > 1

    def fake_run(args, **kwargs):
        commands.append([str(a) for a in args])
        return SimpleNamespace(returncode=0)

    def refuse(args, **kwargs):
        raise AssertionError(args)

    monkeypatch.setattr(setup, "_imports", imports)
    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    got = setup.ensure_venv(tmp_path / "artifixer-venv", {}, refuse)
    assert got == python
    assert commands == [[str(python), "-m", "ensurepip", "--upgrade"]]


def test_missing_ensurepip_bootstraps_with_get_pip(tmp_path, monkeypatch):
    from splat_explorer.scene_runs_ext import setup
    python = _venv_python(tmp_path)
    ready = {"pip": False}
    commands = []

    def imports(py, module, env):
        return ready["pip"] and str(py) == str(python)

    class Body:
        def read(self):
            return b"# get-pip\n"
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    def run(args, **kwargs):
        commands.append([str(a) for a in args])
        ready["pip"] = True

    monkeypatch.setattr(setup, "_imports", imports)
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda url, timeout=0: Body())
    env = {"TMPDIR": str(tmp_path)}
    assert setup.ensure_venv(tmp_path / "artifixer-venv", env, run) == python
    assert commands == [[
        str(python), str(tmp_path / "get-pip.py"), "--index-url", setup.PUBLIC_PYPI,
    ]]
    assert (tmp_path / "get-pip.py").read_bytes() == b"# get-pip\n"


def test_container_pip_seeds_venv_when_get_pip_is_unreachable(tmp_path, monkeypatch):
    from splat_explorer.scene_runs_ext import setup
    python = _venv_python(tmp_path)
    state = {"venv": False}
    commands = []

    def imports(py, module, env):
        if str(py) == str(python):
            return state["venv"]
        return True

    def fake_run(args, **kwargs):
        if kwargs.get("check"):
            return SimpleNamespace(returncode=0, stdout=str(tmp_path / "site") + "\n")
        return SimpleNamespace(returncode=1, stdout="")

    def run(args, **kwargs):
        commands.append([str(a) for a in args])
        if "--target" in commands[-1]:
            state["venv"] = True

    def urlopen(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(setup, "_imports", imports)
    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    monkeypatch.setattr(setup.urllib.request, "urlopen", urlopen)
    assert setup.ensure_venv(tmp_path / "artifixer-venv", {"TMPDIR": str(tmp_path)}, run) == python
    assert "--target" in commands[-1] and commands[-1][-1] == "pip"


def test_venv_creation_falls_back_when_ensurepip_cannot_create_it(tmp_path, monkeypatch):
    from splat_explorer.scene_runs_ext import setup
    envdir = tmp_path / "artifixer-venv"
    commands = []

    def fake_run(args, **kwargs):
        commands.append([str(a) for a in args])
        return SimpleNamespace(returncode=1)

    def run(args, **kwargs):
        commands.append([str(a) for a in args])
        python = envdir / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    monkeypatch.setattr(setup, "_imports", lambda *a, **k: True)
    assert setup.ensure_venv(envdir, {}, run) == envdir / "bin" / "python"
    assert commands[0][1:3] == ["-m", "venv"]
    assert "--clear" in commands[1] and "--without-pip" in commands[1]


def test_bootstrap_failure_reports_missing_pip(tmp_path, monkeypatch):
    from splat_explorer.scene_runs_ext import setup
    _venv_python(tmp_path)

    def urlopen(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(setup, "_imports", lambda *a, **k: False)
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setattr(setup.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="no pip module"):
        setup.ensure_venv(tmp_path / "artifixer-venv", {"TMPDIR": str(tmp_path)}, lambda *a, **k: None)


def test_legacy_intervention_cannot_change_repair_prompt_or_routing():
    a = Action("report_artifact", {"description": "smeared door", "intervention": "appearance"})
    b = Action("report_artifact", {"description": "smeared door", "intervention": "structure"})
    assert proposal(a) == proposal(b)
    assert "intervention" not in proposal(a)
    assert edit_prompt(a) == edit_prompt(b)
    assert validate_options()["fit_iterations"] == 1000


def test_joint_optimizer_updates_geometry_and_opacity(monkeypatch):
    # Exercise real autograd and Adam with a differentiable stand-in rasterizer.
    # This verifies optimizer wiring; CUDA rendering requires the GPU smoke test.
    import sys
    torch = pytest.importorskip("torch")
    from splat_explorer.scene_runs_ext.fitting import fit_views
    def rasterization(**kw):
        value = (kw["colors"].mean() + kw["means"].mean()
                 + kw["scales"].mean() + kw["opacities"].mean()
                 + kw["quats"][:, 1:].mean()) / 5
        return value.expand(1, kw["height"], kw["width"], 3), None, None
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=rasterization))
    s = scene()
    s.opacities[:] = .5
    before = s.copy()
    metrics = fit_views(s, [camera()], [np.full((32,32,3), 200, np.uint8)],
                        iterations=3, should_stop=lambda: False,
                        on_progress=lambda _: None, device="cpu")
    for name in ("means", "scales", "quats", "opacities", "colors"):
        assert not np.array_equal(getattr(s,name), getattr(before,name)), name
    assert metrics["after"]["target_l1"] < metrics["before"]["target_l1"]
    assert metrics["view_updates"] == [3]
    assert s.num_gaussians == before.num_gaussians
