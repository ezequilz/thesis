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
from splat_explorer.scene_runs_ext.config import configure_policy, edit_prompt, validate_options
from splat_explorer.scene_runs_ext.bundle import camera_bundle, transforms
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
    assert "intervention" in next(t for t in policy._tools if t["function"]["name"]=="report_artifact")["function"]["parameters"]["properties"]
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
    with pytest.raises(SceneRunValidationError): extended.validate_config({"width":641})
    with pytest.raises(SceneRunValidationError): extended.validate_config({"image_edit_backend":"qwen-image-edit"})


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
    count=len(json.loads((root/"bundle.json").read_text())["transforms"]["frames"])
    for i in range(count): Image.new("RGB",(32,32),(100,110,120)).save(out/f"{i:05d}.png")
    return out


def test_repair_fits_every_propagated_view_and_commits_only_clone(tmp_path):
    s=scene(); Image.new("RGB",(64,64)).save(tmp_path/"anchor.png")
    phases=[]
    def fitter(candidate,views,targets,**kw):
        assert len(views)==len(targets)==10
        assert kw["intervention"]=="appearance"
        candidate.colors[:]=.8
        return {"n_iters":3}
    candidate, metrics=repair(s,camera(),tmp_path/"anchor.png",tmp_path,
        options={"frames":9,"fit_iterations":3},runtime={},proposal={"intervention":"appearance"},
        should_stop=lambda:False,on_progress=lambda p:phases.append(p["phase"]),
        renderer_factory=Renderer,propagator=propagate,fitter=fitter)
    assert phases==["bundle_render","artifixer_propagate","multiview_fit"]
    np.testing.assert_allclose(s.colors,.4)
    np.testing.assert_allclose(candidate.colors,.8)
    assert metrics["generated_frames"]==9
    assert len(list((tmp_path/"extended/targets").glob("*.png")))==9


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
