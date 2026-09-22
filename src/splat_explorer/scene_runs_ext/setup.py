"""Explicit one-time ArtiFixer provisioning, invoked by the GPU setup button.

Uses an isolated venv on DSS; baseline /workspace/python is never upgraded.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from .config import UPSTREAM_REVISION


def provision(workspace="/workspace"):
    root = Path(workspace)
    repo = root / "third_party/ArtiFixer"
    envdir = root / "artifixer-venv"
    python = envdir / "bin/python"
    checkpoint_dir = root / "models/artifixer"
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    stamp = root / "models/artifixer/setup.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HOME"] = str(root / "models/huggingface")
    env["MAX_JOBS"] = "1"
    def run(args, **kwargs):
        print("ARTIFIXER_SETUP " + " ".join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), check=True, env=env, **kwargs)
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/nv-tlabs/ArtiFixer.git", repo])
    current = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    if current != UPSTREAM_REVISION:
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               check=True, capture_output=True, text=True).stdout.strip()
        if dirty:
            raise RuntimeError(f"ArtiFixer checkout has local changes: {repo}; refusing to overwrite them")
        run(["git", "-C", repo, "fetch", "origin", UPSTREAM_REVISION])
        run(["git", "-C", repo, "checkout", "--detach", UPSTREAM_REVISION])
    run(["git", "-C", repo, "submodule", "update", "--init", "--recursive"])
    if not python.is_file():
        run([sys.executable, "-m", "venv", envdir])
    ready = {}
    if stamp.is_file():
        ready = json.loads(stamp.read_text())
    if ready.get("revision") != UPSTREAM_REVISION:
        run([python, "-m", "pip", "install", "--upgrade", "pip"])
        run([python, "-m", "pip", "install", "torch==2.11.0", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu128"])
        run([python, "-m", "pip", "install", "-r", repo / "thirdparty/3DGRUT-ArtiFixer/requirements.txt"])
        run([python, "-m", "pip", "install", "--no-deps", "-e", repo / "thirdparty/3DGRUT-ArtiFixer"])
        run([python, "-m", "pip", "install", "accelerate==1.13.0", "diffusers==0.37.1",
             "transformers==5.5.0", "ftfy", "einops", "scipy", "wandb", "tqdm",
             "Pillow", "matplotlib", "opencv-python-headless", "pyyaml", "torchmetrics",
             "imageio-ffmpeg", "h5py", "av", "torch-fidelity"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # This download only happens in the explicitly requested setup, never in a run.
    code = (
        "from huggingface_hub import hf_hub_download, snapshot_download; "
        f"hf_hub_download('nvidia/ArtiFixer', 'artifixer-1.3b.pt', local_dir={str(checkpoint_dir)!r}); "
        f"snapshot_download({model_id!r}, allow_patterns=['vae/*', 'transformer/config.json', 'scheduler/*'])"
    )
    run([python, "-c", code])
    runtime = {"repo": str(repo), "python": str(python),
               "checkpoint": str(checkpoint_dir / "artifixer-1.3b.pt"),
               "model_id": model_id, "hf_home": env["HF_HOME"]}
    probe_env = dict(env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([str(python), str(Path(__file__).with_name("artifixer_bridge.py")),
                    "--repo", str(repo), "--checkpoint", runtime["checkpoint"],
                    "--model-id", model_id, "--preflight"],
                   check=True, env=probe_env, cwd=repo)
    stamp.write_text(json.dumps({"revision": UPSTREAM_REVISION, "runtime": runtime}, indent=2))
    return runtime
