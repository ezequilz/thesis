"""Explicit one-time ArtiFixer provisioning, invoked by the GPU setup button.

Uses an isolated venv on DSS; baseline /workspace/python is never upgraded.
Imported GPU images often create that venv without a pip module (Debian/Ubuntu
ships ensurepip in python3-venv, which NGC and CUDA images omit). Setup seeds
pip itself so the extended-run load does not depend on rebuilding the image.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from .config import UPSTREAM_REVISION

GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
PUBLIC_PYPI = "https://pypi.org/simple"


def use_public_pypi(env):
    """Force PyPI and ignore the NGC pip index baked into pytorch images.

    Those images set ``extra-index-url = https://pypi.ngc.nvidia.com`` in
    ``/etc/pip.conf`` or ``PIP_EXTRA_INDEX_URL``. LRZ cannot resolve that host,
    and current pip fails the whole install when the extra index does not resolve.
    """
    for key, value in list(env.items()):
        if key.startswith("PIP_") and "ngc.nvidia.com" in str(value):
            env.pop(key, None)
    env["PIP_INDEX_URL"] = PUBLIC_PYPI
    env["PIP_EXTRA_INDEX_URL"] = PUBLIC_PYPI
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _imports(python, module, env):
    return subprocess.run(
        [str(python), "-c", f"import {module}"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _seed_pip(python, env, run):
    """Install a pip module into an existing venv interpreter."""
    print(f"ARTIFIXER_SETUP {python} -m ensurepip --upgrade", flush=True)
    seeded = subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], env=env)
    if seeded.returncode == 0 and _imports(python, "pip", env):
        return
    errors = []
    dest = Path(env.get("TMPDIR") or "/tmp") / "get-pip.py"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"ARTIFIXER_SETUP fetching {GET_PIP_URL}", flush=True)
        with urllib.request.urlopen(GET_PIP_URL, timeout=120) as response:
            dest.write_bytes(response.read())
        run([python, dest])
    except Exception as exc:
        errors.append(f"get-pip: {exc}")
        print(f"ARTIFIXER_SETUP get-pip failed: {exc}", flush=True)
    if _imports(python, "pip", env):
        return
    if _imports(Path(sys.executable), "pip", env):
        try:
            site = subprocess.run(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                check=True, capture_output=True, text=True, env=env,
            ).stdout.strip()
            Path(site).mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "pip", "install", "--upgrade", "--target", site, "pip"])
        except Exception as exc:
            errors.append(f"container pip --target: {exc}")
            print(f"ARTIFIXER_SETUP container pip seed failed: {exc}", flush=True)
    else:
        errors.append(f"{sys.executable} has no pip module")
    if not _imports(python, "pip", env):
        detail = "; ".join(errors) or "ensurepip failed"
        raise RuntimeError(
            f"{python} has no pip module after GPU setup bootstrap ({detail}). "
            "The image needs python3-venv/ensurepip, or HTTPS to bootstrap.pypa.io."
        )


def ensure_venv(envdir, env, run):
    """Return the venv interpreter, creating it and seeding pip when needed.

    A previous failed ``python -m venv`` leaves ``bin/python`` on DSS without
    pip. Treating that file as a finished environment makes every later
    ``python -m pip`` fail and the extended run stay at waiting_gpu.
    """
    envdir = Path(envdir)
    python = envdir / "bin/python"
    if not python.is_file():
        command = [sys.executable, "-m", "venv", str(envdir)]
        print("ARTIFIXER_SETUP " + " ".join(command), flush=True)
        created = subprocess.run(command, env=env)
        if created.returncode != 0 or not python.is_file():
            print("ARTIFIXER_SETUP venv ensurepip unavailable; recreating --without-pip", flush=True)
            run([sys.executable, "-m", "venv", "--clear", "--without-pip", str(envdir)])
    if not _imports(python, "pip", env):
        print(f"ARTIFIXER_SETUP {python} has no pip module; bootstrapping", flush=True)
        _seed_pip(python, env, run)
    return python


def provision(workspace="/workspace"):
    root = Path(workspace)
    repo = root / "third_party/ArtiFixer"
    envdir = root / "artifixer-venv"
    checkpoint_dir = root / "models/artifixer"
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    stamp = root / "models/artifixer/setup.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HOME"] = str(root / "models/huggingface")
    env["MAX_JOBS"] = "1"
    use_public_pypi(env)
    print(f"ARTIFIXER_SETUP pip index {PUBLIC_PYPI} (ignoring pypi.ngc.nvidia.com)", flush=True)
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
    python = ensure_venv(envdir, env, run)
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
