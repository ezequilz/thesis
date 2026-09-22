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
PYTORCH_CU128 = "https://download.pytorch.org/whl/cu128"


def use_public_pypi(env):
    """Force PyPI and ignore the NGC pip index baked into pytorch images.

    Those images set ``extra-index-url = https://pypi.ngc.nvidia.com`` in
    ``/etc/pip.conf`` or ``PIP_EXTRA_INDEX_URL``. LRZ cannot resolve that host,
    and current pip fails the whole install when the extra index does not resolve.
    ``extra-index-url`` is additive, so dropping the env var is not enough:
    callers also pass ``--isolated`` via :func:`pip_install_command`.
    """
    for key, value in list(env.items()):
        if key.startswith("PIP_") and "ngc.nvidia.com" in str(value):
            env.pop(key, None)
    env["PIP_INDEX_URL"] = PUBLIC_PYPI
    env["PIP_EXTRA_INDEX_URL"] = PUBLIC_PYPI
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def write_public_pip_config(directory):
    """Pip config that lists only public PyPI, for nested installs inside setup."""
    path = Path(directory) / "pip-public.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[global]\n"
        f"index-url = {PUBLIC_PYPI}\n"
        f"extra-index-url = {PUBLIC_PYPI}\n"
    )
    return path


def split_torch_extension_requirements(text):
    """Keep PyPI deps isolated, and pull out CUDA extensions that import torch.

    ``fused-ssim`` has no ``pyproject.toml``. Its ``setup.py`` imports torch
    while pip is still collecting build requirements. The default isolated
    build env does not contain the torch installed in the venv, so that step
    fails with ``ModuleNotFoundError: No module named 'torch'``. Those specs
    are installed afterwards with ``--no-build-isolation``.
    """
    kept = []
    extensions = []
    for line in str(text).splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec.startswith("git+") or "fused-ssim" in spec:
            extensions.append(spec)
            continue
        kept.append(line)
    filtered = "\n".join(kept)
    if kept:
        filtered += "\n"
    return filtered, extensions


def pip_install_command(python, *args, index=PUBLIC_PYPI, extra_index=None):
    """``pip install`` that cannot see ``/etc/pip.conf`` or ``PIP_*`` from the image.

    ``--isolated`` ignores the NGC extra index. Dependencies that are not on
    ``index`` (PyTorch wheels, for example) need ``extra_index`` on the command
    line because isolated mode also ignores ``PIP_EXTRA_INDEX_URL``.
    """
    command = [
        str(python), "-m", "pip", "--isolated", "--disable-pip-version-check",
        "install", "--index-url", str(index),
    ]
    if extra_index:
        command.extend(["--extra-index-url", str(extra_index)])
    command.extend(str(arg) for arg in args)
    return command


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
        run([python, dest, "--index-url", PUBLIC_PYPI])
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
            run(pip_install_command(
                sys.executable, "--upgrade", "--target", site, "pip",
            ))
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
    env["PIP_CONFIG_FILE"] = str(write_public_pip_config(root / "python"))
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
        run(pip_install_command(python, "--upgrade", "pip"))
        run(pip_install_command(
            python, "torch==2.11.0", "torchvision",
            index=PYTORCH_CU128, extra_index=PUBLIC_PYPI,
        ))
        req_file = repo / "thirdparty/3DGRUT-ArtiFixer/requirements.txt"
        filtered, extensions = split_torch_extension_requirements(req_file.read_text())
        filtered_file = root / "tmp" / "3dgrut-requirements.txt"
        filtered_file.parent.mkdir(parents=True, exist_ok=True)
        filtered_file.write_text(filtered)
        run(pip_install_command(python, "-r", filtered_file))
        if extensions:
            # setuptools<72.1.0 is in the filtered file and must already be
            # installed; fused-ssim's CUDA build uses that interpreter.
            run(pip_install_command(python, "ninja"))
            for spec in extensions:
                run(pip_install_command(python, "--no-build-isolation", spec))
        # 3DGRUT declares Python >=3.11; LRZ PyTorch images are often 3.10.
        run(pip_install_command(
            python, "--no-deps", "--ignore-requires-python", "-e",
            repo / "thirdparty/3DGRUT-ArtiFixer",
        ))
        run(pip_install_command(
            python, "accelerate==1.13.0", "diffusers==0.37.1",
            "transformers==5.5.0", "ftfy", "einops", "scipy", "wandb", "tqdm",
            "Pillow", "matplotlib", "opencv-python-headless", "pyyaml", "torchmetrics",
            "imageio-ffmpeg", "h5py", "av", "torch-fidelity",
        ))
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
