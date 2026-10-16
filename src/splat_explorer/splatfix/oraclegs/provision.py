"""Pinned source and isolated model/trainer environments for CUDA workers."""
import json
import os
from pathlib import Path
import subprocess
import sys


def huggingface_environment():
    """Use the authenticated account, independent of LRZ's redirected HOME."""
    import pwd
    home = Path(pwd.getpwuid(os.getuid()).pw_dir) / '.cache/huggingface'
    return {'HF_HOME': str(home), 'HF_HUB_CACHE': str(home/'hub'),
            'HUGGINGFACE_HUB_CACHE': str(home/'hub'), 'HF_TOKEN_PATH': str(home/'token')}


def runtime_environment(prefix):
    env = {**os.environ, **huggingface_environment(), 'PYTHONNOUSERSITE':'1',
           'LD_LIBRARY_PATH':str(Path(prefix)/'lib')+':/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
    for key in ('PYTHONPATH', 'PYTHONHOME', 'TORCH_CUDA_ARCH_LIST'):
        env.pop(key, None)
    return env


def ensure_cuda(base, mamba, run):
    """Install real NVIDIA packages, avoiding cudatoolkit-dev's runfile hook.

    An existing Python executable does not establish that CUDA installed: the
    conda-forge post-link hook can fail while mamba reports a successful create.
    Keep the compiler separate from both Python environments and check it on
    every setup attempt, including retries of an incomplete environment.
    """
    cuda = Path(base) / 'oraclegs-cuda-11.7'
    def ready():
        if not all((cuda / p).is_file() for p in
                   ('bin/nvcc', 'include/cuda.h', 'include/cuda_runtime.h', 'lib/libcudart.so')):
            return False
        try:
            return 'release 11.7,' in subprocess.check_output(
                [str(cuda / 'bin/nvcc'), '--version'], text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError):
            return False
    if not ready():
        existing = (cuda / 'conda-meta/history').exists()
        run(mamba, 'install' if existing else 'create', '-y', '-p', cuda,
            '-c', 'nvidia/label/cuda-11.7.1', '-c', 'nvidia/label/cuda-11.7.0',
            '-c', 'conda-forge', *(['--force-reinstall'] if existing else []),
            'cuda-toolkit=11.7.1', 'cuda-nvcc=11.7.99', 'cuda-cudart-dev=11.7.99')
    if not ready():
        raise RuntimeError(f'CUDA 11.7 compiler/headers/runtime incomplete at {cuda}; refusing extension build')
    return cuda


def main():
    base, mamba = Path(sys.argv[1]).resolve(), sys.argv[2]
    lock = json.loads(Path(__file__).with_name('upstream-lock.json').read_text())
    def run(*argv, **kwargs):
        subprocess.run([str(x) for x in argv],check=True,**kwargs)
    repos = {}
    for name, directory in [('oraclegs','OracleGS'),('seva','oraclegs-seva'),('vggt','oraclegs-vggt'),('depth','oraclegs-depth')]:
        repo = base/'third_party'/directory
        repo.parent.mkdir(parents=True,exist_ok=True)
        if not repo.exists():
            run('git','clone',lock[name]['url'],repo)
            run('git','-C',repo,'checkout','--detach',lock[name]['revision'])
        head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if head != lock[name]['revision']:
            raise ValueError(f'Refusing to overwrite incompatible runtime: {repo}')
        if name == 'oraclegs':
            run('git','-C',repo,'submodule','update','--init','--recursive','submodules/simple-knn','submodules/diff-gaussian-rasterization')
        import hashlib
        for relative, digest in lock[name]['files'].items():
            if hashlib.sha256((repo/relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Modified runtime source: {repo/relative}')
        repos[name] = repo
    run('git','-C',repos['oraclegs'],'submodule','update','--init','--recursive','submodules/simple-knn','submodules/diff-gaussian-rasterization')
    trainer, models = base/'oraclegs-env',base/'oraclegs-model-env'
    if not (trainer/'bin/python').exists():
        run(mamba,'create','-y','-p',trainer,'-c','conda-forge','python=3.10','pip','gxx_linux-64=11','cmake<4','libgl','libglib')
    if not (models/'bin/python').exists():
        run(mamba,'create','-y','-p',models,'-c','conda-forge','python=3.10','pip','colmap=3.9.1','libgl','libglib')
    tp, mp = trainer/'bin/python',models/'bin/python'
    cuda = ensure_cuda(base, mamba, run)
    run(tp,'-m','pip','install','torch==1.13.1+cu117','torchvision==0.14.1+cu117','--extra-index-url','https://download.pytorch.org/whl/cu117')
    run(tp,'-m','pip','install','numpy==1.24.4','plyfile==1.0.3','tqdm','lpips','opencv-python-headless<4.12','scipy<1.14','joblib','tensorboard','ninja','setuptools<70','wheel')
    env = {**runtime_environment(trainer),'CUDA_HOME':str(cuda),'CUDA_PATH':str(cuda),
           'PATH':str(cuda/'bin')+os.pathsep+str(trainer/'bin')+os.pathsep+os.environ['PATH'],
           'LD_LIBRARY_PATH':str(cuda/'lib')+':'+str(trainer/'lib')+':/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
           'CC':str(trainer/'bin/x86_64-conda-linux-gnu-cc'),
           'CXX':str(trainer/'bin/x86_64-conda-linux-gnu-c++'),'MAX_JOBS':'2'}
    # NGC exports architectures for its newer PyTorch. Let the pinned trainer
    # detect this allocation's GPU rather than inherit unsupported targets.
    for name in ('diff-gaussian-rasterization','simple-knn'):
        run(tp,'-m','pip','install','--no-build-isolation',repos['oraclegs']/'submodules'/name,env=env)
    run(mp,'-m','pip','install','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124')
    run(mp,'-m','pip','install','-e',repos['seva'],'numpy==1.24.4','opencv-python<4.12',
        'einops','huggingface-hub==0.28.1','diffusers==0.32.2',
        'safetensors','matplotlib<3.9','scipy<1.14','timm')
    weights = base/'models/oraclegs/depth_anything_v2_vitl.pth'
    weights.parent.mkdir(parents=True,exist_ok=True)
    if not weights.exists():
        temporary = weights.with_suffix('.part')
        run('curl','--fail','--location','--retry','3','https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth','-o',temporary)
        temporary.rename(weights)
    # Match model_worker launch isolation; the container's libpython/libtorch
    # search paths must not leak into these separately pinned environments.
    run(tp,'-c','import torch, diff_gaussian_rasterization, simple_knn._C; assert torch.cuda.is_available()',env=runtime_environment(trainer))
    model_env = runtime_environment(models)
    run(mp,'-c','import torch; from seva.modules.autoencoder import AutoEncoder; '
        'from seva.modules.conditioner import CLIPConditioner; import seva.eval; '
        'assert torch.cuda.is_available()',env=model_env)
    print('OracleGS environments installed. SEVA/VGGT weights download on first run; SEVA may require Hugging Face access.')


if __name__ == '__main__':
    main()
