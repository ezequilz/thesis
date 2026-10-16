"""Pinned source and isolated model/trainer environments for CUDA workers."""
import json
import os
from pathlib import Path
import subprocess
import sys


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
        run(mamba,'create','-y','-p',trainer,'-c','conda-forge','python=3.10','pip','cudatoolkit-dev=11.7','gxx_linux-64=11','cmake<4','libgl','libglib')
    if not (models/'bin/python').exists():
        run(mamba,'create','-y','-p',models,'-c','conda-forge','python=3.10','pip','colmap=3.9.1','libgl','libglib')
    tp, mp = trainer/'bin/python',models/'bin/python'
    run(tp,'-m','pip','install','torch==1.13.1+cu117','torchvision==0.14.1+cu117','--extra-index-url','https://download.pytorch.org/whl/cu117')
    run(tp,'-m','pip','install','numpy==1.24.4','plyfile==1.0.3','tqdm','lpips','opencv-python-headless<4.12','scipy<1.14','joblib','tensorboard','ninja','setuptools<70','wheel')
    env = {**os.environ,'CUDA_HOME':str(trainer),'CUDA_PATH':str(trainer),
           'PATH':str(trainer/'bin')+os.pathsep+os.environ['PATH'],
           'LD_LIBRARY_PATH':str(trainer/'lib')+':/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
           'CC':str(trainer/'bin/x86_64-conda-linux-gnu-cc'),
           'CXX':str(trainer/'bin/x86_64-conda-linux-gnu-c++'),'MAX_JOBS':'2'}
    for name in ('diff-gaussian-rasterization','simple-knn'):
        run(tp,'-m','pip','install','--no-build-isolation',repos['oraclegs']/'submodules'/name,env=env)
    run(mp,'-m','pip','install','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124')
    run(mp,'-m','pip','install','-e',repos['seva'],'numpy==1.24.4','opencv-python<4.12',
        'einops','huggingface-hub','safetensors','matplotlib<3.9','scipy<1.14','timm')
    weights = base/'models/oraclegs/depth_anything_v2_vitl.pth'
    weights.parent.mkdir(parents=True,exist_ok=True)
    if not weights.exists():
        temporary = weights.with_suffix('.part')
        run('curl','--fail','--location','--retry','3','https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth','-o',temporary)
        temporary.rename(weights)
    run(tp,'-c','import torch, diff_gaussian_rasterization, simple_knn._C; assert torch.cuda.is_available()')
    run(mp,'-c','import torch; assert torch.cuda.is_available()')
    print('OracleGS environments installed. SEVA/VGGT weights download on first run; SEVA may require Hugging Face access.')


if __name__ == '__main__':
    main()
