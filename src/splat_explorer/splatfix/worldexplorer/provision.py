"""Install only WorldExplorer's isolated runtime; share account model blobs."""
import json
from pathlib import Path
import subprocess
import sys
from worker import model_environment, atomic

REVISION = 'e661fcc71fb6cf087bcd9cc676345e30e11aa6b2'

def main():
    base, mamba = Path(sys.argv[1]), sys.argv[2]
    (base/'worldexplorer-ready.json').unlink(missing_ok=True)
    repo = base/'third_party/worldexplorer'
    prefix = base/'worldexplorer-env'
    def run(*args, **kwargs):
        subprocess.run([str(a) for a in args], check=True, **kwargs)
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        run('git', 'clone', 'https://github.com/mschneider456/worldexplorer.git', repo)
        run('git', '-C', repo, 'checkout', '--detach', REVISION)
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    run('git', '-c', 'core.fileMode=false', '-C', repo, 'diff', '--exit-code', '--quiet')
    if head != REVISION:
        raise RuntimeError('WorldExplorer source revision differs; refusing to overwrite it')
    if not (prefix/'bin/python').exists():
        run(mamba,'create','-y','-p',prefix,'-c','conda-forge','python=3.10','pip','libgl','libglib','ffmpeg','gxx_linux-64=11','cmake<4')
    cuda = base/'worldexplorer-cuda'
    system_cuda = Path('/usr/local/cuda/bin/nvcc')
    reuse_cuda = system_cuda.exists() and 'release 12.8,' in subprocess.check_output([str(system_cuda),'--version'],text=True)
    if not reuse_cuda and not (cuda/'bin/nvcc').exists():
        run(mamba,'create','-y','-p',cuda,'-c','nvidia/label/cuda-12.8.1','-c','conda-forge','cuda-toolkit=12.8')
    env = model_environment(base)
    env.update(CC=str(prefix/'bin/x86_64-conda-linux-gnu-cc'), CXX=str(prefix/'bin/x86_64-conda-linux-gnu-c++'))
    python = prefix/'bin/python'
    constraints = base/'worldexplorer-cache/constraints.txt'
    constraints.write_text('numpy==1.24.4\ntorch==2.8.0\ntorchvision==0.23.0\nxformers==0.0.32.post2\nopen-clip-torch==2.20.0\nprotobuf==3.20.3\n')
    def pip(*args):
        run(python,'-m','pip','install','-c',constraints,*args,env=env)
    pip('torch==2.8.0','torchvision==0.23.0','xformers==0.0.32.post2','--index-url','https://download.pytorch.org/whl/cu128')
    pip('-e',repo/'model/stable-virtual-camera','diffusers==0.35.1','transformers==4.56.2',
        'huggingface-hub==0.34.4','numpy==1.24.4','opencv-python<4.12','protobuf','sentencepiece','easydict','plyfile','accelerate','open3d','torchmetrics','timm','open-clip-torch==2.20.0','setuptools<81')
    p3d = base/'third_party/worldexplorer-pytorch3d'
    if not p3d.exists():
        run('git','clone','https://github.com/facebookresearch/pytorch3d.git',p3d)
    run('git','-c','core.fileMode=false','-C',p3d,'diff','--exit-code','--quiet')
    run('git','-c','core.fileMode=false','-C',p3d,'checkout','--detach','75ebeeaea0908c5527e7b1e305fbc7681382db47')
    pip('--no-build-isolation',p3d)
    pip('git+https://github.com/nerfstudio-project/nerfstudio.git@50e0e3c')
    # Distinct depth variants: OracleGS's non-metric Large is NOT interchangeable.
    script = '''
from pathlib import Path
from huggingface_hub import hf_hub_download
import sys
repo = Path(sys.argv[1])
sys.path.insert(0,str(repo/'model/stable-virtual-camera'))
for model, filename, directory in [
 ('depth-anything/Video-Depth-Anything-Small','video_depth_anything_vits.pth','Video-Depth-Anything'),
 ('depth-anything/Depth-Anything-V2-Metric-Hypersim-Large','depth_anything_v2_metric_hypersim_vitl.pth','Depth_Anything_V2')]:
    blob = Path(hf_hub_download(model,filename))
    target = repo/'model'/directory/'checkpoints'/filename
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): target.symlink_to(blob)
# Assert SEVA is reused, never download a second copy under another cache.
checkpoint = hf_hub_download('stabilityai/stable-virtual-camera','modelv1.1.safetensors',local_files_only=True)
import torch, pytorch3d, seva.eval
from safetensors import safe_open
from seva.model import Seva, SevaParams
with torch.device('meta'):
    expected = Seva(SevaParams()).state_dict()
with safe_open(checkpoint,framework='pt',device='cpu') as weights:
    assert set(weights.keys()) == set(expected), 'Cached SEVA checkpoint keys differ'
    assert all(tuple(weights.get_slice(k).get_shape()) == tuple(v.shape) for k,v in expected.items()), 'Cached SEVA checkpoint shapes differ'
from model.scaffold_generation import run_scaffold_generation
from model.scene_expansion import run_scene_expansion
assert torch.cuda.is_available(), 'CUDA unavailable'
from pytorch3d.ops import knn_points
x=torch.zeros((1,4,3),device='cuda'); knn_points(x,x,K=1); torch.cuda.synchronize()
print('CUDA kernel, model imports and shared SEVA cache checks passed')
'''
    run(python,'-c',script,repo,cwd=repo,env=env)
    run(python,'-m','pip','check',env=env)
    # Exercise the previously missing dependency, including an actual GPU step.
    smoke = '''
import sys, torch
sys.path.insert(0,sys.argv[1])
from model_assets import verified_inpainting
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
pipe = StableDiffusionInpaintPipeline.from_pretrained(str(verified_inpainting()), torch_dtype=torch.float16, local_files_only=True, use_safetensors=True).to('cuda')
image = pipe(prompt='a blue room', image=Image.new('RGB',(64,64)), mask_image=Image.new('L',(64,64),255), height=64, width=64, num_inference_steps=2, generator=torch.Generator('cuda').manual_seed(0)).images[0]
assert image.size == (64,64)
print('Verified SD2 inpainting checkpoint and GPU inference passed')
'''
    run(python,'-c',smoke,Path(__file__).parent,env=env)
    run(prefix/'bin/ns-train','splatfacto-big','--help',env=env)
    run(python,'-c',"from huggingface_hub import hf_hub_download; hf_hub_download('black-forest-labs/FLUX.1-dev','model_index.json'); print('Gated FLUX access passed')",env=env)
    packages = subprocess.check_output([str(python),'-m','pip','freeze'],text=True,env=env)
    atomic(base/'worldexplorer-ready.json', {'revision':REVISION,'packages':packages,
           'hf_home':env['HF_HOME'],'cuda_home':env['CUDA_HOME']})

if __name__ == '__main__':
    main()
