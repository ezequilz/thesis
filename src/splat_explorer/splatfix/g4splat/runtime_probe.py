"""Executed with the official interpreter; fails before expensive training."""
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    repo = Path(sys.argv[1]).resolve()
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / 'mast3r'), str(repo / 'mast3r/dust3r'),
                   str(repo / '2d-gaussian-splatting'), str(repo / 'Depth-Anything-V2')]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('G4Splat requires an NVIDIA CUDA GPU in the worker environment')
    # Exercise a CUDA kernel as well as import the compiled modules.
    assert (torch.ones(1, device='cuda') + 1).item() == 2
    for name in ('diff_surfel_rasterization', 'simple_knn._C',
                 'tetranerf.utils.extension', 'pytorch3d._C', 'asmk', 'faiss',
                 'segment_anything', 'detectron2', 'depth_anything_v2.dpt',
                 'mast3r.model', 'models.curope', 'gradio', 'diffusers', 'transformers', 'xformers.ops'):
        importlib.import_module(name)
    from simple_knn._C import distCUDA2
    distances = distCUDA2(torch.rand(1024, 3, device='cuda'))
    if not torch.isfinite(distances).all().item():
        raise RuntimeError('G4Splat KNN CUDA kernel returned non-finite distances')
    print(json.dumps({'knn_cuda_kernel': 'passed', 'python': sys.version, 'torch': torch.__version__,
                      'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}))


if __name__ == '__main__':
    main()
