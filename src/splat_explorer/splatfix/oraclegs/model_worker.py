"""Isolated official-model entry points. Executed in the configured CUDA env."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def install_snapshot_marker():
    """Observe upstream Scene.save; mark only after its PLY write has completed."""
    import scene
    original = scene.Scene.save
    def save(self, iteration):
        result = original(self, iteration)
        path = Path(self.model_path)/f'point_cloud/iteration_{iteration}/point_cloud.ply'
        marker = Path(self.model_path)/'saved-snapshot.json'
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps({'iteration':int(iteration), 'size_bytes':path.stat().st_size}))
        temporary.replace(marker)
        return result
    scene.Scene.save = save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['assets','seva','attention','depth','train','script'])
    parser.add_argument('repo')
    parser.add_argument('request')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    repo = Path(args.repo).resolve()
    sys.path.insert(0,str(repo))
    os.chdir(repo)
    if args.action == 'assets':
        from model_assets import check_seva_access, verified_vae, record_vae
        check_seva_access()
        vae = verified_vae()
        record_vae(request['output'], vae)
    elif args.action == 'seva':
        stride = request.pop('pad_to_stride', None)
        if stride:
            from adapter import pad_seva_inputs
            request['data_path'], request['W'], request['H'] = pad_seva_inputs(
                Path(request['data_path']), Path(request['work_dir']).parent/'seva-padded-inputs', stride)
        from model_assets import verified_vae, vae_source, record_vae
        from diffusers.models import AutoencoderKL
        vae = verified_vae()
        record_vae(Path(request['work_dir']).parent/'vae-provenance.json', vae)
        with vae_source(AutoencoderKL, vae):
            import demo
        demo.WORK_DIR = request.pop('work_dir')
        demo.main(**request)
    elif args.action == 'attention':
        import numpy as np
        import torch
        np.random.seed(request.pop('seed'))
        torch.manual_seed(0)
        import visualize_attn
        visualize_attn.main(argparse.Namespace(**request))
    elif args.action == 'depth':
        import cv2
        import numpy as np
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
        model.load_state_dict(torch.load(request['weights'],map_location='cpu',weights_only=True))
        model.to('cuda').eval()
        destination = Path(request['output'])
        destination.mkdir(parents=True,exist_ok=True)
        for path in sorted(Path(request['images']).glob('*.png')):
            with torch.inference_mode():
                depth = model.infer_image(cv2.imread(str(path)))
            if not np.isfinite(depth).all() or depth.max() <= depth.min():
                raise ValueError(f'Invalid estimated inverse depth for {path.name}')
            # Store pseudo inverse depth, not a colorized visualization. Robust
            # scale/offset fitting to SfM follows in the authors' own utility.
            scaled = (depth-depth.min())/(depth.max()-depth.min())
            if not cv2.imwrite(str(destination/path.name),(scaled*65535).astype(np.uint16)):
                raise IOError('Could not save depth')
    elif args.action == 'train':
        install_snapshot_marker()
        import scene.dataset_readers as readers
        original = readers.readColmapCameras
        references = set(json.loads(Path(request['references']).read_text()))
        def calibrated_views(*a, **kw):
            from adapter import reference_only_ground_truth
            return reference_only_ground_truth(original(*a, **kw), references)
        readers.readColmapCameras = calibrated_views
        sys.argv = [str(repo/'train.py'), *request['argv']]
        runpy.run_path(str(repo/'train.py'),run_name='__main__')
    else:
        if request.get('script') == 'train.py':
            install_snapshot_marker()
        sys.argv = [str(repo/request['script']),*request['argv']]
        sys.path.insert(0,str((repo/request['script']).parent))
        runpy.run_path(sys.argv[0],run_name='__main__')


if __name__ == '__main__':
    main()
