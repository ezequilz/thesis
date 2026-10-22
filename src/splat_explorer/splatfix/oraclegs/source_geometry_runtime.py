"""Optional in-memory adapters; pinned upstream files stay unchanged."""
import json
from pathlib import Path
import numpy as np


def load_depth(path, shape):
    with np.load(path, allow_pickle=False) as data:
        inverse, valid = data['invdepth'].copy(), data['valid'].astype(bool)
    if inverse.shape != tuple(shape) or valid.shape != inverse.shape:
        raise ValueError('Source inverse depth does not match training image dimensions')
    if not valid.any() or not np.isfinite(inverse).all() or (inverse[valid] <= 0).any():
        raise ValueError('Invalid source inverse depth')
    return inverse, valid


def initialize_gaussians(model, path, spatial_lr_scale):
    import torch
    model.load_ply(str(path))
    model.spatial_lr_scale = spatial_lr_scale
    model.max_radii2D = torch.zeros((model.get_xyz.shape[0],), device=model.get_xyz.device)


def install_source_geometry(request):
    report = {'depth_source': 'source_splat' if request.get('source_depth') else 'predicted',
              'reference_views': [], 'initialization': None}
    def publish():
        if request.get('source_geometry_report'):
            path = Path(request['source_geometry_report'])
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(path)
    if request.get('source_depth'):
        import torch
        # scene.__init__ imports camera_utils, which imports scene.cameras.
        # Enter through scene first to avoid partially initialized camera_utils.
        import scene
        import utils.camera_utils as cameras
        original = cameras.loadCam
        directory = Path(request['source_depth'])
        def load(args, index, info, scale, synthetic_scene):
            # No predicted-depth decoding, quantization, or scale heuristic.
            camera = original(args,index,info._replace(depth_path='',depth_params=None),scale,synthetic_scene)
            if not info.is_synthetic and info.image_name not in request.get('rgb_only_references', []):
                inverse, valid = load_depth(directory/(Path(info.image_name).stem+'.npz'),
                                            (camera.image_height,camera.image_width))
                camera.invdepthmap = torch.as_tensor(inverse[None],device=camera.original_image.device)
                camera.depth_mask = torch.as_tensor(valid[None],device=camera.original_image.device,dtype=camera.original_image.dtype)
                camera.depth_reliable = True
                report['reference_views'].append({'image': info.image_name,
                    'width': camera.image_width, 'height': camera.image_height,
                    'valid_fraction': float(valid.mean()), 'depth_reliable': True})
                publish()
            return camera
        cameras.loadCam = load
    if request.get('source_initialization'):
        from scene.gaussian_model import GaussianModel
        path = request['source_initialization']
        original_create = GaussianModel.create_from_pcd
        mode = request.get('oracle_initialization', 'source_splat')
        def create(self, pcd, spatial_lr_scale):
            if mode == 'point_cloud':
                from plyfile import PlyData
                from utils.graphics_utils import BasicPointCloud
                from utils.sh_utils import SH2RGB
                vertices = PlyData.read(path)['vertex']
                points = np.column_stack([vertices[key] for key in ('x', 'y', 'z')])
                dc = np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)])
                colors = np.clip(SH2RGB(dc), 0., 1.)
                # Only positions and decoded RGB enter the standard fresh initializer.
                cloud = BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))
                original_create(self, cloud, spatial_lr_scale)
            else:
                initialize_gaussians(self,path,spatial_lr_scale)
            report['initialization'] = {'source': path, 'mode': mode, 'gaussian_count': int(self.get_xyz.shape[0])}
            publish()
        GaussianModel.create_from_pcd = create
