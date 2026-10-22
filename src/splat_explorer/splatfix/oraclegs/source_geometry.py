"""Stage optional source geometry locally; portable assets travel with the job."""
import hashlib
import math
from pathlib import Path
import shutil
import numpy as np
from PIL import Image


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def inverse_depth(depth, valid):
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    inverse = np.zeros_like(depth)
    np.divide(1., depth, out=inverse, where=mask)
    mask &= np.isfinite(inverse) & (inverse > 0)
    inverse[~mask] = 0
    return inverse, mask


def fit_depth(depth, valid, source_k, frame):
    """Nearest-neighbor calibrated warp, without blending depths across edges."""
    target_k = np.array([[frame['fl_x'],0,frame['cx']], [0,frame['fl_y'],frame['cy']], [0,0,1.]])
    mapping = np.asarray(source_k) @ np.linalg.inv(target_k)
    coefficients = tuple(mapping[:2].ravel())
    size = (frame['w'],frame['h'])
    def warp(array):
        return np.asarray(Image.fromarray(array).transform(size, Image.Transform.AFFINE,
            coefficients, resample=Image.Resampling.NEAREST, fillcolor=0)).copy()
    return inverse_depth(warp(np.where(valid,depth,0).astype(np.float32)), warp(valid.astype(np.uint8)).astype(bool))


def inspect_ply(path):
    from ...scene.ply_loader import _parse_header, _REQUIRED
    with Path(path).open('rb') as stream:
        count, props, offset = _parse_header(stream)
    names = {name for name,_ in props}
    if count <= 0 or not set(_REQUIRED).issubset(names):
        raise ValueError('Source initialization requires a nonempty standard 3D Gaussian PLY')
    rest = {name for name in names if name.startswith('f_rest_')}
    degree = math.isqrt(len(rest)//3+1)-1
    if degree not in range(4) or rest != {f'f_rest_{i}' for i in range(3*((degree+1)**2-1))}:
        raise ValueError('Unsupported source spherical-harmonic layout')
    data = np.memmap(path, dtype=np.dtype(props), mode='r', offset=offset, shape=(count,))
    for name in set(_REQUIRED) | rest:
        if not np.isfinite(data[name]).all():
            raise ValueError('Nonfinite source Gaussian parameters: '+name)
    return count, degree


def stage_source_geometry(checkpoint, root, manifest, frames, *, should_stop=lambda:False):
    use_depth = (manifest['oracle_depth_source'] == 'source_splat' or
                 manifest.get('seva_geometry_guidance', 'off') == 'source_splat')
    use_init = manifest['oracle_initialization'] in ('source_splat', 'point_cloud')
    if not (use_depth or use_init):
        return
    from ..checkpoint import validate_source, camera_from_record
    from ...scene import load_scene, save_ply
    from ...rendering.cpu_splat_renderer import CpuSplatRenderer
    from ..g4splat.source_depth import conservative_mask
    validate_source(checkpoint)
    root = Path(root)
    source = Path(checkpoint.manifest['scene_path'])
    cached_depth = use_depth and all(v.get('source_depth') for v in checkpoint.views)
    scene = load_scene(source) if (use_depth and not cached_depth) or (use_init and source.suffix.lower() != '.ply') else None
    if use_depth:
        directory = root/'source-depth'
        directory.mkdir()
        renderer = None if cached_depth else CpuSplatRenderer(scene)
        records = []
        for view, frame in zip(checkpoint.views, frames):
            if should_stop():
                raise InterruptedError('Cancelled while rendering OracleGS source depth')
            if not np.allclose([frame['cx'],frame['cy']], [frame['w']/2,frame['h']/2], rtol=0,atol=1e-4):
                raise ValueError('Source-depth supervision requires centered staged intrinsics for the OracleGS renderer')
            camera = camera_from_record(view)
            if cached_depth:
                from .captures import asset_path
                with np.load(asset_path(checkpoint, view['source_depth']), allow_pickle=False) as cached:
                    depth, mask = cached['depth'], cached['valid']
                    inv, valid = fit_depth(depth, mask, camera.intrinsics, frame)
            else:
                depth = renderer.render_depth(camera)
                inv, valid = fit_depth(depth, conservative_mask(depth), camera.intrinsics, frame)
            if not valid.any():
                raise ValueError('No reliable source depth for '+str(view['id']))
            name = Path(frame['file_path']).stem
            path = directory/(name+'.npz')
            np.savez_compressed(path, invdepth=inv, valid=valid)
            records.append({'image':name+'.png', 'path':str(path.relative_to(root)),
                            'sha256':digest(path), 'valid_fraction':float(valid.mean())})
        manifest['source_depth'] = {'version':1, 'views':records,
            'convention':'inverse_positive_camera_z_scene_units',
            'conversion':'1/z on valid pixels; no normalization or SfM scale fitting',
            'mask':'front-surface coverage >=0.15; 1px erosion; local depth spread <=10%',
            'renderer':'CpuSplatRenderer.render_depth', 'reused_checkpoint_depth':bool(cached_depth),
            'fallback':'none; invalid pixels excluded',
            'source_fingerprint':checkpoint.manifest.get('source_fingerprint')}
    if use_init:
        if should_stop():
            raise InterruptedError('Cancelled before source initialization export')
        path = root/'source-initialization.ply'
        if source.suffix.lower() == '.ply':
            shutil.copyfile(source,path)
            representation = 'original PLY parameters including spherical harmonics'
        else:
            save_ply(scene,path)
            representation = 'decoded viewer geometry and DC colors; higher-order SH unavailable'
        count, degree = inspect_ply(path)
        manifest['source_initialization'] = {'path':path.name, 'sha256':digest(path),
            'gaussian_count':count, 'sh_degree':degree, 'representation':representation,
            'optimizer':'fresh; standard OracleGS training schedule',
            'source_fingerprint':checkpoint.manifest.get('source_fingerprint')}
        manifest['source_splat_used_for_initialization'] = True


def validate_assets(root, manifest):
    root = Path(root).resolve()
    records = []
    if (manifest.get('oracle_depth_source', 'predicted') == 'source_splat' or
            manifest.get('seva_geometry_guidance', 'off') == 'source_splat'):
        records = manifest.get('source_depth',{}).get('views',[])
        if {r['image'] for r in records} != {Path(v['image']).name for v in manifest['views']}:
            raise ValueError('Missing source depth assets; restage inputs')
    if manifest.get('oracle_initialization', 'sfm') in ('source_splat', 'point_cloud'):
        if not manifest.get('source_initialization'):
            raise ValueError('Missing source initialization; restage inputs')
        records = [*records,manifest['source_initialization']]
    for record in records:
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != record['sha256']:
            raise ValueError('Source geometry asset missing or checksum mismatch')


def stage_saved_pose_depth(root, frames):
    """Render reference depth from the snapshotted PLY at staged OpenGL poses."""
    from ...scene import load_scene
    from ...rendering.base import Camera
    from ...rendering.cpu_splat_renderer import CpuSplatRenderer
    from ..g4splat.source_depth import conservative_mask
    root = Path(root)
    renderer = CpuSplatRenderer(load_scene(root/'source-initialization.ply'))
    directory = root/'source-depth'
    directory.mkdir()
    records = []
    for frame in frames:
        if not np.allclose([frame['cx'], frame['cy']], [frame['w']/2, frame['h']/2], rtol=0, atol=1e-4):
            raise ValueError('Source-depth supervision requires centered staged intrinsics')
        pose = np.asarray(frame['transform_matrix']) @ np.diag([1., -1., -1., 1.])
        camera = Camera(pose[:3, 3], pose[:3, :3], width=frame['w'], height=frame['h'],
                        fov_deg=float(np.degrees(2*np.arctan(frame['w']/(2*frame['fl_x'])))))
        depth = renderer.render_depth(camera)
        inverse, valid = fit_depth(depth, conservative_mask(depth), camera.intrinsics, frame)
        if not valid.any():
            raise ValueError('No reliable source depth for '+frame['file_path'])
        name = Path(frame['file_path']).name
        path = directory/(Path(name).stem+'.npz')
        np.savez_compressed(path, invdepth=inverse, valid=valid)
        records.append({'image':name, 'path':str(path.relative_to(root)),
                        'sha256':digest(path), 'valid_fraction':float(valid.mean())})
    return {'version':1, 'views':records,
            'convention':'inverse_positive_camera_z_scene_units',
            'conversion':'1/z on valid pixels; no normalization or SfM scale fitting',
            'mask':'front-surface coverage >=0.15; 1px erosion; local depth spread <=10%',
            'renderer':'CpuSplatRenderer.render_depth', 'fallback':'none; invalid pixels excluded'}
