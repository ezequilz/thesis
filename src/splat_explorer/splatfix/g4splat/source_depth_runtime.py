"""Python 3.9-compatible, opt-in upstream input-depth extension."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

HASHES = {
    'matcha/pointmap/depthanythingv2.py': '6471f990df5d750e63e4579bb03304bf2cd916d46164df5aef3d1ab003f0db23',
    'matcha/dm_trainers/charts_alignment.py': '28aa9cbdba1c1c718135b035b5bbfa42a7740317143147c7bface2f5594eaad2',
    '2d-gaussian-splatting/planes/refine_depth_with_planes.py': '70418a06e9cedcdf9c8bc045246fe5ede4fac9b9481297aed801883b206ed65e',
}


def calibrated_depth(name, shape):
    """Load checked input maps on precisely the cropped image's pixel grid."""
    root = Path(os.environ['SPLATFIX_SOURCE_DEPTH_INPUTS'])
    scene = Path(os.environ['SPLATFIX_SOURCE_DEPTH_SCENE'])
    manifest = json.loads((root / 'inputs.json').read_text())
    records = {Path(r['image']).stem: r for r in manifest['source_depth']['views']}
    key = Path(str(name)).stem
    if key not in records:
        raise ValueError('Missing source depth for input view ' + key)
    record = records[key]
    path = root / record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Source depth checksum mismatch')
    cameras = json.loads((scene / 'cameras.json').read_text())
    index = [Path(p).stem for p in cameras['filepaths']].index(key)
    cal = cameras['image_calibration'][index]
    with np.load(path, allow_pickle=False) as data:
        depth, valid = data['depth'], data['valid']
        if list(depth.shape[::-1]) != cal['source_wh'] or valid.shape != depth.shape:
            raise ValueError('Source depth dimensions disagree with RGB calibration')
        if not np.allclose(data['intrinsics'], cal['source_intrinsics'], atol=1e-5):
            raise ValueError('Source depth intrinsics disagree with RGB calibration')
        if not np.allclose(data['c2w'], cameras['cams2world'][index], atol=1e-5):
            raise ValueError('Source depth pose disagrees with calibrated camera')
        valid = valid & (data['confidence'] > 0) & np.isfinite(depth) & (depth > 0)
    x, y = cal['crop_xy']; w, h = cal['output_wh']
    depth, valid = depth[y:y+h, x:x+w], valid[y:y+h, x:x+w]
    th, tw = map(int, shape)
    # Nearest preserves surface ordering; bilinear would blend across holes.
    depth = np.asarray(Image.fromarray(depth).resize((tw, th), Image.Resampling.NEAREST)).copy()
    valid = np.asarray(Image.fromarray(valid).resize((tw, th), Image.Resampling.NEAREST)).copy()
    if not valid.any():
        raise ValueError('No trusted source depth remains after calibrated crop/resize')
    return depth, valid


def merge_source(name, fallback, scale=1.):
    import torch
    depth, valid = calibrated_depth(name, fallback.shape[-2:])
    source = torch.as_tensor(depth, device=fallback.device, dtype=fallback.dtype) * scale
    mask = torch.as_tensor(valid, device=fallback.device)
    return torch.where(mask, source, fallback)


def source_or_predict(name, image, cameras, cam_idx, points, model_loader, predictor, fitter):
    import torch
    depth, valid = calibrated_depth(name, image.shape[:2])
    source = torch.as_tensor(depth, device=image.device, dtype=image.dtype)
    if valid.all():
        return source
    fallback = fitter(disp=predictor(model_loader(), image), pts=points,
                      training_cameras=cameras, camera_idx=cam_idx, image=None,
                      pt_colors=None, return_alpha_beta=False, use_rasterizer=False)
    # Do not let invalid fallback depth poison downstream chart optimization.
    used = fallback[~torch.as_tensor(valid, device=image.device)]
    if not torch.isfinite(used).all() or not (used > 0).all():
        raise ValueError('Predicted fallback depth must be finite and positive')
    return torch.where(torch.as_tensor(valid, device=image.device), source, fallback)


def calibrated_rays(name, shape, scale=1.):
    scene = Path(os.environ['SPLATFIX_SOURCE_DEPTH_SCENE'])
    cameras = json.loads((scene / 'cameras.json').read_text())
    index = [Path(p).stem for p in cameras['filepaths']].index(Path(str(name)).stem)
    cal = cameras['image_calibration'][index]
    k = np.asarray(cal['intrinsics'], dtype=float)
    h, w = shape[-2:]
    k[0] *= w / cal['output_wh'][0]
    k[1] *= h / cal['output_wh'][1]
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([(xx-k[0,2])/k[0,0], (yy-k[1,2])/k[1,1], np.ones_like(xx)], -1)
    pose = np.asarray(cameras['cams2world'][index])
    return rays @ pose[:3,:3].T, pose[:3,3] * scale


def calibrated_points(name, depth, scale=1.):
    import torch
    rays, center = calibrated_rays(name, depth.shape, scale)
    rays = torch.as_tensor(rays, device=depth.device, dtype=depth.dtype)
    center = torch.as_tensor(center, device=depth.device, dtype=depth.dtype)
    return rays * depth[..., None] + center


def restore_charts(scene_pm, depths, points, prior, cameras, scale):
    for i, name in enumerate(scene_pm.img_paths):
        depths[i] = merge_source(name, depths[i], scale)
        prior[i] = merge_source(name, prior[i], scale)
        points[i] = calibrated_points(name, depths[i], scale).reshape(points[i].shape)


def _local_file(root, relative):
    """Detach directory symlinks before modifying an overlay file."""
    target = root / relative
    current = root
    for part in Path(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            original = current.resolve()
            current.unlink(); current.mkdir()
            for child in original.iterdir():
                (current / child.name).symlink_to(child, target_is_directory=child.is_dir())
    if target.is_symlink():
        original = target.resolve(); target.unlink(); shutil.copyfile(original, target)
    return target


def patch_source_depth_runtime(overlay):
    root = Path(overlay)
    sources = {}
    for relative, expected in HASHES.items():
        source = (root / relative).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise ValueError('Unexpected upstream source-depth patch target: ' + relative)
        sources[relative] = source
    relative = 'matcha/pointmap/depthanythingv2.py'
    source = sources[relative]
    start = source.index('def get_pointmap_from_sfm_data_with_depthanything(')
    end = source.index('\ndef get_pointmap_from_colmap_scene_with_depthanything(', start)
    body = source[start:end]
    model_start = body.index('    model = load_model(')
    model_end = body.index('    # Get indices', model_start)
    body = body[:model_start] + '''    model = None
    def source_model_loader():
        nonlocal model
        if model is None:
            model = load_model(checkpoint_dir=depthanything_checkpoint_dir,
                               encoder=depthanything_encoder, device=device)
        return model

''' + body[model_end:]
    a = body.index('        inv_depth = apply_depthanything(')
    b = body.index('        points3d_i =', a)
    body = body[:a] + '''        from splatfix_source_depth import source_or_predict
        depth = source_or_predict(image_name, gt_image, pointmap_cameras, cam_idx,
                                  cam_sfm_pts, source_model_loader, apply_depthanything,
                                  fit_depth_to_point_cloud)
''' + body[b:]
    body = body.replace('points3d_i = pointmap_cameras.backproject_depth(cam_idx=cam_idx, depth=depth).view(_height, _width, 3)', 'points3d_i = __import__("splatfix_source_depth").calibrated_points(image_name, depth)')
    sources[relative] = source[:start] + body + source[end:]
    relative = 'matcha/dm_trainers/charts_alignment.py'
    anchor = '    if use_learnable_confidence:\n        with torch.no_grad():'
    assert sources[relative].count(anchor) == 1
    sources[relative] = sources[relative].replace(anchor, '''    from splatfix_source_depth import restore_charts
    restore_charts(scene_pm, output_depths, output_verts, initial_depths,
                   lowres_cameras, scale_factor)

''' + anchor)
    relative = '2d-gaussian-splatting/planes/refine_depth_with_planes.py'
    anchor = '        save_img_f32(depth_list[view_id].cpu().numpy(),'
    assert sources[relative].count(anchor) == 1
    sources[relative] = sources[relative].replace(anchor, '''        if view_id < input_view_num:
            from splatfix_source_depth import merge_source
            depth_list[view_id] = merge_source(train_viewpoints[view_id].image_name,
                                               depth_list[view_id])
''' + anchor)
    records = {}
    for relative, source in sources.items():
        compile(source, relative, 'exec')
        _local_file(root, relative).write_text(source)
        records[relative] = {'original_sha256': HASHES[relative],
                             'patched_sha256': hashlib.sha256(source.encode()).hexdigest()}
    helper = _local_file(root, 'splatfix_source_depth.py')
    shutil.copyfile(__file__, helper)
    (root.parent / 'source-depth-extension.json').write_text(json.dumps({
        'version': 'source-splat-depth-v1', 'patched_files': records,
        'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
        'shared_checkout_modified': False}, indent=2) + '\n')


def apply_source_depth_pointmaps(scene, report_path):
    """Inject trusted metric geometry before the mandatory reprojection gate."""
    scene = Path(scene)
    cameras = json.loads((scene / 'cameras.json').read_text())
    report = {'version': 'source-splat-depth-v1', 'views': []}
    for name in cameras['filepaths']:
        path = scene / 'pointmaps' / (Path(name).stem + '.json')
        body = json.loads(path.read_text())
        shape = np.asarray(body['confs']).shape
        depth, valid = calibrated_depth(name, shape)
        rays, center = calibrated_rays(name, shape)
        pts = np.asarray(body['points'], dtype=float).reshape(*shape, 3)
        pts[valid] = (rays * depth[..., None] + center)[valid]
        body['points'] = pts.reshape(-1, 3).tolist()
        path.write_text(json.dumps(body) + '\n')
        report['views'].append({'image': Path(name).name, 'pointmap_wh': list(shape[::-1]),
                                'trusted_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
                                'trusted_fraction': float(valid.mean())})
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    return report
