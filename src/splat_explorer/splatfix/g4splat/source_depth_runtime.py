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
    '2d-gaussian-splatting/render_chart_views.py': '74ddaa2d1494ca8cc2341e16934178618607b6bb2fcdb160683a8d108bee1aee',
}
VERSION = 'source-splat-depth-v2'


def fit_source_disparity(disparity, depth, valid):
    """Align relative inverse depth to trusted camera-z, never invalid SfM.

    Robust affine fitting follows the upstream inverse-depth convention. If
    extrapolating its offset would cross zero, use a positive scale-only fit.
    Neither option invents a near-plane clamp or copies depth across an edge.
    """
    disp = np.asarray(disparity, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if disp.shape != depth.shape or valid.shape != depth.shape:
        raise ValueError('Depth alignment grids disagree')
    if not np.isfinite(disp).all() or (disp < 0).any():
        raise ValueError('Predicted disparity must be finite and nonnegative')
    anchors = valid & np.isfinite(depth) & (depth > 1e-6) & (disp > 0)
    if anchors.sum() < 32:
        raise ValueError('Insufficient source-depth anchors for prediction')
    x, y = disp[anchors], 1. / depth[anchors]
    keep = np.ones(x.shape, dtype=bool)
    alpha, beta = 0., 0.
    for _ in range(4):
        design = np.column_stack([np.ones(keep.sum()), x[keep]])
        coeff, _, rank, _ = np.linalg.lstsq(design, y[keep], rcond=None)
        if rank < 2:
            break
        alpha, beta = coeff
        residual = y - (alpha + beta*x)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        revised = np.abs(residual - center) <= max(3 * 1.4826 * mad, 1e-8)
        if revised.sum() < 32 or np.array_equal(revised, keep):
            break
        keep = revised
    aligned = alpha + beta * disp
    mode = 'robust-affine-inverse-depth'
    if beta <= 0 or not np.isfinite(aligned).all() or (aligned <= 0).any():
        alpha, beta = 0., float(np.median(y / x))
        aligned = beta * disp
        mode = 'positive-scale-inverse-depth'
    if not np.isfinite(aligned).all() or (aligned <= 0).any():
        raise ValueError('Source-aligned predicted depth must be finite and positive')
    return 1. / aligned, {'fit': mode, 'anchors': int(anchors.sum()),
                          'alpha': float(alpha), 'beta': float(beta)}


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


def source_or_predict(name, image, model_loader, predictor):
    import torch
    depth, valid = calibrated_depth(name, image.shape[:2])
    source = torch.as_tensor(depth, device=image.device, dtype=image.dtype)
    if valid.all():
        return source
    disparity = predictor(model_loader(), image)
    fitted, _ = fit_source_disparity(disparity.detach().cpu().numpy(), depth, valid)
    fallback = torch.as_tensor(fitted, device=image.device, dtype=image.dtype)
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


def source_plane_depths(cameras, depths):
    """Plane extraction consumes source geometry, with chart depth in holes.

    This runs BEFORE normals, visibility, plane point clouds and fitting. The
    fitted plane depths and optimized charts are never overwritten afterward.
    The render-chart stage has already returned depths to original scene units.
    """
    for i, camera in enumerate(cameras):
        depths[i] = merge_source(camera.image_name, depths[i])
    return depths


def source_plane_visibility(cameras, visibility):
    """Require both upstream multi-view support and trusted source geometry."""
    import torch
    for i, camera in enumerate(cameras):
        _, valid = calibrated_depth(camera.image_name, visibility[i].shape[-2:])
        visibility[i] *= torch.as_tensor(valid, device=visibility[i].device)
    return visibility


def reference_mask(name, shape):
    """Completed predictions are initializers, not measured SfM targets."""
    scene = Path(os.environ['SPLATFIX_SOURCE_DEPTH_SCENE'])
    body = json.loads((scene / 'pointmaps' / (Path(name).stem + '.json')).read_text())
    excluded = np.asarray(body['source_depth_completed'], dtype=bool)
    if excluded.shape != tuple(shape):
        raise ValueError('Completed-depth mask disagrees with chart grid')
    return ~excluded


def chart_reference_masks(scene_pm, reference_data, masks):
    import torch
    reliable = np.stack([reference_mask(name, reference_data.shape[-2:])
                         for name in scene_pm.img_paths])
    reliable = torch.as_tensor(reliable, device=reference_data.device)
    return reliable if masks is None else masks * reliable


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
        depth = source_or_predict(image_name, gt_image, source_model_loader,
                                  apply_depthanything)
''' + body[b:]
    body = body.replace('points3d_i = pointmap_cameras.backproject_depth(cam_idx=cam_idx, depth=depth).view(_height, _width, 3)', 'points3d_i = __import__("splatfix_source_depth").calibrated_points(image_name, depth)')
    sources[relative] = source[:start] + body + source[end:]
    relative = 'matcha/dm_trainers/charts_alignment.py'
    anchor = '    pa.optimize(\n'
    assert sources[relative].count(anchor) == 1
    sources[relative] = sources[relative].replace(anchor, '''    from splatfix_source_depth import chart_reference_masks
    masks = chart_reference_masks(scene_pm, reference_data, masks)

''' + anchor)
    relative = '2d-gaussian-splatting/render_chart_views.py'
    anchor = "    charts_depths = charts_priors['depths']"
    assert sources[relative].count(anchor) == 1
    sources[relative] = sources[relative].replace(anchor, anchor + '''
    from splatfix_source_depth import source_plane_depths, source_plane_visibility
    charts_depths = source_plane_depths(train_viewpoints, charts_depths)''')
    anchor = "    charts_mono_normals = charts_priors['normals']"
    assert sources[relative].count(anchor) == 1
    sources[relative] = sources[relative].replace(anchor,
        '    charts_mono_normals = charts_depth_normals')
    anchor = '    vis_points = []'
    assert sources[relative].count(anchor) == 1
    sources[relative] = sources[relative].replace(anchor, '''    visibility_times_masks = source_plane_visibility(train_viewpoints, visibility_times_masks)
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
        'version': VERSION, 'patched_files': records,
        'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
        'shared_checkout_modified': False}, indent=2) + '\n')


def apply_source_depth_pointmaps(scene, report_path, predictor=None):
    """Inject trusted depth and complete invalid fallback before the strict gate.

    Prediction is lazy and used only for nonfinite/behind-camera fallback
    samples. Positive but incorrectly projected points still fail the gate.
    Completed pixels remain predictions, never part of the trusted source mask.
    """
    scene = Path(scene)
    cameras = json.loads((scene / 'cameras.json').read_text())
    report = {'version': VERSION, 'views': []}
    for index, name in enumerate(cameras['filepaths']):
        path = scene / 'pointmaps' / (Path(name).stem + '.json')
        body = json.loads(path.read_text())
        shape = np.asarray(body['confs']).shape
        depth, valid = calibrated_depth(name, shape)
        rays, center = calibrated_rays(name, shape)
        pts = np.asarray(body['points'], dtype=float).reshape(*shape, 3)
        pts[valid] = (rays * depth[..., None] + center)[valid]
        pose = np.asarray(cameras['cams2world'][index])
        local = (pts - pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
        invalid = ~np.isfinite(local).all(axis=-1) | (local[..., 2] <= 1e-6)
        fit = None
        if invalid.any():
            if predictor is None:
                # A short-lived subprocess releases all CUDA allocations before
                # the official chart stage starts (the wrapper stays resident).
                import subprocess
                import sys
                cache = scene / 'source-fallback-disparity'
                cache.mkdir(exist_ok=True)
                target = cache / (Path(name).stem + '.npy')
                subprocess.run([sys.executable, 'splatfix_source_depth.py',
                                str(scene / 'images' / Path(name).name), str(target),
                                str(shape[0]), str(shape[1])], check=True)
                disparity = np.load(target, allow_pickle=False)
            else:
                disparity = predictor(name, shape)
            fallback, fit = fit_source_disparity(disparity, depth, valid)
            pts[invalid] = (rays * fallback[..., None] + center)[invalid]
        previous = np.asarray(body.get('source_depth_completed', np.zeros(shape)), dtype=bool)
        body['source_depth_completed'] = ((previous | invalid) & ~valid).tolist()
        body['points'] = pts.reshape(-1, 3).tolist()
        path.write_text(json.dumps(body) + '\n')
        report['views'].append({'image': Path(name).name, 'pointmap_wh': list(shape[::-1]),
                                'trusted_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
                                'trusted_fraction': float(valid.mean()),
                                'completed_invalid_fallback_pixels': int(invalid.sum()),
                                'fallback_alignment': fit})
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import sys
    import torch
    from matcha.pointmap.depthanythingv2 import load_model, apply_depthanything
    image_path, output, height, width = sys.argv[1:]
    with Image.open(image_path) as opened:
        rgb = np.asarray(opened.convert('RGB').resize((int(width), int(height)),
                                                     Image.Resampling.LANCZOS)).copy()
    image = torch.as_tensor(rgb, device='cuda', dtype=torch.float32) / 255.
    disparity = apply_depthanything(load_model(encoder='vitl', device='cuda'), image)
    np.save(output, disparity.detach().cpu().numpy())
