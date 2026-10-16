"""Opt-in source-splat geometry, reusing the episodes front-surface renderer."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from ..checkpoint import camera_from_record, validate_source

VERSION = 'source-splat-depth-v2'


def conservative_mask(depth):
    """Reject holes, their 1px boundary, and >10% local depth discontinuities.

    Episodes already rejects front layers below 0.15 coverage. This binary
    reliability mask is NOT a calibrated probability or renderer alpha map.
    """
    good = np.isfinite(depth) & (depth > 0)
    near = minimum_filter(np.where(good, depth, np.inf), size=3, mode='constant', cval=np.inf)
    far = maximum_filter(np.where(good, depth, 0), size=3, mode='constant', cval=0)
    interior = minimum_filter(good.astype(np.uint8), size=3, mode='constant', cval=0).astype(bool)
    return interior & (far <= near * 1.1)


def export_source_depth(checkpoint, root, *, should_stop=lambda: False, on_progress=None):
    from ...scene import load_scene
    from ...rendering.cpu_splat_renderer import CpuSplatRenderer
    validate_source(checkpoint)
    renderer = CpuSplatRenderer(load_scene(checkpoint.manifest['scene_path']))
    directory = Path(root) / 'source-depth'
    directory.mkdir()
    records = []
    for index, view in enumerate(checkpoint.views):
        if should_stop():
            raise InterruptedError('Cancelled while rendering source-splat depth')
        camera = camera_from_record(view)
        depth = renderer.render_depth(camera)
        valid = conservative_mask(depth)
        if not valid.any():
            raise ValueError('No reliable source-splat depth for view ' + str(view['id']))
        path = directory / ('%06d.npz' % index)
        np.savez_compressed(path, depth=np.where(valid, depth, 0).astype(np.float32),
                            valid=valid, confidence=valid.astype(np.float32),
                            intrinsics=camera.intrinsics, c2w=camera.c2w)
        records.append({'image': '%06d.png' % index, 'path': str(path.relative_to(root)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'valid_fraction': float(valid.mean())})
        if on_progress:
            on_progress(list(records))
    return {'enabled': True, 'version': VERSION, 'depth_convention': 'positive_camera_z_scene_units',
            'renderer': 'episodes CpuSplatRenderer.render_depth',
            'confidence': 'binary reliability; front-layer coverage >=0.15, 1px eroded, local depth spread <=10%',
            'fallback': 'source-aligned prediction at input holes; upstream prediction at novel views',
            'source_fingerprint': checkpoint.manifest.get('source_fingerprint'), 'views': records}
