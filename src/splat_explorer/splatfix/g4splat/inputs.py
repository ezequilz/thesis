"""Export calibrated OpenCV cameras to the official PINHOLE COLMAP reader."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from ..checkpoint import atomic_json, camera_from_record
from .backend import readiness


def export_inputs(checkpoint, destination, mode):
    if mode not in ('baseline', 'edited'):
        raise ValueError('mode must be baseline or edited')
    state = readiness(checkpoint)
    if not state['ready']:
        raise ValueError(state['reason'])
    if mode == 'edited':
        checkpoint.require_viser_images()
    root = Path(destination)
    (root / 'images').mkdir(parents=True, exist_ok=False)
    sparse = root / 'sparse' / '0'
    sparse.mkdir(parents=True)
    cameras, images, records = [], [], []
    for index, view in enumerate(checkpoint.views, 1):
        camera = camera_from_record(view)
        source = checkpoint.image_path(view, repaired=mode == 'edited')
        name = f'{index - 1:06d}.png'
        target = root / 'images' / name
        with Image.open(source) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError('Input RGB dimensions disagree with calibrated camera')
            image.convert('RGB').save(target)
        k = camera.intrinsics
        cameras.append(f'{index} PINHOLE {camera.width} {camera.height} ' +
                       ' '.join(format(float(x), '.17g') for x in (k[0,0], k[1,1], k[0,2], k[1,2])))
        w2c = np.linalg.inv(camera.c2w.astype(np.float64))
        xyzw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        q = xyzw[[3, 0, 1, 2]]
        images.append(f'{index} ' + ' '.join(format(float(x), '.17g') for x in [*q, *w2c[:3,3]]) +
                      f' {index} {name}\n')  # Empty second line: no fabricated observations.
        records.append({'view_id': view['id'], 'image': 'images/' + name,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'source_image': str(source), 'camera': view['camera']})
    (sparse / 'cameras.txt').write_text('\n'.join(cameras) + '\n')
    (sparse / 'images.txt').write_text('\n'.join(images) + '\n')
    (sparse / 'points3D.txt').write_text('# Empty: geometry is reconstructed by MASt3R.\n')
    atomic_json(root / f'split-{len(records)}views.json', {'train': list(range(len(records))), 'test': []})
    manifest = {'method': 'g4splat', 'mode': mode, 'camera_convention': 'opencv_c2w',
                'colmap_convention': 'opencv_w2c', 'views': records,
                'source_checkpoint': str(checkpoint.root), 'source_splat_used_for_initialization': False,
                'resolution': checkpoint.manifest.get('metadata', {}).get('resolution')}
    atomic_json(root / 'inputs.json', manifest)
    return manifest
