"""Viser conditioning captures for the ArtiFixer3D+ pass."""
import json
from pathlib import Path
from ..viser_capture import capture_rgb


def capture_plus_rgb(root, trajectory, *, up_axis='-y', should_stop=lambda: False):
    """Replace reconstructed conditioning RGB with same-pose Viser captures."""
    import numpy as np
    from ...rendering.base import Camera
    root = Path(root)
    distilled = json.loads((root / 'distillation.json').read_text())
    tf = trajectory['transforms']
    cameras = [Camera(np.asarray(f['transform_matrix'])[:3, 3],
                      np.asarray(f['transform_matrix'])[:3, :3],
                      width=tf['w'], height=tf['h'],
                      fov_deg=float(np.degrees(2 * np.arctan(tf['w'] / (2 * tf['fl_x'])))))
               for f in tf['frames']]
    capture_rgb(distilled['splat_path'], cameras,
                [Path(distilled['render_dir']) / 'renders' / f'{i:05d}.png' for i in range(len(cameras))],
                up_axis=up_axis, should_stop=should_stop)

