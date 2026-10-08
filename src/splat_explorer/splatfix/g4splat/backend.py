"""Independent G4Splat method contract; no ArtiFixer runtime imports."""
import numpy as np
from ..checkpoint import camera_from_record


def validate_options(raw):
    return {}  # The pinned authors' recipe has no dashboard tuning overrides.


def readiness(checkpoint):
    cameras = [camera_from_record(v) for v in checkpoint.views]
    ready = checkpoint.complete and len(cameras) >= 2 and any(
        np.linalg.norm(c.position - cameras[0].position) > 1e-6 for c in cameras[1:])
    return {'ready': ready, 'reason': 'Calibrated cameras ready for G4Splat' if ready else
            'G4Splat needs a complete selection with at least two different camera positions'}


def execute_remote(*args, **kwargs):
    from .execution import execute_remote as run
    return run(*args, **kwargs)


def execute_worker(root):
    from .worker import execute
    return execute(root)


def run_repair(*args, **kwargs):
    from .repair import run_repair as run
    return run(*args, **kwargs)
