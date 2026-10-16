"""Independent OracleGS method contract; model imports stay on CUDA workers."""
import numpy as np
from ..checkpoint import camera_from_record


def validate_options(raw):
    adapter = raw.get('oracle_adapter', 'trajectory')
    if adapter not in ('trajectory', 'saved_poses'):
        raise ValueError('oracle_adapter must be trajectory or saved_poses')
    resolution = raw.get('seva_resolution', '576p')
    if resolution not in ('576p', '720p'):
        raise ValueError('seva_resolution must be 576p or 720p')
    frames = raw.get('oracle_target_frames', 80)
    if type(frames) is not int or not 2 <= frames <= 1000:
        raise ValueError('oracle_target_frames must be an integer from 2 to 1000')
    order = raw.get('oracle_path_order')
    if order is not None and (not isinstance(order, list) or len(order) < 2 or
                              any(type(i) is not int or i < 0 for i in order) or len(set(order)) != len(order)):
        raise ValueError('oracle_path_order must contain distinct nonnegative checkpoint view indices')
    return {'oracle_adapter': adapter, 'seva_resolution': resolution, 'oracle_target_frames': frames, 'oracle_path_order': order}


def readiness(checkpoint):
    cameras = [camera_from_record(v) for v in checkpoint.views]
    ready = checkpoint.complete and 2 <= len(cameras) <= 8 and any(
        np.linalg.norm(c.position-cameras[0].position) > 1e-6 for c in cameras[1:])
    reason = 'OracleGS needs 2–8 complete, calibrated reference views at distinct positions'
    if ready and any(abs(c.width/c.height-16/9) > 1/max(c.width,c.height) for c in cameras):
        ready, reason = False, 'OracleGS 576p/720p presets require 16:9 checkpoint cameras; select 16:9 views to preserve framing'
    return {'ready': ready, 'reason': 'Calibrated references ready for OracleGS' if ready else reason}


def execute_remote(*args, **kwargs):
    from .execution import execute_remote
    return execute_remote(*args, **kwargs)


def execute_worker(root):
    from .worker import execute
    return execute(root)


def run_repair(*args, **kwargs):
    from .repair import run_repair
    return run_repair(*args, **kwargs)
