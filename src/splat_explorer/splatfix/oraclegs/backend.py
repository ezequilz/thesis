"""Independent OracleGS method contract; model imports stay on CUDA workers."""
import numpy as np
from ..checkpoint import camera_from_record


def validate_options(raw):
    iterations_testing = raw.get('oracle_iterations_testing', False)
    if type(iterations_testing) is not bool:
        raise ValueError('oracle_iterations_testing must be a boolean')
    assignment = raw.get('oracle_artifixer_reference_assignment')
    if assignment is not None and assignment not in ('trajectory', 'camera_proximity'):
        raise ValueError('oracle_artifixer_reference_assignment must be trajectory or camera_proximity')
    artifixer_input = raw.get('oracle_artifixer_input')
    if artifixer_input is not None and artifixer_input not in ('new_only', 'original_and_new'):
        raise ValueError('oracle_artifixer_input must be new_only or original_and_new')
    # Preserve historical jobs without this field; new jobs explicitly select SEVA.
    capture_source = raw.get('oracle_capture_source', 'saved')
    if capture_source not in ('saved', 'seva'):
        raise ValueError('oracle_capture_source must be saved or seva')
    reference_mode = raw.get('oracle_reference_mode', 'all')
    if reference_mode not in ('scene_memory', 'all', 'single'):
        raise ValueError('oracle_reference_mode must be scene_memory, all or single')
    source_mode = raw.get('oracle_source_reference', 'nearest')
    if source_mode not in ('nearest', 'fixed_initial'):
        raise ValueError('oracle_source_reference must be nearest or fixed_initial')
    split_mode = raw.get('oracle_split_mode', 'single-split')
    if split_mode not in ('single-split', 'double-split'):
        raise ValueError('oracle_split_mode must be single-split or double-split')
    if reference_mode == 'single':
        split_mode = 'double-split'
    guidance = raw.get('seva_geometry_guidance', 'off')
    if guidance not in ('off', 'source_splat'):
        raise ValueError('seva_geometry_guidance must be off or source_splat')
    depth = raw.get('oracle_depth_source', 'predicted')
    initialization = raw.get('oracle_initialization', 'sfm')
    if depth not in ('predicted', 'source_splat'):
        raise ValueError('oracle_depth_source must be predicted or source_splat')
    if initialization not in ('sfm', 'source_splat', 'point_cloud'):
        raise ValueError('oracle_initialization must be sfm, source_splat, or point_cloud')
    batching = raw.get('oracle_attention_batching', 'within_trajectory')
    if batching not in ('within_trajectory', 'global'):
        raise ValueError('oracle_attention_batching must be within_trajectory or global')
    world_attention = raw.get('oracle_worldexplorer_attention', 'weighted')
    if world_attention not in ('weighted', 'unweighted'):
        raise ValueError('oracle_worldexplorer_attention must be weighted or unweighted')
    fit = raw.get('seva_fit', 'crop')
    if fit not in ('crop', 'pad'):
        raise ValueError('seva_fit must be crop or pad')
    adapter = raw.get('oracle_adapter', 'trajectory')
    if adapter not in ('trajectory', 'saved_poses', 'per_view_preset'):
        raise ValueError('oracle_adapter must be trajectory, saved_poses or per_view_preset')
    if adapter == 'saved_poses' and reference_mode == 'single':
        raise ValueError('Single-image references require the trajectory or per_view_preset adapter')
    preset = raw.get('oracle_preset', 'spiral_expanding')
    if preset not in ('orbit', 'spiral', 'spiral_expanding', 'lemniscate', 'shuffle_orbit'):
        raise ValueError('oracle_preset must be orbit, spiral, spiral_expanding, lemniscate or shuffle_orbit')
    scale = raw.get('oracle_preset_scale', 0.25)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('oracle_preset_scale must be a finite positive number in scene units')
    resolution = raw.get('seva_resolution', '576p')
    if resolution not in ('576p', '720p'):
        raise ValueError('seva_resolution must be 576p or 720p')
    frames = raw.get('oracle_target_frames', 240)
    if type(frames) is not int or not 2 <= frames <= 1000:
        raise ValueError('oracle_target_frames must be an integer from 2 to 1000')
    order = raw.get('oracle_path_order')
    if order is not None and (not isinstance(order, list) or len(order) < 2 or
                              any(type(i) is not int or i < 0 for i in order) or len(set(order)) != len(order)):
        raise ValueError('oracle_path_order must contain distinct nonnegative checkpoint view indices')
    return {'oracle_iterations_testing':iterations_testing,
            **({'oracle_artifixer_reference_assignment':assignment} if assignment is not None else {}),
            **({'oracle_artifixer_input':artifixer_input} if artifixer_input is not None else {}),
            'oracle_source_reference':source_mode, 'oracle_capture_source':capture_source, 'oracle_reference_mode':reference_mode, 'oracle_split_mode':split_mode, 'oracle_preset':preset, 'oracle_preset_scale':float(scale), 'seva_geometry_guidance':guidance, 'oracle_depth_source':depth, 'oracle_initialization':initialization, 'oracle_attention_batching':batching, 'oracle_worldexplorer_attention':world_attention, 'seva_fit': fit, 'oracle_adapter': adapter, 'seva_resolution': resolution, 'oracle_target_frames': frames, 'oracle_path_order': order}


def readiness(checkpoint):
    cameras = [camera_from_record(v) for v in checkpoint.views]
    ready = checkpoint.complete and 2 <= len(cameras) <= 8 and any(
        np.linalg.norm(c.position-cameras[0].position) > 1e-6 for c in cameras[1:])
    reason = 'OracleGS needs 2–8 complete, calibrated reference views at distinct positions'
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
