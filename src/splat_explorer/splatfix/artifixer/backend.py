"""Public adapter for selecting ArtiFixer in the shared Splatfix workflow."""
import math


def validate_options(raw):
    from .pipeline_profile import apply_pipeline
    if raw['stage'] == 'repair':
        from .official_worker import validate_regularization, validate_reconstruction_options
        validate_regularization(raw.get('regularization_profile', 'artifixer'))
        validate_reconstruction_options(raw)
        raw = apply_pipeline(raw)
    from .official_worker import validate_regularization
    from .author_trajectory import validate_split_mode
    result = {
        'split_mode': validate_split_mode(raw.get('split_mode', 'double-split')),
        'regularization_profile': validate_regularization(raw.get('regularization_profile', 'artifixer')),
    }
    if raw['stage'] == 'benchmark':
        model = raw.get('model', '1.3b')
        if model not in ('1.3b', '14b'):
            raise ValueError('model must be 1.3b or 14b')
        result['model'] = model
        runs = raw.get('benchmark_runs', 1)
        if type(runs) is not int or runs < 1:
            raise ValueError('benchmark_runs must be a positive integer')
        result['benchmark_runs'] = runs
    if raw['stage'] == 'repair':
        from .handoff import enabled, context_mode, backward_enabled
        result['handoff_generation'] = enabled(raw)
        result['handoff_backward_pass'] = backward_enabled({'handoff_backward_pass': True, **raw})
        result['gpt_image_context'] = context_mode({'gpt_image_context': 'closest_repaired_image', **raw})
        from .handoff_noise import noise_options
        default_noise_mode = ('repaired_pixel_sprinkle' if result['handoff_generation']
                              and result['gpt_image_context'] != 'single_last_frame' else 'gaussian')
        result.update(noise_options({'handoff_add_noise': True, 'handoff_noise_vignette': True,
                                     'handoff_noise_mode': default_noise_mode,
                                     'handoff_vignette_size': 'very_large', 'handoff_blur_radius': 0.25,
                                     'handoff_noise_strength': 35, **raw}))
        if result['handoff_generation']:
            if raw.get('mode', 'edited') != 'edited' or raw.get('image_trajectory') == 'endpoints':
                raise ValueError('Handoff generation requires edited checkpoint trajectories')
            result['split_mode'] = 'single-split'
        result['artifixer_pipeline'] = raw['artifixer_pipeline']
        from .checkpoint_policy import validate_image_trajectory
        result['image_trajectory'] = validate_image_trajectory(raw.get('image_trajectory', 'checkpoint'))
        if result['image_trajectory'] == 'endpoints':
            result['split_mode'] = 'no-split'
        reference_mode = raw.get('endpoint_reference_mode', 'both')
        if reference_mode not in ('both', 'initial'):
            raise ValueError('endpoint_reference_mode must be both or initial')
        result['endpoint_reference_mode'] = reference_mode
        frames = raw.get('frames', 25)
        if type(frames) is not int or frames < 9 or frames > 101 or (frames - 1) % 4:
            raise ValueError('frames must be 1 + 4*n, between 9 and 101')
        span = float(raw.get('span_fraction', .04))
        if not math.isfinite(span) or not 0 < span <= .15:
            raise ValueError('span_fraction must be in (0, .15]')
        insertion = raw.get('image_cache_insertion', False)
        if type(insertion) is not bool:
            raise ValueError('image_cache_insertion must be a boolean')
        seeding = raw.get('initial_image_seeding', True)
        if type(seeding) is not bool:
            raise ValueError('initial_image_seeding must be a boolean')
        result['initial_image_seeding'] = seeding or insertion
        from .official_worker import validate_reconstruction_options
        result.update(validate_reconstruction_options(raw))
        result.update(frames=frames, span_fraction=span, image_cache_insertion=insertion)
    return result


def readiness(checkpoint):
    from ..checkpoint import camera_from_record
    distinct = len({tuple(camera_from_record(view).c2w.ravel()) for view in checkpoint.views})
    ready = checkpoint.complete and distinct >= 2 and distinct == len(checkpoint.views)
    reason = ('Finish selecting all views before reconstruction' if not checkpoint.complete
              else 'Authors smooth orbit requires at least two distinct saved camera poses' if distinct < 2
              else 'Saved anchor poses must be distinct for authors smooth orbit' if distinct != len(checkpoint.views)
              else 'Saved cameras are ready for authors smooth orbit')
    return {'ready': ready, 'reason': reason}


def execute_remote(executor, run_id, options, root, stop, update):
    from .execution import execute_remote as execute
    return execute(executor, run_id, options, root, stop, update)


def execute_worker(root):
    from .worker import execute
    return execute(root)


def run_repair(checkpoint_dir, output_dir, **kwargs):
    from .repair import run_repair as run
    return run(checkpoint_dir, output_dir, **kwargs)
