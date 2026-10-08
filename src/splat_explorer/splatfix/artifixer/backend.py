"""Public adapter for selecting ArtiFixer in the shared Splatfix workflow."""
import math


def validate_options(raw):
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
    if raw['stage'] == 'repair':
        frames = raw.get('frames', 25)
        if type(frames) is not int or frames < 9 or frames > 101 or (frames - 1) % 4:
            raise ValueError('frames must be 1 + 4*n, between 9 and 101')
        span = float(raw.get('span_fraction', .04))
        if not math.isfinite(span) or not 0 < span <= .15:
            raise ValueError('span_fraction must be in (0, .15]')
        insertion = raw.get('image_cache_insertion', False)
        if type(insertion) is not bool:
            raise ValueError('image_cache_insertion must be a boolean')
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
