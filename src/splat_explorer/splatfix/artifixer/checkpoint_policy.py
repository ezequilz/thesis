"""Run-local interpretation of saved checkpoints; never rewrite the source."""
from copy import deepcopy

from ..checkpoint import Checkpoint


def validate_image_trajectory(value):
    if value not in ('checkpoint', 'endpoints'):
        raise ValueError('image_trajectory must be checkpoint or endpoints')
    return value


def checkpoint_for_run(checkpoint, image_trajectory='checkpoint'):
    validate_image_trajectory(image_trajectory)
    if image_trajectory != 'endpoints' and not checkpoint.endpoint_only:
        return checkpoint
    if not checkpoint.complete or len(checkpoint.views) < 2:
        raise ValueError('Endpoint image trajectory requires a completed checkpoint with at least two views')
    manifest = deepcopy(checkpoint.manifest)
    manifest.setdefault('metadata', {})['repair_policy'] = 'endpoints'
    # Intermediate edits remain on disk in the source checkpoint but are not
    # copied into the run-local prepared checkpoint or exposed as supervision.
    for view in manifest['views'][1:-1]:
        view.pop('repaired_rgb', None)
        view.pop('image_repair', None)
    return Checkpoint(checkpoint.root, manifest)
