"""Optional single-view GSFix3D initialization of an isolated scene candidate."""
from __future__ import annotations


def prefit_anchor(scene, camera, rendered, edited, *, should_stop, on_progress):
    # Reuse the scene-run baseline's actual 20-step ADC recipe, not the
    # densification-free extended fitter or a similarly named legacy variant.
    from ..scene_runs.gpu_worker import _default_repair_factory, _jsonable
    if should_stop():
        raise InterruptedError('Stopped before anchor prefit')
    backend = _default_repair_factory({'repair_type': 'original'})
    backend.on_progress = lambda value: on_progress({**value, 'phase': 'anchor_prefit'})
    stats = backend.apply_until(scene, camera, rendered, edited, should_stop=should_stop)
    if should_stop():
        raise InterruptedError('Stopped during anchor prefit; candidate discarded')
    return _jsonable({k: v for k, v in stats.items() if k != 'render_rgb'})
