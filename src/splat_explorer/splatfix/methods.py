"""Reconstruction method registry, independent of any model runtime.

A backend implements validate_options(raw), readiness(checkpoint),
execute_remote(executor, run_id, options, root, stop, update),
execute_worker(root), and run_repair(checkpoint_dir, output_dir, **kwargs).
It owns staging, runtime configuration, caches, evaluation and cancellation.
The shared queue supplies scheduling, a GPU lease, status and stop callbacks.
"""
from importlib import import_module

DEFAULT_METHOD = 'artifixer'
# Explicit allowlist: request values must never become arbitrary import paths.
METHODS = {
    'artifixer': {
        'module': 'splat_explorer.splatfix.artifixer.backend',
        'label': 'ArtiFixer',
        'stages': ['repair', 'benchmark'],
        'description': 'Generate a camera trajectory and new views, then reconstruct a fresh splat.',
    },
}


def get_method(method=None, *, stage=None):
    method = DEFAULT_METHOD if method is None else method
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f'Unknown reconstruction method: {method!r}')
    if stage is not None and stage not in METHODS[method].get('stages', ['repair']):
        raise ValueError(f'Reconstruction method {method!r} does not support stage {stage!r}')
    return import_module(METHODS[method]['module'])


def available_methods():
    return [{'id': key, **{k: v for k, v in info.items() if k != 'module'}}
            for key, info in METHODS.items()]
