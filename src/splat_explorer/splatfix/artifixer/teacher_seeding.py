"""Preserve a clean starter latent in the pinned bidirectional UniPC sampler.

The teacher has no autoregressive KV cache. This scoped adapter restores only
the first latent after each scheduler update; all other latents use the original
sampler. Its first RGB must be the trusted reference at its actual camera pose.
This is an experimental conditioning change, not the unmodified teacher recipe.
"""
from contextlib import contextmanager


@contextmanager
def preserve_starting_latent(pipe):
    import torch

    if hasattr(pipe, '_initialize_kv_cache'):
        raise ValueError('Teacher latent preservation requires the bidirectional pipeline')
    prepare, step = pipe.prepare_latents, pipe.scheduler.step
    audit = {'policy': 'clean_reference_latent_restored_after_every_unipc_step',
             'persistent_kv_cache': False, 'scheduler_steps': 0,
             'encoded_starter_preserved': False}
    state = {}

    def prepare_seed(condition, opacity, is_first_chunk):
        if state or not is_first_chunk or not torch.all(opacity[:, 0] == 1):
            raise ValueError('Teacher preservation requires one opaque reference starter per clip')
        seed = condition[:, :, :1].detach().clone()
        latents = prepare(condition, opacity, is_first_chunk)
        if not torch.equal(latents[:, :, :1], seed):
            raise RuntimeError('Teacher initial latent differs from the encoded reference')
        state['seed'] = seed
        return latents

    def step_preserving_seed(*args, **kwargs):
        if 'seed' not in state or kwargs.get('return_dict') is not False:
            raise RuntimeError('Unexpected teacher scheduler contract')
        result = step(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 1:
            raise RuntimeError('Unexpected teacher scheduler output')
        latents = result[0]
        latents[:, :, :1].copy_(state['seed'])
        audit['scheduler_steps'] += 1
        audit['encoded_starter_preserved'] = torch.equal(latents[:, :, :1], state['seed'])
        return result

    # Restore the exact instance state even on OOM or cancellation. No upstream
    # source edits and no scheduler state carried between inference processes.
    prior_prepare = pipe.__dict__.get('prepare_latents')
    prior_step = pipe.scheduler.__dict__.get('step')
    pipe.prepare_latents = prepare_seed
    pipe.scheduler.step = step_preserving_seed
    try:
        yield audit
    finally:
        if prior_prepare is None:
            del pipe.prepare_latents
        else:
            pipe.prepare_latents = prior_prepare
        if prior_step is None:
            del pipe.scheduler.step
        else:
            pipe.scheduler.step = prior_step
        state.clear()
