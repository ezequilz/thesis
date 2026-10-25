"""Two-boundary RGB seeding and optional immutable latent/KV conditioning.

Wan uses one leading RGB frame followed by groups of four. Seeding the four
overlaps therefore touches shared latent groups, determined from the real window.
The standard RGB/opacity encoding is reused. No reverse-leg latents or caches
are reused; the optional preservation adapter wraps the normal sampler.
"""
from contextlib import contextmanager


MODES = ('off', 'seed', 'seed_preserve')


def seed_layout(real_frame_count, overlap_slots):
    """Map real seeded RGB slots into Wan's first-singleton / groups-of-four layout.

    Padding is model context, never an extra real seed or diagnostic. A latent
    group containing a real overlap may also contain rendered middle frames.
    """
    if (type(real_frame_count) is not int or not 3 <= real_frame_count <= 25
            or not overlap_slots or any(type(i) is not int or not 0 <= i < real_frame_count for i in overlap_slots)):
        raise ValueError('Bridge seed slots must belong to a real window of 3–25 RGB frames')
    def latent_for(i):
        return (i + 3) // 4
    groups = sorted({latent_for(i) for i in overlap_slots})
    constrained = [i for i in range(real_frame_count) if latent_for(i) in groups]
    return {'seed_latent_indices': groups, 'seed_rgb_local_indices': sorted(set(overlap_slots)),
            'constrained_rgb_local_indices': constrained,
            'all_real_latent_groups_preserved': len(constrained) == real_frame_count}


def mode(options, *, default='off'):
    value = options.get('connected_bridge_seeding', default)
    if value not in MODES:
        raise ValueError('connected_bridge_seeding must be off, seed or seed_preserve')
    return value


class PreservedBoundaryAttention:
    """Single-block adapter using the pinned model's QKV/RoPE/attention helpers.

    A clean t=0 pass captures only boundary K/V. During capture every query sees
    only boundary keys, so unknown middle tokens cannot contaminate deeper seed
    states. Denoising then attends the complete block with those slots immutable.
    """
    def __init__(self, original, helpers, latent_indices):
        if getattr(original, '_cp_mesh', None) is not None:
            raise ValueError('Bridge seed preservation requires single-rank attention')
        self.original, self.helpers = original, helpers
        self.latent_indices = latent_indices
        self.capture = True
        self.frozen_k = self.frozen_v = self.indices = None
        self.verified_calls = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask,
                 rotary_emb, kv_cache, current_start, frame_seqlen):
        import torch
        if torch.is_grad_enabled() or current_start != 0 or hidden_states.shape[1] != 7 * frame_seqlen:
            raise ValueError('Preserved bridge attention requires one complete seven-frame latent block')
        if attention_mask is not None or encoder_hidden_states is not None:
            raise ValueError('Unexpected mask or encoder input for preserved bridge self-attention')
        query, key, value = self.helpers._get_qkv_projections(attn, hidden_states, None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = self.helpers._apply_rotary_emb(query, *rotary_emb)
            key = self.helpers._apply_rotary_emb(key, *rotary_emb)
        if self.capture:
            if self.frozen_k is not None:
                raise RuntimeError('Bridge boundary keys were captured more than once')
            self.indices = torch.cat([torch.arange(i * frame_seqlen, (i + 1) * frame_seqlen,
                                                  device=key.device) for i in self.latent_indices])
            self.frozen_k = key.index_select(1, self.indices).clone()
            self.frozen_v = value.index_select(1, self.indices).clone()
        elif self.frozen_k is None:
            raise RuntimeError('Clean bridge boundary keys were not captured')
        else:
            key.index_copy_(1, self.indices, self.frozen_k)
            value.index_copy_(1, self.indices, self.frozen_v)
        end = key.shape[1]
        kv_cache['k'][:, :end] = key
        kv_cache['v'][:, :end] = value
        kv_cache['global_end_index'].fill_(end)
        kv_cache['local_end_index'].fill_(end)
        cached_k, cached_v = kv_cache['k'][:, :end], kv_cache['v'][:, :end]
        if self.capture:
            cached_k, cached_v = self.frozen_k, self.frozen_v
        else:
            if (not torch.equal(cached_k.index_select(1, self.indices), self.frozen_k)
                    or not torch.equal(cached_v.index_select(1, self.indices), self.frozen_v)):
                raise RuntimeError('A preserved bridge key/value changed')
            self.verified_calls += 1
        output = self.helpers.dispatch_attention_fn(query, cached_k, cached_v, attn_mask=None,
            dropout_p=0.0, is_causal=False, backend=self.original._attention_backend)
        output = output.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


@contextmanager
def preserved_attention(pipe, latent_indices):
    from model_training.net import transformer as helpers
    installed = []
    try:
        for block in pipe.transformer.blocks:
            original = block.attn1.processor
            if not isinstance(original, helpers.KvCacheWanSelfAttnProcessor):
                raise ValueError('Bridge preservation requires the pinned KV-cache self-attention processor')
            adapter = PreservedBoundaryAttention(original, helpers, latent_indices)
            block.attn1.set_processor(adapter)
            installed.append((block.attn1, original, adapter))
        if not installed:
            raise ValueError('Bridge preservation requires transformer attention blocks')
        yield [adapter for _, _, adapter in installed]
    finally:
        for attention, original, _ in reversed(installed):
            attention.set_processor(original)


@contextmanager
def temporary_method(obj, name, replacement):
    """Restore the exact instance/class method binding, including on exceptions."""
    owned = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def preserve_samples(pipe, normal_sampler, audit, condition, rendered_opacity,
                     neighbors_condition, camera_rays, w2cs, neighbor_w2cs, Ks,
                     neighbor_Ks, encoded_prompt, num_inference_steps, use_exit_flag,
                     *, ignore_neighbors=False, show_progress=False, progress_bar_leave=True):
    """Project seed groups around the unchanged upstream four-step joint sampler."""
    import torch
    if torch.is_grad_enabled() or use_exit_flag or ignore_neighbors or neighbors_condition is None:
        raise ValueError('Bridge preservation requires no-grad inference with reference conditioning')
    if getattr(pipe.transformer, '_cp_world_size', 1) != 1:
        raise ValueError('Bridge preservation currently supports a single GPU')
    if (condition.shape[2] != 7 or pipe.transformer.patch_size[0] != 1
            or pipe.frames_per_block != 7 or pipe.vae.config.scale_factor_temporal != 4
            or rendered_opacity.shape[1] != 25 or num_inference_steps != 4
            or (pipe.local_attn_size != -1 and pipe.local_attn_size < 7)):
        raise ValueError('Bridge preservation requires one full 25-RGB / seven-latent block and four steps')
    indices = torch.tensor(audit['seed_latent_indices'], device=pipe.vae.device)
    clean = condition.to(pipe.vae.device)
    seed = clean.index_select(2, indices).clone()
    forward, step, add_noise = pipe.transformer.forward, pipe.scheduler.step, pipe.scheduler.add_noise
    calls = 0

    def project(latents):
        latents.index_copy_(2, indices, seed.to(latents))
        return latents

    with preserved_attention(pipe, audit['seed_latent_indices']) as adapters:
        def anchored_forward(*, hidden_states, timestep, **kwargs):
            nonlocal calls
            if calls == 0:
                # Upstream has now initialized fresh caches. Capture clean keys
                # at their original temporal positions with only seeds visible.
                forward(hidden_states=clean, timestep=torch.zeros_like(timestep), **kwargs)
                for adapter in adapters:
                    if adapter.frozen_k is None:
                        raise RuntimeError('A transformer layer did not capture bridge boundary keys')
                    adapter.capture = False
            calls += 1
            return forward(hidden_states=project(hidden_states), timestep=timestep, **kwargs)

        def anchored_step(*args, **kwargs):
            return project(step(*args, **kwargs))

        def anchored_noise(*args, **kwargs):
            return project(add_noise(*args, **kwargs))

        with temporary_method(pipe.transformer, 'forward', anchored_forward), \
                temporary_method(pipe.scheduler, 'step', anchored_step), \
                temporary_method(pipe.scheduler, 'add_noise', anchored_noise):
            output = normal_sampler(condition, rendered_opacity, neighbors_condition,
                camera_rays, w2cs, neighbor_w2cs, Ks, neighbor_Ks, encoded_prompt,
                num_inference_steps, use_exit_flag, ignore_neighbors=ignore_neighbors,
                show_progress=show_progress, progress_bar_leave=progress_bar_leave)
        if (not torch.equal(output.index_select(2, indices), seed.to(output))
                or calls != 4 or any(a.verified_calls != calls for a in adapters)):
            raise RuntimeError('Bridge latent/KV preservation was not verified for every step')
        audit.update(seed_latents_preserved=True, seed_kv_preserved=True,
            verified_attention_layers=len(adapters), verified_denoising_steps=calls, seed_cache_timestep=0)
    return output


@contextmanager
def preserved_bridge(pipe, normal_sampler, *, real_frame_count, overlap_slots):
    """Adapt only the stronger manual mode; ordinary seeding uses the original sampler."""
    audit = {**seed_layout(real_frame_count, overlap_slots),
             'seed_latents_preserved': False, 'seed_kv_preserved': False, 'sampler_called': False}

    def sample(*args, **kwargs):
        audit['sampler_called'] = True
        return preserve_samples(pipe, normal_sampler, audit, *args, **kwargs)

    with temporary_method(pipe, 'generate_samples_from_batch', sample):
        yield audit
        if not audit['sampler_called']:
            raise RuntimeError('Bridge preservation was requested but the sampler was not called')
