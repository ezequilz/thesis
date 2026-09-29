"""GPT-image-seeded causal inference for the pinned ArtiFixer KV pipeline.

The clean first latent is temporal context, not a denoised prediction. Later
frames use calibrated cameras, edited references, and optional scene renders.
This is an inference adaptation; it does not guarantee multiview consistency.
"""
from __future__ import annotations
from types import MethodType


def starter_reference(manifest, segment):
    matches = [r for r in manifest['references'] if r['frame_index'] == segment['start']]
    if len(matches) != 1 or matches[0].get('kind') != 'edited_render':
        raise ValueError('Each starter-image trajectory requires one calibrated GPT-edited reference')
    return matches[0]


def latent_chunks(total, block, refresh_every=0):
    if total < 1 or block < 1:
        raise ValueError('Latent frame count and block size must be positive')
    yield 0, 1
    for start in range(1, total, block):
        end = min(start + block, total)
        while start < end:
            boundary = (1 + ((start - 1) // refresh_every + 1) * refresh_every
                        if refresh_every else end)
            stop = min(end, boundary)
            yield start, stop
            start = stop


def starter_inputs(seed, count):
    """No scene render argument: only the edited first RGB frame is observed."""
    import torch
    if seed.ndim != 3 or seed.shape[0] != 3 or count < 1:
        raise ValueError('Expected a CHW RGB starter and a positive frame count')
    rgb = seed.new_zeros((count, *seed.shape))
    rgb[0] = seed
    opacity = seed.new_zeros((count, *seed.shape[-2:]))
    opacity[0] = 1
    return rgb, opacity


def generate_from_starter(self, condition, rendered_opacity, neighbors_condition,
                          camera_rays, w2cs, neighbor_w2cs, Ks, neighbor_Ks,
                          encoded_prompt, num_inference_steps, use_exit_flag, *,
                          ignore_neighbors=False, show_progress=False,
                          progress_bar_leave=True):
    """Inference-only replacement for generate_samples_from_batch.

    Uses upstream encoding, scheduler, transformer and decoding. A dedicated
    t=0 forward pass inserts the starter into KV memory before any new frame is
    sampled. Generated-block cache refresh is an explicit ablation: upstream
    retains the last denoising input; the original adaptation refreshes at t=0.
    """
    import torch
    if torch.is_grad_enabled() or use_exit_flag or ignore_neighbors:
        raise ValueError('Starter inference requires no-grad inference with edited references')
    if getattr(self.transformer, '_cp_world_size', 1) != 1:
        raise ValueError('Starter inference currently supports a single GPU only')
    batch, _, total, height, width = condition.shape
    pt, ph, pw = self.transformer.patch_size
    if pt != 1 or neighbors_condition is None:
        raise ValueError('Starter inference requires temporal patch size 1 and edited references')
    device = self.vae.device
    temporal = self.vae.config.scale_factor_temporal
    tokens = (height // ph) * (width // pw)
    # Long single-starter paths retain frame zero as an attention sink while
    # rolling the remaining context, as supported by the upstream cache.
    window = self.local_attn_size
    if window != -1 and (window < self.frames_per_block + 1 or self.sink_size != 1):
        raise ValueError('Starter rolling cache requires sink_size=1 and space for a generated block')
    self._initialize_kv_cache(batch, tokens, total if window == -1 else min(total, window))
    self._initialize_crossattn_cache('crossattn_cache')
    self._initialize_crossattn_cache('neighbor_crossattn_cache')
    output = torch.zeros_like(condition, device=device)
    neighbor_w2cs = neighbor_w2cs.to(device)
    neighbor_Ks = neighbor_Ks.to(device)
    timesteps = self.create_denoising_step_list(num_inference_steps)
    refresh = getattr(self, 'starter_refresh', None)
    interval = 20 // temporal if refresh else 0
    if refresh and 20 % temporal:
        raise ValueError('Periodic refresh requires a VAE temporal scale dividing 20')
    for start, end in latent_chunks(total, self.frames_per_block, interval):
        rgb_start = 0 if start == 0 else 1 + (start - 1) * temporal
        rgb_end = 1 + (end - 1) * temporal
        opacity = rendered_opacity[:, rgb_start:rgb_end].to(device)
        kwargs = dict(
            encoder_hidden_states=encoded_prompt,
            neighbor_hidden_states=neighbors_condition, ignore_neighbors=False,
            opacity=opacity, camera_rays=camera_rays[:, start:end].to(device),
            w2cs=w2cs[:, start:end].to(device), neighbor_w2cs=neighbor_w2cs,
            Ks=Ks[:, start:end].to(device), neighbor_Ks=neighbor_Ks,
            kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
            neighbor_crossattn_cache=self.neighbor_crossattn_cache,
            current_start=start * tokens, frame_offset=start, return_dict=False)
        if start == 0:
            # Wan's causal VAE encodes the first RGB frame as one latent frame.
            latents = condition[:, :, :1].to(device).clone()
        else:
            chunk_condition = condition[:, :, start:end].to(device)
            # Match the training source distribution at every denoising step.
            # Zero opacity reduces this to independent Gaussian noise.
            latents = self.prepare_latents(chunk_condition, opacity, False)
            for index, timestep in enumerate(timesteps):
                noise = self.transformer(hidden_states=latents,
                    timestep=timestep.expand(batch).to(latents.dtype), **kwargs)[0]
                latents = self.scheduler.step(noise, timestep.expand(batch), latents, to_final=True)
                if index + 1 < len(timesteps):
                    latents = self.scheduler.add_noise(latents,
                        self.prepare_latents(chunk_condition, opacity, False),
                        timesteps[index + 1] * torch.ones(batch, device=device, dtype=torch.long))
        output[:, :, start:end] = latents
        refreshed = (refresh is not None and end > 1 and (end - 1) % interval == 0
                     and rgb_end - 1 < self.starter_rgb_count)
        if refreshed:
            # Generate normally first. Re-encode the repaired RGB in its causal
            # video context; a standalone-image latent has different semantics.
            replacement = refresh(self, output[:, :, :end], rgb_end - 1)
            if replacement.shape != output[:, :, end-1:end].shape:
                raise ValueError('Periodic repair returned an incompatible latent')
            output[:, :, end-1:end] = replacement
            latents = output[:, :, start:end]
        if start == 0 or refreshed or getattr(self, 'starter_generated_cache', 'clean') == 'clean':
            self.transformer(hidden_states=latents,
                             timestep=torch.zeros(batch, device=device, dtype=latents.dtype), **kwargs)
    return output


def install_starter_inference(pipe, *, generated_cache='clean'):
    """Patch only this disposable inference instance, never upstream files."""
    if not hasattr(pipe, 'generate_samples_from_batch') or not hasattr(pipe, '_initialize_kv_cache'):
        raise ValueError('Starter inference requires the ArtiFixer KV-cache pipeline')
    if generated_cache not in ('clean', 'last_denoising'):
        raise ValueError('Unknown generated-cache policy')
    pipe.starter_generated_cache = generated_cache
    pipe.generate_samples_from_batch = MethodType(generate_from_starter, pipe)


def generation_policy(options):
    """Describe the effective schedule, including legacy exact-starter requests."""
    schedule = options.get('block_schedule', 'exact_starter')
    if schedule not in ('exact_starter', 'periodic_starter', 'upstream'):
        raise ValueError('Unknown block schedule')
    exact = schedule != 'upstream'
    return {
        'block_schedule': schedule,
        'conditioning_mode': ('gpt-periodic-starter-kv-v1' if schedule == 'periodic_starter'
                              else 'gpt-starter-kv-v1' if exact else 'gpt-reference-upstream-v1'),
        'generated_cache': options.get('generated_cache', 'clean') if exact else 'last_denoising',
        'anchor_role': ('clean_temporal_starter_and_direct_reconstruction_target' if exact
                        else 'reference_and_direct_reconstruction_target'),
        'exact_starter_preserved': exact,
        **({'refresh_rgb_interval': 20,
            'refresh_cache': 'clean corrected block; other blocks use generated_cache',
            'refresh_schedule': 'split original blocks at VAE-aligned RGB indices 20,40,60,...'}
           if schedule == 'periodic_starter' else {}),
    }
