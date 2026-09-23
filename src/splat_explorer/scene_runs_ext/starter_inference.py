"""GPT-image-seeded causal inference for the pinned ArtiFixer KV pipeline.

The clean first latent is temporal context, not a denoised prediction. Later
frames use noise, calibrated cameras and edited references, never splat RGB.
This is an inference adaptation; it does not guarantee multiview consistency.
"""
from __future__ import annotations
from types import MethodType


def starter_reference(manifest, segment):
    matches = [r for r in manifest['references'] if r['frame_index'] == segment['start']]
    if len(matches) != 1 or matches[0].get('kind') != 'edited_render':
        raise ValueError('Each starter-image trajectory requires one calibrated GPT-edited reference')
    return matches[0]


def latent_chunks(total, block):
    if total < 1 or block < 1:
        raise ValueError('Latent frame count and block size must be positive')
    yield 0, 1
    for start in range(1, total, block):
        yield start, min(start + block, total)


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
    sampled. Every generated block also refreshes KV memory with its clean
    output rather than leaving the last noisy denoising input in the cache.
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
    for start, end in latent_chunks(total, self.frames_per_block):
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
            shape = condition[:, :, start:end].to(device)
            latents = torch.randn_like(shape)
            for index, timestep in enumerate(timesteps):
                noise = self.transformer(hidden_states=latents,
                    timestep=timestep.expand(batch).to(latents.dtype), **kwargs)[0]
                latents = self.scheduler.step(noise, timestep.expand(batch), latents, to_final=True)
                if index + 1 < len(timesteps):
                    latents = self.scheduler.add_noise(latents, torch.randn_like(latents),
                        timesteps[index + 1] * torch.ones(batch, device=device, dtype=torch.long))
        output[:, :, start:end] = latents
        self.transformer(hidden_states=latents,
                         timestep=torch.zeros(batch, device=device, dtype=latents.dtype), **kwargs)
    return output


def install_starter_inference(pipe):
    """Patch only this disposable inference instance, never upstream files."""
    if not hasattr(pipe, 'generate_samples_from_batch') or not hasattr(pipe, '_initialize_kv_cache'):
        raise ValueError('Starter inference requires the ArtiFixer KV-cache pipeline')
    pipe.generate_samples_from_batch = MethodType(generate_from_starter, pipe)
