"""Serializable model/sampling choices for Step 4, independent of GPU setup."""
from pathlib import Path
import math


MODELS = {
    '1.3b': ('artifixer-1.3b.pt', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'),
    '14b': ('artifixer-14b.pt', 'Wan-AI/Wan2.1-T2V-14B-Diffusers'),
    '14b-teacher': ('artifixer-s1-14b.pt', 'Wan-AI/Wan2.1-T2V-14B-Diffusers'),
}
TEACHER_DISABLED = ('connected_trajectory_generation', 'handoff_generation', 'generated_context_references')


def new_job_options(raw):
    """Default new submissions to the teacher; explicit model choices still work."""
    enabled = raw.get('teacher_inference', raw.get('inference_model', '14b-teacher') == '14b-teacher')
    if type(enabled) is not bool:
        raise ValueError('teacher_inference must be a boolean')
    model = '14b-teacher' if enabled else raw.get('inference_model', '1.3b')
    if not enabled and model == '14b-teacher':
        raise ValueError('Enable teacher inference or choose a legacy distilled model')
    defaults = {'initial_image_seeding': True, 'image_cache_insertion': True,
                'teacher_max_frames': 25, 'teacher_clip_layout': 'full-leg'} if enabled else {}
    return {'teacher_inference': enabled, **inference_options({**defaults, **raw, 'inference_model': model})}


def inference_options(raw):
    model = raw.get('inference_model', '1.3b')
    if model not in MODELS:
        raise ValueError('inference_model must be 1.3b, 14b or 14b-teacher')
    result = {'inference_model': model}
    if model == '14b-teacher':
        # Historical teacher requests without these flags remain reference-only.
        seeding = raw.get('initial_image_seeding', False)
        preserve = raw.get('image_cache_insertion', False)
        if type(seeding) is not bool or type(preserve) is not bool:
            raise ValueError('initial_image_seeding and image_cache_insertion must be booleans')
        frames = raw.get('teacher_max_frames', 37)
        steps = raw.get('teacher_steps', 50)
        resolution = raw.get('teacher_resolution', 'native')
        layout = raw.get('teacher_clip_layout', 'segment')
        if layout not in ('segment', 'full-leg'):
            raise ValueError('teacher_clip_layout must be segment or full-leg')
        extension = raw.get('teacher_endpoint_extension', 0)
        opacity = raw.get('teacher_opacity_scale', 1.0)
        reference_opacity = raw.get('teacher_reference_opacity', 1.0)
        warmup_supervision = raw.get('teacher_warmup_supervision', True)
        if type(warmup_supervision) is not bool:
            raise ValueError('teacher_warmup_supervision must be a boolean')
        if type(extension) is not int or extension not in (0, 4, 8):
            raise ValueError('teacher_endpoint_extension must be 0, 4 or 8')
        if type(opacity) not in (int, float) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError('teacher_opacity_scale must be between 0 and 1')
        if type(reference_opacity) not in (int, float) or not math.isfinite(reference_opacity) or not 0 <= reference_opacity <= 1:
            raise ValueError('teacher_reference_opacity must be between 0 and 1')
        if extension and (layout != 'full-leg' or extension >= frames - 1):
            raise ValueError('Endpoint extension requires full-leg clips and room for target views')
        if extension:
            # A natural warm-up path places the repaired reference inside the
            # video. Its temporal latent cannot be locked as an isolated frame.
            seeding, preserve = True, False
        if preserve and reference_opacity != 1:
            raise ValueError('Preserving the reference latent requires reference opacity 1; choose seed once to adjust it')
        if type(frames) is not int or not 5 <= frames <= 81 or (frames - 1) % 4:
            raise ValueError('teacher_max_frames must be 1 + 4*n, between 5 and 81')
        if type(steps) is not int or not 1 <= steps <= 150:
            raise ValueError('teacher_steps must be between 1 and 150 (50 by default; 1 only for smoke tests)')
        if resolution not in ('480p', 'native'):
            raise ValueError('teacher_resolution must be 480p or native')
        result.update(teacher_max_frames=frames, teacher_steps=steps, teacher_resolution=resolution,
                      teacher_clip_layout=layout,
                      teacher_endpoint_extension=extension, teacher_opacity_scale=float(opacity),
                      teacher_reference_opacity=float(reference_opacity),
                      teacher_warmup_supervision=warmup_supervision,
                      initial_image_seeding=seeding or preserve, image_cache_insertion=preserve,
                      **{key: False for key in TEACHER_DISABLED})
    return result


def model_runtime(runtime):
    """Missing selection preserves historical/custom paths; explicit selection is paired."""
    cfg = dict(runtime)
    if 'teacher_inference' in cfg:
        # New-submission defaults must not reinterpret a saved request.
        cfg.update(new_job_options({**cfg, 'teacher_max_frames': cfg.get('teacher_max_frames', 37),
                                    'teacher_clip_layout': cfg.get('teacher_clip_layout', 'segment')}))
    selection = cfg.get('inference_model')
    if selection is None:
        name = Path(cfg.get('checkpoint', '')).name
        selection = next((key for key, (filename, _) in MODELS.items() if name == filename), '1.3b')
    options = inference_options({**cfg, 'inference_model': selection})
    if 'inference_model' in cfg:
        filename, model_id = MODELS[selection]
        cfg.update(checkpoint=str(Path(cfg['checkpoint']).with_name(filename)), model_id=model_id)
    cfg.update(options)
    if selection == '14b-teacher' and cfg.get('model_id') != MODELS[selection][1]:
        raise ValueError('14B teacher requires its matching 14B base model')
    return cfg


def inference_size(width, height, resolution):
    """Keep the field of view, align VAE dimensions; never enlarge inputs."""
    if resolution == 'native':
        if width % 16 or height % 16:
            raise ValueError('Native teacher dimensions must be divisible by 16')
        return width, height
    scale = min(1., 480 / height, 832 / width)
    return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)
