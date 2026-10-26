"""Serializable model/sampling choices for Step 4, independent of GPU setup."""
from pathlib import Path


MODELS = {
    '1.3b': ('artifixer-1.3b.pt', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'),
    '14b': ('artifixer-14b.pt', 'Wan-AI/Wan2.1-T2V-14B-Diffusers'),
    '14b-teacher': ('artifixer-s1-14b.pt', 'Wan-AI/Wan2.1-T2V-14B-Diffusers'),
}
TEACHER_DISABLED = ('connected_trajectory_generation', 'handoff_generation',
                    'image_cache_insertion', 'initial_image_seeding', 'generated_context_references')


def inference_options(raw):
    model = raw.get('inference_model', '1.3b')
    if model not in MODELS:
        raise ValueError('inference_model must be 1.3b, 14b or 14b-teacher')
    result = {'inference_model': model}
    if model == '14b-teacher':
        frames = raw.get('teacher_max_frames', 17)
        steps = raw.get('teacher_steps', 50)
        resolution = raw.get('teacher_resolution', '480p')
        if type(frames) is not int or not 5 <= frames <= 81 or (frames - 1) % 4:
            raise ValueError('teacher_max_frames must be 1 + 4*n, between 5 and 81')
        if type(steps) is not int or not 1 <= steps <= 100:
            raise ValueError('teacher_steps must be between 1 and 100 (50 for quality; 1 only for smoke tests)')
        if resolution not in ('480p', 'native'):
            raise ValueError('teacher_resolution must be 480p or native')
        result.update(teacher_max_frames=frames, teacher_steps=steps, teacher_resolution=resolution,
                      **{key: False for key in TEACHER_DISABLED})
    return result


def model_runtime(runtime):
    """Missing selection preserves historical/custom paths; explicit selection is paired."""
    cfg = dict(runtime)
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
