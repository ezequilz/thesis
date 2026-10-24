"""Deterministic pixel-space noise for handoff repair inputs."""
import math
from pathlib import Path


def noise_options(options):
    enabled = options.get('handoff_add_noise', False)
    strength = options.get('handoff_noise_strength', 25)
    if type(enabled) is not bool:
        raise ValueError('handoff_add_noise must be a boolean')
    validate_strength(strength)
    return {'handoff_add_noise': enabled, 'handoff_noise_strength': float(strength)}


def validate_strength(strength):
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 100):
        raise ValueError('Handoff noise strength must be between 0 and 100 percent')


def add_gaussian_noise(source, destination, *, strength, seed):
    """Add independent RGB Gaussian noise; strength is sigma as percent of 255.

    This is image preprocessing, not a diffusion timestep or model temperature.
    The original is never overwritten. A stable seed makes retries reproducible.
    """
    import numpy as np
    from PIL import Image
    validate_strength(strength)
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Handoff noise must preserve the original image')
    with Image.open(source) as image:
        pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, strength * 255 / 100, pixels.shape)
    noisy = np.rint(np.clip(pixels + noise, 0, 255)).astype(np.uint8)
    Image.fromarray(noisy).save(destination, format='PNG')
    return {'algorithm': 'additive_gaussian_rgb', 'strength_percent': strength,
            'sigma_8bit': strength * 255 / 100, 'seed': seed,
            'clipping': '0_255', 'color_space': 'encoded_rgb'}
