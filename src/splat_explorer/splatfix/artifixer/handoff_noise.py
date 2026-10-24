"""Deterministic pixel-space noise for handoff repair inputs."""
import math
from pathlib import Path


def noise_options(options):
    enabled = options.get('handoff_add_noise', False)
    strength = options.get('handoff_noise_strength', 25)
    if type(enabled) is not bool:
        raise ValueError('handoff_add_noise must be a boolean')
    vignette = options.get('handoff_noise_vignette', False)
    if type(vignette) is not bool:
        raise ValueError('handoff_noise_vignette must be a boolean')
    mode = options.get('handoff_noise_mode', 'gaussian')
    if mode not in ('gaussian', 'repaired_pixel_sprinkle'):
        raise ValueError('handoff_noise_mode must be gaussian or repaired_pixel_sprinkle')
    if enabled and mode == 'repaired_pixel_sprinkle' and options.get('gpt_image_context') == 'single_last_frame':
        raise ValueError('Repaired pixel sprinkle requires closest/initial repaired image or original RGB capture context')
    validate_strength(strength)
    return {'handoff_add_noise': enabled, 'handoff_noise_strength': float(strength),
            'handoff_noise_vignette': vignette, 'handoff_noise_mode': mode}


def validate_strength(strength):
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 100):
        raise ValueError('Handoff noise strength must be between 0 and 100 percent')


def vignette_strength(height, width, strength):
    """Elliptical smoothstep: flat central half-radius, full noise at the rim."""
    import numpy as np
    x = np.linspace(-1, 1, width) if width > 1 else np.zeros(1)
    y = np.linspace(-1, 1, height) if height > 1 else np.zeros(1)
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    t = np.clip((radius - .5) / .5, 0, 1)
    feather = t * t * (3 - 2 * t)
    # Do not reduce a base strength that already exceeds the edge target.
    return strength + (max(strength, 80) - strength) * feather


def add_gaussian_noise(source, destination, *, strength, seed, vignette=False):
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
    if type(vignette) is not bool:
        raise ValueError('vignette must be a boolean')
    sigma = (vignette_strength(*pixels.shape[:2], strength)[..., None] * 255 / 100
             if vignette else strength * 255 / 100)
    noise = np.random.default_rng(seed).normal(0, sigma, pixels.shape)
    noisy = np.rint(np.clip(pixels + noise, 0, 255)).astype(np.uint8)
    Image.fromarray(noisy).save(destination, format='PNG')
    return {'algorithm': 'additive_gaussian_rgb', 'strength_percent': strength,
            'sigma_8bit': strength * 255 / 100, 'seed': seed,
            'clipping': '0_255', 'color_space': 'encoded_rgb',
            'vignette': {'enabled': vignette, 'shape': 'ellipse', 'falloff': 'smoothstep',
                         'inner_radius': .5, 'outer_radius': 1.,
                         'edge_strength_percent': max(strength, 80) if vignette else strength}}


def sprinkle_repaired_pixels(source, reference, destination, *, strength, seed, vignette=False):
    """Replace whole RGB pixels with the same-position repaired-reference pixels.

    Strength is replacement probability in percent, not Gaussian sigma. Uses the
    same elliptical strength profile as Gaussian mode, without additive noise.
    """
    import numpy as np
    from PIL import Image, ImageOps
    validate_strength(strength)
    if type(vignette) is not bool:
        raise ValueError('vignette must be a boolean')
    source, reference, destination = map(Path, (source, reference, destination))
    if destination.resolve() in (source.resolve(), reference.resolve()):
        raise ValueError('Pixel sprinkle must preserve the original and reference images')
    with Image.open(source) as image:
        original = np.asarray(image.convert('RGB'))
        size = image.size
    with Image.open(reference) as image:
        reference_size = list(image.size)
        repaired = np.asarray(ImageOps.fit(image.convert('RGB'), size, Image.Resampling.LANCZOS))
    probability = (vignette_strength(*original.shape[:2], strength) if vignette else strength) / 100
    selected = np.random.default_rng(seed).random(original.shape[:2]) < probability
    combined = np.where(selected[..., None], repaired, original)
    Image.fromarray(combined).save(destination, format='PNG')
    return {'algorithm': 'repaired_pixel_sprinkle', 'strength_percent': strength,
            'strength_semantics': 'whole_pixel_replacement_probability', 'seed': seed,
            'replaced_pixels': int(selected.sum()), 'total_pixels': int(selected.size),
            'reference_source_size': reference_size, 'reference_fit': 'cover_center_crop',
            'vignette': {'enabled': vignette, 'shape': 'ellipse', 'falloff': 'smoothstep',
                         'inner_radius': .5, 'outer_radius': 1.,
                         'edge_strength_percent': max(strength, 80) if vignette else strength}}
