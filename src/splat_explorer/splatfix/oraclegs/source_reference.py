"""Optional fixed global-image source; all spatial RGB conditioning stays native."""
import functools
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image


class FixedSourceConditioner:
    """Delegate model lifecycle; encode only the clip's initial saved image.

    Native SEVA averages this result over images. Returning the single source's
    embedding preserves its API while excluding other references from that mean.
    Their VAE latents, cameras, masks, and multiview attention remain unchanged.
    """
    def __init__(self, conditioner, image):
        self.conditioner, self.image = conditioner, image

    def __getattr__(self, name):
        return getattr(self.conditioner, name)

    def __call__(self, images):
        return self.conditioner(self.image.to(images))


def load_source(data_path, request):
    scene = (Path(data_path)/'scene').resolve()
    frames = json.loads((scene/'transforms.json').read_text())['frames']
    index = request['frame_index']
    if type(index) is not int or not 0 <= index < len(frames):
        raise ValueError('Invalid fixed source frame index')
    frame = frames[index]
    path = (scene/frame['file_path']).resolve()
    if frame.get('camera_only') or not path.is_relative_to(scene):
        raise ValueError('Fixed source must be a saved reference image')
    # Validate that this is the requested saved reference, not an on-path
    # generated anchor that happens to share the initial camera pose.
    if path.stem != f"ref_{request['reference_index']:06d}":
        raise ValueError('Fixed source is not the preset starting reference')
    with Image.open(path) as image:
        if image.size != (frame['w'], frame['h']):
            raise ValueError('Fixed source camera/image dimensions disagree')
        pixels = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2,0,1)/127.5-1
    return pixels[None], {'source_image':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'reference_index':request['reference_index'], 'mode':request['mode']}


def install_fixed_source(evaluation, data_path, request, report_path):
    import torch
    pixels, source = load_source(data_path, request)
    image = torch.as_tensor(pixels)
    original = evaluation.do_sample
    signature = inspect.signature(original)
    report = {**source, 'policy':'fixed_initial', 'global_embedding':'initial-saved-image-only',
              'spatial_context':'all native conditioned RGB latents and cameras retained', 'chunks':[]}

    @functools.wraps(original)
    def sample(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if tuple(image.shape[-2:]) != (bound.arguments['H'], bound.arguments['W']):
            raise ValueError('Fixed source resolution differs from SEVA conditioning')
        values = bound.arguments['value_dict']
        bound.arguments['conditioner'] = FixedSourceConditioner(bound.arguments['conditioner'], image)
        result = original(*bound.args, **bound.kwargs)
        report['chunks'].append({'frame_count':bound.arguments['T'],
            'conditioned_count':int(values['cond_frames_mask'].sum().item()), 'completed':result is not None})
        path = Path(report_path)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)
        return result
    evaluation.do_sample = sample
