"""Manual framing corrections, always derived from the full GPT response."""
from pathlib import Path
import hashlib
import uuid

from PIL import Image, ImageOps

from .checkpoint import utc_now


def crop_source(checkpoint, view):
    if not view.get('repaired_rgb'):
        raise ValueError('Select a saved GPT repair')
    relative = (view.get('image_repair') or {}).get('response_image') or view.get('full_repaired_rgb') or view['repaired_rgb']
    path = checkpoint.image_path({'original_rgb': relative})
    token = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, token


def save_crop(checkpoint, view, box, token):
    source, current_token = crop_source(checkpoint, view)
    if token != current_token:
        raise ValueError('The repair changed. Reopen Resize for input.')
    if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
        raise ValueError('Crop must contain four integer pixel coordinates')
    x, y, width, height = box
    with Image.open(source) as raw:
        image = raw.convert('RGB')
    if width < 1280 or height < 720 or width * 9 != height * 16:
        raise ValueError('Crop must be 16:9 and at least 1280 × 720')
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        raise ValueError('Crop must stay inside the full repaired image')
    directory = Path(view['original_rgb']).parent / ('manual-' + uuid.uuid4().hex)
    (checkpoint.root / directory).mkdir(parents=True)
    # Keep the normalized legacy repair as well as the untouched response bytes.
    view.setdefault('full_repaired_rgb', view['repaired_rgb'])
    crop = image.crop((x, y, x + width, y + height))
    relative = str(directory / 'manual_resized.png')
    crop.save(checkpoint.root / relative)
    for name, size in [('720p', (1280, 720)), ('480p', (832, 480))]:
        ImageOps.fit(crop, size, Image.Resampling.LANCZOS).save(checkpoint.root / directory / (name + '.png'))
    display = str(directory / 'repaired.png')
    size = (view['camera']['width'], view['camera']['height'])
    ImageOps.fit(crop, size, Image.Resampling.LANCZOS).save(checkpoint.root / display)
    view['manual_resized_rgb'] = relative
    view['repaired_rgb'] = display
    view['manual_crop'] = {'box_xywh': box, 'source_size': list(image.size),
                           'source_sha256': token, 'created_at': utc_now()}
    checkpoint.save()
