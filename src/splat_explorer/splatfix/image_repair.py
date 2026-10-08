"""Resumable GPT image editing, independent of selection and splat repair."""
from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageOps

from .checkpoint import Checkpoint, utc_now

DEFAULT_PROMPT = (
    'Repair this rendered 3D reconstruction into a clean, realistic image. '
    'Remove floaters, holes, blur, and reconstruction artifacts while preserving '
    'the exact camera viewpoint, perspective, scene geometry, objects, lighting, '
    'and framing. Do not invent or rearrange objects. Return the same aspect ratio.'
)


def repair_images(checkpoint, *, backend=None, cfg=None, prompt=DEFAULT_PROMPT):
    """Edit missing views, committing each success so retries avoid paid repeats.

    The supplied backend is injectable for offline tests. By default this uses
    the repository's GPT image backend. Original response bytes are saved before
    decoding. Images are scaled proportionally and center-cropped to the camera
    resolution, without padding or stretching.
    Response payloads containing base64 images are deliberately not copied.
    """
    if not isinstance(checkpoint, Checkpoint):
        checkpoint = Checkpoint.load(checkpoint)
    if not checkpoint.complete:
        raise ValueError('Finish selecting all requested views before image repair')
    checkpoint.require_viser_images()
    pending = [v for v in checkpoint.views if not v.get('repaired_rgb')]
    for view in checkpoint.views:
        if view.get('repaired_rgb'):
            checkpoint.image_path(view, repaired=True)
    if not pending:
        return checkpoint
    if backend is None:
        from ..image_edit import make_image_edit_backend
        backend = make_image_edit_backend(cfg=cfg)
    for view in pending:
        width, height = view['camera']['width'], view['camera']['height']
        metadata = {'backend': backend.name, 'model': getattr(backend, 'model', backend.name),
                    'prompt': prompt, 'target_size': [width, height], 'started_at': utc_now()}
        view['image_repair'] = metadata
        checkpoint.save()
        try:
            result = backend.edit(checkpoint.image_path(view), prompt)
            if result.error or not result.images:
                raise RuntimeError(result.error or 'Image edit returned no image')
            # Keep the exact response independently of the normalized PNG, even
            # if decoding or resizing fails. The bytes determine the image format.
            response_relative = f"views/{view['id']}/response.image"
            response_path = checkpoint.root / response_relative
            response_tmp = response_path.with_suffix('.tmp')
            try:
                response_tmp.write_bytes(result.images[0])
                os.replace(response_tmp, response_path)
            finally:
                response_tmp.unlink(missing_ok=True)
            metadata['response_image'] = response_relative
            checkpoint.save()
            with Image.open(io.BytesIO(result.images[0])) as edited:
                image = edited.convert('RGB')
            metadata['response_size'] = list(image.size)
            scale = max(width / image.width, height / image.height)
            crop_width, crop_height = width / scale, height / scale
            left, top = (image.width - crop_width) / 2, (image.height - crop_height) / 2
            metadata['resize'] = {'method': 'cover_center_crop', 'scale': scale,
                                  'source_crop_box': [left, top, left + crop_width, top + crop_height]}
            image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS,
                                 centering=(0.5, 0.5))
            relative = f"views/{view['id']}/repaired.png"
            destination = checkpoint.root / relative
            temporary = destination.with_suffix('.tmp')
            try:
                image.save(temporary, format='PNG')
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            view['repaired_rgb'] = relative
            metadata.update(status='complete', completed_at=utc_now())
            checkpoint.save()
        except Exception as exc:
            view.pop('repaired_rgb', None)
            metadata.update(status='failed', error=str(exc))
            checkpoint.save()
            raise
    return checkpoint
