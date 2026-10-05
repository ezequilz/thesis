"""Resumable GPT image editing, independent of selection and splat repair."""
from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image

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
    the repository's GPT image backend. Responses are decoded once and saved at
    the original camera resolution; changed aspect ratios are rejected. Response
    payloads (which can contain huge base64 images) are deliberately not copied.
    """
    if not isinstance(checkpoint, Checkpoint):
        checkpoint = Checkpoint.load(checkpoint)
    if not checkpoint.complete:
        raise ValueError('Finish selecting all requested views before image repair')
    pending = [v for v in checkpoint.views if not v.get('repaired_rgb')]
    for view in checkpoint.views:
        if view.get('repaired_rgb'):
            checkpoint.image_path(view, repaired=True)
    if not pending:
        return checkpoint
    if backend is None:
        from ..image_edit_gpt import GptImageEditBackend
        backend = GptImageEditBackend.from_config(cfg)
    for view in pending:
        metadata = {'backend': backend.name, 'prompt': prompt, 'started_at': utc_now()}
        view['image_repair'] = metadata
        checkpoint.save()
        try:
            result = backend.edit(checkpoint.image_path(view), prompt)
            if result.error or not result.images:
                raise RuntimeError(result.error or 'Image edit returned no image')
            with Image.open(io.BytesIO(result.images[0])) as edited:
                image = edited.convert('RGB')
            width, height = view['camera']['width'], view['camera']['height']
            if image.width * height != image.height * width:
                raise ValueError('Image edit changed the calibrated camera aspect ratio')
            metadata['response_size'] = list(image.size)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
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
