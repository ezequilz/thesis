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


def repair_images(checkpoint, *, backend=None, cfg=None, prompt=DEFAULT_PROMPT, image_set='saved', view_id=None,
                  endpoint_repair_context='initial_repair'):
    """Edit missing views, committing each success so retries avoid paid repeats.

    The supplied backend is injectable for offline tests. By default this uses
    the repository's GPT image backend. Original response bytes are saved before
    decoding. Images are scaled proportionally and center-cropped to the camera
    resolution. SEVA retains the full response and resizes locally without cropping;
    original-capture repairs retain their established center-crop policy.
    Response payloads containing base64 images are deliberately not copied.
    """
    if endpoint_repair_context not in ('initial_repair', 'prior_repair', 'rgb_only'):
        raise ValueError('endpoint_repair_context must be initial_repair, prior_repair or rgb_only')
    if not isinstance(checkpoint, Checkpoint):
        checkpoint = Checkpoint.load(checkpoint)
    if not checkpoint.complete:
        raise ValueError('Finish selecting all requested views before image repair')
    if image_set == '77':
        checkpoint.require_horizontal_fov(77)
    if checkpoint.endpoint_only and image_set not in ('saved', '77'):
        raise ValueError('Endpoint checkpoints use their saved RGB captures')
    if image_set == 'seva':
        from .oraclegs.captures import validate_capture
        selected = [v for v in checkpoint.views if view_id is None or v['id'] == view_id]
        if not selected:
            raise ValueError('Unknown saved view')
        views = [validate_capture(checkpoint, v) for v in selected]
        # Nested records remain owned by the original checkpoint for saving.
        for original, view in zip(selected, views):
            view['id'] = original['id']
    elif image_set in ('saved', '77'):
        checkpoint.require_viser_images()
        views = checkpoint.repair_views
    else:
        raise ValueError('Unknown image set')
    pending = [v for v in views if not v.get('repaired_rgb')]
    for view in views:
        if view.get('repaired_rgb'):
            checkpoint.image_path(view, repaired=True)
    if not pending:
        return checkpoint
    if backend is None:
        from ..image_edit import make_image_edit_backend
        backend = make_image_edit_backend(cfg=cfg)
    use_context = endpoint_repair_context != 'rgb_only'
    if use_context and any(v['id'] != views[0]['id'] for v in pending) and not callable(getattr(backend, 'edit_with_references', None)):
        raise ValueError('Image backend does not support repair context; select RGB only')
    for view in pending:
        width, height = view['camera']['width'], view['camera']['height']
        metadata = {'backend': backend.name, 'model': getattr(backend, 'model', backend.name),
                    'prompt': prompt, 'image_set':image_set,
                    'endpoint_repair_context': endpoint_repair_context, 'target_size': [width, height], 'started_at': utc_now()}
        view['image_repair'] = metadata
        checkpoint.save()
        try:
            if use_context and view['id'] != views[0]['id']:
                index = next(i for i, candidate in enumerate(views) if candidate['id'] == view['id'])
                context_view = views[index - 1] if endpoint_repair_context == 'prior_repair' else views[0]
                reference = checkpoint.image_path(context_view, repaired=True)
                metadata['context_repaired_rgb'] = context_view['repaired_rgb']
                result = backend.edit_with_references(checkpoint.image_path(view), [reference], prompt)
            else:
                result = backend.edit(checkpoint.image_path(view), prompt)
            if result.error or not result.images:
                raise RuntimeError(result.error or 'Image edit returned no image')
            # Keep the exact response independently of the normalized PNG, even
            # if decoding or resizing fails. The bytes determine the image format.
            response_relative = str(Path(view['original_rgb']).parent / 'response.image')
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
            if image_set == 'seva':
                original_response = Path(view['original_rgb']).parent / 'response-original.png'
                image.save(checkpoint.root / original_response, format='PNG')
                view['original_response_rgb'] = str(original_response)
            if image_set == 'seva':
                # Keep the model's original bytes above, and resize its full frame
                # locally. Never crop away image content to force the SEVA canvas.
                metadata['dimension_check'] = {'input_size':[width,height], 'output_size':list(image.size),
                    'exact_dimensions':image.size == (width,height),
                    'same_aspect_ratio':image.width * height == image.height * width,
                    'policy':'preserve original response; resize full frame locally'}
                metadata['resize'] = {'method':'resize_full_frame',
                    'scale_xy':[width/image.width, height/image.height],
                    'source_crop_box':[0,0,image.width,image.height]}
                image = image.resize((width,height), Image.Resampling.LANCZOS)
            else:
                scale = max(width / image.width, height / image.height)
                crop_width, crop_height = width / scale, height / scale
                left, top = (image.width - crop_width) / 2, (image.height - crop_height) / 2
                metadata['resize'] = {'method': 'cover_center_crop', 'scale': scale,
                                      'source_crop_box': [left, top, left + crop_width, top + crop_height]}
                image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS,
                                     centering=(0.5, 0.5))
            relative = str(Path(view['original_rgb']).parent / 'repaired.png')
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
