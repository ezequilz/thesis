"""Publish a six-view repair set atomically; retain previous versions on disk."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import re
import uuid
from pathlib import Path

from PIL import Image

from ..image_edit_multiview import REASONING_MODEL, VIEW_COUNT, validate_response
from .checkpoint import atomic_json, utc_now


def repair_shared_views(checkpoint, views, *, backend, prompt, image_set, request_id=None,
                        checkpoint_before=None, repair_mode='shared_six'):
    if len(views) != VIEW_COUNT:
        raise ValueError('Shared Thinking repair requires exactly six views; use sequential repair for other counts')
    if not callable(getattr(backend, 'edit_views', None)):
        raise ValueError('This image backend does not support shared Thinking repair')
    request_id = request_id or uuid.uuid4().hex
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', request_id):
        raise ValueError('Invalid image repair request ID')
    directory = checkpoint.root / 'image-repairs' / ('shared-' + request_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (checkpoint.root / '.shared-image-repair.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A shared image repair is already running for this checkpoint') from None
        return _repair(checkpoint, views, backend, prompt, image_set, request_id, directory,
                       checkpoint_before if checkpoint_before is not None else checkpoint.manifest,
                       repair_mode)


def _repair(checkpoint, views, backend, prompt, image_set, request_id, directory, checkpoint_before, repair_mode):
    manifest_path = checkpoint.root / 'checkpoint.json'
    before_bytes = manifest_path.read_bytes()
    if json.loads(before_bytes) != checkpoint_before:
        raise ValueError('Checkpoint changed; reload it before image repair')
    response_path = directory / 'response.json'
    status_path = directory / 'status.json'
    targets = [dict(id=v['id'], path=checkpoint.image_path(v),
                    width=v['camera']['width'], height=v['camera']['height']) for v in views]
    signature = {'image_set': image_set, 'prompt': prompt, 'model': backend.model, 'repair_mode': repair_mode,
                 'targets': [{'id': t['id'], 'width': t['width'], 'height': t['height'],
                              'sha256': hashlib.sha256(t['path'].read_bytes()).hexdigest()}
                             for t in targets]}
    status = {'status': 'running', 'request_id': request_id, 'started_at': utc_now(),
              'request': signature, 'reasoning_model': REASONING_MODEL, 'reasoning_effort': 'high',
              'checkpoint_sha256': hashlib.sha256(before_bytes).hexdigest()}
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior.get('request') != signature:
            raise ValueError('Repair request ID was already used for different inputs')
        if all(v.get('image_repair', {}).get('request_id') == request_id for v in views):
            for v in views:
                checkpoint.image_path(v, repaired=True)
            return checkpoint
        if prior.get('checkpoint_sha256') != status['checkpoint_sha256']:
            raise ValueError('Checkpoint changed since this shared repair started; start a new run')
        if not response_path.exists():
            raise ValueError('Earlier shared request has no saved response; start a new run explicitly to retry')
        status['started_at'] = prior['started_at']
    else:
        atomic_json(directory / 'checkpoint-before.json', checkpoint_before)
    atomic_json(status_path, status)
    original_manifest = copy.deepcopy(checkpoint_before)
    try:
        if response_path.exists():
            results = validate_response(json.loads(response_path.read_text()), [v['id'] for v in views],
                                        expected_model=backend.model)
        else:
            results = backend.edit_views(targets, prompt, response_path)
        if set(results) != {v['id'] for v in views}:
            raise ValueError('Shared repair did not return exactly the six requested view IDs')
        updates = []
        for view in views:
            result = results[view['id']]
            folder = directory / view['id']
            folder.mkdir(exist_ok=True)
            (folder / 'response.image').write_bytes(result['image'])
            width, height = view['camera']['width'], view['camera']['height']
            with Image.open(io.BytesIO(result['image'])) as source:
                image = source.convert('RGB')
            # Cropping a multi-view repair changes its camera. Allow only small
            # model dimension rounding, then resize the full frame.
            if abs(image.width * height / (image.height * width) - 1) > .01:
                raise ValueError(f"Repair {view['id']} changed its aspect ratio; existing repairs were kept")
            response_size = list(image.size)
            if image_set == 'seva':
                image.save(folder / 'response-original.png')
            image.resize((width, height), Image.Resampling.LANCZOS).save(folder / 'repaired.png')
            relative = folder.relative_to(checkpoint.root)
            metadata = {'status': 'complete', 'repair_mode': repair_mode, 'request_id': request_id,
                        'backend': backend.name, 'model': backend.model,
                        'reasoning_model': REASONING_MODEL, 'reasoning_effort': 'high',
                        'prompt': prompt, 'image_set': image_set, 'target_size': [width, height],
                        'response_size': response_size, 'response_image': str(relative / 'response.image'),
                        'call_id': result.get('call_id'), 'revised_prompt': result.get('revised_prompt'),
                        'requested_quality': 'high', 'reported_quality': result.get('reported_quality'),
                        'reported_size': result.get('reported_size'),
                        'context_view_ids': [v['id'] for v in views],
                        'previous_repaired_rgb': view.get('repaired_rgb'),
                        'started_at': status['started_at'], 'completed_at': utc_now(),
                        'resize': {'method': 'resize_full_frame',
                                   'scale_xy': [width / image.width, height / image.height],
                                   'source_crop_box': [0, 0, image.width, image.height]}}
            updates.append((view, relative, metadata))
        if manifest_path.read_bytes() != before_bytes or any(
                hashlib.sha256(t['path'].read_bytes()).hexdigest() != s['sha256']
                for t, s in zip(targets, signature['targets'])):
            raise ValueError('Checkpoint or source images changed during repair; outputs retained without publishing')
        for view, relative, metadata in updates:
            # Resolution staging prefers manual_resized_rgb over repaired_rgb.
            # A previous crop must not silently override this new generation.
            for key in ('manual_resized_rgb', 'manual_crop', 'full_repaired_rgb'):
                view.pop(key, None)
            view['repaired_rgb'] = str(relative / 'repaired.png')
            view['image_repair'] = metadata
            if image_set == 'seva':
                view['original_response_rgb'] = str(relative / 'response-original.png')
        checkpoint.save()  # One atomic manifest switch for all six views.
    except Exception as exc:
        checkpoint.manifest = original_manifest
        atomic_json(status_path, {**status, 'status': 'failed', 'error': str(exc), 'finished_at': utc_now()})
        raise
    atomic_json(status_path, {**status, 'status': 'complete', 'finished_at': utc_now()})
    return checkpoint
