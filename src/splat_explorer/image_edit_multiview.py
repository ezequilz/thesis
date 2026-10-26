"""Six separate camera repairs in one Responses conversation.

Shared reasoning/image inputs do not imply joint diffusion or cross-output
attention. Never use Images API `n` to assign variations to different cameras.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re

from PIL import Image

from .splatfix.checkpoint import atomic_json

VIEW_COUNT = 6
REASONING_MODEL = 'gpt-6-astra'
NATIVE_IMAGE_MODEL = 'gpt-image-2-codex'


def check_image_model(payload, requested_model):
    models = {t.get('model') for t in payload.get('tools', []) if t.get('type') == 'image_generation'}
    if models != {requested_model}:
        reported = ', '.join(sorted(str(m) for m in models)) or 'unreported'
        raise ValueError(f'Shared Thinking requested {requested_model}, but the relay reports {reported}. '
                         'Refusing a silent model substitution. Use sequential repair or configure a '
                         'Responses route that preserves the requested image model.')


def request_repairs(client, model, targets, prompt, response_path):
    """Persist the complete response before validating output-to-camera mapping.

    Disable SDK retries: an ambiguous network failure must not automatically
    purchase another six-image request. The caller retains existing repairs.
    """
    if len(targets) != VIEW_COUNT or len({t['id'] for t in targets}) != VIEW_COUNT:
        raise ValueError('Shared Thinking repair requires exactly six distinct views')
    if any(not re.fullmatch(r'[0-9]{3}', t['id']) for t in targets):
        raise ValueError('Expected three-digit checkpoint view IDs')
    client = client.with_options(max_retries=0, timeout=900)
    tool = {'type': 'image_generation', 'model': model, 'action': 'edit',
            'quality': 'high', 'output_format': 'png'}
    # ChatGPT-backed relays may silently replace the requested image model.
    # Check without generating images before purchasing a six-image response.
    preflight = client.responses.create(
        model=REASONING_MODEL, reasoning={'effort': 'high'},
        input='Reply OK only. Do not generate any images.', tools=[tool],
        tool_choice='none', store=False, timeout=60,
    ).model_dump(mode='json')
    atomic_json(response_path.with_name('preflight.json'), preflight)
    check_image_model(preflight, model)
    instructions = (
        'Repair all six supplied views of the SAME physical 3D scene as a consistent set. '
        'Study all six before editing. ' + prompt + ' '
        'Keep object identity, materials, colors, geometry and lighting consistent wherever '
        'views overlap, while preserving each exact camera. Produce SIX SEPARATE full-frame '
        'high-quality images. Choose the highest supported image quality for EVERY view; '
        'do not reduce quality on later views to save time. Make '
        'exactly one image_generation call per target view. No collage, grid, text '
        'labels or alternate variations. In each image_generation call, START the revised '
        'prompt with TARGET_VIEW= followed by the corresponding three-digit view ID, then '
        'describe that target and shared scene constraints. The other five images are scene '
        'references, not additional panels. Return all six repaired images in this response. '
        'View IDs: ' + ', '.join(t['id'] for t in targets)
    )
    content = [{'type': 'input_text', 'text': instructions}]
    for target in targets:
        # Normalize the input encoding, without resizing or modifying its pixels.
        with Image.open(target['path']) as source:
            buf = io.BytesIO()
            source.convert('RGB').save(buf, format='PNG')
        content += [
            {'type': 'input_text', 'text':
             f"Target view {target['id']} (exact camera and framing), "
             f"{target['width']}x{target['height']}:"},
            {'type': 'input_image', 'detail': 'high', 'image_url':
             'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')},
        ]
    response = client.responses.create(
        model=REASONING_MODEL, reasoning={'effort': 'high'},
        input=[{'role': 'user', 'content': content}],
        tools=[tool],
        # CliRelay's ChatGPT transport rejects max_tool_calls. Enforce the
        # requested count in the prompt and strictly validate the returned set.
        tool_choice={'type': 'image_generation'},
        parallel_tool_calls=True, store=False,
    )
    payload = response.model_dump(mode='json')
    atomic_json(response_path, payload)
    return validate_response(payload, [t['id'] for t in targets], expected_model=model)


def validate_response(payload, view_ids, *, expected_model=None):
    """Reject missing, ambiguous, duplicate or failed outputs; never guess order."""
    if payload.get('status') != 'completed':
        raise ValueError('Shared repair response did not complete; existing repairs were kept')
    if expected_model is not None:
        check_image_model(payload, expected_model)
    calls = [o for o in payload.get('output', []) if o.get('type') == 'image_generation_call']
    if len(calls) != len(view_ids):
        raise ValueError(f'Expected {len(view_ids)} separate repairs, received {len(calls)}; '
                         'existing repairs were kept')
    mapped, hashes = {}, set()
    for call in calls:
        match = re.match(r'^\s*TARGET_VIEW=([0-9]{3})(?![0-9])', call.get('revised_prompt') or '')
        if not match or match[1] not in view_ids or match[1] in mapped:
            raise ValueError('Missing or ambiguous target view label in image tool response')
        if call.get('status') != 'completed' or not call.get('result'):
            raise ValueError(f'Image repair for view {match[1]} did not complete')
        data = base64.b64decode(call['result'], validate=True)
        # Pixel hashes also catch identical images encoded differently.
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert('RGB')
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Shared repair returned duplicate images for different cameras')
        hashes.add(digest)
        mapped[match[1]] = {'image': data, 'call_id': call.get('id'),
                           'revised_prompt': call['revised_prompt'],
                           'reported_quality': call.get('quality'), 'reported_size': call.get('size')}
    return mapped
