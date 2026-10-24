"""Opt-in sequential repairs. GPU workers exchange files, never relay credentials."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import time
import uuid


def enabled(options):
    value = options.get('handoff_generation', False)
    if type(value) is not bool:
        raise ValueError('handoff_generation must be a boolean')
    return value


def backward_enabled(options):
    value = options.get('handoff_backward_pass', False)
    if type(value) is not bool:
        raise ValueError('handoff_backward_pass must be a boolean')
    return value


def context_mode(options):
    value = options.get('gpt_image_context', 'single_last_frame')
    if value not in ('single_last_frame', 'closest_repaired_image', 'initial_repaired_image', 'original_rgb_capture'):
        raise ValueError('gpt_image_context must be single_last_frame, closest_repaired_image, initial_repaired_image or original_rgb_capture')
    return value


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repair_boundary(root, source, frame_index, camera, *, phase, reference=None, reference_frame_index=None, noise=None, reference_context='closest_repaired_image'):
    """Block until the manager commits a repair, retaining the exact camera/index."""
    from PIL import Image
    mailbox = Path(os.environ['SPLATFIX_HANDOFF_MAILBOX'])
    ident = uuid.uuid4().hex
    folder = mailbox / ident
    folder.mkdir(parents=True)
    shutil.copy2(source, folder / 'before.png')
    request = {'id': ident, 'frame_index': frame_index, 'camera': camera,
               'phase': phase, 'input_sha256': digest(source)}
    if noise is not None:
        if __package__:
            from .handoff_noise import noise_options
        else:
            from handoff_noise import noise_options
        request.update(noise_options(noise))
    if reference is not None:
        shutil.copy2(reference, folder / 'reference.png')
        request.update(gpt_image_context=context_mode({'gpt_image_context': reference_context}),
                       reference_frame_index=reference_frame_index,
                       reference_sha256=digest(folder / 'reference.png'))
    print(f'HANDOFF_REPAIR_WAIT: {phase} frame {frame_index}', flush=True)
    write_json(mailbox / 'request.json', request)
    response = folder / 'response.json'
    deadline = time.monotonic() + 1800
    while not response.is_file():
        if (mailbox.parent / 'STOP').exists():
            raise InterruptedError('Stopped waiting for handoff repair')
        if time.monotonic() >= deadline:
            raise TimeoutError('Timed out waiting for handoff repair')
        time.sleep(.25)
    result = json.loads(response.read_text())
    if result.get('error'):
        raise RuntimeError(result['error'])
    fixed = folder / 'repaired.png'
    if (result.get('id') != ident or result.get('input_sha256') != request['input_sha256']
            or result.get('sha256') != digest(fixed)):
        raise ValueError('Handoff repair identity/checksum mismatch')
    with Image.open(source) as original, Image.open(fixed) as edited:
        if original.size != edited.size:
            raise ValueError('Handoff repair dimensions changed')
    archive = Path(root) / 'handoffs' / phase / f'{frame_index:05d}'
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('before.png', 'repaired.png', 'response.image', 'response.json'):
        shutil.copy2(folder / name, archive / name)
    if (folder / 'repair-input.png').is_file():
        shutil.copy2(folder / 'repair-input.png', archive / 'repair-input.png')
    if (folder / 'reference.png').is_file():
        shutil.copy2(folder / 'reference.png', archive / 'reference.png')
    if (folder / 'response-original.png').is_file():
        shutil.copy2(folder / 'response-original.png', archive / 'response-original.png')
    write_json(archive / 'camera.json', request)
    # Replace in-place: no duplicate camera, no additional index or gap.
    shutil.copy2(fixed, source)
    print(f'HANDOFF_REPAIR_READY: {phase} frame {frame_index}', flush=True)
    return archive / 'repaired.png'


def edit_request(folder, request, cfg=None, backend=None):
    """Use Step 3's backend/prompt with the requested 2.5 model; retain raw bytes."""
    from PIL import Image, ImageOps
    from ..image_repair import DEFAULT_PROMPT
    from ...image_edit_gpt import GptImageEditBackend
    from ...image_edit import SCENE_RUN_GPT_IMAGE_MODEL
    mode = context_mode(request)
    from .handoff_noise import noise_options, add_gaussian_noise, sprinkle_repaired_pixels
    noise = noise_options(request)
    sprinkle = noise['handoff_add_noise'] and noise['handoff_noise_mode'] == 'repaired_pixel_sprinkle'
    attach_reference = mode != 'single_last_frame' and not sprinkle
    prompt = DEFAULT_PROMPT
    if attach_reference:
        prompt += ' Align color grading and visual quality details with the reference image in the second attachment.'
    result = {'id': request['id'], 'input_sha256': request['input_sha256'],
              'model': SCENE_RUN_GPT_IMAGE_MODEL, 'prompt': prompt, 'gpt_image_context': mode,
              'reference_attached': attach_reference, 'attachment_count': 2 if attach_reference else 1}
    try:
        if digest(folder / 'before.png') != request['input_sha256']:
            raise ValueError('Handoff input checksum mismatch')
        if mode != 'single_last_frame' and digest(folder / 'reference.png') != request['reference_sha256']:
            raise ValueError('Handoff reference checksum mismatch')
        repair_input = folder / 'before.png'
        if noise['handoff_add_noise']:
            repair_input = folder / 'repair-input.png'
            options = {'strength': noise['handoff_noise_strength'],
                       'seed': int(request['input_sha256'][:16], 16),
                       'vignette': noise['handoff_noise_vignette'],
                       'vignette_size': noise['handoff_vignette_size'],
                       'edge_strength': noise['handoff_vignette_edge_strength']}
            result['noise'] = (sprinkle_repaired_pixels(folder / 'before.png', folder / 'reference.png', repair_input, **options)
                               if sprinkle else add_gaussian_noise(folder / 'before.png', repair_input, **options))
            if sprinkle:
                result['noise']['reference_sha256'] = request['reference_sha256']
            result['noise']['enabled'] = True
            result['noise']['input_sha256'] = digest(repair_input)
        else:
            result['noise'] = {'enabled': False}
        raw = folder / 'response.image'
        if not raw.exists():
            backend = backend or GptImageEditBackend.from_config(cfg, model=SCENE_RUN_GPT_IMAGE_MODEL)
            answer = (backend.edit_with_references(repair_input, [folder / 'reference.png'], prompt)
                      if attach_reference else backend.edit(repair_input, prompt))
            if answer.error or not answer.images:
                raise RuntimeError(answer.error or 'Handoff repair returned no image')
            temporary = raw.with_suffix('.tmp')
            temporary.write_bytes(answer.images[0])
            temporary.replace(raw)
        with Image.open(folder / 'before.png') as original, Image.open(raw) as edited:
            edited.convert('RGB').save(folder / 'response-original.png')
            # Match prepare_repair_checkpoint: preserve full response first,
            # then fit to the already resolution-prepared boundary's canvas.
            width, height = original.size
            scale = max(width / edited.width, height / edited.height)
            crop_width, crop_height = width / scale, height / scale
            left, top = (edited.width - crop_width) / 2, (edited.height - crop_height) / 2
            result['reconstruction_resize'] = {
                'source': 'response.image', 'source_kind': 'full_response',
                'method': 'cover_center_crop', 'target_size': [width, height],
                'source_size': list(edited.size), 'scale': scale,
                'source_crop_box': [left, top, left + crop_width, top + crop_height]}
            ImageOps.fit(edited.convert('RGB'), original.size, Image.Resampling.LANCZOS,
                         centering=(.5, .5)).save(folder / 'repaired.png')
        result['sha256'] = digest(folder / 'repaired.png')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    write_json(folder / 'response.json', result)


class HandoffService:
    """Nonblocking manager service; cancellation/deadline polling remains active."""
    def __init__(self, root, *, cfg=None, remote=None, endpoint=None, ssh=None, sync=None):
        from concurrent.futures import ThreadPoolExecutor
        self.root = Path(root)
        self.cfg, self.remote, self.endpoint, self.ssh, self.sync = cfg, remote, endpoint, ssh, sync
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.done = set()

    def poll(self):
        import shlex
        if self.pending:
            ident, future = self.pending
            if not future.done():
                return
            future.result()
            folder = self.root / ident
            if self.remote:
                target = self.endpoint + ':' + self.remote + '/' + ident + '/'
                for name in ('response.image', 'response-original.png', 'repaired.png', 'repair-input.png'):
                    if (folder / name).exists():
                        self.sync(str(folder / name), target + name)
                self.sync(str(folder / 'response.json'), target + 'response.json')
            self.done.add(ident)
            self.pending = None
        if self.remote:
            path = shlex.quote(self.remote + '/request.json')
            raw = self.ssh(f'if [ -f {path} ]; then cat {path}; fi')
        else:
            path = self.root / 'request.json'
            raw = path.read_text() if path.exists() else ''
        if not raw.strip():
            return
        request = json.loads(raw)
        ident = request.get('id', '')
        if not isinstance(ident, str) or not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Invalid handoff request id')
        if ident in self.done:
            return
        folder = self.root / ident
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'request.json', request)
        if self.remote:
            marker = shlex.quote(self.remote + '/' + ident + '/response.json')
            if self.ssh(f'if [ -f {marker} ]; then cat {marker}; fi').strip():
                self.done.add(ident)
                return
            self.sync(self.endpoint + ':' + self.remote + '/' + ident + '/before.png', str(folder / 'before.png'))
        if self.remote and context_mode(request) != 'single_last_frame':
            self.sync(self.endpoint + ':' + self.remote + '/' + ident + '/reference.png', str(folder / 'reference.png'))
        if (folder / 'response.json').is_file():
            from concurrent.futures import Future
            future = Future()
            future.set_result(None)
        else:
            future = self.pool.submit(edit_request, folder, request, self.cfg)
        self.pending = (ident, future)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
