"""File handshake between disposable GPU inference and the local GPT relay.

Only the original rendered view and the original anchor are sent to GPT.
No credentials or network client are needed in the ArtiFixer environment.
"""
from pathlib import Path
import json
import time


REFERENCE_INSTRUCTION = (
    '\nThe FIRST image is the degraded render to repair. Preserve its camera, '
    'perspective, framing, and scene geometry. The SECOND image is the original '
    'repaired anchor: use it only as a visual reference for consistent materials, '
    'colors, lighting, and repair details. Return only the repaired FIRST view; '
    'do not copy the anchor viewpoint or create a collage.'
)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class PeriodicRefresh:
    def __init__(self, root, indices):
        self.root = Path(root)
        self.indices = indices
        self.records = []
        self.prepared_references = []

    def prepare(self):
        """Finish every real (unpadded) planned repair before inference starts."""
        self.prepared_references = [self.prepare_frame(frame)
                                    for frame in range(20, len(self.indices), 20)]
        return self.prepared_references

    def prepare_frame(self, frame):
        from PIL import Image
        index = self.indices[frame]
        folder = self.root / 'refresh' / f'{index:05d}'
        folder.mkdir(parents=True, exist_ok=True)
        # Reuse the immutable bundle inputs; refresh folders contain outputs only.
        source_path = self.root / 'inputs' / f'{index:05d}.png'
        # Publish after bundle rendering has finished.
        write_json(self.root / 'refresh-request.json', {'frame_index': index})
        response = folder / 'response.json'
        until = time.monotonic() + 1800
        while not response.is_file():
            if time.monotonic() >= until:
                raise TimeoutError('Timed out waiting for periodic GPT-image repair')
            # Parent propagate() enforces request cancellation/deadline by
            # terminating this disposable process, including during this wait.
            time.sleep(.25)
        result = json.loads(response.read_text())
        if result.get('status') != 'ok':
            raise RuntimeError(result.get('error') or 'Periodic GPT-image repair failed')
        with Image.open(source_path) as source, Image.open(folder / 'regenerated.png') as edited:
            if source.width * edited.height != source.height * edited.width:
                raise ValueError('Periodic repair changed camera aspect ratio')
            fixed = edited.convert('RGB').resize(source.size, Image.Resampling.LANCZOS)
            fixed.save(folder / 'fixed.png')
        return {'frame_index': index, 'path': f'refresh/{index:05d}/fixed.png',
                'kind': 'edited_render', 'reference_encoding': 'independent_image',
                'reference_camera': 'exact_rgb_pose', 'available_from': 'generation_start'}

    def __call__(self, pipe, prefix, frame):
        import numpy as np
        import torch
        from PIL import Image
        index = self.indices[frame]
        matches = [r for r in self.prepared_references if r['frame_index'] == index]
        if len(matches) != 1:
            raise ValueError('Periodic replacement requires a prepared anchor')
        reference = matches[0]
        folder = self.root / 'refresh' / f'{index:05d}'
        with Image.open(self.root / reference['path']) as fixed:
            seed = torch.from_numpy(np.array(fixed.convert('RGB'))).permute(2, 0, 1).float() / 255
        # decode_latents_to_video clears ALL inference caches upstream. Use the
        # underlying VAE decoder to preserve temporal and reference KV memory.
        video = pipe.video_processor.postprocess_video(pipe.latents_to_rgb(prefix), output_type='pt')
        if video.shape[1] != frame + 1 or video.shape[2:] != seed.shape:
            raise ValueError('Periodic repair decoded an unexpected video shape')
        before = video[0, -1].detach().float().cpu()
        Image.fromarray(before.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()).save(folder / 'before.png')
        video[:, -1] = seed.to(device=video.device, dtype=video.dtype)
        encoded = pipe.encode_video_frames(video.to(pipe.vae.device))
        if encoded.shape != prefix.shape:
            raise ValueError('Periodic repair VAE changed latent dimensions')
        self.records.append({**reference, 'rgb_interval': 20,
                             'cache': 'clean corrected block; original opacity and cameras'})
        return encoded[:, :, -1:].to(device=prefix.device, dtype=prefix.dtype)


def service_refresh(transport, request_id, *, deadline, should_stop):
    """Service at most one GPU request on the host that can reach CliRelay."""
    from .. import repair_lrz
    if should_stop() or time.time() >= deadline:
        return
    pending = transport._remote_json(f'requests/{request_id}/extended/refresh-request.json')
    if not pending:
        return
    index = pending.get('frame_index')
    if type(index) is not int or not 0 < index < 100000:
        raise ValueError('Invalid periodic repair frame index')
    key = (request_id, index)
    completed = getattr(transport, '_completed_refreshes', set())
    if key in completed:
        return
    local_request = transport.run_dir / 'requests' / request_id
    folder = local_request / 'extended' / 'refresh' / f'{index:05d}'
    folder.mkdir(parents=True, exist_ok=True)
    remote = f"{transport.cfg['user']}@{transport.cfg['host']}"
    remote_folder = f'{transport.remote_dir}/requests/{request_id}/extended/refresh/{index:05d}/'
    def transfer(arguments):
        result = repair_lrz._mux_run(['rsync', '-az', '-e', repair_lrz.rsync_ssh_cmd(transport.cfg), *arguments])
        if result.returncode:
            raise RuntimeError('Periodic repair image transfer failed')
    bundle = local_request / 'extended'
    source = bundle / 'inputs' / f'{index:05d}.png'
    anchor = bundle / 'anchor.png'
    remote_bundle = f'{transport.remote_dir}/requests/{request_id}/extended'
    response = folder / 'response.json'
    # A completed local result can be re-uploaded after a transfer failure
    # without purchasing another GPT edit.
    if not response.is_file():
        try:
            # Each immutable input is downloaded once per repair bundle. Never
            # use a prior scene's render after fitting has changed the splat.
            for image, relative in ((source, f'inputs/{index:05d}.png'), (anchor, 'anchor.png')):
                if not image.is_file():
                    image.parent.mkdir(parents=True, exist_ok=True)
                    transfer([f'{remote}:{remote_bundle}/{relative}', str(image)])
            request = json.loads((local_request / 'request.json').read_text())
            prompt = str(request.get('prompt') or transport._run_config.get('image_edit_prompt') or '')
            transport._progress('image_edit', f'Periodic GPT repair at frame {index + 1}')
            payload = transport._edit_with_clirelay(source, folder / 'regenerated.png',
                                                   prompt + REFERENCE_INSTRUCTION,
                                                   references=[anchor])
            if should_stop() or time.time() >= deadline:
                raise InterruptedError('Stopped during periodic image repair')
            write_json(response, {'status': 'ok', 'frame_index': index, 'image_edit': payload})
        except Exception as exc:
            write_json(response, {'status': 'error', 'frame_index': index, 'error': str(exc)})
    # Publish response separately after image upload; GPU waits on this file.
    if (folder / 'regenerated.png').is_file():
        transfer([str(folder / 'regenerated.png'), f'{remote}:{remote_folder}'])
    transfer([str(response), f'{remote}:{remote_folder}'])
    completed.add(key)
    transport._completed_refreshes = completed
