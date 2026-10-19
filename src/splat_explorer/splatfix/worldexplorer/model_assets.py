"""Pinned recovery of the authors' unavailable SD2 inpainting checkpoint."""
import hashlib
import json
from pathlib import Path

INPAINTING = json.loads(Path(__file__).with_name('inpainting-lock.json').read_text())


def verified_inpainting(download=None, cache=None):
    if download is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import HF_HUB_CACHE
        download = hf_hub_download
        cache = Path(HF_HUB_CACHE)
    root = None
    for name, expected in INPAINTING['files'].items():
        # Reuse identical blobs already cached by OracleGS, notably the SD2 VAE.
        if cache is not None and name.endswith('.safetensors'):
            target = Path(cache)/('models--'+INPAINTING['repo'].replace('/', '--'))/'snapshots'/INPAINTING['revision']/name
            if not target.exists():
                existing = next((p for p in Path(cache).glob('models--*/blobs/'+expected) if p.is_file()), None)
                if existing is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        target.symlink_to(existing.resolve())
                    except FileExistsError:
                        pass
        path = Path(download(repo_id=INPAINTING['repo'], revision=INPAINTING['revision'], filename=name))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Inpainting checkpoint checksum mismatch: {name}')
        candidate = path.parent if '/' not in name else path.parent.parent
        if root is not None and root != candidate:
            raise ValueError('Inpainting files must belong to one snapshot')
        root = candidate
    return root


def inpainting_loader(pipeline, root):
    """A local facade avoids globally changing Diffusers' class methods."""
    class Loader:
        @staticmethod
        def from_pretrained(model, *args, **kwargs):
            if model == INPAINTING['original_repo']:
                model = str(root)
                kwargs.update(local_files_only=True, use_safetensors=True)
            return pipeline.from_pretrained(model, *args, **kwargs)
    return Loader
