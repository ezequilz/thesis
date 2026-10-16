"""Content-pinned recovery of SEVA's unavailable SD 2.1 VAE dependency."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


VAE = {
    'original_repo': 'stabilityai/stable-diffusion-2-1-base',
    'repo': 'sd2-community/stable-diffusion-2-1-base',
    'revision': '4e63672c03103b6c636b8fb4119ba982469b2955',
    'files': {
        'vae/config.json': '424117cb534ce03497c41305ed868980123917b2b6abba4bbaa615e968772903',
        'vae/diffusion_pytorch_model.safetensors': 'a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815',
    },
}


def check_seva_access(download=None):
    """Fail before geometry/CLIP downloads when the official model is gated."""
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    try:
        return download(repo_id='stabilityai/stable-virtual-camera', filename='config.yaml')
    except Exception as exc:
        # Avoid exposing HTTP request details or authentication material in status.
        response = getattr(exc, 'response', None)
        if getattr(response, 'status_code', None) in (401, 403):
            raise RuntimeError(
                'SEVA model access denied. Accept access at '
                'https://huggingface.co/stabilityai/stable-virtual-camera and '
                'authenticate the model environment with huggingface-cli login. '
                'Do not put tokens in run configuration or logs.') from None
        raise


def verified_vae(download=None):
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    root = None
    for name, expected in VAE['files'].items():
        path = Path(download(repo_id=VAE['repo'], revision=VAE['revision'], filename=name))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'SEVA VAE checksum mismatch: {name}')
        candidate = path.parent.parent
        if root is not None and root != candidate:
            raise ValueError('SEVA VAE files must belong to the same snapshot')
        root = candidate
    return root


@contextmanager
def vae_source(autoencoder_class, root):
    """Redirect only this exact model lookup; retain upstream encode/decode code."""
    original = autoencoder_class.from_pretrained
    descriptor = autoencoder_class.__dict__.get('from_pretrained')
    def load(cls, model, *args, **kwargs):
        if model == VAE['original_repo'] and kwargs.get('subfolder') == 'vae':
            model = str(root)
            kwargs.update(local_files_only=True, use_safetensors=True)
        return original(model, *args, **kwargs)
    autoencoder_class.from_pretrained = classmethod(load)
    try:
        yield
    finally:
        if descriptor is None:
            delattr(autoencoder_class, 'from_pretrained')
        else:
            autoencoder_class.from_pretrained = descriptor


def record_vae(output, root):
    Path(output).write_text(json.dumps({**VAE, 'snapshot': str(root),
        'verification': 'SHA-256 of config and safetensors; SD 2.1 mirror, unchanged SEVA codec'}, indent=2))
