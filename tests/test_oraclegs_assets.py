import hashlib
from pathlib import Path
import pytest
from splat_explorer.splatfix.oraclegs import model_assets as assets


def test_vae_download_verifies_both_files(tmp_path, monkeypatch):
    files = {'vae/config.json': b'config', 'vae/diffusion_pytorch_model.safetensors': b'weights'}
    monkeypatch.setitem(assets.VAE, 'files', {k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    def download(**kw):
        assert kw['revision'] == assets.VAE['revision']
        p=tmp_path/kw['filename']; p.parent.mkdir(exist_ok=True);p.write_bytes(files[kw['filename']]);return str(p)
    assert assets.verified_vae(download) == tmp_path
    files['vae/diffusion_pytorch_model.safetensors'] = b'wrong weights'
    with pytest.raises(ValueError, match='checksum mismatch'):
        assets.verified_vae(download)


def test_vae_redirect_is_scoped_and_restored_on_failure(tmp_path):
    class Base:
        @classmethod
        def from_pretrained(cls, model, **kw):
            return model,kw
    class VAE(Base):
        pass
    with pytest.raises(RuntimeError):
        with assets.vae_source(VAE,tmp_path):
            model,kw=VAE.from_pretrained(assets.VAE['original_repo'],subfolder='vae',low_cpu_mem_usage=False)
            assert model == str(tmp_path) and kw['local_files_only'] and kw['use_safetensors']
            assert kw['low_cpu_mem_usage'] is False
            assert VAE.from_pretrained('other')[0] == 'other'
            raise RuntimeError('failed model load')
    assert 'from_pretrained' not in VAE.__dict__
    assert VAE.from_pretrained(assets.VAE['original_repo'])[0] == assets.VAE['original_repo']


@pytest.mark.parametrize('status', [401,403])
def test_model_access_failure_is_actionable_and_does_not_echo_http_details(status):
    from types import SimpleNamespace
    class Denied(Exception):
        response=SimpleNamespace(status_code=status)
    def download(**kw):
        raise Denied('private HTTP details')
    with pytest.raises(RuntimeError, match='huggingface-cli login') as error:
        assets.check_seva_access(download)
    assert 'private HTTP details' not in str(error.value)


def test_model_access_keeps_network_errors_distinct():
    def download(**kw):
        raise TimeoutError('network timeout')
    with pytest.raises(TimeoutError):
        assets.check_seva_access(download)
