import pytest


@pytest.fixture(autouse=True)
def no_detached_download_processes(monkeypatch):
    """Unit tests explicitly drive transfers without starting persistent workers."""
    monkeypatch.setattr('splat_explorer.splatfix.downloads.ensure_worker', lambda root: None)
