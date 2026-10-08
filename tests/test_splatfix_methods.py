"""Method dispatch must isolate backends and preserve legacy saved jobs."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from splat_explorer.splatfix import methods
from splat_explorer.splatfix.executor import SplatfixExecutor
from splat_explorer.splatfix.job_worker import execute
from splat_explorer.splatfix.jobs import validate_job


@pytest.mark.parametrize('method', ['', 'unknown', '../repair', [], 3])
def test_unknown_method_rejected_before_queueing(method):
    with pytest.raises(ValueError, match='Unknown reconstruction method'):
        validate_job({'stage': 'repair', 'checkpoint': 'saved', 'reconstruction_method': method})


def test_legacy_defaults_and_import_identity():
    job = validate_job({'stage': 'repair', 'checkpoint': 'saved'})
    assert job['reconstruction_method'] == 'artifixer'
    assert methods.get_method() is methods.get_method('artifixer')
    assert importlib.import_module('splat_explorer.splatfix.repair') is importlib.import_module(
        'splat_explorer.splatfix.artifixer.repair')
    assert job['split_mode'] == 'double-split'
    assert job['regularization_profile'] == 'artifixer'


def test_registered_backend_receives_validation_remote_and_worker(tmp_path, monkeypatch):
    calls = []
    backend = SimpleNamespace(
        validate_options=lambda raw: {'custom_option': raw['custom_option']},
        execute_remote=lambda *args: calls.append(('remote', args)) or {'results': 'custom'},
        execute_worker=lambda root: calls.append(('worker', root)),
    )
    monkeypatch.setitem(sys.modules, 'test_backend', backend)
    monkeypatch.setitem(methods.METHODS, 'test', {'module': 'test_backend', 'label': 'Test'})
    options = validate_job({'stage': 'repair', 'checkpoint': 'saved', 'mode': 'baseline',
                            'reconstruction_method': 'test', 'custom_option': 7})
    assert options['custom_option'] == 7
    assert 'split_mode' not in options and 'frames' not in options and 'regularization_profile' not in options
    executor = SplatfixExecutor({}, None)
    result = executor._remote('run', options, tmp_path, lambda: False, lambda **kw: None)
    assert result == {'results': 'custom'}
    assert calls[0][1][2] is options
    (tmp_path / 'worker-request.json').write_text(json.dumps(options))
    execute(tmp_path)
    assert calls[1] == ('worker', tmp_path)
    with pytest.raises(ValueError, match='does not support'):
        validate_job({'stage': 'benchmark', 'source': 'saved', 'reconstruction_method': 'test'})


def test_unknown_worker_method_publishes_error(tmp_path):
    (tmp_path / 'worker-request.json').write_text(json.dumps({'reconstruction_method': 'missing'}))
    execute(tmp_path)
    status = json.loads((tmp_path / 'worker-status.json').read_text())
    assert status['status'] == 'error'
    assert 'Unknown reconstruction method' in status['message']


@pytest.mark.parametrize('selection', [{}, {'reconstruction_method': 'artifixer'}])
def test_worker_dispatch_preserves_legacy_requests(tmp_path, monkeypatch, selection):
    calls = []
    monkeypatch.setattr(methods.get_method(), 'execute_worker', calls.append)
    (tmp_path / 'worker-request.json').write_text(json.dumps(selection))
    execute(tmp_path)
    assert calls == [tmp_path]


@pytest.mark.parametrize('value', [False, True])
def test_image_cache_insertion_setting(value):
    from splat_explorer.splatfix.artifixer.backend import validate_options
    assert validate_options({'stage': 'repair'})['image_cache_insertion'] is False
    assert validate_options({'stage': 'repair', 'image_cache_insertion': value})['image_cache_insertion'] is value


@pytest.mark.parametrize('value', ['false', 'on', 1, None])
def test_image_cache_insertion_rejects_non_boolean(value):
    from splat_explorer.splatfix.artifixer.backend import validate_options
    with pytest.raises(ValueError, match='image_cache_insertion'):
        validate_options({'stage': 'repair', 'image_cache_insertion': value})
