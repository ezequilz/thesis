"""The optional image-only pass must not gate the repaired PLY result."""
import argparse
import json

import pytest

from splat_explorer.splatfix.artifixer.inference_policy import MODELS, model_runtime, new_job_options


@pytest.mark.parametrize('model', ['1.3b', '14b', '14b-teacher'])
def test_new_jobs_skip_plus_and_saved_requests_keep_their_policy(model):
    assert new_job_options({'inference_model': model})['artifixer_plus_pass'] is False
    assert new_job_options({'inference_model': model, 'artifixer_plus_pass': True})['artifixer_plus_pass'] is True
    saved = {'inference_model': model, 'checkpoint': '/models/' + MODELS[model][0],
             'model_id': MODELS[model][1], 'teacher_inference': model == '14b-teacher'}
    assert model_runtime(saved)['artifixer_plus_pass'] is True
    assert model_runtime({**saved, 'artifixer_plus_pass': False})['artifixer_plus_pass'] is False


@pytest.mark.parametrize('bad', [None, 0, 1, 'false'])
def test_plus_setting_requires_a_boolean(bad):
    with pytest.raises(ValueError, match='artifixer_plus_pass'):
        new_job_options({'artifixer_plus_pass': bad})


def test_stored_history_does_not_acquire_the_new_skip_default(tmp_path):
    from splat_explorer.scene_runs.store import SceneRunStore
    store = SceneRunStore(tmp_path)
    job = store.create_run({'pipeline': 'splatfix', 'splatfix': {
        'stage': 'repair', 'checkpoint': '/saved/cp', 'reconstruction_method': 'artifixer'}})
    assert store.get_run(job.run_id).config.splatfix['artifixer_plus_pass'] is False
    path = tmp_path / job.run_id / 'config.json'
    historical = json.loads(path.read_text())
    del historical['splatfix']['artifixer_plus_pass']
    path.write_text(json.dumps(historical))
    assert store.get_run(job.run_id).config.splatfix['artifixer_plus_pass'] is True
    assert 'artifixer_plus_pass' not in json.loads(path.read_text())['splatfix']


@pytest.mark.parametrize('flags,expected', [([], False), (['--artifixer-plus-pass'], True),
                                          (['--no-artifixer-plus-pass'], False)])
def test_cli_passes_effective_plus_setting(tmp_path, monkeypatch, flags, expected):
    from splat_explorer.config import Config
    from splat_explorer.splatfix import repair
    from splat_explorer.splatfix.cli import add_parser, repair_splat
    received = {}
    def run(*args, **kwargs):
        received.update(kwargs['runtime'])
        return {}
    monkeypatch.setattr(repair, 'run_repair', run)
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(['splatfix', 'repair', str(tmp_path), '--output', str(tmp_path / 'out'),
                              '--mode', 'baseline', *flags])
    repair_splat(Config({}), args)
    assert received['artifixer_plus_pass'] is expected
