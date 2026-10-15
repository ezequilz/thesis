import pytest
from splat_explorer.splatfix.jobs import validate_job
from splat_explorer.splatfix.g4splat.execution import job_runtime
from splat_explorer.splatfix.g4splat.baselines import baseline_catalog


def test_guidance_defaults_and_runtime_override():
    raw = {'stage': 'repair', 'checkpoint': 'saved', 'reconstruction_method': 'g4splat'}
    assert validate_job(raw)['source_depth'] is False
    for value in (False, True):
        options = validate_job({**raw, 'source_depth': value})
        runtime = job_runtime({'splatfix': {'g4splat_runtime': {'source_depth': not value}}}, options)
        assert runtime['source_depth'] is value
    assert job_runtime({'splatfix': {'g4splat_runtime': {'source_depth': True}}},
                       {'resolution_profile': 'training'})['source_depth'] is True


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None, {}, []])
def test_guidance_rejects_non_boolean(value):
    with pytest.raises(ValueError, match='boolean'):
        validate_job({'stage': 'repair', 'checkpoint': 'saved', 'reconstruction_method': 'g4splat', 'source_depth': value})


def test_author_baselines_are_selectable_but_not_false_ready(tmp_path):
    rows = baseline_catalog(tmp_path, str)
    assert [r['id'] for r in rows] == ['scene1', 'scene4']
    assert all(not r['ready'] and r['reason'] and not r['audit_url'] for r in rows)
    evidence = tmp_path/'g4splat-published-baseline/evidence'
    evidence.mkdir(parents=True)
    (evidence/'scene1_thumbnail.png').write_bytes(b'preview')
    (evidence.parent/'manifest.json').write_text('{}')
    rows = baseline_catalog(tmp_path, str)
    assert rows[0]['thumbnail_url'] and rows[0]['audit_url']
    assert not rows[1]['thumbnail_url']
    assert all(not r['ready'] for r in rows)  # Reference media is not training data.
