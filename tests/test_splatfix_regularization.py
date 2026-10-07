import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from splat_explorer.splatfix import official_worker as worker
from splat_explorer.splatfix.jobs import validate_job


def test_job_regularization_selection():
    for stage, extra in [('repair', {'checkpoint': 'saved'}), ('benchmark', {'source': 'dataset'})]:
        assert validate_job({'stage': stage, **extra})['regularization_profile'] == 'artifixer'
        assert validate_job({'stage': stage, **extra, 'regularization_profile': 'base_mcmc'})['regularization_profile'] == 'base_mcmc'
        with pytest.raises(ValueError, match='regularization_profile'):
            validate_job({'stage': stage, **extra, 'regularization_profile': 'typo'})


@pytest.mark.parametrize('override,lpips', [(True, True), (False, True), (True, False)])
def test_regularization_added_once(override, lpips):
    class Upstream:
        conf = SimpleNamespace(loss={'use_lpips_override': lpips})
        def get_losses(self, batch, outputs):
            return dict(total_loss=10.0 if override and lpips else 10.7,
                        opacity_loss=0.2, scale_loss=0.5)
    result = worker.regularized_trainer_class(Upstream)().get_losses(SimpleNamespace(is_override=override), {})
    assert result['total_loss'] == pytest.approx(10.7)


@pytest.mark.parametrize('profile', ['artifixer', 'base_mcmc'])
def test_config_inheritance_is_scoped_and_preserves_other_settings(monkeypatch, profile):
    sparse = SimpleNamespace(loss=dict(use_opacity=False, lambda_opacity=0.0, use_scale=False,
        lambda_scale=0.0, lambda_lpips_override=0.1), model={'density': 0.1}, schedule={'stop': 25000})
    base = SimpleNamespace(loss=dict(use_opacity=True, lambda_opacity=0.037, use_scale=True, lambda_scale=0.029))
    calls = []
    def compose(name, overrides, directory):
        calls.append(name)
        return copy.deepcopy(base if name == 'base_mcmc' else sparse)
    training = ModuleType('data_processing.threedgrut_training')
    training.compose_3dgrut_config, training.Trainer3DGRUT = compose, object
    package = ModuleType('data_processing')
    package.threedgrut_training = training
    monkeypatch.setitem(sys.modules, 'data_processing', package)
    monkeypatch.setitem(sys.modules, training.__name__, training)
    with pytest.raises(RuntimeError, match='training failed'):
        with worker.reconstruction_settings(profile):
            config = training.compose_3dgrut_config(worker.ARTIFIXER3D_CONFIG, [], '/pinned')
            assert config.loss == ({**sparse.loss, **base.loss} if profile == 'base_mcmc' else sparse.loss)
            assert config.model == sparse.model and config.schedule == sparse.schedule
            raise RuntimeError('training failed')
    assert training.compose_3dgrut_config is compose and training.Trainer3DGRUT is object
    assert calls == [worker.ARTIFIXER3D_CONFIG] + (['base_mcmc'] if profile == 'base_mcmc' else [])
