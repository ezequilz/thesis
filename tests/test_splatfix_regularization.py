import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from splat_explorer.splatfix import official_worker as worker
from splat_explorer.splatfix.jobs import validate_job


def test_job_regularization_selection():
    for stage, extra in [('repair', {'checkpoint': 'saved', 'artifixer_pipeline': 'legacy_gsplat'}), ('benchmark', {'source': 'dataset'})]:
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


@pytest.mark.parametrize('source_ply', [None, '/original.ply'])
@pytest.mark.parametrize('profile', ['artifixer', 'base_mcmc'])
def test_config_inheritance_is_scoped_and_preserves_other_settings(monkeypatch, profile, source_ply):
    sparse = SimpleNamespace(loss=dict(use_opacity=False, lambda_opacity=0.0, use_scale=False,
        lambda_scale=0.0, lambda_lpips_override=0.1), model={'density': 0.1}, schedule={'stop': 25000}, resume='old.ckpt',
        import_ingp=SimpleNamespace(enabled=True),
        import_ply=SimpleNamespace(enabled=False, path='', init_global_step=30000))
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
        with worker.reconstruction_settings(profile, source_ply=source_ply):
            config = training.compose_3dgrut_config(worker.ARTIFIXER3D_CONFIG, [], '/pinned')
            assert config.loss == ({**sparse.loss, **base.loss} if profile == 'base_mcmc' else sparse.loss)
            if source_ply:
                assert config.resume == ''
                assert config.import_ingp.enabled is False
                assert config.import_ply.enabled is True
                assert config.import_ply.path == source_ply
                assert config.import_ply.init_global_step == 0
            else:
                assert config.import_ply.enabled is False
            assert config.model == sparse.model and config.schedule == sparse.schedule
            raise RuntimeError('training failed')
    assert training.compose_3dgrut_config is compose and training.Trainer3DGRUT is object
    assert calls == [worker.ARTIFIXER3D_CONFIG] + (['base_mcmc'] if profile == 'base_mcmc' else [])


@pytest.mark.parametrize('pipeline', ['authors_3dgut', 'authors_3dgut_viser_rgb', 'legacy_gsplat'])
@pytest.mark.parametrize('weight', [0.1, 1.0])
@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('downscale', [False, True])
def test_generated_loss_job_choices_survive_pipeline(pipeline, rgb, downscale, weight):
    options = validate_job(dict(stage='repair', checkpoint='saved', artifixer_pipeline=pipeline,
                                generated_rgb_loss=rgb, generated_lpips_downscale=downscale, generated_rgb_loss_weight=weight))
    policy = worker.validate_reconstruction_options(options)
    assert policy['generated_rgb_loss'] is rgb
    assert policy['generated_lpips_downscale'] is downscale
    recipe = worker.reconstruction_recipe(**worker.generated_loss_options(policy))
    assert recipe['lambda_reconlosses_override'] == float(rgb) * weight
    assert recipe['lambda_lpips_override'] == (0.1 if downscale else 1.0)
    assert recipe['custom_reconstruction_overrides'] == (rgb or not downscale)


def test_generated_loss_defaults_and_validation():
    options = validate_job(dict(stage='repair', checkpoint='saved'))
    assert worker.generated_loss_options(options) == dict(generated_rgb_loss=False, generated_lpips_downscale=True, generated_rgb_loss_weight=1.0)
    for key in ('generated_rgb_loss', 'generated_lpips_downscale'):
        for value in ['off', 'on', 0, 1, None]:
            with pytest.raises(ValueError, match=key):
                validate_job(dict(stage='repair', checkpoint='saved', **{key: value}))


@pytest.mark.parametrize('weight', [0.1, 1.0])
@pytest.mark.parametrize('rgb', [False, True])
@pytest.mark.parametrize('downscale', [False, True])
@pytest.mark.parametrize('profile', ['artifixer', 'base_mcmc'])
def test_generated_loss_config_and_reference_isolation(monkeypatch, rgb, downscale, profile, weight):
    # Model the pinned upstream branches with nonzero synthetic losses so that
    # toggling either weight independently changes the generated total only.
    sparse = SimpleNamespace(loss=dict(use_lpips_override=True, lambda_l1=0.8, lambda_ssim=0.2,
        lambda_l1_override=0.8, lambda_ssim_override=0.2, lambda_reconlosses_override=0.0,
        lambda_lpips_override=0.1, use_opacity=False, lambda_opacity=0., use_scale=False, lambda_scale=0.))
    base = SimpleNamespace(loss=dict(use_opacity=True, lambda_opacity=0.01, use_scale=True, lambda_scale=0.01))
    def compose(name, overrides, directory):
        return copy.deepcopy(base if name == 'base_mcmc' else sparse)
    class Trainer:
        def get_losses(self, batch, outputs):
            c = self.conf.loss
            regularization = c['lambda_opacity'] * 2 + c['lambda_scale'] * 3
            total = (c['lambda_lpips_override'] * 5 + c['lambda_reconlosses_override'] *
                     (c['lambda_l1_override'] * 2 + c['lambda_ssim_override'] * 3)) if batch.is_override else (
                     c['lambda_l1'] * 2 + c['lambda_ssim'] * 3 + regularization)
            return dict(total_loss=total, opacity_loss=c['lambda_opacity'] * 2, scale_loss=c['lambda_scale'] * 3)
    training = ModuleType('data_processing.threedgrut_training')
    training.compose_3dgrut_config, training.Trainer3DGRUT = compose, Trainer
    package = ModuleType('data_processing')
    package.threedgrut_training = training
    monkeypatch.setitem(sys.modules, 'data_processing', package)
    monkeypatch.setitem(sys.modules, training.__name__, training)
    with worker.reconstruction_settings(profile, generated_rgb_loss=rgb, generated_lpips_downscale=downscale, generated_rgb_loss_weight=weight):
        trainer = training.Trainer3DGRUT()
        trainer.conf = training.compose_3dgrut_config(worker.ARTIFIXER3D_CONFIG, [], '/pinned')
        reg = 0.05 if profile == 'base_mcmc' else 0
        assert trainer.get_losses(SimpleNamespace(is_override=True), {})['total_loss'] == pytest.approx(
            (0.5 if downscale else 5) + (2.2 * weight if rgb else 0) + reg)
        assert trainer.get_losses(SimpleNamespace(is_override=False), {})['total_loss'] == pytest.approx(2.2 + reg)
        for key in ('lambda_l1', 'lambda_ssim', 'use_lpips_override'):
            assert trainer.conf.loss[key] == sparse.loss[key]
    assert training.compose_3dgrut_config is compose
    assert training.Trainer3DGRUT is Trainer


@pytest.mark.parametrize('pipeline', ['authors_3dgut', 'authors_3dgut_viser_rgb', 'legacy_gsplat'])
def test_gaussian_limit_survives_pipeline_and_runtime_validation(pipeline):
    options = validate_job(dict(stage='repair', checkpoint='saved', artifixer_pipeline=pipeline,
                                max_gaussians=2_500_000))
    assert options['max_gaussians'] == 2_500_000
    overrides = worker.reconstruction_overrides(options)
    assert worker.validate_reconstruction_options(options)['max_gaussians'] == 2_500_000
    recipe = worker.reconstruction_recipe(**overrides)
    assert recipe['max_gaussians'] == 2_500_000
    assert recipe['custom_reconstruction_overrides'] is True
    assert validate_job(dict(stage='repair', checkpoint='saved'))['max_gaussians'] == 1_000_000


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '2000000', None])
def test_invalid_gaussian_limits_rejected(limit):
    with pytest.raises(ValueError, match='max_gaussians must be a positive integer'):
        validate_job(dict(stage='repair', checkpoint='saved', max_gaussians=limit))


@pytest.mark.parametrize('limit', [250_000, 2_000_000])
def test_gaussian_limit_reaches_training_and_restores_compose(monkeypatch, limit):
    sparse = SimpleNamespace(strategy=SimpleNamespace(add=SimpleNamespace(max_n_gaussians=1_000_000)),
                             loss={'lambda_lpips_override': 0.1})
    def compose(*args):
        return copy.deepcopy(sparse)
    training = ModuleType('data_processing.threedgrut_training')
    training.compose_3dgrut_config, training.Trainer3DGRUT = compose, object
    package = ModuleType('data_processing'); package.threedgrut_training = training
    monkeypatch.setitem(sys.modules, 'data_processing', package)
    monkeypatch.setitem(sys.modules, training.__name__, training)
    with worker.reconstruction_settings(max_gaussians=limit):
        config = training.compose_3dgrut_config(worker.ARTIFIXER3D_CONFIG, [], '/pinned')
        assert config.strategy.add.max_n_gaussians == limit
        assert config.loss == sparse.loss
    assert training.compose_3dgrut_config is compose
    assert sparse.strategy.add.max_n_gaussians == 1_000_000


@pytest.mark.parametrize('weight', [False, True, 0, -0.1, 0.5, float('nan'), float('inf'), '0.1', None])
def test_invalid_generated_rgb_weight_rejected(weight):
    with pytest.raises(ValueError, match='generated_rgb_loss_weight'):
        validate_job(dict(stage='repair', checkpoint='saved', generated_rgb_loss=True,
                          generated_rgb_loss_weight=weight))
