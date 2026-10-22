"""Explicit native versus historical ArtiFixer rendering/reconstruction policy."""
DEFAULT_PIPELINE = 'authors_3dgut'
LEGACY_PIPELINE = 'legacy_gsplat'


def pipeline_profile(value):
    if value not in (DEFAULT_PIPELINE, LEGACY_PIPELINE):
        raise ValueError('artifixer_pipeline must be authors_3dgut or legacy_gsplat')
    return value


def apply_pipeline(raw):
    result = dict(raw)
    profile = pipeline_profile(raw.get('artifixer_pipeline', DEFAULT_PIPELINE))
    result['artifixer_pipeline'] = profile
    if profile == DEFAULT_PIPELINE:
        # The upstream ArtiFixer3D endpoint is a fresh fit, using references and
        # generated frames, with the published sparse LPIPS recipe unchanged.
        result.update(reconstruction_initialization='point_cloud',
                      reconstruction_images='distilled', regularization_profile='artifixer')
    return result


def rgb_backend(raw):
    profile = pipeline_profile(raw.get('artifixer_pipeline', LEGACY_PIPELINE))
    return '3dgut' if profile == DEFAULT_PIPELINE else 'viser'
