"""Explicit native versus historical ArtiFixer rendering/reconstruction policy."""
AUTHORS_PIPELINE = 'authors_3dgut'
HYBRID_PIPELINE = 'authors_3dgut_viser_rgb'
LEGACY_PIPELINE = 'legacy_gsplat'
DEFAULT_PIPELINE = LEGACY_PIPELINE


def pipeline_profile(value):
    if value not in (AUTHORS_PIPELINE, HYBRID_PIPELINE, LEGACY_PIPELINE):
        raise ValueError('artifixer_pipeline must be authors_3dgut, authors_3dgut_viser_rgb or legacy_gsplat')
    return value


def apply_pipeline(raw):
    result = dict(raw)
    profile = pipeline_profile(raw.get('artifixer_pipeline', DEFAULT_PIPELINE))
    result['artifixer_pipeline'] = profile
    if profile != LEGACY_PIPELINE:
        # The upstream ArtiFixer3D endpoint is a fresh fit, using references and
        # generated frames. Generated-image loss choices remain user-controlled.
        result.update(reconstruction_initialization='point_cloud',
                      reconstruction_images='distilled', regularization_profile='artifixer')
    return result


def rgb_backend(raw):
    profile = pipeline_profile(raw.get('artifixer_pipeline', LEGACY_PIPELINE))
    return '3dgut' if profile == AUTHORS_PIPELINE else 'viser'


def geometry_backend(raw):
    profile = pipeline_profile(raw.get('artifixer_pipeline', LEGACY_PIPELINE))
    return 'gsplat' if profile == LEGACY_PIPELINE else '3dgut'
