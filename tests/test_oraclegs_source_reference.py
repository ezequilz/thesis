"""Fixed source is opt-in; historical queues and spatial context stay unchanged."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_splatfix_oraclegs import checkpoint
from splat_explorer.splatfix.oraclegs.backend import validate_options
from splat_explorer.splatfix.oraclegs.execution import job_runtime
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from splat_explorer.splatfix.oraclegs.repair import seva_request, DEFAULT_RUNTIME
from splat_explorer.splatfix.oraclegs.source_reference import load_source, install_fixed_source
from splat_explorer.splatfix.oraclegs.adapter import pad_seva_inputs


def test_queued_jobs_without_option_keep_native_behavior(tmp_path):
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path), root, runtime={
        'oracle_adapter':'per_view_preset', 'oracle_reference_mode':'all', 'oracle_target_frames':4})
    del manifest['oracle_source_reference']  # Existing staged job, before new option.
    options = {'oracle_adapter':'per_view_preset', 'oracle_reference_mode':'all'}
    runtime = job_runtime({}, options)
    assert runtime['oracle_source_reference'] == 'nearest'
    for segment in manifest['seva_segments']:
        request = seva_request(root/segment['input_subdir'], tmp_path/'out', manifest, runtime, segment)
        assert 'fixed_initial_source' not in request
        # A worker-level default must never opt an old staged run into the change.
        other = seva_request(root/segment['input_subdir'], tmp_path/'out', manifest,
                             {**runtime, 'oracle_source_reference':'fixed_initial'}, segment)
        assert other == request
    assert validate_options({})['oracle_source_reference'] == 'nearest'
    with pytest.raises(ValueError, match='oracle_source_reference'):
        validate_options({'oracle_source_reference':'invalid'})


@pytest.mark.parametrize('mode,color', [('edited',[0,0,255]), ('baseline',[255,0,0])])
@pytest.mark.parametrize('resolution',['576p','720p'])
def test_fixed_initial_uses_each_clips_saved_image_and_guidance(tmp_path, mode, color, resolution):
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path), root, mode=mode, runtime={
        'oracle_adapter':'per_view_preset', 'oracle_reference_mode':'all', 'oracle_target_frames':4,
        'oracle_source_reference':'fixed_initial', 'seva_resolution':resolution})
    first = manifest['seva_segments'][0]
    unguided = seva_request(root/first['input_subdir'], tmp_path/'out', manifest, DEFAULT_RUNTIME, first)
    assert 'fixed_initial_source' in unguided and 'source_geometry_guidance' not in unguided
    manifest['seva_geometry_guidance'] = 'source_splat'  # Request wiring, no model/depth run.
    for segment in manifest['seva_segments']:
        native = root/segment['input_subdir']
        request = seva_request(native, tmp_path/'out', manifest, DEFAULT_RUNTIME, segment)
        source = request['fixed_initial_source']
        assert source['reference_index'] == segment['from_reference']
        assert segment['frame_order'][source['frame_index']] == segment['from_reference']
        assert request['source_geometry_guidance']['source_reference'] == segment['from_reference']
        if resolution == '720p':
            native, _, _ = pad_seva_inputs(native, tmp_path/f"padded-{segment['index']}")
        pixels, report = load_source(native, source)
        np.testing.assert_allclose(pixels[0,:,pixels.shape[-2]//2,0], np.array(color)/127.5-1, atol=1e-6)
        assert report['mode'] == mode
        bad = {**source, 'reference_index':99}
        with pytest.raises(ValueError, match='starting reference'):
            load_source(native, bad)


class Tensor(np.ndarray):
    """CPU shim for the narrow conditioner dispatch contract; no CUDA needed."""
    def to(self, other):
        return self


def test_both_passes_use_fixed_embedding_and_keep_all_spatial_context(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(as_tensor=lambda x: np.asarray(x).view(Tensor)))
    root = tmp_path/'inputs'
    manifest = export_inputs(checkpoint(tmp_path), root, runtime={
        'oracle_adapter':'per_view_preset', 'oracle_reference_mode':'all', 'oracle_target_frames':4,
        'oracle_source_reference':'fixed_initial'})
    segment = manifest['seva_segments'][1]
    request = seva_request(root/segment['input_subdir'], tmp_path/'out', manifest, DEFAULT_RUNTIME, segment)
    seen = []
    def conditioner(images):
        seen.append(images.copy())
        return images.mean(axis=(2,3))
    def native(conditioner, value_dict, H, W, T):
        # Native SEVA keeps masked VAE latents and averages the global embedding.
        embedding = conditioner(value_dict['cond_frames'][value_dict['cond_frames_mask']]).mean(0)
        return embedding, value_dict
    evaluation = SimpleNamespace(do_sample=native)
    report = tmp_path/'source.json'
    install_fixed_source(evaluation, root/segment['input_subdir'], request['fixed_initial_source'], report)
    for total, conditioned in [(27,19),(21,8)]:
        # Small dummy inputs: the source encoder must ignore context pixels.
        values = {'cond_frames':np.zeros((total,3,1,1)).view(Tensor),
                  'cond_frames_mask':np.arange(total)<conditioned, 'K':object(), 'c2w':object()}
        before = values['cond_frames'].copy()
        embedding, returned = evaluation.do_sample(conditioner, values, 576, 1024, total)
        np.testing.assert_allclose(embedding, [-1,-1,1])
        assert returned is values
        np.testing.assert_array_equal(values['cond_frames'], before)
        assert values['cond_frames_mask'].sum() == conditioned
    assert all(image.shape == (1,3,576,1024) for image in seen)
    assert [c['frame_count'] for c in json.loads(report.read_text())['chunks']] == [27,21]
