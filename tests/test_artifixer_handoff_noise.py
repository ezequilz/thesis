from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from splat_explorer.splatfix.artifixer.handoff_noise import add_gaussian_noise


def test_noise_is_reproducible_preserves_source_and_matches_sigma(tmp_path):
    source = tmp_path/'before.png'
    Image.new('RGB', (256,256), (128,128,128)).save(source)
    original = source.read_bytes()
    target = tmp_path/'noisy.png'
    metadata = add_gaussian_noise(source,target,strength=8,seed=42)
    first = target.read_bytes()
    delta = np.asarray(Image.open(target), dtype=float) - 128
    assert abs(delta.mean()) < .2
    assert delta.std() == pytest.approx(20.4, abs=.2)
    assert metadata['sigma_8bit'] == 20.4
    add_gaussian_noise(source,target,strength=8,seed=42)
    assert first == target.read_bytes()
    assert source.read_bytes() == original
    add_gaussian_noise(source,target,strength=0,seed=42)
    np.testing.assert_array_equal(np.asarray(Image.open(source)),np.asarray(Image.open(target)))
    with pytest.raises(ValueError,match='preserve'):
        add_gaussian_noise(source,source,strength=8,seed=42)


@pytest.mark.parametrize('strength', [-1, 101, float('nan'), float('inf'), True, '8'])
def test_invalid_strength_is_rejected(tmp_path,strength):
    with pytest.raises(ValueError,match='strength'):
        add_gaussian_noise(tmp_path/'before.png',tmp_path/'noisy.png',strength=strength,seed=42)


@pytest.mark.parametrize('with_context', [False, True])
def test_repair_sends_noisy_first_image_and_clean_reference(tmp_path, with_context):
    import json
    from types import SimpleNamespace
    from splat_explorer.splatfix.artifixer.handoff import edit_request, digest
    source = tmp_path/'before.png'
    reference = tmp_path/'reference.png'
    Image.new('RGB', (32,16), (100,110,120)).save(source)
    Image.new('RGB', (32,16), 'blue').save(reference)
    original, clean_reference = source.read_bytes(), reference.read_bytes()
    request = {'id':'a'*32, 'input_sha256':digest(source), 'handoff_add_noise':True,
               'handoff_noise_strength':25}
    if with_context:
        request.update(gpt_image_context='closest_repaired_image', reference_sha256=digest(reference))
    calls = []
    def record(image, references, prompt):
        assert image == tmp_path/'repair-input.png'
        assert image.read_bytes() != original
        assert [p.read_bytes() for p in references] == ([clean_reference] if with_context else [])
        calls.append(image.read_bytes())
        return SimpleNamespace(error=None,images=[original])
    backend = SimpleNamespace(edit=lambda image,prompt:record(image,[],prompt),edit_with_references=record)
    edit_request(tmp_path,request,backend=backend)
    result = json.loads((tmp_path/'response.json').read_text())
    assert 'error' not in result
    assert result['noise']['strength_percent'] == 25
    assert result['noise']['input_sha256'] == digest(tmp_path/'repair-input.png')
    assert source.read_bytes() == original and reference.read_bytes() == clean_reference
    edit_request(tmp_path,request,backend=backend)
    assert len(calls) == 1  # Cached response never triggers another paid request.
    assert (tmp_path/'repair-input.png').read_bytes() == calls[0]


def test_new_jobs_default_to_noise_but_historical_runtime_stays_off():
    from splat_explorer.splatfix.artifixer.backend import validate_options
    from splat_explorer.splatfix.artifixer.handoff_noise import noise_options
    options = validate_options({'stage':'repair', 'mode':'edited', 'handoff_generation':True})
    assert options['handoff_add_noise'] is True
    assert options['handoff_noise_strength'] == 25
    assert noise_options({})['handoff_add_noise'] is False
    assert noise_options({'handoff_add_noise':False})['handoff_add_noise'] is False
    with pytest.raises(ValueError,match='boolean'):
        noise_options({'handoff_add_noise':'yes'})
