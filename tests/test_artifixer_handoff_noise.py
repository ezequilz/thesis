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
               'handoff_noise_strength':25, 'handoff_noise_vignette':True}
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
    assert result['noise']['vignette']['enabled'] is True
    assert result['noise']['vignette']['edge_strength_percent'] == 80
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
    assert options['handoff_noise_vignette'] is True
    assert noise_options({})['handoff_noise_vignette'] is False
    assert noise_options({})['handoff_add_noise'] is False
    assert noise_options({'handoff_add_noise':False})['handoff_add_noise'] is False
    with pytest.raises(ValueError,match='boolean'):
        noise_options({'handoff_add_noise':'yes'})


def test_vignette_is_elliptical_feathered_and_capped():
    from splat_explorer.splatfix.artifixer.handoff_noise import vignette_strength, noise_options
    field = vignette_strength(721,1281,25)
    assert field[360,640] == 25
    assert field[360,800] == 25  # Quarter-radius is still the flat center.
    assert field[360,960] == 25  # Transition begins at half-radius.
    assert field[360,1120] == pytest.approx(52.5)
    assert field[630,640] == pytest.approx(52.5)  # Same normalized elliptical radius.
    assert np.all(field[0] == 80) and np.all(field[-1] == 80)
    assert np.all(field[:,0] == 80) and np.all(field[:,-1] == 80)
    assert field.max() == 80
    assert np.max(np.abs(np.diff(field[360]))) < .3  # No hard band at the border.
    np.testing.assert_allclose(field,field[::-1,::-1])
    assert np.all(vignette_strength(10,20,90) == 90)
    with pytest.raises(ValueError,match='vignette'):
        noise_options({'handoff_noise_vignette':'yes'})


def test_vignette_preserves_center_and_increases_edge_noise(tmp_path):
    source = tmp_path/'before.png'
    Image.new('RGB',(640,360),(128,128,128)).save(source)
    add_gaussian_noise(source,tmp_path/'uniform.png',strength=25,seed=7)
    add_gaussian_noise(source,tmp_path/'vignette.png',strength=25,seed=7,vignette=True)
    uniform = np.asarray(Image.open(tmp_path/'uniform.png'),dtype=float)
    vignette = np.asarray(Image.open(tmp_path/'vignette.png'),dtype=float)
    np.testing.assert_array_equal(uniform[150:210,280:360],vignette[150:210,280:360])
    assert vignette[:20].std() > uniform[:20].std() * 1.35
