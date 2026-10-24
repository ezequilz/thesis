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
    assert result['noise']['vignette']['edge_strength_percent'] == 95
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
    assert options['handoff_noise_mode'] == 'repaired_pixel_sprinkle'
    assert noise_options({})['handoff_noise_mode'] == 'gaussian'
    assert validate_options({'stage':'repair', 'mode':'edited', 'handoff_generation':True,
                             'handoff_noise_mode':'gaussian'})['handoff_noise_mode'] == 'gaussian'
    assert options['handoff_noise_vignette'] is True
    assert noise_options({})['handoff_noise_vignette'] is False
    assert noise_options({})['handoff_add_noise'] is False
    assert noise_options({'handoff_add_noise':False})['handoff_add_noise'] is False
    with pytest.raises(ValueError,match='boolean'):
        noise_options({'handoff_add_noise':'yes'})


def test_vignette_is_elliptical_feathered_and_capped():
    from splat_explorer.splatfix.artifixer.handoff_noise import vignette_strength, noise_options
    field = vignette_strength(2001,2001,25)
    assert field[1000,1000] == 12.5
    assert field[1000,1252] == pytest.approx(18.75)  # Halfway through inner radius.
    assert field[1700,1000] == pytest.approx(25)  # Full selected strength at inner boundary.
    assert field[1000,1612] == pytest.approx(60)
    assert field[1850,1000] == pytest.approx(60)  # Same elliptical midpoint.
    assert np.all(field[0] == 95) and np.all(field[-1] == 95)
    assert np.all(field[:,0] == 95) and np.all(field[:,-1] == 95)
    assert field.max() == 95
    assert np.max(np.abs(np.diff(field[1000]))) < 1.0  # Soft despite the narrower band.
    np.testing.assert_allclose(field,field[::-1,::-1])
    assert vignette_strength(10,20,98).max() == 98
    with pytest.raises(ValueError,match='vignette'):
        noise_options({'handoff_noise_vignette':'yes'})


def test_vignette_reduces_center_and_increases_edge_noise(tmp_path):
    source = tmp_path/'before.png'
    Image.new('RGB',(640,360),(128,128,128)).save(source)
    add_gaussian_noise(source,tmp_path/'uniform.png',strength=25,seed=7)
    add_gaussian_noise(source,tmp_path/'vignette.png',strength=25,seed=7,vignette=True)
    uniform = np.asarray(Image.open(tmp_path/'uniform.png'),dtype=float)
    vignette = np.asarray(Image.open(tmp_path/'vignette.png'),dtype=float)
    assert vignette[170:190,310:330].std() == pytest.approx(uniform[170:190,310:330].std() * .5, rel=.06)
    assert vignette[:20].std() > uniform[:20].std() * 1.35


def test_pixel_sprinkle_replaces_whole_pixels_with_elliptical_probability(tmp_path):
    from splat_explorer.splatfix.artifixer.handoff_noise import sprinkle_repaired_pixels
    source, reference, output = [tmp_path/n for n in ('source.png','reference.png','output.png')]
    Image.new('RGB',(1280,720),(10,20,30)).save(source)
    Image.new('RGB',(1280,720),(210,220,230)).save(reference)
    original, ref_bytes = source.read_bytes(), reference.read_bytes()
    meta = sprinkle_repaired_pixels(source,reference,output,strength=25,seed=42,vignette=True)
    pixels = np.asarray(Image.open(output))
    selected = pixels[:,:,0] == 210
    assert np.all(pixels[selected] == [210,220,230])
    assert np.all(pixels[~selected] == [10,20,30])
    assert selected[340:380,620:660].mean() == pytest.approx(.125,abs=.025)
    assert selected[:20].mean() == pytest.approx(.95,abs=.03)
    assert meta['replaced_pixels'] == selected.sum()
    first = output.read_bytes()
    sprinkle_repaired_pixels(source,reference,output,strength=25,seed=42,vignette=True)
    assert output.read_bytes() == first
    assert source.read_bytes() == original and reference.read_bytes() == ref_bytes
    for strength,color in [(0,[10,20,30]),(100,[210,220,230])]:
        sprinkle_repaired_pixels(source,reference,output,strength=strength,seed=42)
        assert np.all(np.asarray(Image.open(output)) == color)


@pytest.mark.parametrize('context', ['closest_repaired_image', 'initial_repaired_image', 'original_rgb_capture'])
def test_sprinkle_sends_only_composite_with_single_image_prompt(tmp_path,context):
    import json
    from types import SimpleNamespace
    from splat_explorer.splatfix.artifixer.handoff import edit_request, digest
    from splat_explorer.splatfix.image_repair import DEFAULT_PROMPT
    source, reference = tmp_path/'before.png', tmp_path/'reference.png'
    Image.new('RGB',(64,32),'red').save(source)
    Image.new('RGB',(64,32),'blue').save(reference)
    request = {'id':'b'*32, 'input_sha256':digest(source), 'reference_sha256':digest(reference),
               'gpt_image_context':context,'handoff_add_noise':True,'handoff_noise_strength':25,
               'handoff_noise_vignette':True,'handoff_noise_mode':'repaired_pixel_sprinkle'}
    calls=[]
    def edit(path,prompt):
        assert prompt == DEFAULT_PROMPT and 'second attachment' not in prompt
        assert path.name == 'repair-input.png'
        assert path.read_bytes() != source.read_bytes()
        assert path.read_bytes() != reference.read_bytes()
        calls.append(path.read_bytes())
        return SimpleNamespace(error=None,images=[source.read_bytes()])
    backend=SimpleNamespace(edit=edit,edit_with_references=lambda *args:pytest.fail('Sprinkle must send one attachment'))
    edit_request(tmp_path,request,backend=backend)
    result=json.loads((tmp_path/'response.json').read_text())
    assert 'error' not in result
    assert result['attachment_count'] == 1 and result['reference_attached'] is False
    assert result['noise']['reference_sha256'] == digest(reference)
    edit_request(tmp_path,request,backend=backend)
    assert len(calls)==1 and calls[0]==(tmp_path/'repair-input.png').read_bytes()


def test_sprinkle_requires_a_reference():
    from splat_explorer.splatfix.artifixer.handoff_noise import noise_options
    with pytest.raises(ValueError,match='requires closest/initial'):
        noise_options({'handoff_add_noise':True,'handoff_noise_mode':'repaired_pixel_sprinkle',
                       'gpt_image_context':'single_last_frame'})
    with pytest.raises(ValueError,match='handoff_noise_mode'):
        noise_options({'handoff_noise_mode':'unknown'})


def test_smaller_vignette_restores_previous_profile():
    from splat_explorer.splatfix.artifixer.handoff_noise import vignette_strength, noise_options
    field = vignette_strength(721, 1281, 25, 'smaller')
    x, y = np.linspace(-1, 1, 1281), np.linspace(-1, 1, 721)
    t = np.clip((np.sqrt(y[:, None] ** 2 + x[None, :] ** 2) - .5) / .5, 0, 1)
    outer = t > 0
    np.testing.assert_array_equal(field[outer], (25 + 55 * (t * t * (3 - 2 * t)))[outer])
    assert field[360,640] == 12.5
    assert field[360,1120] == pytest.approx(52.5)
    assert field.max() == 80
    assert noise_options({})['handoff_vignette_size'] == 'larger'
    with pytest.raises(ValueError, match='handoff_vignette_size'):
        noise_options({'handoff_vignette_size': 'invalid'})


@pytest.mark.parametrize('mode', ['gaussian', 'repaired_pixel_sprinkle'])
def test_smaller_profile_reaches_repair_request(tmp_path, mode):
    import json
    from types import SimpleNamespace
    from splat_explorer.splatfix.artifixer.handoff import edit_request, digest
    source = tmp_path/'before.png'
    reference = tmp_path/'reference.png'
    Image.new('RGB', (64,32), 'red').save(source)
    Image.new('RGB', (64,32), 'blue').save(reference)
    request = {'id': 'c'*32, 'input_sha256': digest(source),
               'reference_sha256': digest(reference), 'gpt_image_context': 'closest_repaired_image',
               'handoff_add_noise': True, 'handoff_noise_mode': mode,
               'handoff_noise_vignette': True, 'handoff_vignette_size': 'smaller',
               'handoff_vignette_edge_strength': 65}
    def edit(*args): return SimpleNamespace(error=None, images=[source.read_bytes()])
    edit_request(tmp_path, request, backend=SimpleNamespace(edit=edit, edit_with_references=edit))
    result = json.loads((tmp_path/'response.json').read_text())
    assert 'error' not in result
    assert result['noise']['vignette']['size'] == 'smaller'
    assert result['noise']['vignette']['inner_radius'] == .5
    assert result['noise']['vignette']['edge_strength_percent'] == 65


@pytest.mark.parametrize('size,default', [('larger',95), ('smaller',80)])
def test_edge_strength_is_independent_and_defaults_follow_size(size,default):
    from splat_explorer.splatfix.artifixer.handoff_noise import noise_options, vignette_strength
    assert noise_options({'handoff_vignette_size':size})['handoff_vignette_edge_strength'] == default
    for edge in (0,10,70,100):
        field = vignette_strength(101,201,25,size,edge_strength=edge)
        assert field[50,100] == 12.5
        np.testing.assert_allclose(field[0],edge,atol=1e-12)
        np.testing.assert_allclose(field[:,0],edge,atol=1e-12)
        assert noise_options({'handoff_vignette_edge_strength':edge})['handoff_vignette_edge_strength'] == edge
    for bad in (-1,101,True,'95',None,float('nan')):
        with pytest.raises(ValueError,match='strength'):
            noise_options({'handoff_vignette_edge_strength':bad})


def test_very_large_ellipse_extends_vertically_and_is_new_job_default():
    from splat_explorer.splatfix.artifixer.handoff_noise import vignette_strength, vignette_profile
    from splat_explorer.splatfix.artifixer.backend import validate_options
    larger = vignette_strength(721,1281,25,'larger')
    very_large = vignette_strength(721,1281,25,'very_large')
    np.testing.assert_array_equal(very_large[360], larger[360])
    assert very_large[360,640] == 12.5
    assert very_large[60,640] < larger[60,640]
    assert very_large.max() == 95
    np.testing.assert_allclose(very_large, very_large[::-1,::-1])
    assert vignette_profile('very_large')['axis_scale_y'] == 1.15
    options = validate_options({'stage':'repair', 'handoff_generation':True})
    assert options['handoff_vignette_size'] == 'very_large'
    assert options['handoff_vignette_edge_strength'] == 95
    assert validate_options({'stage':'repair', 'handoff_vignette_size':'smaller'})['handoff_vignette_size'] == 'smaller'
