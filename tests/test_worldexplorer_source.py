import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.worldexplorer.source import discover, source_frames, stage_source, load_staged


def fixture(tmp_path):
    root = tmp_path / 'worldexplorer/we_20261020_215533_02b07c345ec5/results'
    dataset = root / 'scenes/room/img2trajvid'
    dataset.mkdir(parents=True)
    ply = root / 'nerfstudio_output/room/splatfacto/date/exports/splat/splat.ply'
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b'ply')
    transform = np.eye(4)[:3]
    transform[:, 3] = [1, 2, 3]
    (ply.parents[2] / 'dataparser_transforms.json').write_text(json.dumps({'transform': transform.tolist(), 'scale': 2}))
    frames = []
    for i in range(2):
        path = dataset / f'{i:03d}/input/000.png'
        path.parent.mkdir(parents=True)
        Image.new('RGB', (16, 12), (i, 20, 30)).save(path)
        pose = np.eye(4)
        pose[0, 3] = i
        frames.append({'file_path': str(Path('/workspace/worldexplorer-runs') / root.parent.name / path.relative_to(root)),
                       'transform_matrix': pose.tolist()})
    (dataset / 'transforms.json').write_text(json.dumps({'w': 16, 'h': 12, 'fl_x': 10, 'fl_y': 11, 'cx': 8, 'cy': 6, 'frames': frames}))
    return ply


def test_catalog_and_normalized_originals(tmp_path):
    ply = fixture(tmp_path)
    scene, = discover(tmp_path)
    assert scene.path == ply and scene.up_axis == '+z'
    data = source_frames(ply)
    assert len(data['frames']) == 2
    np.testing.assert_allclose(np.array(data['frames'][1]['transform_matrix'])[:3, 3], [4, 4, 6])
    np.testing.assert_allclose(np.array(data['frames'][1]['transform_matrix'])[:3, :3], np.eye(3))
    bundle = stage_source(ply, tmp_path / 'staged')
    staged = load_staged(bundle)
    assert len(staged['frames']) == 2
    assert staged['frames'][0]['fl_y'] == 11
    assert Path(staged['frames'][0]['file_path']).read_bytes() == Path(data['frames'][0]['file_path']).read_bytes()
    Path(staged['frames'][0]['file_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_staged(bundle)


def test_missing_original_fails_instead_of_dropping(tmp_path):
    ply = fixture(tmp_path)
    Path(source_frames(ply)['frames'][0]['file_path']).unlink()
    with pytest.raises(ValueError, match='Missing WorldExplorer source image'):
        stage_source(ply, tmp_path / 'staged')
    assert source_frames(tmp_path / 'other.ply') is None


def test_artifixer_appends_originals_without_replacing_repaired_references(tmp_path):
    from splat_explorer.splatfix.artifixer.official_worker import append_worldexplorer_supervision
    bundle = stage_source(fixture(tmp_path), tmp_path / 'staged')
    transforms = {'frames': [{'file_path': 'repaired.png'}, {'file_path': 'generated.png'}]}
    supervision = {'selected_indices': [0], 'groups': [
        {'reference': 'repaired.png', 'source_index': 0}, {'reference': None, 'source_index': 1}]}
    append_worldexplorer_supervision({'worldexplorer_source': bundle}, transforms, supervision)
    assert supervision['selected_indices'] == [0, 2, 3]
    assert supervision['groups'][:2] == [{'reference': 'repaired.png', 'source_index': 0}, {'reference': None, 'source_index': 1}]
    assert len(transforms['frames']) == supervision['distillation_frame_count'] == 4
    assert transforms['frames'][2]['fl_x'] == 10


def test_oracle_appends_originals_to_synthetic_pool(tmp_path):
    from splat_explorer.splatfix.oraclegs.adapter import append_worldexplorer
    inputs = tmp_path / 'inputs'
    stage_source(fixture(tmp_path), inputs / 'worldexplorer-source')
    (inputs / 'inputs.json').write_text(json.dumps({'worldexplorer_source': 'worldexplorer-source'}))
    output = tmp_path / 'output'
    dataset = output / 'dataset'
    sparse = dataset / 'sparse/1'
    sparse.mkdir(parents=True)
    (dataset / 'images').mkdir()
    (dataset / 'synthetic_1').mkdir()
    (dataset / 'synthetic_1_attention_maps').mkdir()
    (sparse / 'cameras.txt').write_text('1 PINHOLE 16 12 10 11 8 6\n')
    (sparse / 'images.txt').write_text('1 1 0 0 0 0 0 0 1 repaired.png\n\n')
    (dataset / 'reference_names.json').write_text('["repaired.png"]')
    total, names = append_worldexplorer(inputs, output, 1)
    assert total == 1 and len(names) == 2
    assert json.loads((dataset / 'reference_names.json').read_text()) == ['repaired.png']
    assert (dataset / 'synthetic_1_attention_maps').is_dir()
    assert 'worldexplorer_000001.png' in (sparse / 'images.txt').read_text()
    for name in names:
        assert (dataset / 'images' / name).read_bytes() == (dataset / 'synthetic_1' / name).read_bytes()
    report = json.loads((output / 'worldexplorer-supervision.json').read_text())
    assert report['sampling_pool'] == 'synthetic'
    assert report['synthetic_names'] == names


@pytest.mark.parametrize('initialization', ['full_splat', 'point_cloud'])
def test_artifixer_distilled_only_ignores_even_a_stale_original_bundle(initialization):
    from splat_explorer.splatfix.artifixer.official_worker import append_worldexplorer_supervision
    transforms = {'frames': [{'file_path': 'repaired.png'}, {'file_path': 'generated.png'}]}
    supervision = {'selected_indices': [0], 'distillation_frame_count': 2, 'groups': [
        {'reference': 'repaired.png', 'source_index': 0}, {'reference': None, 'source_index': 1}]}
    append_worldexplorer_supervision({'worldexplorer_source': '/missing/old-bundle', 'runtime': {
        'reconstruction_images': 'distilled', 'reconstruction_initialization': initialization}}, transforms, supervision)
    assert len(transforms['frames']) == supervision['distillation_frame_count'] == 2
    assert supervision['selected_indices'] == [0]
    assert supervision['worldexplorer_original_count'] == 0
    assert supervision['groups'][0]['reference'] == 'repaired.png'
    assert supervision['groups'][1]['reference'] is None
