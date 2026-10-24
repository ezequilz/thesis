import json
from pathlib import Path

import numpy as np
from PIL import Image

from splat_explorer.web.artifixer_anchor_chart import anchor_chart
from test_splatfix_results import setup


def fixture(tmp_path):
    studio, root, result = setup(tmp_path)
    cp = result / 'checkpoint'
    trajectory = cp / 'trajectories/used'
    trajectory.mkdir(parents=True)
    anchors = []
    for name, index in [('000', 0), ('003', 64)]:
        Image.new('RGB', (32, 18), 'blue').save(trajectory / f'{name}.png')
        np.save(trajectory / f'{name}.npy', np.ones((18, 32), dtype=np.float32))
        np.save(trajectory / f'{name}-depth.npy', np.linspace(.2, 2, 18*32).reshape(18, 32))
        anchors.append({'view_id': name, 'frame_index': index,
                        'original_rgb': f'trajectories/used/{name}.png',
                        'opacity': f'trajectories/used/{name}.npy',
                        'depth': f'trajectories/used/{name}-depth.npy'})
    data = {'depth_convention': 'expected_camera_z', 'anchors': anchors}
    (trajectory / 'trajectory.json').write_text(json.dumps(data))
    (result / 'request.json').write_text(json.dumps({
        'checkpoint_root': f'/workspace/jobs/run/results/{result.name}/checkpoint',
        'trajectory': f'/workspace/jobs/run/results/{result.name}/checkpoint/trajectories/used/trajectory.json'}))
    return studio, root, result, trajectory, data


def test_chart_uses_recorded_anchors_and_reuses_cache(tmp_path):
    studio, root, result, trajectory, _ = fixture(tmp_path)
    chart = anchor_chart(studio, result / 'result.json', root.name)
    assert chart['anchors'] == ['000', '003']
    png = next((result / '.anchor-chart').glob('*/*.png'))
    assert png.name == root.name + '_intermediate_arr.png'
    with Image.open(png) as image:
        assert image.size == (1556, 856)
    modified = png.stat().st_mtime_ns
    assert anchor_chart(studio, result / 'result.json', root.name) == chart
    assert png.stat().st_mtime_ns == modified
    np.save(trajectory / '000.npy', np.zeros((18, 32)))
    assert anchor_chart(studio, result / 'result.json', root.name)['url'] != chart['url']
    row = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert row['anchor_chart']['anchors'] == ['000', '003']


def test_missing_or_escaping_arrays_omit_chart(tmp_path):
    studio, root, result, trajectory, data = fixture(tmp_path)
    depth = trajectory / '000-depth.npy'
    depth.unlink()
    assert anchor_chart(studio, result / 'result.json', root.name) is None
    external = tmp_path / 'outside.npy'
    np.save(external, np.ones((18, 32)))
    depth.symlink_to(external)
    assert anchor_chart(studio, result / 'result.json', root.name) is None
    assert not (result / '.anchor-chart').exists()


def test_chart_section_precedes_reconstruction_inputs():
    html = Path('src/splat_explorer/web/static/splatfix_result_detail.html').read_text()
    assert html.index('id="anchorChart"') < html.index('id="galleryTitle"')
