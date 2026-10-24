import json
from types import SimpleNamespace

import pytest

from splat_explorer.web.handoff_details import handoff_details
from splat_explorer.web.run_reduction import reduce_artifixer


@pytest.mark.parametrize('mode', ['reduce', 'backup', 'reconstruction'])
@pytest.mark.parametrize('location', ['handoff', 'gpu/handoff', 'gpu/results/edited/handoffs/inference'])
def test_reduction_keeps_handoff_gallery(tmp_path, mode, location):
    root = tmp_path / 'run'
    result = root / 'gpu/results/edited'
    result.mkdir(parents=True)
    expected = {}
    for index in (31, 62):
        folder = root / location / str(index)
        folder.mkdir(parents=True)
        metadata = {'id': str(index), 'phase': 'inference', 'frame_index': index}
        for name in ('request.json', 'camera.json', 'response.json'):
            expected[folder / name] = json.dumps(metadata).encode()
        for name in ('before.png', 'response-original.png', 'repaired.png'):
            expected[folder / name] = f'{index}-{name}'.encode()
    for path, content in expected.items():
        path.write_bytes(content)
    intermediate = result / 'checkpoint.pt'
    intermediate.write_bytes(b'discard')
    history = tmp_path / 'history' if mode == 'backup' else None
    reduce_artifixer(root, [{'root': result, 'ply_copies': [], 'trajectory': {'frames': [1]}}],
                    history, preserve_reconstruction=mode == 'reconstruction')
    assert all(path.read_bytes() == content for path, content in expected.items())
    assert not intermediate.exists()
    gallery = handoff_details(SimpleNamespace(file_url=str), root)
    assert gallery['enabled']
    assert [item['frame_index'] for item in gallery['items']] == [31, 62]
    assert all(item['status'] == 'repaired' and len(item['images']) == 3 for item in gallery['items'])
    if history:
        assert (history / root.name / intermediate.relative_to(root)).read_bytes() == b'discard'
        assert all((history / root.name / path.relative_to(root)).read_bytes() == content
                   for path, content in expected.items())


def test_reduction_rejects_external_handoff_before_pruning(tmp_path):
    root = tmp_path / 'run'
    folder = root / 'handoff/31'
    folder.mkdir(parents=True)
    external = tmp_path / 'external.png'
    external.write_bytes(b'external')
    (folder / 'before.png').symlink_to(external)
    checkpoint = root / 'checkpoint.pt'
    checkpoint.write_bytes(b'weights')
    with pytest.raises(ValueError, match='linked handoff'):
        reduce_artifixer(root, [{'root': root, 'ply_copies': []}])
    assert checkpoint.exists()
    assert external.read_bytes() == b'external'
