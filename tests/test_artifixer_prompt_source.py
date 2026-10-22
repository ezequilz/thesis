"""Prompt source isolation, reference integrity, and portable scene metadata."""
import json
import sys
import types
from pathlib import Path

import pytest

from splat_explorer.splatfix.artifixer import official_worker as worker
from splat_explorer.splatfix.worldexplorer.source import checkpoint_prompt


def test_prompt_options():
    assert worker.validate_reconstruction_options({})['prompt_source'] == 'worldexplorer'
    for source in ('original_anchors', 'original_and_repaired', 'worldexplorer'):
        assert worker.validate_reconstruction_options({'prompt_source': source})['prompt_source'] == source
    with pytest.raises(ValueError, match='prompt_source'):
        worker.validate_reconstruction_options({'prompt_source': 'unknown'})


def test_caption_sources_cache_and_integrity(tmp_path, monkeypatch):
    cp = tmp_path / 'checkpoint'
    cp.mkdir()
    (cp / 'original.png').write_bytes(b'original')
    (cp / 'repair.png').write_bytes(b'repaired')
    trajectory = {'anchors': [{'view_id': '0', 'original_rgb': 'original.png'}],
                  'sha256': {'original.png': worker.caption_digest(cp / 'original.png')}}
    request = {'checkpoint_root': str(cp), 'runtime': {'model_id': 'wan'},
               'upstream_revision': 'pinned', 'mode': 'edited',
               'references': [str(cp / 'repair.png')],
               'reference_sha256': [worker.caption_digest(cp / 'repair.png')]}
    snapshots, generated, encoded = [], [], []
    def snapshot(model, **kwargs):
        snapshots.append(model)
        return Path('/models') / model, {'files': []}
    monkeypatch.setattr(worker, 'caption_model_snapshot', snapshot)
    def generate(**kwargs):
        frames = json.loads((kwargs['input_path'] / 'transforms.json').read_text())['frames']
        generated.append([(kwargs['input_path'] / f['file_path']).read_bytes() for f in frames])
        kwargs['output_path'].write_text(json.dumps(list(range(len(frames)))))
    def encode(prompt, output, model, count):
        encoded.append(prompt)
        output.write_text(json.dumps(list(range(count))))
    monkeypatch.setattr(worker, 'encode_scene_prompt', encode)
    def validate(path, expected_hash=None):
        digest = worker.caption_digest(path)
        if expected_hash is not None:
            assert digest == expected_hash
        return {'sha256': digest, 'image_indices': json.loads(Path(path).read_text())}
    monkeypatch.setattr(worker, 'validate_caption', validate)
    module = types.ModuleType('data_processing.captioning.generate_captions')
    module.generate_caption_hdf5 = generate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    results = []
    for source in ('original_anchors', 'original_and_repaired', 'worldexplorer'):
        root = tmp_path / source
        root.mkdir()
        request.update(prompt_source=source, worldexplorer_prompt='Exact original scene prompt')
        worker.prepare_caption(root, request, trajectory)
        worker.prepare_caption(root, request, trajectory)
        results.append(json.loads((root / 'caption-result.json').read_text()))
    assert generated == [[b'original'], [b'original', b'repaired']]
    assert encoded == ['Exact original scene prompt']
    assert len({r['signature'] for r in results}) == 3
    request['worldexplorer_prompt'] = 'Changed scene prompt'
    worker.prepare_caption(root, request, trajectory)
    assert encoded[-1] == 'Changed scene prompt'
    request['worldexplorer_prompt'] = None
    with pytest.raises(ValueError, match='unavailable'):
        worker.prepare_caption(root, request, trajectory)
    request.update(prompt_source='original_and_repaired', mode='baseline')
    with pytest.raises(ValueError, match='edited'):
        worker.prepare_caption(root, request, trajectory)
    request['mode'] = 'edited'
    (cp / 'repair.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        worker.prepare_caption(root, request, trajectory)


def test_legacy_worldexplorer_prompt_is_saved_and_portable(tmp_path):
    run = tmp_path / 'we_example'
    scene = run / 'results/nerfstudio_output/scene/splat.ply'
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b'ply')
    request = run / 'request.json'
    request.write_text(json.dumps({'prompt': 'The exact generated scene'}))
    saves = []
    cp = types.SimpleNamespace(root=tmp_path, manifest={'scene_path': str(scene)}, save=lambda: saves.append(True))
    assert checkpoint_prompt(cp) == 'The exact generated scene'
    assert saves == [True]
    request.unlink()
    assert checkpoint_prompt(cp) == 'The exact generated scene'
    assert saves == [True]


def test_prompt_recovery_preserves_canonical_checkpoint_views(tmp_path, monkeypatch):
    from splat_explorer.splatfix.worldexplorer import source
    saved = tmp_path / 'checkpoint.json'
    saved.write_text(json.dumps({'views': [{'repaired_rgb': 'keep.png'}], 'metadata': {}}))
    cp = types.SimpleNamespace(root=tmp_path, manifest={'scene_path': 'scene.ply', 'views': []})
    monkeypatch.setattr(source, 'scene_prompt', lambda scene: 'Saved prompt')
    assert checkpoint_prompt(cp) == 'Saved prompt'
    canonical = json.loads(saved.read_text())
    assert canonical['views'] == [{'repaired_rgb': 'keep.png'}]
    assert canonical['metadata']['worldexplorer_prompt'] == 'Saved prompt'


def test_worldexplorer_default_matches_dashboard_and_runtime():
    from html.parser import HTMLParser
    from splat_explorer.splatfix.artifixer.repair import DEFAULT_RUNTIME
    class Options(HTMLParser):
        active = False
        selected = None
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'select':
                self.active = attrs.get('id') == 'artifixer-prompt-source'
            if tag == 'option' and self.active and 'selected' in attrs:
                self.selected = attrs['value']
        def handle_endtag(self, tag):
            if tag == 'select':
                self.active = False
    parser = Options()
    parser.feed((Path(__file__).parents[1] / 'src/splat_explorer/web/static/splatfix.html').read_text())
    assert parser.selected == DEFAULT_RUNTIME['prompt_source'] == 'worldexplorer'
