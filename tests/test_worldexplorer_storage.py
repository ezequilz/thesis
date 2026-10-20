import errno
import os
from pathlib import Path

from splat_explorer.splatfix.worldexplorer.source import link_source_image, stage_source
from test_worldexplorer_source import fixture


def test_staging_shares_archive_bytes_and_survives_source_unlink(tmp_path):
    ply = fixture(tmp_path)
    from splat_explorer.splatfix.worldexplorer.source import source_frames, load_staged
    original = Path(source_frames(ply)['frames'][0]['file_path'])
    bundle = stage_source(ply, tmp_path / 'staged')
    image = Path(load_staged(bundle)['frames'][0]['file_path'])
    assert image.samefile(original)
    original.unlink()
    assert load_staged(bundle)['frames']


def test_cross_device_fallback_does_not_overwrite_shared_inode(tmp_path, monkeypatch):
    source = tmp_path / 'original.png'; source.write_bytes(b'original')
    destination = tmp_path / 'input.png'
    os.link(source, destination)
    replacement = tmp_path / 'new.png'; replacement.write_bytes(b'new')
    def cross_device(*args):
        raise OSError(errno.EXDEV, 'Different filesystem')
    monkeypatch.setattr(os, 'link', cross_device)
    link_source_image(replacement, destination)
    assert source.read_bytes() == b'original'
    assert destination.read_bytes() == b'new'


def test_run_size_counts_internal_links_once_and_excludes_shared_inputs(tmp_path):
    from splat_explorer.scene_runs import SceneRunStore
    store = SceneRunStore(tmp_path / 'runs')
    run = store.create_run()
    root = Path(run.path)
    archive = tmp_path / 'archive.png'; archive.write_bytes(b'x' * 1000)
    os.link(archive, root / 'input.png')
    own = root / 'own.png'; own.write_bytes(b'x' * 100)
    os.link(own, root / 'own-copy.png')
    store.set_status(run.run_id, 'error')
    expected = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and p.name not in ('input.png', 'own-copy.png'))
    assert store.output_size_bytes(run.run_id) == expected


def test_persistence_seeds_shared_inputs_before_transfer(tmp_path):
    import json
    from splat_explorer.splatfix.persistence import persist_remote_run
    from test_splatfix_persistence import transport
    image = tmp_path / 'oraclegs-inputs-test/inputs/worldexplorer-source/images/000.png'
    image.parent.mkdir(parents=True); image.write_bytes(b'original')
    (tmp_path / 'oraclegs-staging.json').write_text(json.dumps({'input_manifest': 'oraclegs-inputs-test/inputs/inputs.json'}))
    lrz, _ = transport(tmp_path)
    transfer = lrz._mux_run
    def verify(argv):
        assert (tmp_path / 'gpu/inputs/worldexplorer-source/images/000.png').samefile(image)
        transfer(argv)
    lrz._mux_run = verify
    persist_remote_run(lrz, {'workspace': '/dss', 'user': 'u', 'host': 'h'}, tmp_path, 'run', '/dss/splatfix-jobs/run')
