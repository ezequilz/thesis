from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from splat_explorer.splatfix.executor import persist_checkpoint_caches
from splat_explorer.splatfix.repair import digest_file


def remote_cache(root):
    caption = root / 'captions' / ('a' * 64)
    caption.mkdir(parents=True)
    (caption / 'caption.h5').write_bytes(b'persist opaque HDF5 fixture; validation tested in worker')
    (caption / 'manifest.json').write_text(json.dumps({
        'signature': caption.name, 'sha256': digest_file(caption / 'caption.h5'),
        'recipe': {'originals': ['same saved views']}}))
    (caption / 'source').mkdir()
    (caption / 'source/original.png').symlink_to('/workspace/splatfix-jobs/old/checkpoint/views/000/original.png')
    measurement = root / 'trajectories/traj/metric_alignment/measurement/images'
    measurement.mkdir(parents=True)
    (measurement / 'anchor.png').symlink_to('/workspace/splatfix-jobs/old/checkpoint/views/000/original.png')
    (measurement.parent.parent / 'scale.json').write_text('{"metric_scale": 2.5}')
    (root / 'trajectories/traj/trajectory.json').write_text('{}')
    return caption


def test_cache_copy_excludes_remote_caption_sources_and_preserves_scale(tmp_path):
    source, target = tmp_path / 'remote', tmp_path / 'canonical'
    target.mkdir()
    caption = remote_cache(source)
    persist_checkpoint_caches(source, target)
    copied = target / 'captions' / caption.name
    assert {p.name for p in copied.iterdir()} == {'caption.h5', 'manifest.json'}
    assert digest_file(copied / 'caption.h5') == digest_file(caption / 'caption.h5')
    assert (target / 'trajectories/traj/metric_alignment/scale.json').is_file()
    assert not (target / 'trajectories/traj/metric_alignment/measurement/images/anchor.png').is_symlink()
    persist_checkpoint_caches(source, target)  # Idempotent reattachment.
    (caption / 'caption.h5').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        persist_checkpoint_caches(source, target)
    assert (copied / 'caption.h5').read_bytes() != b'corrupt'


def test_missing_committed_caption_is_an_error_but_partial_attempt_is_ignored(tmp_path):
    source, target = tmp_path / 'remote', tmp_path / 'canonical'
    target.mkdir()
    partial = source / 'captions/.unfinished'
    partial.mkdir(parents=True)
    (partial / 'caption.tmp').write_bytes(b'partial')
    persist_checkpoint_caches(source, target)
    assert list((target / 'captions').iterdir()) == []
    (source / 'captions' / ('b' * 64)).mkdir()
    with pytest.raises(ValueError, match='missing its manifest or HDF5'):
        persist_checkpoint_caches(source, target)


def test_remote_caption_returns_to_checkpoint_and_is_staged_for_other_mode(tmp_path, monkeypatch):
    from splat_explorer import repair_lrz as lrz
    from splat_explorer.config import Config
    from splat_explorer.rendering.base import Camera
    from splat_explorer.scene_runs.lrz_transport import LrzSceneRunTransport
    from splat_explorer.scene_runs.store import SceneRunStore
    from splat_explorer.splatfix.checkpoint import Checkpoint
    from splat_explorer.splatfix.executor import SplatfixExecutor
    from splat_explorer.web.splatfix_studio import SplatfixStudio
    cfg = Config({'output': {'dir': str(tmp_path)}, 'agent': {'model': 'test'}})
    store = SceneRunStore(tmp_path / 'scene-runs')
    studio = SplatfixStudio(SimpleNamespace(cfg=cfg), store)
    scene = tmp_path / 'scene.ply'
    scene.write_text('source fixture')
    cp = Checkpoint.create(studio.checkpoint_root, scene, target_views=2)
    for index in range(2):
        cp.add_view(Image.new('RGB', (16, 16)), Camera(np.array([float(index), 0., 0.]), np.eye(3), width=16, height=16, fov_deg=60))
        edit = cp.image_path(cp.views[index]).with_name('repaired.png')
        Image.new('RGB', (16, 16), 'red').save(edit)
        cp.views[index]['repaired_rgb'] = str(edit.relative_to(cp.root))
    cp.save()
    fixture = tmp_path / 'remote-cache'
    caption = remote_cache(fixture)
    staged = []
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'workspace': '/dss/work', 'user': 'u', 'host': 'h'})
    monkeypatch.setattr(LrzSceneRunTransport, 'validate', lambda self: {})
    monkeypatch.setattr(lrz, 'sync_code_to_dss', lambda cfg: None)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(lrz, '_remote_pythonpath_exports', lambda cfg: '')
    monkeypatch.setattr(lrz, 'container_srun_prefix', lambda cfg: 'srun ')
    def transfer(argv):
        source, destination = argv[-2:]
        if source == str(cp.root) + '/':
            copied = tmp_path / ('staged-' + str(len(staged)))
            shutil.copytree(cp.root, copied, symlinks=True)
            staged.append(copied)
        elif source.startswith('u@h:'):
            shutil.copytree(fixture, Path(destination) / 'checkpoint', symlinks=True, dirs_exist_ok=True)
    monkeypatch.setattr(lrz, '_mux_run', transfer)
    monkeypatch.setattr(lrz, '_ssh_run', lambda cfg, command, **kwargs: SimpleNamespace(
        returncode=0, stderr='', stdout=json.dumps({'status': 'completed'}) if 'worker-status.json' in command else ''))
    executor = SplatfixExecutor(cfg, store)
    for mode in ('baseline', 'edited'):
        job = studio.create({'stage': 'repair', 'mode': mode, 'checkpoint': str(cp.root)})
        executor.execute(job['run_id'])
        assert store.get_run(job['run_id']).state.status.value == 'completed'
    assert len(staged) == 2
    assert not (staged[0] / 'captions').exists()
    reused = staged[1] / 'captions' / caption.name
    assert (reused / 'caption.h5').read_bytes() == (caption / 'caption.h5').read_bytes()
    assert not (reused / 'source').exists()
    assert (staged[1] / 'trajectories/traj/metric_alignment/scale.json').is_file()
