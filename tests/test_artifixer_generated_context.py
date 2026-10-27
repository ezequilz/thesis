"""Reference selection uses scene cameras and only completed real predictions."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from splat_explorer.splatfix.artifixer import generated_context as context


def pose(x=0, angle=0):
    a = np.radians(angle)
    result = np.eye(4)
    result[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    result[0, 3] = x
    return result.tolist()


def memory(tmp_path, poses):
    return context.GeneratedContext(tmp_path, {'generated_context_references': True},
                                    {'w': 8, 'h': 8, 'frames': [{'transform_matrix': p} for p in poses]})


def complete(pool, tmp_path, ids):
    pool.augment(ids, [], [], label='source')
    folder = tmp_path / 'pred'
    folder.mkdir(exist_ok=True)
    for i in ids:
        Image.new('RGB', (8, 8), (i, 0, 0)).save(folder / f'{i:05d}.png')
    pool.complete_sequence(ids, folder)
    return folder


@pytest.mark.parametrize('bad', ['on', 'off', 1, None, []])
def test_boolean_validation(bad):
    from splat_explorer.splatfix.artifixer.backend import validate_options
    with pytest.raises(ValueError, match='generated_context_references must be a boolean'):
        validate_options({'stage': 'repair', 'generated_context_references': bad})


def test_off_does_not_inspect_cameras_or_read_old_outputs(tmp_path):
    pool = context.GeneratedContext(tmp_path, {}, {})
    ids, paths = [5], ['unchanged']
    actual = pool.augment([6], ids, paths, required=[123])
    assert actual[0] is ids and actual[1] is paths
    pool.complete_sequence([6], tmp_path / 'missing')
    pool.finish()
    assert not list(tmp_path.iterdir())


def test_rotation_dominates_position_and_position_breaks_ties(tmp_path):
    pool = memory(tmp_path, [pose(0, 180), pose(100, 1), pose(2, 40), pose(10, 1), pose(0, 1)])
    complete(pool, tmp_path, [0, 1, 2, 3])
    ids, _ = pool.augment([4], [], [])
    assert ids == [3, 1, 2]  # Facing away is omitted, even at the target position.
    selected = pool.records[-1]['generated_references']
    assert selected[0]['min_rotation_degrees'] == pytest.approx(0, abs=1e-5)


def test_votes_cover_the_target_sequence_and_selection_is_scale_invariant(tmp_path):
    poses = [pose(1, 1), pose(2, 60), pose(3, 110), pose(4, 150)]
    poses += [pose(0, 1)] * 8 + [pose(0, 60)]
    selections = []
    for scale in [1, .001, 1000]:
        scaled = np.asarray(poses); scaled[:, :3, 3] *= scale
        root = tmp_path / str(scale)
        pool = memory(root, scaled.tolist())
        complete(pool, root, [0, 1, 2, 3])
        selections.append(pool.augment(list(range(4, 13)), [], [])[0])
        assert pool.records[-1]['generated_references'][0]['votes'] == 9
    assert selections[0] == selections[1] == selections[2]
    assert selections[0][:2] == [0, 1]


def test_boundaries_have_priority_and_budget_and_exact_camera_ids(tmp_path):
    pool = memory(tmp_path, [pose(i, i) for i in range(20)])
    complete(pool, tmp_path, list(range(18)))
    ids, paths = pool.augment([18], [19], ['repaired'], required=[16, 17, 16])
    assert ids[:3] == [19, 16, 17] and len(ids) == 7 and len(set(ids)) == 7
    assert paths[0] == 'repaired'
    assert [r['reason'] for r in pool.records[-1]['generated_references'][:2]] == ['boundary'] * 2
    with pytest.raises(ValueError, match='has not completed'):
        pool.augment([19], [], [], required=[18])


def test_duplicate_poses_and_bridge_interior_are_excluded(tmp_path):
    pool = memory(tmp_path, [pose(0), pose(0), pose(1), pose(2), pose(3), pose(4)])
    complete(pool, tmp_path, [0, 1, 2, 3, 4])
    ids, _ = pool.augment([3, 4], [5], ['ref'], exclude=[3, 4])
    assert len({0, 1} & set(ids)) == 1
    assert {3, 4}.isdisjoint(ids)
    ids, _ = pool.augment([3], [0], ['ref'])
    assert 1 not in ids  # An exact repaired pose always wins.


def test_snapshots_survive_source_replacement_and_fresh_run_starts_empty(tmp_path):
    poses = [pose(i) for i in range(4)]
    pool = memory(tmp_path, poses)
    folder = complete(pool, tmp_path, [1, 2])
    ids, paths = pool.augment([3], [0], ['ref'])
    before = paths[1].read_bytes()
    Image.new('RGB', (8, 8), 'blue').save(folder / f'{ids[1]:05d}.png')
    assert paths[1].read_bytes() == before
    with pytest.raises(ValueError, match='unfinished'):
        pool.finish()
    pool.complete_sequence([], folder)
    pool.finish()
    manifest = json.loads((tmp_path / 'inference-context.json').read_text())
    assert manifest['complete']
    assert manifest['sequences'][1]['generated_references'][0]['source_sequence'] == 0
    fresh = memory(tmp_path, poses)
    assert fresh.augment([3], [0], ['ref'])[0] == [0]


def test_invalid_or_partial_sequence_never_enters_memory(tmp_path):
    pool = memory(tmp_path, [pose(i) for i in range(4)])
    pool.augment([1, 2], [], [])
    Image.new('RGB', (8, 8)).save(tmp_path / '00001.png')
    with pytest.raises(FileNotFoundError):
        pool.complete_sequence([1, 2], tmp_path)
    assert not pool.pool and not pool.records[-1]['complete']
    Image.new('RGB', (9, 8)).save(tmp_path / '00002.png')
    with pytest.raises(ValueError, match='dimensions'):
        pool.complete_sequence([1, 2], tmp_path)
    assert not pool.pool


def test_stale_prediction_cannot_satisfy_a_new_call(tmp_path):
    pool = memory(tmp_path, [pose(), pose(1)])
    folder = tmp_path / 'inference/splatfix/frames/batch_0000/pred'
    folder.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(folder / '00001.png')
    pool.augment([1], [0], ['ref'])
    with pytest.raises(RuntimeError, match='every real target'):
        pool.run_sequence(lambda *args: None, None,
                          {'frame_indices': [1], 'valid_frames_mask': [True]},
                          SimpleNamespace(), tmp_path / 'inference')
    assert not pool.pool


def test_cli_can_override_config_in_both_directions(tmp_path, monkeypatch):
    import argparse
    from splat_explorer.config import Config
    from splat_explorer.splatfix.cli import add_parser, repair_splat
    from splat_explorer.splatfix.artifixer import repair
    received = []
    monkeypatch.setattr(repair, 'run_repair', lambda *a, **kw: received.append(kw['runtime']) or {})
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    for flag, value in [('--generated-context-references', True), ('--no-generated-context-references', False)]:
        args = parser.parse_args(['splatfix', 'repair', '--no-teacher-inference', str(tmp_path), '--output', str(tmp_path), flag])
        repair_splat(Config({'splatfix': {'runtime': {'generated_context_references': not value}}}), args)
        assert received[-1]['generated_context_references'] is value
