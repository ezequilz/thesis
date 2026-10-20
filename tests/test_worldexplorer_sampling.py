"""Regression coverage for the authors' trajectory sampling contract."""
import json

import pytest

from splat_explorer.splatfix.worldexplorer import runner, service


def scene(tmp_path, lengths):
    root = tmp_path/'input_scaffold_3.0_test'/'img2trajvid'
    frames = []
    for index, length in enumerate(lengths):
        trajectory = root/f'{index:03}_in'
        folders = [('samples-rgb', length)]
        if index == 0:
            folders.insert(0, ('input', 8))
        for folder, count in folders:
            directory = trajectory/folder
            directory.mkdir(parents=True)
            for number in range(count):
                image = directory/f'{number:03}.png'
                image.write_bytes(f'{index}/{folder}/{number}'.encode())
                frames.append({'file_path': str(image)})
    (root/'transforms.json').write_text(json.dumps({'frames': frames}))
    return root


def test_failed_run_layout_preserves_inputs_and_samples_each_trajectory(tmp_path):
    # The real run has 8 scaffold + 32*~79 generated frames.
    root = scene(tmp_path, [79]*24+[78]*8)
    runner.select_images_for_vggt(root, 40)
    selected = list((root/'selected_for_vggt').iterdir())
    assert len(selected) == 40  # 8 inputs + one first frame per trajectory
    assert sum('_input_' in p.name for p in selected) == 8
    for index in range(32):
        assert (root/'selected_for_vggt'/f'{index:03}_in_samples-rgb_000.png').read_bytes() == f'{index}/samples-rgb/0'.encode()


def test_preserves_authors_floor_stride_including_rounding(tmp_path):
    root = scene(tmp_path, [10, 10])
    runner.select_images_for_vggt(root, 6)
    # Upstream uses stride=10//3, intentionally retained rather than
    # replacing it with midpoint, linspace, or exact-budget sampling.
    for index in range(2):
        names = {p.name for p in (root/'selected_for_vggt').glob(f'{index:03}_in_samples*')}
        assert names == {f'{index:03}_in_samples-rgb_{n:03}.png' for n in (0, 3, 6, 9)}


def test_short_trajectory_and_repeat_selection(tmp_path):
    root = scene(tmp_path, [2, 8])
    runner.select_images_for_vggt(root, 20)
    assert len(list((root/'selected_for_vggt').iterdir())) == 18
    runner.select_images_for_vggt(root, 2)
    assert len(list((root/'selected_for_vggt').iterdir())) == 10


def test_empty_trajectories_and_insufficient_budget_are_explicit(tmp_path):
    root = scene(tmp_path, [0, 0])
    with pytest.raises(ValueError, match='No generated trajectory'):
        runner.select_images_for_vggt(root, 40)
    root = scene(tmp_path/'second', [2, 2])
    with pytest.raises(ValueError, match='at least 2'):
        runner.select_images_for_vggt(root, 1)


def test_missing_source_preserves_previous_selection(tmp_path):
    root = scene(tmp_path, [3])
    runner.select_images_for_vggt(root, 1)
    before = {p.name: p.read_bytes() for p in (root/'selected_for_vggt').iterdir()}
    (root/'000_in/samples-rgb/000.png').unlink()
    with pytest.raises(FileNotFoundError):
        runner.select_images_for_vggt(root, 1)
    assert {p.name: p.read_bytes() for p in (root/'selected_for_vggt').iterdir()} == before


@pytest.mark.parametrize('budget', [8, 31])
def test_reject_insufficient_budget_before_gpu_submission(budget):
    with pytest.raises(ValueError, match='all 32 trajectories'):
        service.validate({'num_images_for_vggt': budget})
