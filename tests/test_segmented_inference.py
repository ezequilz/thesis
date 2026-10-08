from types import SimpleNamespace

import pytest

from splat_explorer.splatfix.segmented_inference import segment_dataset


@pytest.mark.parametrize('reference_count', [3, 6])
def test_separate_series_keep_every_reference_and_global_output_index(reference_count):
    dataset = SimpleNamespace(scene_ids=['scene'],
        target_ids_by_scene_id={'scene': {0, 1, 2, 3, 4}},
        train_ids_by_scene_id={'scene': set(range(5, 5 + reference_count))})
    provenance = {'segments': [{'target_indices': [0, 1]}, {'target_indices': [2]}, {'target_indices': [3, 4]}]}
    assert segment_dataset(dataset, provenance, SimpleNamespace) is dataset
    pairs = [pair for _, pair in dataset.inference_items]
    assert [pair.test_indices for pair in pairs] == [[0, 1], [2], [3, 4]]
    assert [pair.chunk_idx for pair in pairs] == [0, 1, 2]
    for pair in pairs:
        assert pair.neighbor_indices == list(range(5, 5 + reference_count))
        assert pair.is_test_frame == [True] * len(pair.test_indices)
        assert not pair.reversed


@pytest.mark.parametrize('groups', [[], [[0], [0, 1]], [[0]], [[0, 1, 2]], [[], [0, 1]], [[-1, 0, 1]]])
def test_invalid_coverage_fails_instead_of_silently_reusing_long_history(groups):
    dataset = SimpleNamespace(scene_ids=['scene'], target_ids_by_scene_id={'scene': {0, 1}},
                              train_ids_by_scene_id={'scene': {2, 3, 4}})
    with pytest.raises(ValueError):
        segment_dataset(dataset, {'segments': [{'target_indices': g} for g in groups]}, SimpleNamespace)


def test_cli_passes_segmented_dataset_to_original_inference(tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType
    from splat_explorer.splatfix.segmented_inference import main
    provenance = tmp_path / 'provenance.json'
    provenance.write_text(json.dumps({'segments': [{'target_indices': [0]}, {'target_indices': [1, 2]}]}))
    args = SimpleNamespace(evalset='reconstructed_colmap', render_trajectory='trajectory',
                           inference_pipeline='kv_cache', context_parallel_size=1)
    dataset = SimpleNamespace(scene_ids=['scene'], target_ids_by_scene_id={'scene': {0, 1, 2}},
                              train_ids_by_scene_id={'scene': {3, 4, 5}})
    invocations = []
    def upstream_main(received, dataset_factory):
        assert received is args
        invocations.append(dataset_factory(received, 0))
    upstream = SimpleNamespace(parse_args=lambda argv: args,
                               create_dataset=lambda args, rank: dataset, main=upstream_main)
    for name in ('model_eval', 'model_training', 'model_training.data', 'model_training.data.utils'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['model_eval'].run_inference = upstream
    sys.modules['model_training.data.utils'].InferencePair = SimpleNamespace
    main(['--repo', str(tmp_path), '--trajectory-provenance', str(provenance),
          '--render_trajectory', 'trajectory'])
    assert invocations == [dataset]
    assert [pair.test_indices for _, pair in dataset.inference_items] == [[0], [1, 2]]


def test_seeded_halves_keep_rgb_pose_and_reverse_output_indices_together(monkeypatch):
    import numpy as np
    import sys
    from types import ModuleType
    calls = []
    def camera_rays(**kwargs):
        calls.append(kwargs)
        return {'camera_rays': np.array(kwargs['frame_indices'])}
    utils = ModuleType('model_training.data.utils')
    utils.compute_camera_rays = camera_rays
    monkeypatch.setitem(sys.modules, 'model_training.data.utils', utils)
    class Dataset:
        scene_ids = ['scene']
        target_ids_by_scene_id = {'scene': {0, 1, 2, 3}}
        train_ids_by_scene_id = {'scene': {4, 5, 6}}
        transforms_by_scene_id = {'scene': {'frames': [{'transform_matrix': [[i]]} for i in range(7)]}}
        scenes_by_scene_id = {'scene': SimpleNamespace(camera_scale=2.5)}
        def __len__(self):
            return len(self.inference_items)
        def __getitem__(self, index):
            pair = self.inference_items[index][1]
            # Match upstream: reference images exist only as neighbors, not
            # numbered trajectory renders or opacity files.
            if any(i not in self.target_ids_by_scene_id['scene'] for i in pair.test_indices):
                raise FileNotFoundError('Reference has no trajectory render')
            return {'rgb_rendered': np.zeros((len(pair.test_indices), 3, 2, 2)),
                    'rgb_neighbors': np.stack([np.full((3, 2, 2), n) for n in pair.neighbor_indices]),
                    'opacity': np.zeros((len(pair.test_indices), 2, 2)),
                    'valid_frames_mask': np.ones(len(pair.test_indices), dtype=bool),
                    'frame_indices': np.array(pair.test_indices)}
    original = Dataset()
    dataset = segment_dataset(original, {'segments': [
        {'target_indices': [0, 1], 'seed_transform_matrix': [[4]]},
        {'target_indices': [3, 2], 'seed_transform_matrix': [[5]]},
    ]}, SimpleNamespace)
    assert len(dataset) == 2
    for index, (seed, targets) in enumerate([(4, [0, 1]), (5, [3, 2])]):
        item = dataset[index]
        assert item['frame_indices'].tolist() == [seed, *targets]
        assert item['camera_rays'].tolist() == [seed, *targets]
        assert calls[-1]['scale'] == 2.5
        assert calls[-1]['image_shape'] == (2, 2)
        assert calls[-1]['skip_vae_check'] is True
        assert item['frame_indices'][item['valid_frames_mask']].tolist() == targets
        assert (item['rgb_rendered'][0] == seed).all()
        assert (item['rgb_rendered'][1:] == 0).all()
        assert (item['opacity'][0] == 1).all()
        assert dataset.inference_items[index][1].neighbor_indices == [4, 5, 6]
        assert original.inference_items[index][1].test_indices == [seed, *targets]
        assert dataset[index]['frame_indices'].tolist() == [seed, *targets]
    with pytest.raises(ValueError, match='trusted reference'):
        segment_dataset(original, {'segments': [
            {'target_indices': [0, 1, 2, 3], 'seed_transform_matrix': [[0]]}]}, SimpleNamespace)


@pytest.mark.parametrize('stage,extra', [('repair', {'checkpoint': '/saved'}), ('benchmark', {'source': '/source'})])
def test_jobs_default_double_split_and_allow_single_split(stage, extra):
    from splat_explorer.splatfix.jobs import validate_job
    job = {'stage': stage, **extra}
    assert validate_job(job)['split_mode'] == 'double-split'
    assert validate_job({**job, 'split_mode': 'single-split'})['split_mode'] == 'single-split'
    with pytest.raises(ValueError, match='split_mode'):
        validate_job({**job, 'split_mode': 'invalid'})
