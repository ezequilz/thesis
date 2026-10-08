"""Run the pinned authors' inference with one dataset item per camera leg.

Only dataset item construction changes. Padding, denoising, rolling/sink KV
cache, decoding and global-index PNG output remain upstream implementations.
The pipeline initializes fresh caches for each item and clears them on decode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


class SeededDataset:
    """Prepend trusted RGB conditioning without exporting generated anchors."""
    def __init__(self, dataset, seeds):
        self.dataset, self.seeds = dataset, seeds

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        seed = self.seeds[index]
        if seed is not None:
            neighbors = self.dataset.inference_items[index][1].neighbor_indices
            item['rgb_rendered'][0] = item['rgb_neighbors'][neighbors.index(seed)]
            item['opacity'][0] = 1
            item['valid_frames_mask'][0] = False
        return item


def segment_dataset(dataset, provenance, pair_type):
    segments = provenance.get('segments')
    if not segments:
        raise ValueError('Independent inference requires trajectory segment provenance')
    groups = [s['target_indices'] for s in segments]
    flattened = [i for group in groups for i in group]
    if (any(not group for group in groups) or
            any(type(i) is not int or i < 0 for i in flattened) or
            len(set(flattened)) != len(flattened)):
        raise ValueError('Trajectory segments must contain unique nonnegative target indices')
    if len(dataset.scene_ids) != 1:
        raise ValueError('Trajectory provenance must describe exactly one scene')
    scene = dataset.scene_ids[0]
    if set(flattened) != set(dataset.target_ids_by_scene_id[scene]):
        raise ValueError('Trajectory segments do not cover the prepared targets exactly')
    references = sorted(dataset.train_ids_by_scene_id[scene])
    if not references or set(references) & set(flattened):
        raise ValueError('References must be nonempty and separate from targets')
    seeds, items = [], []
    for index, (segment, group) in enumerate(zip(segments, groups)):
        seed = None
        if 'seed_transform_matrix' in segment:
            frames = dataset.transforms_by_scene_id[scene]['frames']
            matches = [i for i in references if frames[i]['transform_matrix'] == segment['seed_transform_matrix']]
            if len(matches) != 1:
                raise ValueError('Series seed must uniquely match a trusted reference camera')
            seed = matches[0]
        seeds.append(seed)
        indices = [seed, *group] if seed is not None else list(group)
        items.append((scene, pair_type(
            neighbor_indices=list(references), test_indices=indices, reversed=False,
            chunk_idx=index, scene_id=scene,
            is_test_frame=([False] if seed is not None else []) + [True] * len(group))))
    dataset.inference_items = items
    return SeededDataset(dataset, seeds) if any(s is not None for s in seeds) else dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--trajectory-provenance', type=Path, required=True)
    adapter, remaining = parser.parse_known_args(argv)
    sys.path.insert(0, str(adapter.repo))
    from model_eval import run_inference as upstream
    from model_training.data.utils import InferencePair
    args = upstream.parse_args(remaining)
    if (args.evalset != 'reconstructed_colmap' or args.render_trajectory != 'trajectory'
            or args.inference_pipeline != 'kv_cache' or args.context_parallel_size != 1):
        raise ValueError('Segmented inference requires reconstructed_colmap trajectory with single-rank KV cache')
    provenance = json.loads(adapter.trajectory_provenance.read_text())

    def dataset_factory(args, rank):
        return segment_dataset(upstream.create_dataset(args, rank), provenance, InferencePair)

    upstream.main(args, dataset_factory=dataset_factory)


if __name__ == '__main__':
    main()
