"""Run-local RGB reference memory, updated only between independent sequences.

Camera indices always refer to the shared calibrated C2W transforms. Generated
images are conditioning only: this module never promotes them to repaired GT.
"""
from pathlib import Path
from copy import copy
import shutil
import tempfile

import numpy as np

if __package__:
    from .handoff import digest, write_json
else:  # Standalone worker in the authors' environment.
    from handoff import digest, write_json


MAX_ADDITIONAL_REFERENCES = 6
BRIDGE_ADDITIONAL_REFERENCES = 2
POSITION_WEIGHT = 0.05
POLICY = {'version': 1, 'max_additional_references': MAX_ADDITIONAL_REFERENCES,
          'max_additional_bridge_references': BRIDGE_ADDITIONAL_REFERENCES,
          'rotation_metric': 'SO(3) geodesic / pi', 'position_weight': POSITION_WEIGHT,
          'selection': 'nearest-camera votes across all real target poses',
          'max_rotation_degrees': 90}


def enabled(options):
    value = options.get('generated_context_references', False)
    if type(value) is not bool:
        raise ValueError('generated_context_references must be a boolean')
    return value


class GeneratedContext:
    """An empty pool per inference/plus invocation, never populated by a disk scan.

    Selection snapshots are immutable even if a later bridge replaces a source
    prediction. The caller registers only successfully completed sequences.
    Off is a no-op, including for historical requests without the setting.
    """

    def __init__(self, root, options, cameras, *, phase='inference'):
        self.enabled = enabled(options)
        self.root = Path(root)
        self.phase = phase
        self.pool = {}
        self.records = []
        if not self.enabled:
            return
        self.poses = np.asarray([f['transform_matrix'] for f in cameras['frames']], dtype=float)
        if (self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4)
                or not np.isfinite(self.poses).all()):
            raise ValueError('Generated context requires finite calibrated 4x4 C2W cameras')
        rotations = self.poses[:, :3, :3]
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
            raise ValueError('Generated context requires proper camera rotations')
        extent = float(np.linalg.norm(np.ptp(self.poses[:, :3, 3], axis=0)))
        self.position_scale = extent if extent > 0 else 1.0
        self.size = (cameras['w'], cameras['h'])
        self.root.mkdir(parents=True, exist_ok=True)
        self._write(False)

    def _write(self, complete):
        write_json(self.root / f'{self.phase}-context.json', {
            'complete': complete, 'policy': POLICY, 'phase': self.phase,
            'position_scale': self.position_scale, 'sequences': self.records})

    def _same_pose(self, index, others):
        return any(np.allclose(self.poses[index], self.poses[i], rtol=0, atol=1e-7) for i in others)

    def augment(self, targets, reference_ids, reference_paths, *, required=(), exclude=(),
                limit=MAX_ADDITIONAL_REFERENCES, label='sequence'):
        """Append required boundary images, then rotation-dominant nearest views.

        Like WorldExplorer/SEVA, each pending camera votes for its nearest
        references. The secondary position term is bounded: it can change the
        score by at most nine degrees of rotation, regardless of scene units.
        Callers can exclude predictions being replaced (the bridge interior).
        Prior sequences may otherwise supply an already visited target pose.
        """
        if not self.enabled:
            return reference_ids, reference_paths
        if type(limit) is not int or not 0 <= limit <= MAX_ADDITIONAL_REFERENCES:
            raise ValueError('Invalid generated context reference limit')
        ids, paths = list(reference_ids), list(reference_paths)
        if len(ids) != len(paths) or len(ids) != len(set(ids)):
            raise ValueError('Context references must have distinct aligned camera indices and paths')
        selected = []
        for index in dict.fromkeys(required):
            if index in ids:
                continue
            if index not in self.pool:
                raise ValueError(f'Required context frame {index} has not completed generation')
            # Explicit overlaps must retain their exact camera ID for seeding.
            selected.append({'frame_index': index, 'reason': 'boundary'})
        if len(selected) > limit:
            raise ValueError('Too many required generated context references')
        chosen = ids + [r['frame_index'] for r in selected]
        candidates = [i for i in sorted(self.pool) if i not in exclude and i not in chosen
                      and not self._same_pose(i, chosen)]
        remaining = limit - len(selected)
        if candidates and targets and remaining:
            source, target = self.poses[candidates], self.poses[list(targets)]
            relative = source[:, None, :3, :3] @ target[None, :, :3, :3].transpose(0, 1, 3, 2)
            angles = np.arccos(np.clip((np.trace(relative, axis1=2, axis2=3) - 1) / 2, -1, 1))
            distances = np.linalg.norm(source[:, None, :3, 3] - target[None, :, :3, 3], axis=-1)
            scores = angles / np.pi + POSITION_WEIGHT * distances / (distances + self.position_scale)
            scores[angles > np.pi / 2 + 1e-8] = np.inf
            nearest = np.argsort(scores, axis=0, kind='stable')[:remaining]
            votes = np.zeros(len(candidates), dtype=int)
            for column in range(len(targets)):
                rows = nearest[:, column]
                np.add.at(votes, rows[np.isfinite(scores[rows, column])], 1)
            ranked = sorted(range(len(candidates)), key=lambda row: (
                -int(votes[row]), float(np.min(scores[row])), candidates[row]))
            for row in ranked:
                index = candidates[row]
                if not votes[row] or self._same_pose(index, chosen):
                    continue
                selected.append({'frame_index': index, 'reason': 'nearest_camera',
                                 'votes': int(votes[row]), 'best_score': float(np.min(scores[row])),
                                 'min_rotation_degrees': float(np.degrees(np.min(angles[row])))})
                chosen.append(index)
                if len(selected) == limit:
                    break
        folder = self.root / f'{self.phase}-context' / f'sequence-{len(self.records):03d}'
        folder.mkdir(parents=True, exist_ok=True)
        for record in selected:
            index = record['frame_index']
            source = self.pool[index]
            destination = folder / f'{index:05d}.png'
            shutil.copyfile(source['path'], destination)
            record.update(source_path=str(source['path']), source_sequence=source['sequence'],
                          path=str(destination), sha256=digest(destination))
            ids.append(index)
            paths.append(destination)
        self.records.append({'label': label, 'target_indices': list(targets),
                             'base_reference_indices': list(reference_ids), 'reference_indices': ids,
                             'required_reference_indices': list(dict.fromkeys(required)),
                             'additional_reference_limit': limit,
                             'generated_references': selected, 'complete': False})
        self._write(False)
        print(f'ARTIFIXER_CONTEXT: {label}, generated references {[r["frame_index"] for r in selected]}', flush=True)
        return ids, paths

    def complete_sequence(self, indices, folder):
        """Validate all real exports before making any of them selectable."""
        if not self.enabled:
            return
        from PIL import Image
        additions = {}
        for index in indices:
            path = Path(folder) / f'{index:05d}.png'
            with Image.open(path) as image:
                image.load()
                if image.size != self.size:
                    raise ValueError('Generated context image dimensions disagree with calibration')
            additions[index] = {'path': path, 'sequence': len(self.records) - 1}
        self.pool.update(additions)
        self.records[-1].update(complete=True, exported_indices=list(indices))
        self._write(False)

    def finish(self):
        if self.enabled:
            if any(not row['complete'] for row in self.records):
                raise ValueError('Generated context contains unfinished sequences')
            self._write(True)

    def run_sequence(self, process, pipe, item, args, output, *arguments):
        """Enabled calls publish only freshly produced outputs, never retry debris."""
        if not self.enabled:
            return process(pipe, item, args, output, *arguments)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        indices = [int(i) for i, valid in zip(item['frame_indices'], item['valid_frames_mask']) if valid]
        call_args = copy(args)
        call_args.replace_if_exists = True
        with tempfile.TemporaryDirectory(prefix='.context-call-', dir=output.parent) as temporary:
            staging = Path(temporary) / output.name
            result = process(pipe, item, call_args, staging, *arguments)
            folder = staging / 'splatfix/frames/batch_0000/pred'
            if {p.name for p in folder.glob('*.png')} != {f'{i:05d}.png' for i in indices}:
                raise RuntimeError('Context sequence did not produce every real target image')
            from PIL import Image
            for path in folder.glob('*.png'):
                with Image.open(path) as image:
                    image.load()
                    if image.size != self.size:
                        raise ValueError('Generated context image dimensions disagree with calibration')
            for path in staging.rglob('*'):
                if path.is_file():
                    destination = output / path.relative_to(staging)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)
        self.complete_sequence(indices, output / 'splatfix/frames/batch_0000/pred')
        return result
