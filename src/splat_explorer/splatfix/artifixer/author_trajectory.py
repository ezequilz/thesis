"""Thin trajectory adapter for the pinned authors' 3DGRUT implementation.

Interpolation lives in ``threedgrut.render.Renderer``; importing that module is
intentionally deferred until this adapter runs in the authors' environment.
"""
from __future__ import annotations

import math


SPLIT_MODES = ('single-split', 'double-split')


def validate_split_mode(value):
    if value not in SPLIT_MODES:
        raise ValueError('split_mode must be single-split or double-split')
    return value


def build_author_orbit(reference_poses, intrinsics, metric_scale, *, target_poses=None,
                       loop=False, spacing=0.1, renderer=None, split_mode='double-split'):
    """Return target-only OpenGL C2W ``trajectory`` and auditable ``provenance``.

    ``reference_poses`` and optional ``target_poses`` are OpenGL C2W matrices.
    ``spacing`` has exactly the authors' CLI semantics: their combined
    translation/rotation distance is limited to ``spacing / metric_scale``.
    At least two distinct input cameras are required. With three or more nodes,
    the first reference starts the authored orbit. With two, authors' PCA sort
    determines the endpoints. ``loop`` is explicit and defaults to their False.
    If spacing yields only anchors, bounded refinement through the same helper
    creates intermediate targets; provenance records requested/effective spacing.

    ``split_mode`` defaults to double-split: disjoint inward halves between
    trusted references, each with its endpoint image as initial conditioning.
    Single-split retains one forward series per original waypoint leg.

    Exact references are context cameras, never generated supervision. Input
    and output deduplication is exact, so nearby genuine cameras stay distinct.
    ``renderer`` is an optional instance for tests; production needs no model
    construction, checkpoint, or CUDA allocation to invoke the original helper.
    """
    import numpy as np

    validate_split_mode(split_mode)
    if not isinstance(loop, bool):
        raise ValueError('loop must be a bool')
    metric_scale, spacing = float(metric_scale), float(spacing)
    if not math.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError('metric_scale must be finite and positive')
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be finite and positive')
    camera = {key: intrinsics[key] for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')}
    if any(not math.isfinite(float(value)) for value in camera.values()):
        raise ValueError('Camera intrinsics must be finite')
    if any(float(camera[key]) <= 0 for key in ('w', 'h', 'fl_x', 'fl_y')):
        raise ValueError('Image dimensions and focal lengths must be positive')
    if any(float(camera[key]) != int(camera[key]) for key in ('w', 'h')):
        raise ValueError('Image dimensions must be integers')
    camera = {key: int(value) if key in ('w', 'h') else float(value)
              for key, value in camera.items()}
    # The authors' transforms reader requires this tag even for an undistorted
    # pinhole camera. Distortion coefficients default to zero in that reader.
    camera['camera_model'] = 'OPENCV'

    def validate(poses, name):
        result = np.asarray(poses, dtype=np.float64)
        if result.size == 0:
            return np.empty((0, 4, 4), dtype=np.float64)
        if result.ndim != 3 or result.shape[1:] != (4, 4) or not np.isfinite(result).all():
            raise ValueError(f'{name} must contain finite 4x4 poses')
        if not np.allclose(result[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError(f'{name} must contain homogeneous C2W poses')
        rotations = result[:, :3, :3]
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
            raise ValueError(f'{name} must contain rigid camera rotations')
        return result

    references = validate(reference_poses, 'reference_poses')
    targets = validate([] if target_poses is None else target_poses, 'target_poses')
    if not len(references):
        raise ValueError('At least one reference camera is required')

    def key(pose):
        return tuple(pose.reshape(-1).tolist())

    # Consolidate nodes before interpolation, including targets equal to anchors.
    nodes, node_lookup = [], {}
    for pose in list(references) + list(targets):
        if key(pose) not in node_lookup:
            node_lookup[key(pose)] = len(nodes)
            nodes.append(pose)
    if len(nodes) < 2:
        raise ValueError('At least two distinct camera poses are required')
    reference_nodes = [node_lookup[key(p)] for p in references]
    target_nodes = [node_lookup[key(p)] for p in targets]
    reference_keys = {key(p) for p in references}
    flip = np.diag([1., -1., -1., 1.])
    cv_nodes = np.asarray(nodes) @ flip
    if renderer is None:
        from threedgrut.render import Renderer
        renderer = Renderer.__new__(Renderer)
    # Match render_orbit_trajectory exactly whenever there are at least two
    # genuine target nodes: all references are contexts and the center of
    # interest is estimated from targets. Never repeat anchors as targets.
    unique_reference_nodes = list(dict.fromkeys(reference_nodes))
    genuine_target_nodes = [i for i in dict.fromkeys(target_nodes) if i not in unique_reference_nodes]
    if len(genuine_target_nodes) >= 2:
        input_nodes = genuine_target_nodes
        training_nodes = unique_reference_nodes
        node_roles = 'all_references_as_training'
    elif len(nodes) >= 3:
        # The original helper requires >=2 poses besides training_poses. For
        # reference-only/few-target paths, designate the first reference as
        # the start context and let the remaining distinct nodes define orbit.
        input_nodes = list(range(1, len(nodes)))
        training_nodes = [0]
        node_roles = 'first_reference_as_training_remaining_nodes_as_waypoints'
    else:
        input_nodes = list(range(len(nodes)))
        training_nodes = []
        node_roles = 'two_node_authors_sort'
    starts_at_reference = bool(training_nodes)
    # Ask the same helper for its ordered nodes without intermediate samples.
    # Pose equality alone cannot identify boundaries: a leg may pass through a
    # different input camera on its way to its actual endpoint.
    waypoint_cv, _ = renderer.interpolate_orbit_poses(
        cv_nodes[input_nodes], loop=loop, interp_distance=np.finfo(np.float64).max,
        training_poses=cv_nodes[training_nodes] if training_nodes else None)
    waypoint_keys = [key(pose) for pose in np.asarray(waypoint_cv) @ flip]
    effective_distance = spacing / metric_scale
    fallback_reason = None
    attempted_distances = []
    # Preserve authors' requested spacing unless it produces only anchors.
    # Use their own distance helper to add an intermediate on the longest
    # segment. Retries remain bounded even for numerically degenerate poses.
    for attempt in range(3):
        attempted_distances.append(effective_distance)
        interpolated_cv, original_indices = renderer.interpolate_orbit_poses(
            cv_nodes[input_nodes], loop=loop, interp_distance=effective_distance,
            training_poses=cv_nodes[training_nodes] if training_nodes else None,
        )
        interpolated = validate(np.asarray(interpolated_cv) @ flip, 'author_output')
        original_indices = np.asarray(original_indices)
        if original_indices.shape != (len(interpolated),):
            raise ValueError('Authors interpolation returned an invalid node mapping')
        if any(key(pose) not in reference_keys for pose in interpolated):
            break
        if attempt == 2:
            raise ValueError('Authors interpolation produced no non-reference targets after bounded spacing refinement')
        fallback_reason = 'requested spacing produced only reference cameras'
        if attempt == 0:
            pairs = list(zip(interpolated_cv[:-1], interpolated_cv[1:]))
            if loop and len(interpolated_cv) > 1:
                pairs.append((interpolated_cv[-1], interpolated_cv[0]))
            distances = [math.hypot(*renderer.compute_pose_distance(first, second))
                         for first, second in pairs]
            effective_distance = max(distances, default=0.0) / 2
        else:
            effective_distance /= 2
        if not math.isfinite(effective_distance) or effective_distance <= 0:
            raise ValueError('Distinct cameras have no measurable distance for authors interpolation')
    # Split at original waypoints, before filtering references. Re-running the
    # orbit sorter on each pair could reverse legs (its PCA is pair-dependent).
    # Keeping the original interpolation preserves exact benchmark cameras.
    boundaries, cursor = [], 0
    for waypoint_key in waypoint_keys:
        while cursor < len(interpolated) and key(interpolated[cursor]) != waypoint_key:
            cursor += 1
        if cursor == len(interpolated):
            raise ValueError('Authors interpolation omitted an ordered waypoint')
        boundaries.append(cursor)
        cursor += 1
    legs = []
    for left, right in zip(boundaries, boundaries[1:]):
        legs.append((left, right))
    if loop and boundaries:
        legs.append((boundaries[-1], len(interpolated)))
    elif legs:
        legs[-1] = (legs[-1][0], len(interpolated))
    segments = []
    for start, end in legs:
        segments.append({'author_start': start, 'author_end_exclusive': end,
                         'target_indices': [], 'full_frame_indices': []})
    frames, mapping, removed, seen = [], [], [], {}
    full_frames, full_lookup = [], {}
    for raw_index, (pose, original_index) in enumerate(zip(interpolated, original_indices)):
        pose_key = key(pose)
        if pose_key not in full_lookup:
            full_lookup[pose_key] = len(full_frames)
            full_frames.append({'file_path': f'orbit_full/{len(full_frames):05d}.png',
                                'transform_matrix': pose.tolist()})
        if pose_key in reference_keys:
            removed.append({'author_frame_index': raw_index, 'reason': 'reference_camera'})
            continue
        if pose_key in seen:
            removed.append({'author_frame_index': raw_index, 'reason': 'duplicate_target',
                            'target_index': seen[pose_key]})
            continue
        frame_index = len(frames)
        seen[pose_key] = frame_index
        frames.append({'transform_matrix': pose.tolist()})
        # A previous leg may already have passed exactly through this node;
        # retain its named-camera mapping even if upstream labels it interpolated.
        node_index = node_lookup.get(pose_key)
        mapping.append({'target_index': frame_index, 'author_frame_index': raw_index,
                        'full_frame_index': full_lookup[pose_key],
                        'input_node_index': node_index,
                        'requested_target_indices': [i for i, n in enumerate(target_nodes) if n == node_index]})
        segment = next((s for s in segments if s['author_start'] <= raw_index < s['author_end_exclusive']), None)
        if segment is None:
            raise ValueError('Authors interpolation target is outside original waypoint legs')
        segment['target_indices'].append(frame_index)
        segment['full_frame_indices'].append(full_lookup[pose_key])
    if not frames:
        raise ValueError('Authors interpolation produced no non-reference targets; use smaller spacing')
    segments = [s for s in segments if s['target_indices']]
    if split_mode == 'double-split':
        # Only trusted reference cameras may seed a half-series. Benchmark
        # test cameras remain targets, including when they are orbit waypoints.
        anchors = [i for i in boundaries if key(interpolated[i]) in reference_keys]
        spans = list(zip(anchors, anchors[1:]))
        if loop:
            spans.append((anchors[-1], len(interpolated)))
        elif anchors[-1] < len(interpolated) - 1:
            spans.append((anchors[-1], len(interpolated)))
        segments = []
        for leg, (left, right) in enumerate(spans):
            entries = [m for m in mapping if left <= m['author_frame_index'] < right]
            end = (0 if loop else None) if right == len(interpolated) else right
            midpoint = (len(entries) + 1) // 2 if end is not None else len(entries)
            halves = [(entries[:midpoint], left, 'forward')]
            if end is not None:
                halves.append((entries[midpoint:][::-1], end, 'reverse'))
            for half, seed, direction in halves:
                if not half:
                    continue
                seed_key = key(interpolated[seed])
                segments.append({
                    'leg_index': leg, 'direction': direction,
                    'open_tail': end is None,
                    'author_start': left, 'author_end_exclusive': right,
                    'seed_full_frame_index': full_lookup[seed_key],
                    'seed_transform_matrix': interpolated[seed].tolist(),
                    'target_indices': [m['target_index'] for m in half],
                    'full_frame_indices': [m['full_frame_index'] for m in half],
                })
        if sorted(i for s in segments for i in s['target_indices']) != list(range(len(frames))):
            raise ValueError('Reference-anchored splits must cover every target exactly once')
    # Use the trusted cameras' up estimate for both paths. Interpolation is
    # continuous within each leg, but azimuth sorting need not produce a flat
    # or globally smooth path when the inputs span multiple capture heights.
    if __package__:
        from .trajectory_diagnostics import summarize_trajectory
    else:  # The GPU worker invokes this file directly in the authors' venv.
        from trajectory_diagnostics import summarize_trajectory
    reference_up = references[:, :3, 1].mean(axis=0)
    diagnostic_up = reference_up if np.linalg.norm(reference_up) >= 1e-8 else None
    motion = {
        'full_path': summarize_trajectory(
            [f['transform_matrix'] for f in full_frames], up=diagnostic_up),
        'generated_targets': summarize_trajectory(
            [f['transform_matrix'] for f in frames], up=diagnostic_up),
        'scope': 'Geometry only; no visibility, image quality, or safe-path guarantee. '
                 'Up is estimated from reference camera +Y when unambiguous.',
    }
    return {
        'trajectory': {**camera, 'camera_convention': 'opengl_c2w', 'frames': frames},
        'full_trajectory': {**camera, 'camera_convention': 'opengl_c2w', 'frames': full_frames},
        'provenance': {
            'implementation': 'threedgrut.render.Renderer.interpolate_orbit_poses',
            'camera_convention': 'opengl_c2w', 'interpolation_camera_convention': 'opencv_c2w',
            'metric_scale': metric_scale, 'spacing': spacing,
            'requested_spacing': spacing, 'effective_spacing': effective_distance * metric_scale,
            'normalized_interp_distance': effective_distance,
            'spacing_fallback_reason': fallback_reason,
            'attempted_normalized_distances': attempted_distances, 'loop': loop,
            'starts_at_first_reference': starts_at_reference, 'node_roles': node_roles,
            'reference_frame_indices': [full_lookup[key(p)] for p in references],
            'reference_node_indices': reference_nodes, 'requested_target_node_indices': target_nodes,
            'unique_input_nodes': len(nodes), 'author_frame_count': len(interpolated),
            'target_frame_count': len(frames), 'frames': mapping, 'removed_frames': removed,
            'split_mode': split_mode,
            'series_policy': ('independent_original_waypoint_legs_v1' if split_mode == 'single-split'
                              else 'reference_anchored_bidirectional_halves_v1'),
            'segments': segments,
            'motion_diagnostics': motion,
        },
    }


def main(argv=None):
    """Generate an orbit in the authors' environment without loading a model."""
    import argparse
    import json
    from pathlib import Path
    import sys
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, required=True)
    parser.add_argument('--selected-indices', type=Path, required=True)
    parser.add_argument('--target-names', type=Path)
    parser.add_argument('--metric-scale', type=float, required=True)
    parser.add_argument('--interp-distance', type=float, default=0.1)
    parser.add_argument('--split-mode', choices=SPLIT_MODES, default='double-split')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--full-output', type=Path)
    args = parser.parse_args(argv)
    source = json.loads(args.transforms.read_text())
    selected = json.loads(args.selected_indices.read_text())
    names = json.loads(args.target_names.read_text()) if args.target_names else []
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError('target-names must contain a JSON list of basenames')
    if len(names) != len(set(names)):
        raise ValueError('Requested target names must be unique')
    frames = source['frames']
    if (not isinstance(selected, list) or not selected
            or any(type(index) is not int or index < 0 or index >= len(frames) for index in selected)):
        raise ValueError('selected-indices must contain valid frame indices')
    by_name = {}
    for index, frame in enumerate(frames):
        by_name.setdefault(Path(frame.get('file_path', '')).name, []).append(index)
    target_indices = []
    for name in names:
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ValueError(f'Requested target {name!r} must uniquely identify a source frame')
        target_indices.append(matches[0])
    chosen = [frames[index] for index in selected + target_indices]
    intrinsics = {key: chosen[0].get(key, source.get(key))
                  for key in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')}
    for frame in chosen:
        if any(frame.get(key, source.get(key)) != value for key, value in intrinsics.items()):
            raise ValueError('Orbit requires one shared camera calibration')
        if any(float(frame.get(key, source.get(key, 0))) != 0 for key in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2')):
            raise ValueError('Orbit requires undistorted cameras')
    convention = source.get('camera_convention', 'opengl_c2w')
    if convention not in ('opengl_c2w', 'opencv_c2w'):
        raise ValueError(f'Unsupported input convention {convention!r}')
    conversion = np.diag([1., -1., -1., 1.]) if convention == 'opencv_c2w' else np.eye(4)
    references = [np.asarray(frames[index]['transform_matrix']) @ conversion for index in selected]
    targets = [np.asarray(frames[index]['transform_matrix']) @ conversion for index in target_indices]
    sys.path.insert(0, str(args.repo / 'thirdparty' / '3DGRUT-ArtiFixer'))
    sys.path.insert(0, str(args.repo))
    result = build_author_orbit(references, intrinsics, args.metric_scale, target_poses=targets,
                                loop=args.loop, spacing=args.interp_distance, split_mode=args.split_mode)
    name_to_index = {}
    for mapping in result['provenance']['frames']:
        requested = mapping['requested_target_indices']
        if requested:
            # Multiple named observations at the identical target pose share one
            # supervision frame; provenance retains every name for evaluation.
            output_index = mapping['target_index']
            for index in requested:
                name_to_index[names[index]] = output_index
    result['provenance'].update({'source_transforms': str(args.transforms),
                                 'selected_source_indices': selected,
                                 'requested_target_source_indices': target_indices,
                                 'target_name_to_index': name_to_index})
    outputs = [(args.output, result['trajectory']), (args.provenance, result['provenance'])]
    if args.full_output:
        outputs.append((args.full_output, result['full_trajectory']))
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + '\n')
    print(json.dumps({'trajectory': str(args.output), 'targets': len(result['trajectory']['frames']),
                      'provenance': str(args.provenance)}))


if __name__ == '__main__':
    main()
