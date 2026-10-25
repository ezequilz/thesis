"""A fresh, single-block bridge conditioned on RGB references from both legs.

All indices below are RGB camera indices, never temporal VAE latent indices.
The checkpoint stays unchanged; optional seed preservation is scoped to the bridge.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import copy
import json
import math
from pathlib import Path
import shutil

if __package__:
    from .handoff import digest, write_json
    from . import bridge_seeding
else:  # Standalone worker in the pinned authors' environment.
    from handoff import digest, write_json
    import bridge_seeding


BLOCK_FRAMES = 25
MAX_POSES = 64
OVERLAP = 2


def enabled(options, *, default=False):
    value = options.get('connected_trajectory_generation', default)
    if type(value) is not bool:
        raise ValueError('connected_trajectory_generation must be a boolean')
    return value


def configure(options, *, default=True):
    """Resolve new-run defaults; explicit fallback modes keep their own policy."""
    compatible = (not options.get('handoff_generation', False)
                  and options.get('image_trajectory', 'checkpoint') != 'endpoints'
                  and options.get('split_mode', 'double-split') == 'double-split')
    result = enabled(options, default=default and compatible)
    if result and not compatible:
        raise ValueError('Connected trajectory generation requires double-split checkpoint '
                         'trajectories with handoff generation off')
    return result


def plan_legs(trajectory):
    """Keep inward halves; bridge a centered middle third plus real overlaps.

    Spans include both endpoint cameras. Upstream pads each logical window to
    one 25-RGB block; padding never becomes trajectory ownership.
    """
    count = len(trajectory['frames'])
    anchors = sorted(a['frame_index'] for a in trajectory['anchors'])
    if count <= 1 and anchors == list(range(count)) and not trajectory['segments']:
        return []  # No interior or pair of distinct endpoints to bridge.
    if (len(anchors) < 2 or len(set(anchors)) != len(anchors)
            or any(type(i) is not int or not 0 <= i < count for i in anchors)):
        raise ValueError('Connected trajectories require distinct endpoint reference cameras')
    loop = trajectory.get('recipe', {}).get('orbit', {}).get('provenance', {}).get('loop', False)
    spans = [list(range(left, right + 1)) for left, right in zip(anchors, anchors[1:])]
    if loop:
        spans.append([*range(anchors[-1], count), *range(anchors[0] + 1)])
    expected, plans = [], []
    for leg, ordered in enumerate(spans):
        n = len(ordered)
        if n > MAX_POSES:
            raise ValueError(f'Connected trajectory {leg} has {n} poses; maximum is {MAX_POSES}. '
                             'Use shorter trajectories or disable connected trajectory generation.')
        interior = ordered[1:-1]
        midpoint = (len(interior) + 1) // 2
        halves = [(ordered[0], interior[:midpoint]), (ordered[-1], interior[midpoint:][::-1])]
        expected.extend((seed, targets) for seed, targets in halves if targets)
        outer = [[seed, *targets[:BLOCK_FRAMES - 1]] for seed, targets in halves if targets]
        plan = {'leg_index': leg, 'trajectory_indices': ordered, 'outer_generation_indices': outer,
                'target_indices': [], 'keep_indices': [], 'overlap_indices': [], 'reference_indices': []}
        if n >= 3:
            bridge_count = max(1, min(n - 2, (n + 1) // 3))
            start = (n - bridge_count) // 2
            stop = start + bridge_count
            window_start, window_stop = max(0, start - OVERLAP), min(n, stop + OVERLAP)
            window = ordered[window_start:window_stop]
            left_end, right_start = start - 1, stop
            left = sorted(i for i in {0, round(.4 * left_end), round(.7 * left_end),
                                     left_end - 2, left_end - 1, left_end} if 0 <= i < start)
            distance = n - 1 - right_start
            right = sorted(i for i in {right_start, right_start + 1, right_start + 2,
                                       right_start + round(.3 * distance),
                                       right_start + round(.6 * distance), n - 1} if stop <= i < n)
            overlap_sides = {'left': ordered[window_start:start], 'right': ordered[stop:window_stop]}
            plan.update(target_indices=window, keep_indices=ordered[start:stop],
                        overlap_indices=overlap_sides['left'] + overlap_sides['right'],
                        overlap_sides=overlap_sides, reference_indices=[ordered[i] for i in left + right],
                        padding_count=BLOCK_FRAMES - len(window), padded_frame_index=window[-1])
        else:
            plan['skip_reason'] = 'no interior poses to bridge; preserve endpoints'
        plans.append(plan)
    actual = []
    for segment in trajectory['segments']:
        indices = (list(segment['indices']) if 'indices' in segment else
                   list(range(segment['start'], segment['start'] + segment['count'])))
        if indices:
            actual.append((segment.get('seed_index', segment.get('cache_seed_index')), indices))
    # Order of the segment records may differ, but direction and exact ownership may not.
    if sorted(actual, key=lambda x: x[1]) != sorted(expected, key=lambda x: x[1]):
        raise ValueError('Connected trajectories require the original double-split inward halves; rebuild the orbit')
    covered = [i for _, group in actual for i in group]
    if sorted(covered) != [i for i in range(count) if i not in anchors]:
        raise ValueError('Connected trajectories must cover every non-reference camera exactly once; open tails are unsupported')
    return plans


def compare_overlap(existing, generated, destination):
    """Save exact boundary pairs and pixel errors; no uncalibrated pass threshold."""
    import numpy as np
    from PIL import Image
    with Image.open(existing) as a, Image.open(generated) as b:
        a, b = a.convert('RGB'), b.convert('RGB')
        if a.size != b.size:
            raise ValueError('Connected overlap dimensions differ from the preserved leg')
        difference = np.asarray(a, dtype=np.float64) / 255 - np.asarray(b, dtype=np.float64) / 255
        mse = float(np.mean(difference ** 2))
        comparison = Image.new('RGB', (a.width * 3, a.height))
        comparison.paste(a, (0, 0))
        comparison.paste(b, (a.width, 0))
        comparison.paste(Image.fromarray(np.rint(np.abs(difference) * 255).astype('uint8')), (2 * a.width, 0))
        comparison.save(destination)
    return {'mae': float(np.mean(np.abs(difference))), 'rmse': math.sqrt(mse),
            'psnr_db': -10 * math.log10(mse) if mse else None, 'identical': mse == 0}


def run(root, request, trajectory, *, pipe, args, device, encoded_prompt, cameras,
        normal_sampler, plans=None, plus=False):
    """Generate both outer legs, then publish only the dynamic middle third.

    Each block gets fresh temporal, prompt and reference caches. Seeding is a
    bridge-specific option, independent of the outer-leg starter mechanism.
    """
    import numpy as np
    import torch
    from PIL import Image
    from model_eval.run_inference import process_item
    from model_training.data.utils import compute_camera_rays

    plans = plan_legs(trajectory) if plans is None else plans
    if (args.inference_pipeline != 'kv_cache' or pipe.frames_per_block != 7
            or pipe.vae.config.scale_factor_temporal != 4):
        raise ValueError('Connected generation requires the causal seven-latent-frame / 25-RGB-frame pipeline')
    root = Path(root)
    phase = 'plus' if plus else 'inference'
    output = root / phase
    prediction = output / 'splatfix/frames/batch_0000/pred'
    cp = Path(request['checkpoint_root'])
    render_dir = Path(json.loads((root / 'distillation.json').read_text())['render_dir']) if plus else None
    anchor_ids = [a['frame_index'] for a in trajectory['anchors']]
    if len(anchor_ids) != len(request['references']):
        raise ValueError('Connected trajectory reference paths do not match the anchor cameras')
    anchors = dict(zip(anchor_ids, map(Path, request['references'])))
    original_hashes = {i: digest(path) for i, path in anchors.items()}
    manifest_path = root / f'{phase}-connected.json'
    selected_seeding = bridge_seeding.mode(request['runtime'])
    manifest = {'bridge_seeding': selected_seeding, 'schema_version': 1, 'complete': False, 'phase': phase, 'legs': [],
                'checkpoint': request['runtime']['checkpoint'], 'num_inference_steps': 4,
                'frames_per_block': 7, 'overlap_policy': 'diagnostic only; no quality acceptance threshold',
                'reference_policy': 'unchanged endpoint references plus generated outer RGBs, at most 12',
                'plan': plans}
    write_json(manifest_path, manifest)
    sampler_args = copy(args)
    sampler_args.num_inference_steps = 4
    sampler_args.replace_if_exists = True
    seeding_audits = []

    def rgb(path):
        with Image.open(path) as image:
            return torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255

    def plus_opacity(index):
        with Image.open(render_dir / 'opacity' / f'{index:05d}.png') as image:
            return torch.from_numpy(np.array(image.convert('L'), dtype=np.float32) / 255)

    def generate(indices, reference_ids, reference_paths, destination, *, seed=None, noise_seed,
                 bridge_plan=None):
        bridge = bridge_plan is not None
        if plus:
            renders = torch.stack([rgb(render_dir / 'renders' / f'{i:05d}.png') for i in indices])
            opacity = torch.stack([plus_opacity(i) for i in indices])
        else:
            renders = torch.stack([rgb(cp / trajectory['frames'][i]['rgb']) for i in indices])
            opacity = torch.stack([torch.from_numpy(np.load(cp / trajectory['frames'][i]['opacity'], allow_pickle=False)) for i in indices])
        references = torch.stack([rgb(p) for p in reference_paths])
        if tuple(references.shape[-2:]) != tuple(renders.shape[-2:]):
            raise ValueError('Connected references must match calibrated target dimensions')
        valid = torch.ones(len(indices), dtype=torch.bool)
        for slot, i in enumerate(indices):
            if i in anchors and not bridge:
                valid[slot] = False
        if seed is not None:
            renders[0] = references[reference_ids.index(seed)]
            opacity[0] = 1
            valid[0] = False
        if bridge and selected_seeding != 'off':
            # The established initial-image path, at every available real overlap.
            # Keep them valid so regenerated overlaps are exported for comparison.
            for i in bridge_plan['overlap_indices']:
                slot = indices.index(i)
                renders[slot] = references[reference_ids.index(indices[slot])]
                opacity[slot] = 1
        item = {'scene_id': 'splatfix', 'rgb_rendered': renders, 'rgb_neighbors': references,
                'opacity': opacity, 'encoded_prompt': encoded_prompt,
                'frame_indices': torch.tensor(indices), 'valid_frames_mask': valid}
        # Compute padded camera groups from the repeated final *RGB pose*, not
        # by repeating an average of the last few real cameras. process_item
        # pads RGB/opacity itself and trims those synthetic outputs afterward.
        camera_indices = indices + [indices[-1]] * (BLOCK_FRAMES - len(indices)) if bridge else indices
        item.update(compute_camera_rays(cameras, camera_indices, reference_ids,
                    scale=request['camera_scale'], image_shape=renders.shape[-2:], skip_vae_check=True))
        torch.manual_seed(noise_seed)
        pipe.clear_inference_caches()
        try:
            context = (bridge_seeding.preserved_bridge(pipe, normal_sampler,
                       real_frame_count=len(indices), overlap_slots=[indices.index(i) for i in bridge_plan['overlap_indices']])
                       if bridge and selected_seeding == 'seed_preserve' else nullcontext(None))
            with context as audit:
                process_item(pipe, item, sampler_args, destination, 0, device, 4)
            return audit
        finally:
            pipe.clear_inference_caches()

    with torch.inference_mode():
        for plan in plans:
            leg = plan['leg_index']
            archive = root / f'{phase}-connected' / f'leg-{leg:03d}'
            archive.mkdir(parents=True, exist_ok=True)
            print(f'CONNECTED_OUTER: {phase} leg {leg}, {len(plan["trajectory_indices"])} poses', flush=True)
            for direction, indices in enumerate(plan['outer_generation_indices']):
                seed = indices[0]
                seeding = request['runtime'].get('initial_image_seeding', True) or request['runtime'].get('image_cache_insertion', False)
                generate(indices, anchor_ids, list(anchors.values()), output,
                         seed=seed if seeding else None, noise_seed=request['seed'] + leg * 3 + direction)
                if request['runtime'].get('image_cache_insertion', False):
                    if not (pipe.starter_inference_audit or {}).get('encoded_starter_preserved'):
                        raise RuntimeError('Connected outer-leg starter preservation was not verified')
                seeding_audits.append({'output': phase, 'enabled': seeding, 'seed_global_index': seed,
                                      'cache_preservation': request['runtime'].get('image_cache_insertion', False),
                                      'generation_global_indices': indices,
                                      'export_global_indices': indices[1:]})
            if not plan['target_indices']:
                manifest['legs'].append({**plan, 'status': 'skipped'})
                write_json(manifest_path, manifest)
                continue
            # Snapshot outer predictions before replacing any interior images.
            outer = archive / 'outer'
            outer.mkdir(exist_ok=True)
            for indices in plan['outer_generation_indices']:
                for i in indices[1:]:
                    shutil.copyfile(prediction / f'{i:05d}.png', outer / f'{i:05d}.png')
            reference_ids = plan['reference_indices']
            reference_paths = [anchors[i] if i in anchors else outer / f'{i:05d}.png' for i in reference_ids]
            reference_records = [{'frame_index': i, 'kind': 'endpoint_reference' if i in anchors else 'generated_outer',
                                  'path': str(p), 'sha256': digest(p)} for i, p in zip(reference_ids, reference_paths)]
            bridge_output = archive / 'bridge'
            print(f'CONNECTED_BRIDGE: {phase} leg {leg}, targets {plan["target_indices"]}, references {reference_ids}', flush=True)
            # Reuse the normal joint sampler even when outer legs use the older
            # prefix-only exact-starter adaptation. Never merge their caches.
            seed_metadata = {'mode': selected_seeding,
                             'seed_frame_indices': plan['overlap_indices'] if selected_seeding != 'off' else []}
            if selected_seeding == 'seed_preserve':
                layout = bridge_seeding.seed_layout(len(plan['target_indices']),
                    [plan['target_indices'].index(i) for i in plan['overlap_indices']])
                seed_metadata.update(layout)
                seed_metadata['constrained_rgb_frame_indices'] = [plan['target_indices'][i]
                    for i in layout['constrained_rgb_local_indices']]
            with bridge_seeding.temporary_method(pipe, 'generate_samples_from_batch', normal_sampler):
                audit = generate(plan['target_indices'], reference_ids, reference_paths, bridge_output,
                                 noise_seed=request['seed'] + leg * 3 + 2, bridge_plan=plan)
                if audit:
                    seed_metadata.update(audit)
            bridge = bridge_output / 'splatfix/frames/batch_0000/pred'
            expected = {f'{i:05d}.png' for i in plan['target_indices']}
            if {p.name for p in bridge.glob('*.png')} != expected:
                raise RuntimeError(f'Connected bridge did not produce exactly {len(expected)} real target images')
            # Validate all outputs before changing the canonical inference set.
            for path in bridge.glob('*.png'):
                with Image.open(path) as image:
                    image.load()
                    if image.size != (cameras['w'], cameras['h']):
                        raise ValueError('Connected bridge output dimensions differ from the calibrated cameras')
            overlaps = []
            for i in plan['overlap_indices']:
                comparison = archive / f'overlap-{i:05d}.png'
                overlaps.append({'frame_index': i, 'side': 'left' if i in plan['overlap_sides']['left'] else 'right',
                                 **compare_overlap(anchors[i] if i in anchors else outer / f'{i:05d}.png', bridge / f'{i:05d}.png', comparison),
                                 'comparison': str(comparison.relative_to(root))})
            side_errors = {side: sum(row['mae'] for row in overlaps if row['side'] == side) / len(plan['overlap_sides'][side])
                           for side in ('left', 'right')}
            retained = {i: digest(outer / f'{i:05d}.png') for indices in plan['outer_generation_indices']
                        for i in indices[1:] if i not in plan['keep_indices']}
            for i in plan['keep_indices']:
                target = prediction / f'{i:05d}.png'
                temporary = target.with_suffix('.tmp')
                shutil.copyfile(bridge / target.name, temporary)
                temporary.replace(target)
            if any(digest(prediction / f'{i:05d}.png') != checksum for i, checksum in retained.items()):
                raise RuntimeError('Connected generation modified a preserved outer image')
            manifest['legs'].append({**plan, 'status': 'complete', 'references': reference_records,
                                     'seeding': seed_metadata,
                                     'bridge_noise_seed': request['seed'] + leg * 3 + 2,
                                     'overlaps': overlaps, 'boundary_mae': side_errors,
                                     'preserved_outer_sha256': retained})
            write_json(manifest_path, manifest)
    if any(digest(anchors[i]) != checksum for i, checksum in original_hashes.items()):
        raise RuntimeError('Connected generation modified an endpoint reference')
    expected = {f'{i:05d}.png' for i in range(len(trajectory['frames'])) if i not in anchors}
    if {p.name for p in prediction.glob('*.png')} != expected:
        raise RuntimeError('Connected generation did not cover every target camera exactly once')
    manifest['complete'] = True
    write_json(manifest_path, manifest)
    write_json(root / f'{phase}-seeding.json', {'series': seeding_audits})
    return seeding_audits
