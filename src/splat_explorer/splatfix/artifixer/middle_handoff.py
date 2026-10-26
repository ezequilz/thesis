"""Pose-exact inward generation and additive midpoint repair references."""
import json
import shutil
from pathlib import Path

if __package__:
    from .handoff import write_json
else:
    from handoff import write_json


def plan_legs(trajectory):
    """Partition targets once; both inward passes include the same midpoint.

    Odd camera-interval counts give the first half the extra interval. The
    central third of the leg intervals (rounded to nearest frame) includes
    the repaired midpoint itself.
    """
    anchors = sorted(a['frame_index'] for a in trajectory['anchors'])
    jobs = []
    owned = set()
    for segment in trajectory['segments']:
        indices = segment.get('indices')
        if indices is None:
            indices = list(range(segment['start'], segment['start'] + segment['count']))
        targets = [i for i in indices if i not in anchors]
        if not targets:
            continue
        seed = segment.get('seed_index', segment.get('cache_seed_index'))
        if seed not in anchors:
            raise ValueError('Middle handoff requires a repaired start camera for each leg')
        end = anchors[(anchors.index(seed) + 1) % len(anchors)]
        if owned.intersection(targets) or len(set(targets)) != len(targets):
            raise ValueError('Middle handoff legs must own distinct target indices')
        owned.update(targets)
        middle = len(targets) // 2
        count = max(1, (len(targets) + 2) // 3)
        left = min(middle, count // 2)
        right = min(len(targets) - middle - 1, count - 1 - left)
        jobs.append({'leg_index': len(jobs), 'midpoint': targets[middle],
                     'forward': [seed, *targets[:middle + 1]],
                     'backward': [end, *targets[middle:][::-1]],
                     'fuse_backward': targets[middle - left:middle + 1][::-1],
                     'fuse_forward': targets[middle:middle + right + 1]})
    expected = set(range(len(trajectory['frames']))) - set(anchors)
    if owned != expected:
        raise ValueError('Middle handoff legs must cover every non-reference camera')
    return jobs


def run(root, request, trajectory, *, pipe, args, device, encoded_prompt, cameras,
        repair_boundary, digest, previous_state=None, plus=False, memory=None):
    import numpy as np
    import torch
    from PIL import Image
    from model_eval.run_inference import process_item
    from model_training.data.utils import compute_camera_rays

    cfg = request['runtime']
    phase = 'plus' if plus else 'inference'
    if __package__:
        from .generated_context import GeneratedContext
    else:
        from generated_context import GeneratedContext
    own_memory = memory is None
    memory = memory or GeneratedContext(root, cfg, cameras, phase=phase)
    cp = Path(request['checkpoint_root'])
    original = [a['frame_index'] for a in trajectory['anchors']]
    state = ({key: list(value) if isinstance(value, list) else value for key, value in previous_state.items()}
             if previous_state else {'original_frame_indices': original, 'frame_indices': list(original),
                                     'paths': list(request['references'])})
    state['policy'] = 'add_midpoint_references'
    prediction = root / phase / 'splatfix/frames/batch_0000/pred'
    prediction.mkdir(parents=True, exist_ok=True)
    render_dir = Path(json.loads((root / 'distillation.json').read_text())['render_dir']) if plus else None
    audits, records, ownership, insertion_audits = [], [], {}, []
    manifest = root / (phase + '-middle.json')
    write_json(manifest, {'complete': False, 'legs': []})

    def rgb(path):
        with Image.open(path) as image:
            return torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255

    def generate(indices, leg, direction, required=()):
        seed = indices[0]
        neighbors, paths = state['frame_indices'], state['paths']
        neighbors, paths = memory.augment(indices, neighbors, paths, required=required,
                                          label=f'leg-{leg}-{direction}')
        references = torch.stack([rgb(path) for path in paths])
        if plus:
            renders = torch.stack([rgb(render_dir / 'renders' / f'{i:05d}.png') for i in indices])
            opacity = torch.stack([torch.from_numpy(np.array(Image.open(render_dir / 'opacity' / f'{i:05d}.png').convert('L'), dtype=np.float32) / 255) for i in indices])
        else:
            renders = torch.stack([rgb(cp / trajectory['frames'][i]['rgb']) for i in indices])
            opacity = torch.stack([torch.from_numpy(np.load(cp / trajectory['frames'][i]['opacity'], allow_pickle=False)) for i in indices])
        renders[0] = references[neighbors.index(seed)]
        opacity[0] = 1
        valid = torch.ones(len(indices), dtype=torch.bool)
        valid[0] = False
        output = root / (phase + '-middle') / str(leg) / direction
        item = {'scene_id': 'splatfix', 'rgb_rendered': renders, 'rgb_neighbors': references,
                'opacity': opacity, 'encoded_prompt': encoded_prompt,
                'frame_indices': torch.tensor(indices), 'valid_frames_mask': valid}
        item.update(compute_camera_rays(cameras, indices, neighbors, scale=request['camera_scale'],
                    image_shape=renders.shape[-2:], skip_vae_check=True))
        pipe.clear_inference_caches()
        if cfg.get('image_cache_insertion'):
            pipe.starter_inference_audit = None
        print(f'HANDOFF_MIDDLE: {phase} leg {leg} {direction} seed {seed}', flush=True)
        if len(indices) > 1:
            memory.run_sequence(process_item, pipe, item, args, output, 0, device, pipe.vae.config.scale_factor_temporal)
            if cfg.get('image_cache_insertion') and not (pipe.starter_inference_audit or {}).get('encoded_starter_preserved'):
                raise RuntimeError('Middle handoff starter preservation was not verified')
        else:
            memory.complete_sequence([], output / 'splatfix/frames/batch_0000/pred')
        record = {'output': phase, 'enabled': True, 'leg_index': leg, 'direction': direction,
                  'seed_global_index': seed, 'generation_global_indices': indices,
                  'export_global_indices': indices[1:], 'reference_frame_indices': list(neighbors),
                  'reference_paths': list(map(str, paths)), 'cache_preservation': cfg.get('image_cache_insertion', False)}
        if cfg.get('image_cache_insertion') and len(indices) > 1:
            reference = paths[neighbors.index(seed)]
            insertion_audits.append({**pipe.starter_inference_audit, **record,
                'reference_path': reference, 'reference_sha256': digest(reference),
                'starter_exported': False})
            write_json(root / (phase + '-insertion-audit.json'), {'series': insertion_audits})
        audits.append(record)
        folder = output / 'splatfix/frames/batch_0000/pred'
        for index in indices[1:]:
            shutil.copy2(folder / f'{index:05d}.png', prediction / f'{index:05d}.png')
            ownership[index] = len(audits) - 1
        return folder

    with torch.inference_mode():
        for job in plan_legs(trajectory):
            leg, midpoint = job['leg_index'], job['midpoint']
            forward = generate(job['forward'], leg, 'forward')
            backward = generate(job['backward'], leg, 'backward', required=job['forward'][1:][-2:])
            source = prediction / f'{midpoint:05d}.png'
            if midpoint not in state['frame_indices']:
                # The two images have the exact same global camera. Reuse the
                # existing center-old / vignette-new pixel-sprinkle preparation.
                shutil.copy2(forward / source.name, source)
                noise = {key: value for key, value in cfg.items() if key.startswith('handoff_')}
                noise.update(handoff_add_noise=True, handoff_noise_mode='repaired_pixel_sprinkle',
                             handoff_noise_vignette=True)
                repaired = repair_boundary(root, source, midpoint,
                    {**{k: v for k, v in cameras.items() if k != 'frames'}, **cameras['frames'][midpoint]},
                    phase=phase, reference=backward / source.name, reference_frame_index=midpoint,
                    reference_context='closest_repaired_image', noise=noise)
                state['frame_indices'].append(midpoint)
                state['paths'].append(str(repaired))
            else:
                shutil.copy2(state['paths'][state['frame_indices'].index(midpoint)], source)
            generate(job['fuse_backward'], leg, 'fuse_backward')
            generate(job['fuse_forward'], leg, 'fuse_forward')
            ownership.pop(midpoint, None)
            records.append(job)
            write_json(manifest, {'complete': False, 'legs': records})
    state['sha256'] = [digest(path) for path in state['paths']]
    # Record final ownership, excluding predictions overwritten by fusion.
    for slot, audit in enumerate(audits):
        audit['export_global_indices'] = [i for i in audit['export_global_indices'] if ownership.get(i) == slot]
    write_json(root / (phase + '-references.json'), state)
    write_json(root / (phase + '-seeding.json'), {'series': audits})
    write_json(manifest, {'complete': True, 'legs': records})
    if own_memory:
        memory.finish()
    return audits
