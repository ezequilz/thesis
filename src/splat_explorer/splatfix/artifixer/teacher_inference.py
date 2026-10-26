"""Bounded single-GPU stage-1 inference, using the pinned authors' sampler.

One clip is read from disk at a time. References are encoded one at a time.
Short clips change temporal context; this is an explicit experimental policy,
not an exact equivalent of full-trajectory bidirectional inference.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

if __package__:
    from .inference_policy import inference_options, inference_size, model_runtime
else:
    from inference_policy import inference_options, inference_size, model_runtime


def load_pipe(args, device):
    # Shared with setup preflight, without importing any scene-runs sampler.
    path = Path(__file__).resolve().parents[2] / 'scene_runs_ext/model_loading.py'
    spec = importlib.util.spec_from_file_location('splatfix_model_loading', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_eval_pipe(args, device)


def resized_cameras(cameras, size):
    result = copy.deepcopy(cameras)
    width, height = size
    for source, dest in zip([cameras, *cameras['frames']], [result, *result['frames']]):
        sx, sy = width / source.get('w', cameras['w']), height / source.get('h', cameras['h'])
        for key, scale in (('fl_x', sx), ('fl_y', sy), ('cx', sx), ('cy', sy)):
            dest[key] = source.get(key, cameras[key]) * scale
        dest.update(w=width, h=height)
    return result


def clip_jobs(jobs, maximum):
    """Partition ordered targets exactly once, including reversed endpoint runs."""
    for segment, references, directory in jobs:
        indices = (list(segment['indices']) if 'indices' in segment else
                   list(range(segment['start'], segment['start'] + segment['count'])))
        for offset in range(0, len(indices), maximum):
            yield indices[offset:offset + maximum], references, directory


def memory_record(torch, device, phase, **extra):
    free, total = torch.cuda.mem_get_info(device)
    return {'phase': phase, 'time': time.time(),
            'allocated_gib': torch.cuda.memory_allocated(device) / 2**30,
            'reserved_gib': torch.cuda.memory_reserved(device) / 2**30,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
            'device_free_gib': free / 2**30, 'device_total_gib': total / 2**30,
            'process_peak_host_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            **extra}


def run(root, request, trajectory, *, plus=False, max_clips=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from model_eval.run_inference import build_parser, process_item
    from model_training.data.utils import compute_camera_rays, load_encoded_prompt
    if __package__:
        from .official_worker import inference_jobs, opengl_transforms, validate_caption
    else:
        from official_worker import inference_jobs, opengl_transforms, validate_caption

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = model_runtime(request['runtime'])
    if cfg['inference_model'] != '14b-teacher':
        raise ValueError('Teacher inference requires the stage-1 14B checkpoint')
    if not Path(cfg['checkpoint']).is_file():
        raise FileNotFoundError(f"Download the teacher checkpoint first: {cfg['checkpoint']}")
    policy = inference_options(cfg)
    request = {**request, 'runtime': cfg}
    phase = 'plus' if plus else 'inference'
    validate_caption(request['caption_path'], request['caption_sha256'])
    encoded_prompt, _ = load_encoded_prompt([Path(request['caption_path'])])
    cameras = opengl_transforms(trajectory)
    native_size = (cameras['w'], cameras['h'])
    size = inference_size(*native_size, cfg['teacher_resolution'])
    cameras = resized_cameras(cameras, size)
    args = build_parser().parse_args([
        '--checkpoint_pt', cfg['checkpoint'], '--model_id', cfg['model_id'],
        '--save_dir', str(root / phase), '--save_frame_outputs_only', '--replace_if_exists',
        '--evalset', 'reconstructed_colmap', '--render_trajectory', 'trajectory',
        '--inference_pipeline', 'bidirectional', '--attention_backend', 'native',
        '--num_inference_steps', str(cfg['teacher_steps']), '--max_neighbors_per_encode', '1',
        '--bidirectional_chunk_size', str(cfg['teacher_max_frames']),
    ])
    # create_dataset is bypassed by Step 4: enforce the clip bound here ourselves.
    jobs = clip_jobs(inference_jobs(request, trajectory, plus=plus), cfg['teacher_max_frames'])
    cp = Path(request['checkpoint_root'])
    anchors = [a['frame_index'] for a in trajectory['anchors']]
    render_dir = Path(json.loads((root / 'distillation.json').read_text())['render_dir']) if plus else None

    def rgb(path):
        with Image.open(path) as image:
            if image.size != native_size:
                raise ValueError(f'RGB dimensions disagree with trajectory: {path}')
            return torch.from_numpy(np.array(image.convert('RGB').resize(size, Image.Resampling.BILINEAR))).permute(2, 0, 1).float() / 255

    def alpha(index):
        if plus:
            with Image.open(render_dir / 'opacity' / f'{index:05d}.png') as image:
                values = np.array(image.convert('L'), dtype=np.float32) / 255
        else:
            values = np.array(np.load(cp / trajectory['frames'][index]['opacity'], mmap_mode='r', allow_pickle=False), copy=True)
        if values.shape != native_size[::-1] or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError('Opacity must match calibrated RGB and be finite in [0, 1]')
        tensor = torch.from_numpy(values).float()[None, None]
        return F.interpolate(tensor, size=size[::-1], mode='bilinear', align_corners=False)[0, 0]

    device = torch.device('cuda:0')
    torch.manual_seed(request['seed'])
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    manifest = root / f'{phase}-teacher.json'
    telemetry = root / f'{phase}-memory.jsonl'

    def record(stage, **extra):
        row = memory_record(torch, device, stage, **extra)
        with telemetry.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print('ARTIFIXER_MEMORY ' + json.dumps(row), flush=True)
        return row

    def save(complete, **extra):
        manifest.write_text(json.dumps({'complete': complete, **policy,
            'checkpoint': cfg['checkpoint'], 'model_id': cfg['model_id'],
            'inference_size': size, 'output_size': native_size,
            'output_resampling': 'bilinear' if size != native_size else None,
            'sampling': 'stage-1 bidirectional UniPC; CFG 5; independent bounded clips',
            'references': 'selected anchors according to endpoint-reference policy; sequential VAE encoding',
            'seeding': 'reference cross-attention only; no autoregressive starter/cache adapters',
            'clips': records, **extra}, indent=2))

    save(False)
    stage = 'model_loading'
    try:
        record(stage)
        with torch.inference_mode():
            pipe = load_pipe(args, device)
            record('model_loaded')
            for indices, reference_order, directory in jobs:
                if max_clips is not None and len(records) >= max_clips:
                    break
                # Exclude anchors from prediction ownership, as in the normal path.
                indices = [i for i in indices if i not in anchors]
                if not indices:
                    continue
                stage = 'clip_loading'
                start = time.monotonic()
                neighbors = [anchors[i] for i in reference_order]
                renders = torch.stack([rgb(render_dir / 'renders' / f'{i:05d}.png' if plus else
                                          cp / trajectory['frames'][i]['rgb']) for i in indices])
                item = {'scene_id': 'splatfix', 'rgb_rendered': renders,
                        'rgb_neighbors': torch.stack([rgb(request['references'][i]) for i in reference_order]),
                        'opacity': torch.stack([alpha(i) for i in indices]), 'encoded_prompt': encoded_prompt,
                        'frame_indices': torch.tensor(indices),
                        'valid_frames_mask': torch.ones(len(indices), dtype=torch.bool),
                        'target_h': native_size[1], 'target_w': native_size[0]}
                item.update(compute_camera_rays(cameras, indices, neighbors, scale=request['camera_scale'],
                            image_shape=size[::-1], skip_vae_check=True))
                stage = 'clip_inference'
                torch.cuda.reset_peak_memory_stats(device)
                process_item(pipe, item, args, root / directory, 0, device, pipe.vae.config.scale_factor_temporal)
                predicted = root / directory / 'splatfix/frames/batch_0000/pred'
                for index in indices:
                    with Image.open(predicted / f'{index:05d}.png') as image:
                        image.load()
                        if image.size != native_size:
                            raise RuntimeError('Teacher output dimensions disagree with saved cameras')
                torch.cuda.synchronize(device)
                row = record('clip_complete', frame_indices=indices, output=directory,
                             elapsed_seconds=time.monotonic() - start)
                records.append(row)
                del item, renders
                save(False)
            save(True, test_only=max_clips is not None)
    except Exception as exc:
        record('error', failed_phase=stage, error=f'{type(exc).__name__}: {exc}')
        save(False, failed_phase=stage, error=f'{type(exc).__name__}: {exc}')
        raise
    return records


def main():
    parser = argparse.ArgumentParser(description='Isolated teacher inference on an existing Step 4 request; no reconstruction')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-clips', type=int, default=1)
    parser.add_argument('--segment', type=int, default=0, help='Sample one saved trajectory segment')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--max-frames', type=int, default=17)
    parser.add_argument('--resolution', choices=('480p', 'native'), default='480p')
    args = parser.parse_args()
    if args.max_clips < 1 or args.output.exists():
        parser.error('Use a new output directory and at least one clip')
    request = json.loads(args.request.read_text())
    request['runtime'] = model_runtime({**request['runtime'], 'inference_model': '14b-teacher',
        'teacher_steps': args.steps, 'teacher_max_frames': args.max_frames, 'teacher_resolution': args.resolution})
    sys.path.insert(0, request['runtime']['repo'])
    trajectory = json.loads(Path(request['trajectory']).read_text())
    if not 0 <= args.segment < len(trajectory['segments']):
        parser.error('segment is outside the saved trajectory')
    trajectory['segments'] = [trajectory['segments'][args.segment]]
    request['runtime']['split_mode'] = 'double-split'  # Honor this selected segment, even from a no-split source.
    request['probe_source'] = {'request': str(args.request), 'segment': args.segment, 'max_clips': args.max_clips}
    args.output.mkdir(parents=True)
    (args.output / 'probe-request.json').write_text(json.dumps(request, indent=2))
    run(args.output, request, trajectory, max_clips=args.max_clips)


if __name__ == '__main__':
    main()
