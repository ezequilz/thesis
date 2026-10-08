"""Portable per-run trajectory/quality charts from recorded cameras and metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..checkpoint import atomic_json
from .trajectory_diagnostics import summarize_trajectory

STAGES = {'baseline': ('Initial reconstruction', '#777777'),
          'artifixer': ('First diffusion pass', '#176b9b'),
          'artifixer3d': ('Distilled 3D', '#bd641b'),
          'artifixer3d_plus': ('Postprocessed 3D+', '#577c46')}


def _read(path):
    return json.loads(Path(path).read_text())


def chart_data(run_dir, metrics=None):
    root = Path(run_dir).resolve()
    result = _read(next(p for p in (root / 'result.json', root / 'inference-preview.json', root / 'partial-result.json') if p.is_file()))
    if not result.get('inference_split') and (root / 'benchmark-run.json').is_file():
        result['inference_split'] = _read(root / 'benchmark-run.json').get('inference_split', 'prepared/bicycle/split.json')
    remote = Path(result.get('output_dir', str(root)))

    def resolve(raw, base=root):
        path = Path(raw)
        if path.is_absolute() and not path.is_relative_to(root):
            path = root / path.relative_to(remote)
        elif not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Chart inputs must belong to the evaluated run')
        return path

    if result.get('inference_split') or (root / 'prepared/bicycle/split.json').is_file():
        split = resolve(result.get('inference_split', 'prepared/bicycle/split.json'))
        scenes = _read(split)['test']
        if len(scenes) != 1:
            raise ValueError('Chart requires one scene per result')
        scene, entry = next(iter(scenes.items()))
        transforms = _read(resolve(entry['transforms_path'], split.parent))
        selected = _read(resolve(entry['selected_indices_path'], split.parent))
        targets = (_read(resolve(entry['target_indices_path'], split.parent)) if entry.get('target_indices_path')
                   else [i for i in range(len(transforms['frames'])) if i not in selected])
        # Official loader sorts target index sets before inference.
        segments = [sorted(targets)]
    else:
        request = _read(root / 'request.json')
        if (root / 'preview-trajectory.json').is_file():
            request['trajectory'] = 'preview-trajectory.json'
        try:
            trajectory_path = resolve(request['trajectory'])
        except ValueError:
            # Historical workers kept the checkpoint beside results/. Only
            # remap that exact job-local layout, never an arbitrary JSON path.
            checkpoint = root.parent.parent / 'checkpoint'
            recorded_checkpoint = Path(request['checkpoint_root'])
            relative = Path(request['trajectory']).relative_to(recorded_checkpoint)
            trajectory_path = (checkpoint / relative).resolve()
            if (root.parent.name != 'results' or recorded_checkpoint.name != 'checkpoint'
                    or not checkpoint.resolve().is_relative_to(root.parent.parent.resolve())
                    or not trajectory_path.is_relative_to(checkpoint.resolve())):
                raise ValueError('Chart trajectory must belong to this job checkpoint')
        trajectory = _read(trajectory_path)
        transforms = trajectory['transforms']
        transforms = {**transforms, 'camera_convention': trajectory.get('camera_convention', 'opencv_c2w')}
        selected = [a['frame_index'] for a in trajectory['anchors']]
        segments = [[i for i in range(s['start'], s['start'] + s['count'])
                     if request.get('inference_target_policy') != 'exclude_saved_anchor_indices' or i not in selected]
                    for s in trajectory['segments']]
        segments = [s for s in segments if s]
        targets = [i for segment in segments for i in segment]
        scene = request.get('scene_id', 'Saved-view reconstruction')
    frames = transforms['frames']
    for indices in (targets, selected):
        if (any(type(i) is not int or not 0 <= i < len(frames) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid or duplicate chart camera indices')
    if not targets:
        raise ValueError('No generated target cameras available')
    poses = np.asarray([f['transform_matrix'] for f in frames], dtype=float)
    convention = transforms.get('camera_convention', 'opengl_c2w')
    if convention == 'opencv_c2w':
        poses = poses @ np.diag([1., -1., -1., 1.])
    elif convention != 'opengl_c2w':
        raise ValueError('Unsupported chart camera convention')
    # Prefer the complete source capture to a few potentially tilted references.
    # This also keeps the height axis stable when the generated path changes.
    up = poses[selected or list(range(len(poses))), :3, 1].mean(axis=0)
    axis_source = 'Mean reference camera +Y'
    source_split = root / 'prepared/bicycle/split.json'
    if source_split.is_file():
        source_scenes = _read(source_split).get('test', {})
        if len(source_scenes) == 1:
            source_entry = next(iter(source_scenes.values()))
            source_transforms = _read(resolve(source_entry['transforms_path'], source_split.parent))
            source_poses = np.asarray([f['transform_matrix'] for f in source_transforms['frames']], dtype=float)
            source_convention = source_transforms.get('camera_convention', 'opengl_c2w')
            if source_convention == 'opencv_c2w':
                source_poses = source_poses @ np.diag([1., -1., -1., 1.])
            elif source_convention != 'opengl_c2w':
                raise ValueError('Unsupported source camera convention')
            up = source_poses[:, :3, 1].mean(axis=0)
            axis_source = 'Mean source capture camera +Y'
    if np.linalg.norm(up) < 1e-8:
        raise ValueError('Ambiguous camera up; chart needs a known scene up direction')
    up /= np.linalg.norm(up)
    heights = poses[:, :3, 3] @ up
    motion = [summarize_trajectory(poses[segment], up=up) for segment in segments]
    metrics = metrics if metrics is not None else (_read(root / 'published-test-metrics.json')
                                                  if (root / 'published-test-metrics.json').is_file() else {})
    target_set = set(targets)
    scores = sorted([r for r in metrics.get('per_image', []) if r['prepared_index'] in target_set],
                    key=lambda r: r['prepared_index'])
    run_id = next((p for p in root.parts if p.startswith('run_')), root.name)
    return {'schema_version': 1, 'scene': scene, 'run_id': run_id, 'target_count': len(targets),
            'height_axis': axis_source, 'up': up.tolist(),
            'segments': [{'indices': s, 'heights': heights[s].tolist(), 'motion': m}
                         for s, m in zip(segments, motion)],
            'height_range': float(np.ptp(heights[targets])),
            'vertical_travel': sum(m['height']['total_travel'] for m in motion),
            'height_reversals': sum(m['height']['reversal_count'] for m in motion),
            'scores': [{'index': r['prepared_index'], 'metrics': r['metrics']} for r in scores],
            'score_scope': metrics.get('scope', 'No held-out photographic scores available for this run.'),
            'caption': 'Camera height and per-view quality from this run. Heights use estimated scene up; '
                       'distances are scene units. Different viewpoints confound sequence position with difficulty.'}


def _font(size):
    for name in ('DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_chart(data, destination):
    """No plotting/GPU dependencies; omit unavailable scores rather than invent them."""
    available = []
    for stage, (label, color) in STAGES.items():
        points = []
        for row in data['scores']:
            value = row['metrics'].get(stage, {}).get('psnr')
            if isinstance(value, (float, int)) and math.isfinite(value):
                points.append((row['index'], value))
        if points:
            available.append((label, points, color))
    im = Image.new('RGB', (1500, 1060 if available else 650), 'white')
    draw = ImageDraw.Draw(im)
    def text(x, y, value, size=19, color='#48545f'):
        draw.text((x, y), value, font=_font(size), fill=color)
    text(50, 24, f"{data['scene']} · trajectory and image quality", 30, '#17212a')
    text(50, 72, f"{data['run_id']} | {data['target_count']} generated targets | zero-based catalogue indices")

    def plot(box, series, title, ylabel):
        x0, y0, x1, y1 = box
        all_points = [p for _, points, _ in series for p in points]
        xs, ys = zip(*all_points)
        xmin, xmax = min(xs), max(xs)
        if xmin == xmax:
            xmin, xmax = xmin - .5, xmax + .5
        ymin, ymax = min(ys), max(ys)
        pad = (ymax - ymin) * .1 or .5
        ymin, ymax = ymin - pad, ymax + pad
        def xy(x, y):
            return x0 + (x-xmin)/(xmax-xmin)*(x1-x0), y1 - (y-ymin)/(ymax-ymin)*(y1-y0)
        text(x0, y0-48, title, 22, '#17212a')
        text(x0, y0-22, ylabel, 16)
        for value in np.linspace(ymin, ymax, 5):
            _, py = xy(xmin, value)
            draw.line((x0, py, x1, py), fill='#e3e7e9')
            text(x0-65, py-8, f'{value:.2g}', 15)
        for value in np.linspace(xmin, xmax, min(5, max(2, int(xmax-xmin)+1))):
            px, _ = xy(value, ymin)
            text(px-15, y1+8, f'{value:.0f}', 15)
        for label, points, color in series:
            pixels = [xy(x,y) for x,y in points]
            if len(pixels) > 1:
                draw.line(pixels, fill=color, width=3)
            for px, py in pixels if len(points) < 30 else []:
                draw.ellipse((px-3,py-3,px+3,py+3),fill=color)
        text(615, y1+32, 'Target catalogue index', 16)

    plot((115,175,1410,410), [('Actual path', list(zip(s['indices'],s['heights'])), '#176b9b')
                            for s in data['segments']],
         'Height along the inference sequence', 'Height along estimated up (scene units)')
    text(115, 475, f"{data['height_reversals']} height reversals · {data['vertical_travel']:.2f} units vertical travel · "
         f"{data['height_range']:.2f} units height range", 20, '#17212a')
    if available:
        plot((115,600,1410,835), available, f"Scores at {len(data['scores'])} evaluated target cameras",
             'RGB PSNR (dB; higher is better)')
        for i, (label, _, color) in enumerate(available):
            text(115+i*320, 903, label, 18, color)
        text(115, 952, 'Different viewpoints confound sequence position with difficulty. Lines connect evaluated samples.', 18)
        text(115, 985, 'Source: saved run cameras and per-image evaluation metrics. Non-finite scores are omitted.', 18)
    else:
        text(115, 535, 'No held-out photographic scores available; this chart reports camera motion only.', 19)
        text(115, 577, 'Source: saved run cameras. Separate inference segments are not joined across video cuts.', 18)
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    im.save(temporary, format='PNG')
    temporary.replace(destination)


def write_evaluation_chart(run_dir, metrics=None):
    root = Path(run_dir)
    data = chart_data(root, metrics)
    render_chart(data, root / 'trajectory-quality.png')
    atomic_json(root / 'trajectory-quality.json', data)
    return {'image': 'trajectory-quality.png', 'data': 'trajectory-quality.json', 'caption': data['caption']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(write_evaluation_chart(args.run_dir)))
