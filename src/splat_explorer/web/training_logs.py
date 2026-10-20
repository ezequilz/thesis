"""Readable, cached exports of locally saved TensorBoard scalar logs."""
import hashlib
import math
import threading
import uuid
from pathlib import Path

from ..splatfix.checkpoint import atomic_json
from ..splatfix.jobs import read_json

_lock = threading.Lock()


def export_training_logs(root, *, event_subdir='artifixer3d/runs'):
    """Decode only event files belonging to this result, without touching originals."""
    root = Path(root).resolve()
    files = sorted(p for p in (root / event_subdir).rglob('events.out.tfevents.*')
                   if p.is_file() and p.resolve().is_relative_to(root))
    if not files:
        return [], None, None
    cache = root / '.training-metrics'
    if cache.is_symlink():
        return [], None, 'Training metric cache must not be a symlink.'
    signature = [[str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns] for p in files]
    with _lock:
        cache.mkdir(exist_ok=True)
        metadata = cache / 'index.json'
        exports = [cache / (hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest()[:16] + '.log')
                   for p in files]
        chart = cache / 'scalars.json'
        if any(p.is_symlink() for p in [metadata, chart, *exports]):
            return [], None, 'Training metric cache contains a symlink.'
        previous = read_json(metadata, {})
        if previous.get('signature') == signature and chart.is_file() and all(p.is_file() for p in exports):
            return list(zip(files, exports)), chart, None
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return [], None, 'Install the viewer dependencies to read TensorBoard training logs.'
        all_points, metrics = {}, set()
        for source, destination in zip(files, exports):
            accumulator = EventAccumulator(str(source), size_guidance={'scalars': 0}, purge_orphaned_data=False)
            try:
                accumulator.Reload()
            except Exception as exc:
                return [], None, f'Cannot read saved TensorBoard log {source.name}: {exc}'
            points = {}
            for tag in accumulator.Tags()['scalars']:
                for event in accumulator.Scalars(tag):
                    if not math.isfinite(event.value) or not math.isfinite(event.wall_time):
                        continue
                    point = points.setdefault(event.step, {'iteration': event.step, 'wall_time': event.wall_time})
                    point['wall_time'] = min(point['wall_time'], event.wall_time)
                    point[tag] = event.value
                    metrics.add(tag)
            lines = ['TensorBoard scalar export: ' + str(source.relative_to(root)),
                     'Recorded values; last value per tag and iteration. Timestamps are Unix seconds.']
            for step, point in sorted(points.items()):
                lines.append(f"Iteration {step} | timestamp={point['wall_time']:.6f} | " +
                             ' | '.join(f'{tag}={value:.9g}' for tag, value in point.items()
                                        if tag not in ('iteration', 'wall_time')))
                # Later event files take precedence for resumed/repeated steps.
                all_points.setdefault(step, {}).update(point)
            temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
            try:
                temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        start = min((p['wall_time'] for p in all_points.values()), default=0)
        points = [dict(p, seconds=p['wall_time'] - start) for _, p in sorted(all_points.items())]
        atomic_json(chart, {'points': points, 'metrics': [[tag, tag] for tag in sorted(metrics)],
                            'source': 'Saved TensorBoard scalars', 'time_origin': 'First recorded scalar',
                            'samples': len(points)})
        atomic_json(metadata, {'signature': signature})
        return list(zip(files, exports)), chart, None
