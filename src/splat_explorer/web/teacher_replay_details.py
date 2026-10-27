"""Read-only review of isolated teacher replays, including live completed clips."""
from pathlib import Path
import re

from ..splatfix.jobs import read_json

PREFIX = 'teacher_replay_'
RUN = re.compile(r'run_\d{8}_\d{6}')
PREDICTIONS = Path('results/inference/splatfix/frames/batch_0000/pred')


def safe(root, path):
    return path.is_file() and path.resolve().is_relative_to(root.resolve())


def allowed_file(output, path):
    path = Path(path).resolve()
    base = Path(output).resolve() / 'artifixer-teacher-replay'
    try:
        relative = path.relative_to(base)
    except ValueError:
        return False
    if len(relative.parts) < 2 or not RUN.fullmatch(relative.parts[0]):
        return False
    root = base / relative.parts[0]
    # Only review artifacts, never the frozen source tree or model/input tensors.
    permitted = relative.parts[1] in ('results', 'inputs') or relative.name in ('status.json', 'provenance.json', 'run-info.json')
    return (permitted and not root.is_symlink() and safe(root, path)
            and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.json', '.jsonl', '.log', '.mp4'))


def teacher_summary(data):
    clips = data.get('clips', [])
    if not clips:
        return 'Teacher: first clip in progress' if data else None
    frames = sum(len(c['frame_indices']) for c in clips if c.get('output') == 'inference')
    expected = data.get('expected_frame_count')
    message = f"Teacher: {len(clips)} clips · {frames}{' / ' + str(expected) if expected else ''} frames"
    peak = max((c.get('peak_reserved_gib', 0) for c in clips), default=0)
    preserved = data.get('image_cache_insertion') and all(
        c.get('seeding', {}).get('encoded_starter_preserved') for c in clips)
    return (message + (f' · {peak:.1f} GiB peak VRAM reserved' if peak else '')
            + (' · repaired starting latent preserved' if preserved else ''))


def entries(catalog, *, key=None):
    if key and not key.startswith(PREFIX):
        return []
    base = Path(catalog.cfg.output.dir).resolve() / 'artifixer-teacher-replay'
    if base.is_symlink():
        return []
    rows = []
    for root in sorted(base.glob('run_*')):
        ident = PREFIX + root.name
        if key and key != ident:
            continue
        if not RUN.fullmatch(root.name) or root.is_symlink():
            continue
        if not all(safe(root, root / p) for p in ('provenance.json', 'status.json', 'results/request.json')):
            continue
        provenance = read_json(root / 'provenance.json', {})
        request = read_json(root / 'results/request.json', {})
        status = read_json(root / 'status.json', {})
        if not all(isinstance(item, dict) for item in (provenance, request, status)):
            continue
        if (provenance.get('source_run') != root.name or not provenance.get('inference_only')
                or request.get('runtime', {}).get('inference_model') != '14b-teacher'):
            continue
        record = {'run_id': ident, 'config': {'pipeline': 'inference_replay', 'scene_id': 'Teacher replay · ' + root.name,
                  'splatfix': {**request['runtime'], 'mode': request.get('mode')}},
                  'state': {'status': status.get('status'), 'created_at': provenance.get('created_at'),
                            'message': status.get('error'), 'details': {'phase': 'inference'}}}
        row = catalog._row(ident, record, None, None, {}, {}, None, None)
        row.update(kind='ArtiFixer inference', external_inference=True, inference_only=True, can_archive=False, can_reduce=False,
                   can_stop=False, source_run=root.name, model='14b-teacher', inference_ready=True,
                   provenance_url=catalog.studio.file_url(root / 'provenance.json'),
                   _manifest=root / 'results/request.json')
        generated = completed_frames(root)
        first = next((root / PREDICTIONS / f'{i:05d}.png' for i in sorted(generated)
                      if safe(root, root / PREDICTIONS / f'{i:05d}.png')), None)
        row.update(preview_url=catalog.studio.file_url(first) if first else None, preview_label='14B teacher inference',
                   frames=status.get('completed_frames', 0), teacher_summary=progress_summary(status),
                   updated_at_epoch=status.get('updated_at_epoch'))
        rows.append(row)
    return rows


def completed_frames(root):
    manifest = root / 'results/inference-teacher.json'
    if not safe(root, manifest):
        return set()
    return {i for c in read_json(manifest, {}).get('clips', []) if c.get('output') == 'inference'
            for i in c.get('frame_indices', []) if type(i) is int and i >= 0}


def progress_summary(status):
    message = f"{status.get('completed_clips', 0)} / {status.get('total_clips', '?')} clips · {status.get('completed_frames', 0)} / {status.get('total_generated_frames', '?')} frames"
    if status.get('status') == 'running' and status.get('denoising_step') is not None:
        message += f" · clip {status.get('current_clip')} step {status['denoising_step']}/{status.get('steps_per_clip', 50)}"
    if status.get('peak_reserved_gib'):
        message += f" · {status['peak_reserved_gib']:.1f} GiB peak VRAM reserved"
    return message


def run_detail(catalog, row):
    root = row.pop('_manifest').parent.parent
    row.pop('_paths')
    request = read_json(root / 'results/request.json', {})
    provenance = read_json(root / 'provenance.json', {})
    from ..splatfix.artifixer.progress import stage
    status = read_json(root / 'status.json', {})
    steps = status.get('steps_per_clip', 50)
    row['progress'] = {'observed_at': status.get('updated_at_epoch'), 'poll_seconds': 60, 'stages': {
        'inference': stage('14B inference', status.get('completed_clips', 0) * steps + status.get('denoising_step', 0),
            status.get('total_clips', 0) * steps, status='completed' if status.get('status') == 'completed' else
            'active' if status.get('status') == 'running' else status.get('status', 'pending'),
            detail=progress_summary(status))}}
    def asset(path, index, name=None):
        if not safe(root, path) or not allowed_file(catalog.cfg.output.dir, path):
            return None
        return {'index': index, 'name': name or path.name, 'url': catalog.studio.file_url(path)}
    def relocate(raw):
        try:
            relative = Path(raw).relative_to(request['checkpoint_root'])
        except (ValueError, KeyError, TypeError):
            return root / '__missing__'
        return root / 'inputs' / relative
    from .teacher_frame_order import frame_order
    ordered = frame_order(read_json(root / 'results/inference-teacher.json', {}))
    frames = [{**item, **ordered.get(i, {})} for i in sorted(completed_frames(root), key=lambda i: ordered.get(i, {}).get('sequence_order', i))
              if (item := asset(root / PREDICTIONS / f'{i:05d}.png', i))]
    references = [item for i, raw in enumerate(request.get('references', []))
                  if (item := asset(relocate(raw), i))]
    trajectory_path = relocate(request.get('trajectory'))
    trajectory = read_json(trajectory_path, {}) if safe(root, trajectory_path) else {}
    captures = [item for i, frame in enumerate(trajectory.get('frames', []))
                if (item := asset(root / 'inputs' / frame['rgb'], i))]
    row.update(gallery={'frames': frames, 'reference_count': len(request.get('references', [])),
                        'verified_inputs': True, 'expected_count': len(provenance.get('expected_frame_indices', [])),
                        'count': len(frames), 'stage': 'ArtiFixer 14B teacher · generated frames'},
               plus_gallery={'frames': []}, reference_images=references, original_rgb_images=captures,
               comparison_gallery=[], settings_snapshot={'effective_request': request},
               settings_url=catalog.studio.file_url(root / 'results/request.json'),
               resolution_setting=' × '.join(map(str, read_json(root / 'status.json', {}).get('resolution', []))),
               json_artifacts=[], logs=[], inference_only=True)
    for name in ('status.json', 'results/inference-teacher.json', 'results/validation.json'):
        path = root / name
        if safe(root, path):
            row['json_artifacts'].append({'name': path.name, 'url': catalog.studio.file_url(path),
                                          'snapshot': read_json(path, {}), 'path': name})
    log = root / 'results/inference.log'
    if safe(root, log):
        row['logs'].append({'name': 'Teacher inference', 'url': catalog.studio.file_url(log),
                            'size': log.stat().st_size, 'modified': log.stat().st_mtime_ns, 'partial': row['status'] == 'running'})
    return row
