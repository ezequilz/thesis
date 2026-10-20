"""Local WorldExplorer artifacts adapted to the shared results UI and renderer."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlencode

from .training_logs import export_training_logs

RUN_ID = re.compile(r'we_\d{8}_\d{6}_[0-9a-f]{12}')
IMAGES = {'.png', '.jpg', '.jpeg', '.webp'}
VIDEOS = {'.mp4', '.webm', '.mov'}


def safe_file(root, path):
    return path.is_file() and path.resolve().is_relative_to(root.resolve())


def file_url(run_id, relative):
    return '/api/splatfix/worldexplorer/file?' + urlencode({'id': run_id, 'path': str(relative)})


def entries(catalog):
    base = Path(catalog.cfg.output.dir).resolve() / 'worldexplorer'
    rows = []
    for path in sorted(base.glob('we_*/job.json')):
        run_id = path.parent.name
        if not RUN_ID.fullmatch(run_id) or path.parent.is_symlink() or not safe_file(path.parent, path):
            continue
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        request = job.get('request', {})
        if request.get('operation') == 'setup':
            continue
        root = path.parent / 'results'
        if root.is_symlink():
            continue
        # Small, fixed searches for cards; enumerate all media only on detail.
        splats = sorted(p for p in root.glob('nerfstudio_output/*/*/*/exports/splat/splat.ply')
                        if safe_file(root, p))
        previews = sorted(p for p in root.glob('scenes/*/img2trajvid/*/input/*.png')
                          if safe_file(root, p))
        if not previews:
            previews = sorted(p for p in root.glob('panoramas/*/*/final/*.png') if safe_file(root, p))
        ply = splats[0] if splats else None
        record = {'run_id': run_id, 'config': {'scene_id': request.get('prompt', run_id)},
                  'state': {'status': job.get('status'),
                            'created_at': datetime.fromtimestamp(job.get('created_at', 0), timezone.utc).isoformat()}}
        row = catalog._row(run_id, record, None, ply, {}, {}, None, None)
        row.update(kind='WorldExplorer', mode=request.get('mode'), model='VGGT → splatfacto-big',
                   phase=job.get('phase'), can_stop=False, can_archive=False,
                   message=job.get('error') or job.get('persistence'),
                   ply_url=file_url(run_id, ply.relative_to(root)) if ply else None,
                   preview_url=file_url(run_id, previews[0].relative_to(root)) if previews else None,
                   preview_label='Scaffold view',
                   provenance_url=file_url(run_id, 'request.json') if safe_file(root, root/'request.json') else None)
        rows.append(row)
    return rows


def run_detail(catalog, row):
    run_id = row['run_id']
    root = Path(catalog.cfg.output.dir).resolve() / 'worldexplorer' / run_id / 'results'
    row.pop('_manifest')
    row.pop('_paths')
    groups, videos, logs = {}, [], []
    for path in sorted(root.rglob('*')):
        if not safe_file(root, path) or '.training-metrics' in path.parts:
            continue
        relative = path.relative_to(root)
        item = {'name': str(relative), 'label': str(relative), 'url': file_url(run_id, relative)}
        if path.suffix.lower() in IMAGES:
            group = groups.setdefault(str(relative.parent), [])
            group.append({**item, 'index': len(group)})
        elif path.suffix.lower() in VIDEOS:
            videos.append(item)
        elif path.suffix.lower() in {'.log', '.txt'}:
            logs.append({**item, 'partial': False, 'size': path.stat().st_size,
                         'modified': path.stat().st_mtime_ns})
    try:
        exports, chart, error = export_training_logs(root, event_subdir='nerfstudio_output')
    except OSError as exc:
        exports, chart, error = [], None, str(exc)
    for source, path in exports:
        logs.append({'name': str(source.relative_to(root)) + ' (decoded scalars)',
                     'url': file_url(run_id, path.relative_to(root)), 'partial': False,
                     'size': path.stat().st_size, 'modified': path.stat().st_mtime_ns})
    if chart:
        row['training_metrics'] = {'url': file_url(run_id, chart.relative_to(root)),
                                   'modified': chart.stat().st_mtime_ns,
                                   'name': 'WorldExplorer · TensorBoard training scalars', 'format': 'scalars'}
    if error:
        row['training_metrics_error'] = error
    ordered = sorted(groups, key=lambda name: ('selected_for_vggt' not in name, name))
    row.update(worldexplorer={'image_groups': [{'name': name, 'frames': groups[name]} for name in ordered],
                              'videos': videos}, logs=logs,
               gallery={'frames': [], 'verified_inputs': True, 'stage': 'WorldExplorer · saved images',
                        'reference_count': None, 'expected_count': None},
               reference_images=[], original_rgb_images=[], comparison_gallery=[])
    return row


def review_views(catalog, row):
    """Map saved OpenGL cameras into the native Nerfstudio PLY frame."""
    import numpy as np

    root = Path(catalog.cfg.output.dir).resolve() / 'worldexplorer' / row['run_id'] / 'results'
    ply = row['_paths'].get('repaired')
    if ply is None:
        return []
    # Use normalization from this exact export, not another training attempt.
    parser_path = ply.parents[2] / 'dataparser_transforms.json'
    transforms = [p for p in root.glob('scenes/*/img2trajvid/transforms.json') if safe_file(root, p)]
    if len(transforms) != 1 or not safe_file(root, parser_path):
        return []
    try:
        data = json.loads(transforms[0].read_text())
        parser = json.loads(parser_path.read_text())
        transform = np.asarray(parser['transform'], dtype=float)
        scale = float(parser['scale'])
        if (transform.shape != (3, 4) or not np.isfinite(transform).all()
                or not np.isfinite(scale) or scale <= 0):
            return []
        rotation, offset = transform[:, :3], transform[:, 3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            return []
        # WorldExplorer emits poses directly. Do not guess coordinates for an
        # unrelated dataset that already applied a transform before saving.
        if 'applied_transform' in data or 'applied_scale' in data:
            return []
        frames = data['frames']
    except (OSError, ValueError, KeyError, TypeError):
        return []

    def order(frame):
        path = Path(frame.get('file_path', ''))
        return (path.parent.name != 'input', path.parent.parent.name,
                int(path.stem) if path.stem.isdigit() else -1, path.name)

    views = []
    for frame in sorted(frames, key=order):
        try:
            image = Path(frame['file_path'])
            if image.parent.name not in ('input', 'samples-rgb'):
                continue
            pose = np.asarray(frame['transform_matrix'], dtype=float)
            fy = float(frame.get('fl_y', data.get('fl_y')))
            height = float(frame.get('h', data.get('h')))
            if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1])
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)
                    or not np.isfinite([fy, height]).all() or fy <= 0 or height <= 0):
                continue
            label = (f'Scaffold {image.stem}' if image.parent.name == 'input'
                     else f'{image.parent.parent.name} · frame {image.stem}')
            views.append({'center': scale * (rotation @ pose[:3, 3] + offset),
                          'forward': -(rotation @ pose[:3, 2]),
                          'up': rotation @ pose[:3, 1],
                          'vfov': float(2 * np.arctan(height / (2 * fy))), 'label': label})
        except (ValueError, KeyError, TypeError, IndexError):
            continue
    return views
