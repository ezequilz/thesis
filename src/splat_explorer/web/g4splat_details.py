"""Bounded, read-only inspection of saved G4Splat artifacts (no model execution)."""
from __future__ import annotations

import hashlib
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

_LOCK = threading.Lock()
MAX_VIEWS = 24
MAX_PIXELS = 4_000_000
SFM = 'official/mast3r_sfm'


def _json(path, default=None):
    try:
        if path is None or path.stat().st_size > 8_000_000:
            return default
        value = json.loads(path.read_text())
        return value if default is None or isinstance(value, type(default)) else default
    except (OSError, ValueError):
        return default


def _safe(root, relative):
    """Artifacts must remain in this run, including every symlink target."""
    path = root / relative
    return path if path.resolve().is_relative_to(root.resolve()) and path.is_file() else None


def _depth(path):
    if path.suffix == '.npz':
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 150_000_000:
                raise ValueError('Source depth archive exceeds preview limit')
        with np.load(path, allow_pickle=False) as archive:
            a = np.asarray(archive['depth'], dtype=np.float32)
            valid = np.asarray(archive['valid'], dtype=bool)
        if a.ndim != 2 or a.size > MAX_PIXELS or valid.shape != a.shape:
            raise ValueError('Invalid source depth dimensions')
        return np.where(valid, a, 0)
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError('Depth preview exceeds four million pixels')
        a = np.asarray(im, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError('Expected scalar camera-z depth')
    return a


def _previews(root, sources):
    """Cache display-only PNGs; shared sampled p99 scaling, never per-image contrast."""
    fingerprint = hashlib.sha256(('depth-preview-v1' + repr([
        (key, str(p), p.stat().st_size, p.stat().st_mtime_ns)
        for key, p in sources.items()])).encode()).hexdigest()[:20]
    cache = root / 'debug-previews' / fingerprint
    if not cache.resolve().is_relative_to(root.resolve()):
        return {}, {'error': 'Unsafe preview cache path'}
    with _LOCK:
        metadata = _json(_safe(root, str(cache.relative_to(root) / 'index.json')))
        if isinstance(metadata, dict) and isinstance(metadata.get('files'), dict):
            expected = {key: hashlib.sha256(key.encode()).hexdigest()[:16] + '.png' for key in sources}
            if all(k in expected and v == expected[k] and _safe(root, str(cache.relative_to(root) / v))
                   for k, v in metadata['files'].items()):
                return {k: cache / v for k, v in metadata['files'].items()}, metadata
        cache.mkdir(parents=True, exist_ok=True)
        samples, errors = [], {}
        for key, path in sources.items():
            try:
                a = _depth(path)
                values = a[np.isfinite(a) & (a > 0)]
                if values.size:
                    samples.append(values[::max(1, values.size // 4096)])
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                errors[key] = str(exc)
        upper = float(np.percentile(np.concatenate(samples), 99)) if samples else 1.0
        files, stats = {}, {}
        for key, path in sources.items():
            if key in errors:
                continue
            try:
                a = _depth(path)
                valid = np.isfinite(a) & (a > 0)
                gray = (np.clip(np.nan_to_num(a, nan=0, posinf=0) / upper, 0, 1) * 255).astype('uint8')
                rgb = np.repeat(gray[:, :, None], 3, axis=2)
                rgb[~valid] = [187, 58, 150]
                image = Image.fromarray(rgb)
                image.thumbnail((640, 400))
                filename = hashlib.sha256(key.encode()).hexdigest()[:16] + '.png'
                target = cache / filename
                if target.is_symlink():
                    raise ValueError('Unsafe preview target')
                image.save(target)
                files[key] = filename
                stats[key] = {'valid_fraction': float(valid.mean()),
                              'clipped_fraction': float((valid & (a > upper)).mean()),
                              'width': a.shape[1], 'height': a.shape[0]}
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                errors[key] = str(exc)
        metadata = {'files': files, 'stats': stats, 'errors': errors, 'min': 0,
                    'max': upper, 'units': 'scene units (not established meters)',
                    'normalization': 'Shared sampled 99th percentile across displayed depth maps; values above white are clipped.',
                    'invalid_color': 'magenta = nonfinite or nonpositive depth'}
        index = cache / 'index.json'
        if index.is_symlink():
            raise ValueError('Unsafe preview index')
        index.write_text(json.dumps(metadata, indent=2))
    return {k: cache / v for k, v in files.items()}, metadata


def build_details(studio, manifest, result):
    root = manifest.parent
    def local(name):
        return _safe(root, name)
    def url(path):
        return studio.file_url(path) if path else None
    def data(name, default=None):
        return _json(local(name), default)
    recipe = data('recipe.json', {})
    inputs = recipe.get('inputs') or {}
    # Exported inputs are siblings of results in local and downloaded jobs.
    job_root = root.parent.parent
    input_manifest = _safe(job_root, 'inputs/inputs.json')
    if not inputs:
        inputs = _json(input_manifest, {})
    cameras = data(SFM + '/cameras.json', {})
    report = data('reprojection-check.json', {})
    views = inputs.get('views', [])
    if not views:
        views = [{'image': str(p), 'view_id': str(i)} for i, p in enumerate(cameras.get('filepaths', [])[:MAX_VIEWS])]
    native_cameras = data('official/free_gaussians/cameras.json', [])
    # The native training renderer enumerates sorted train cameras. Do not pair
    # indices if the saved camera list cannot establish this exact input order.
    names = [Path(v.get('image', '')).stem for v in views]
    native_names = [str(v.get('img_name')) for v in native_cameras]
    native_order_known = bool(names and native_names == names and names == sorted(names))
    native_root = None
    for candidate in sorted((root / 'official/free_gaussians/train').glob('ours_*'),
                            key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else -1, reverse=True):
        if candidate.resolve().is_relative_to(root.resolve()):
            native_root = candidate.relative_to(root)
            break
    saved_intrinsics = {}
    calibration_file = local(SFM + '/sparse/0/cameras.txt')
    if calibration_file and calibration_file.stat().st_size < 1_000_000:
        for line in calibration_file.read_text().splitlines():
            fields = line.split()
            if len(fields) == 8 and fields[1] == 'PINHOLE':
                try:
                    saved_intrinsics[int(fields[0])] = dict(zip(('width', 'height', 'fx', 'fy', 'cx', 'cy'), map(float, fields[2:])))
                except ValueError:
                    pass
    sources, records = {}, []
    for i, view in enumerate(views[:MAX_VIEWS]):
        name = Path(view.get('image', '')).name
        stem = Path(name).stem
        record = {'view_id': str(view.get('view_id', i)), 'image': name,
                  'input_camera': view.get('camera'), 'sha256': view.get('sha256'), 'panels': {},
                  'calibration': (cameras.get('image_calibration') or [])[i] if i < len(cameras.get('image_calibration') or []) else None,
                  'reprojection': next((v for v in report.get('views', []) if v.get('image') == name), None)}
        image = local(SFM + '/images/' + name)
        record['saved_intrinsics'] = saved_intrinsics.get(i + 1)
        record['calibration_warning'] = None
        if image:
            try:
                with Image.open(image) as im:
                    record['staged_wh'] = list(im.size)
                k = record['saved_intrinsics']
                if k and (abs(k['cx'] - record['staged_wh'][0] / 2) > .01 or abs(k['cy'] - record['staged_wh'][1] / 2) > .01):
                    record['calibration_warning'] = 'Saved principal point is off-center in the staged RGB; the native loader assumes a centered camera.'
                if k and [k['width'], k['height']] != record['staged_wh']:
                    record['calibration_warning'] = 'Saved camera dimensions differ from staged RGB dimensions.'
            except (OSError, ValueError):
                record['calibration_warning'] = 'Staged image could not be inspected.'
        record['panels']['input'] = {'url': url(image), 'label': 'Staged input RGB'}
        if native_root and native_order_known:
            native = local(str(native_root / 'renders' / f'{i:05d}.png'))
            record['panels']['native'] = {'url': url(native), 'label': 'Native training-view render'}
        for diagnostic, label in [('plane_vis', 'Upstream fitted-plane visualization'), ('confident_map', 'Upstream confidence visualization')]:
            record['panels'][diagnostic] = {'url': url(local(f'{SFM}/plane-refine-depths/{diagnostic}_frame{stem}.png')), 'label': label}
        depths = {'chart': f'{SFM}/plane-refine-depths/depth_frame{stem}.tiff',
                  'refined': f'{SFM}/plane-refine-depths/refine_depth_frame{stem}.tiff',
                  'monocular': f'{SFM}/plane-refine-depths/mono_depth_frame{stem}.tiff'}
        if native_root and native_order_known:
            depths['native_depth'] = str(native_root / 'vis' / f'depth_{i:05d}.tiff')
        for stage, raw in depths.items():
            path = local(raw)
            if path:
                sources[f'{i}:{stage}'] = path
        source_metadata = inputs.get('source_depth', {})
        source_view = next((v for v in source_metadata.get('views', []) if Path(v.get('image', '')).name == name), None)
        if source_view:
            source_path = _safe(job_root, 'inputs/' + source_view.get('path', ''))
            if source_path and source_path.suffix == '.npz':
                sources[f'{i}:source'] = source_path
            record['source_depth'] = source_view
        records.append(record)
    try:
        previews, scale = _previews(root, sources) if sources else ({}, {})
    except (OSError, ValueError) as exc:
        previews, scale = {}, {'error': 'Depth previews unavailable: ' + str(exc)}
    labels = {'source': 'Source-splat depth (invalid = magenta)', 'chart': 'Chart depth', 'refined': 'Plane-refined depth', 'monocular': 'Aligned monocular prior', 'native_depth': 'Native rendered depth'}
    for i, record in enumerate(records):
        for stage, label in labels.items():
            key = f'{i}:{stage}'
            record['panels'][stage] = {'url': url(previews.get(key)), 'label': label,
                                      'stats': scale.get('stats', {}).get(key), 'error': scale.get('errors', {}).get(key)}
    artifacts = []
    for name, label in [('recipe.json', 'Recipe and input provenance'), ('result.json', 'Result metadata'),
                        ('reprojection-check.json', 'Reprojection gate report'), ('source-depth-check.json', 'Source-depth pointmap report'),
                        ('commands.jsonl', 'Executed / skipped commands'), ('g4splat.log', 'Pipeline log'),
                        ('preflight.log', 'Preflight log'), ('provision.log', 'Provision log'),
                        (SFM + '/cameras.json', 'Saved cameras and calibration')]:
        path = local(name)
        if path:
            artifacts.append({'label': label, 'url': url(path)})
    generation = []
    for stage in range(1, 4):
        folder = root / SFM / 'see3d_render' / f'stage{stage}' / 'cat_img'
        panels = []
        for p in sorted(folder.glob('*.png'))[:16]:
            safe = local(str(p.relative_to(root)))
            if safe:
                panels.append({'name': p.name, 'url': url(safe)})
        generation.append({'stage': stage, 'panels': panels, 'status': 'Artifacts available' if panels else 'Not available'})
    return {'variant': result.get('method_variant') or recipe.get('method_variant') or 'Historical run · correction provenance unavailable',
            'revision': result.get('upstream_revision') or recipe.get('revision'),
            'correction': result.get('geometry_correction') or recipe.get('geometry_correction'),
            'source_depth': result.get('source_depth') or recipe.get('source_depth') or inputs.get('source_depth') or {'enabled': False},
            'source_depth_check': data('source-depth-check.json', {}),
            'gate_status': 'passed' if report.get('passed') is True else 'failed' if report.get('passed') is False else 'not_available',
            'gate': report, 'views': records, 'view_count': len(views), 'view_limit': MAX_VIEWS,
            'depth_scale': scale, 'native_order_known': native_order_known,
            'generation': generation, 'artifacts': artifacts,
            'fitting': [{'url': url(p), 'label': label} for name, label in [
                ('official/free_gaussians/gaussian_points_count.png', 'Recorded Gaussian count during fitting')]
                if (p := local(name))],
            'evaluation': recipe.get('held_out_evaluation') or 'Held-out evaluation not recorded',
            'settings': {k: recipe.get(k) for k in ('sfm_config', 'select_inpaint_num', 'tetra_downsample_ratio', 'generation_rounds')}}
