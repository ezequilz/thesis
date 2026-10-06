"""Splatfix stages on the existing durable scene-run queue."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


def validate_job(raw):
    if not isinstance(raw, dict):
        raise ValueError('splatfix job must be an object')
    stage = raw.get('stage')
    if stage not in ('select', 'edit', 'repair', 'benchmark'):
        raise ValueError('stage must be select, edit, repair or benchmark')
    from .resolution import profile_size
    profile = raw.get('resolution_profile', 'training')
    profile_size(profile)
    result = {'stage': stage, 'resolution_profile': profile}
    scheduled = raw.get('scheduled_at')
    if scheduled:
        instant = datetime.fromisoformat(str(scheduled).replace('Z', '+00:00'))
        if instant.tzinfo is None:
            raise ValueError('scheduled_at must include a timezone')
        result['scheduled_at'] = instant.astimezone(timezone.utc).isoformat()
    if stage == 'select':
        views = raw.get('views', 6)
        if type(views) is not int or not 1 <= views <= 24:
            raise ValueError('views must be an integer between 1 and 24')
        result['views'] = views
        renderer = raw.get('renderer', 'cpu_splats')
        if renderer not in ('cpu_splats', 'cpu_points'):
            raise ValueError('Dashboard view selection uses a local CPU renderer')
        result['renderer'] = renderer
    elif stage == 'benchmark':
        source = str(raw.get('source') or '')
        if not source:
            raise ValueError('Choose a registered benchmark source')
        if raw.get('mode', 'baseline') != 'baseline':
            raise ValueError('Published-dataset benchmark uses baseline mode only')
        model = raw.get('model', '1.3b')
        if model not in ('1.3b', '14b'):
            raise ValueError('model must be 1.3b or 14b')
        result.update(source=source, mode='baseline', model=model)
        if raw.get('resume_from') is not None and raw.get('preparation_from') is not None:
            raise ValueError('resume_from and preparation_from are mutually exclusive')
        if raw.get('repeat_from') and (raw.get('resume_from') or raw.get('preparation_from')):
            raise ValueError('repeat_from cannot be combined with preparation or resume')
        for key in ('resume_from', 'preparation_from', 'repeat_from'):
            if raw.get(key) is not None:
                from ..scene_runs.store import RUN_ID_PATTERN
                prior = raw[key]
                if not isinstance(prior, str) or RUN_ID_PATTERN.fullmatch(prior) is None:
                    raise ValueError(f'{key} must be an existing job id')
                result[key] = prior
    else:
        checkpoint = str(raw.get('checkpoint') or '')
        if not checkpoint:
            raise ValueError('Choose an existing checkpoint')
        result['checkpoint'] = checkpoint
    if stage == 'repair':
        mode = raw.get('mode', 'edited')
        if mode not in ('baseline', 'edited'):
            raise ValueError('mode must be baseline or edited')
        frames = raw.get('frames', 25)
        if type(frames) is not int or frames < 9 or frames > 101 or (frames - 1) % 4:
            raise ValueError('frames must be 1 + 4*n, between 9 and 101')
        span = float(raw.get('span_fraction', .04))
        if not math.isfinite(span) or not 0 < span <= .15:
            raise ValueError('span_fraction must be in (0, .15]')
        result.update(mode=mode, frames=frames, span_fraction=span)
    return result


def requires_gpu(config):
    return config.get('pipeline') != 'splatfix' or config.get('splatfix', {}).get('stage') in ('repair', 'benchmark')


def is_due(config, now=None):
    scheduled = config.get('splatfix', {}).get('scheduled_at')
    return not scheduled or datetime.fromisoformat(scheduled) <= (now or datetime.now(timezone.utc))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def registered_benchmark(root, source):
    """Resolve an explicitly registered benchmark, keeping its inputs in its root."""
    root = Path(root).resolve()
    raw = Path(source).expanduser()
    source = raw.resolve()
    if (raw.is_symlink() or source.name != 'input' or source.parent.parent != root
            or not source.is_dir()):
        raise ValueError('Choose a registered benchmark under outputs/benchmarks/<name>/input')
    metadata = read_json(source / 'benchmark.json')
    if not isinstance(metadata, dict):
        raise ValueError('Benchmark registration is missing or invalid')
    for entry in source.rglob('*'):
        if entry.is_symlink():
            raise ValueError('Benchmark inputs must contain local files, not symbolic links')
    for required in ('colmap/images', 'colmap/sparse/0', 'selected_images.txt'):
        if not (source / required).exists():
            raise ValueError(f'Benchmark input is incomplete: missing {required}')
    return source, metadata
