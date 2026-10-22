"""Splatfix stages on the existing durable scene-run queue."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def validate_job(raw):
    if not isinstance(raw, dict):
        raise ValueError('splatfix job must be an object')
    stage = raw.get('stage')
    if stage not in ('select', 'edit', 'seva_capture', 'repair', 'benchmark'):
        raise ValueError('stage must be select, edit, seva_capture, repair or benchmark')
    from .resolution import profile_size
    profile = raw.get('resolution_profile', 'training')
    profile_size(profile)
    result = {'stage': stage, 'resolution_profile': profile}
    if stage in ('repair', 'benchmark'):
        from .methods import DEFAULT_METHOD, get_method
        method = raw.get('reconstruction_method')
        method = DEFAULT_METHOD if method is None else method
        backend = get_method(method, stage=stage)
        result['reconstruction_method'] = method
        result.update(backend.validate_options(
            {'oracle_reference_mode':'scene_memory', **raw} if method == 'oraclegs' else raw))
    if raw.get('oracle_artifixer_inputs') is not None:
        snapshot = raw['oracle_artifixer_inputs']
        if (stage != 'repair' or result.get('reconstruction_method') != 'oraclegs'
                or raw.get('mode', 'edited') != 'edited'
                or not isinstance(snapshot, str) or not Path(snapshot).is_absolute()):
            raise ValueError('Invalid OracleGS ArtiFixer input snapshot')
        result['oracle_artifixer_inputs'] = snapshot
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
        policy = raw.get('repair_policy', 'all')
        if policy not in ('all', 'endpoints') or (policy == 'endpoints' and views != 6):
            raise ValueError('Endpoint checkpoints require exactly six views')
        fov = raw.get('horizontal_fov')
        if fov is not None and (type(fov) not in (int, float) or not 0 < fov < 180):
            raise ValueError('Invalid horizontal field of view')
        result.update(views=views, repair_policy=policy, horizontal_fov=fov)
        renderer = raw.get('renderer', 'viser')
        if renderer != 'viser':
            raise ValueError('Splatfix view selection requires Viser captures; other RGB renderers are disabled')
        result['renderer'] = renderer
    elif stage == 'benchmark':
        source = str(raw.get('source') or '')
        if not source:
            raise ValueError('Choose a registered benchmark source')
        if raw.get('mode', 'baseline') != 'baseline':
            raise ValueError('Published-dataset benchmark uses baseline mode only')
        result.update(source=source, mode='baseline')
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
        if not checkpoint and not result.get('oracle_artifixer_inputs'):
            raise ValueError('Choose an existing checkpoint')
        result['checkpoint'] = checkpoint
    if stage == 'seva_capture' and raw.get('view_id') is not None:
        view_id = raw['view_id']
        if not isinstance(view_id, str) or len(view_id) != 3 or not view_id.isascii() or not view_id.isdigit():
            raise ValueError('view_id must be a saved three-digit view id')
        result['view_id'] = view_id
    if stage == 'edit':
        image_set = raw.get('image_set', 'saved')
        if image_set not in ('saved', 'seva'):
            raise ValueError('image_set must be saved or seva')
        result['image_set'] = image_set
    if stage == 'repair':
        mode = raw.get('mode', 'edited')
        if mode not in ('baseline', 'edited'):
            raise ValueError('mode must be baseline or edited')
        result['mode'] = mode
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
