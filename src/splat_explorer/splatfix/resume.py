"""Validate benchmark retry provenance and locate one prior failed GPU result."""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path, PurePosixPath

from ..scene_runs.store import RUN_ID_PATTERN


def validate_resume(store, options):
    prior_id = options.get('resume_from')
    if not isinstance(prior_id, str) or RUN_ID_PATTERN.fullmatch(prior_id) is None:
        raise ValueError('resume_from must be an existing job id')
    prior = store.get_run(prior_id)
    original = prior.config.splatfix
    if prior.config.pipeline != 'splatfix' or original.get('stage') != 'benchmark':
        raise ValueError('Resume requires a previous benchmark job')
    if prior.state.status.value not in ('error', 'stopped'):
        raise ValueError('Only failed or cancelled benchmark jobs can be resumed')
    if (Path(original['source']).resolve() != Path(options['source']).resolve()
            or original.get('model', '1.3b') != options.get('model', '1.3b')):
        raise ValueError('Resume source and model must match the previous benchmark')
    if not prior.state.details.get('remote_dir'):
        raise ValueError('Previous benchmark has no saved remote job directory')
    return prior


_DISCOVER_RESULT = '''import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
results = root / 'results'
comparison = len(sys.argv) > 2 and sys.argv[2] == 'preparation'
phases = ('prepare', 'reconstruct', 'render', 'scale', 'caption') if comparison else ('prepare', 'reconstruct', 'render', 'scale')
statuses = ('complete', 'failed') if comparison else ('failed',)
candidates = []
for manifest in results.glob('*/benchmark-run.json'):
    parent = manifest.parent.resolve()
    if manifest.is_symlink() or not parent.is_relative_to(results) or parent.parent != results:
        continue
    body = json.loads(manifest.read_text())
    stages = body.get('stages', [])
    completed = all(len([s for s in stages if s.get('phase') == phase and s.get('status') == 'complete']) == 1
                    and len([s for s in stages if s.get('phase') == phase]) == 1
                    for phase in phases)
    if body.get('status') in statuses and completed and (parent / 'prepared/bicycle').is_dir():
        candidates.append(str(parent.relative_to(root)))
if len(candidates) != 1:
    raise RuntimeError('Expected exactly one supported benchmark result; found ' + str(len(candidates)))
print(json.dumps({'relative': candidates[0]}))
'''


def resolve_remote_resume(store, options, cfg, ssh):
    return _resolve_remote_result(validate_resume(store, options), cfg, ssh)


def resolve_remote_preparation(store, options, cfg, ssh):
    return _resolve_remote_result(validate_preparation(store, options), cfg, ssh, comparison=True)


def _resolve_remote_result(prior, cfg, ssh, *, comparison=False):
    expected = str(PurePosixPath(str(cfg['workspace'])) / 'splatfix-jobs' / prior.run_id)
    if str(PurePosixPath(prior.state.details['remote_dir'])) != expected:
        raise ValueError('Previous benchmark is outside the currently configured remote workspace')
    raw = ssh(shlex.join(['python3', '-c', _DISCOVER_RESULT, expected] + (['preparation'] if comparison else [])))
    relative = json.loads(raw).get('relative')
    if not isinstance(relative, str):
        raise ValueError('Invalid remote benchmark result path')
    path = PurePosixPath(relative)
    if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != 'results'
            or re.fullmatch(r'benchmark_[A-Za-z0-9_.-]+', path.parts[1]) is None
            or '..' in path.parts):
        raise ValueError('Remote benchmark result escaped its prior job directory')
    return str(PurePosixPath('/workspace/splatfix-jobs') / prior.run_id / path)


def resume_available(run):
    """UI readiness, including manifests saved before resume_supported existed."""
    if run.config.pipeline != 'splatfix' or run.config.splatfix.get('stage') != 'benchmark' or run.state.status.value != 'error':
        return False
    from .jobs import read_json
    manifests = list((Path(run.path) / 'gpu/results').glob('*/benchmark-run.json'))
    if len(manifests) != 1:
        return False
    manifest = manifests[0]
    data = read_json(manifest, {})
    if not isinstance(data, dict) or data.get('status') != 'failed':
        return False
    if 'resume_supported' in data:
        return data['resume_supported'] is True
    stages = data.get('stages', [])
    return (all(len([s for s in stages if s.get('phase') == phase]) == 1
                and next(s for s in stages if s.get('phase') == phase).get('status') == 'complete'
                for phase in ('prepare', 'reconstruct', 'render', 'scale'))
            and (manifest.parent / 'prepared/bicycle').is_dir())


def validate_preparation(store, options):
    prior_id = options.get('preparation_from')
    if not isinstance(prior_id, str) or RUN_ID_PATTERN.fullmatch(prior_id) is None:
        raise ValueError('preparation_from must be an existing job id')
    prior = store.get_run(prior_id)
    original = prior.config.splatfix
    if prior.config.pipeline != 'splatfix' or original.get('stage') != 'benchmark':
        raise ValueError('Shared preparation requires a previous benchmark job')
    if prior.state.status.value not in ('completed', 'error'):
        raise ValueError('Shared preparation requires a completed or failed benchmark job')
    if Path(original['source']).resolve() != Path(options['source']).resolve():
        raise ValueError('Shared preparation source must match the previous benchmark')
    if not prior.state.details.get('remote_dir'):
        raise ValueError('Previous benchmark has no saved remote job directory')
    return prior


def preparation_available(run):
    if (run.config.pipeline != 'splatfix' or run.config.splatfix.get('stage') != 'benchmark'
            or run.state.status.value not in ('completed', 'error')):
        return False
    from .jobs import read_json
    manifests = list((Path(run.path) / 'gpu/results').glob('*/benchmark-run.json'))
    if len(manifests) != 1:
        return False
    manifest = manifests[0]
    data = read_json(manifest, {})
    if not isinstance(data, dict) or data.get('status') not in ('complete', 'failed'):
        return False
    if 'preparation_supported' in data:
        return data['preparation_supported'] is True
    stages = data.get('stages', [])
    return (all(len([s for s in stages if s.get('phase') == phase]) == 1
                and next(s for s in stages if s.get('phase') == phase).get('status') == 'complete'
                for phase in ('prepare', 'reconstruct', 'render', 'scale', 'caption'))
            and (manifest.parent / 'prepared/bicycle').is_dir())
