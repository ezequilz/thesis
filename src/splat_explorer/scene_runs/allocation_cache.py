"""Durable allocation deadline captured by Load GPU setup, never by polling."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid
from zoneinfo import ZoneInfo
from .models import parse_slurm_end, parse_slurm_duration


def cache_path():
    from ..repair_lrz import lrz_local_config_path
    return lrz_local_config_path().parent.parent / 'outputs/gpu-allocation.json'


def identity(cfg):
    return {key: str(cfg.get(key) or '') for key in ('job_id', 'host', 'user', 'workspace')}


def save_loaded_allocation(cfg, allocation, *, path=None):
    if allocation.get('job_id') != str(cfg['job_id']) or allocation.get('state') != 'R':
        raise ValueError('Load GPU setup did not establish the selected running allocation')
    observed = float(allocation.get('observed_at_epoch') or time.time())
    end = parse_slurm_end(allocation.get('expected_end'))
    if end is not None:
        if end.tzinfo is None:
            end = end.replace(tzinfo=ZoneInfo('Europe/Berlin'))
        deadline = end.timestamp()
    else:
        remaining = parse_slurm_duration(allocation.get('time_left'))
        if remaining is None:
            raise ValueError('Slurm did not report an allocation deadline during GPU setup')
        deadline = observed + remaining.total_seconds()
    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError('GPU allocation expired before setup completed')
    body = {**identity(cfg), 'state': 'R', 'deadline_epoch': deadline,
            'expected_end': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
            'observed_at_epoch': observed, 'setup_completed_at_epoch': time.time(),
            'source': 'load_gpu_setup',
            **{key: allocation.get(key, '') for key in ('node', 'partition', 'mem', 'timelimit')}}
    destination = Path(path) if path is not None else cache_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(body, indent=2) + '\n')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return body


def loaded_allocation(cfg, *, path=None, now=None):
    destination = Path(path) if path is not None else cache_path()
    try:
        body = json.loads(destination.read_text())
        if not isinstance(body, dict) or any(body.get(k) != v for k, v in identity(cfg).items()):
            raise ValueError('Cached GPU setup belongs to another allocation')
        deadline = float(body['deadline_epoch'])
        if not math.isfinite(deadline):
            raise ValueError('Invalid cached GPU deadline')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError('Load GPU setup on /gpu to cache the deadline for the connected job') from exc
    if (time.time() if now is None else now) >= deadline:
        raise ValueError('Cached GPU allocation has expired; select a new job and Load GPU setup on /gpu')
    return body
