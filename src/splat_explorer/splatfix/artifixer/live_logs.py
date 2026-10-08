"""Bounded, best-effort phase-log snapshots for detached reconstruction jobs."""
from __future__ import annotations

import os
import shlex
import time
from pathlib import Path, PurePosixPath

from ..checkpoint import atomic_json

LOG_BYTES = 64 * 1024
POLL_SECONDS = 15.0


def phase_log_target(container_root, remote_root, raw):
    """Map a worker-reported absolute .log path into this job's DSS directory."""
    if not isinstance(raw, str) or not raw or '\x00' in raw:
        raise ValueError('Missing or invalid phase log path')
    path = PurePosixPath(raw)
    root = PurePosixPath(container_root)
    if not path.is_absolute() or '..' in path.parts or path.suffix != '.log':
        raise ValueError('Phase log must be an absolute log file inside the job')
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError('Phase log is outside the container job root') from exc
    return str(PurePosixPath(remote_root) / relative), relative.as_posix()


def bounded_tail_command(remote_root, target):
    """Validate real remote paths before reading; symlinks may not escape the job."""
    return ('root=$(realpath ' + shlex.quote(str(remote_root)) + ') && '
            'target=$(realpath ' + shlex.quote(str(target)) + ') && '
            'case "$target" in "$root"/*) tail -c ' + str(LOG_BYTES) + ' -- "$target" | iconv -f UTF-8 -t UTF-8 -c || :;; '
            '*) exit 64;; esac')


class PhaseLogMirror:
    def __init__(self, root, container_root, remote_root, ssh, *, clock=time.monotonic):
        self.root = Path(root)
        self.container_root, self.remote_root, self.ssh = container_root, remote_root, ssh
        self.clock = clock
        self.next_poll = 0.0

    def poll(self, state):
        """Return whether a snapshot was written; log failures never stop a job."""
        now = self.clock()
        if now < self.next_poll or not state.get('log'):
            return False
        self.next_poll = now + POLL_SECONDS
        try:
            target, relative = phase_log_target(self.container_root, self.remote_root, state['log'])
            body = self.ssh(bounded_tail_command(self.remote_root, target))
            data = body.encode('utf-8', errors='replace') if isinstance(body, str) else bytes(body)
            # Bound local storage too, even when a test transport or decoding expands bytes.
            data = data[-LOG_BYTES:]
            directory = self.root / 'gpu'
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / '.live-stage.log.tmp'
            temporary.write_bytes(data)
            os.replace(temporary, directory / 'live-stage.log')
            atomic_json(directory / 'live-stage.json', {'phase': str(state.get('phase') or 'reconstruction'),
                        'relative_path': relative, 'updated_at': time.time(), 'max_bytes': LOG_BYTES})
            return True
        except Exception:
            # SSH timeouts/decoding/filesystem errors are observability failures,
            # never reasons to cancel or relaunch reconstruction.
            return False
