import json
import subprocess
from pathlib import Path

import pytest

from splat_explorer.splatfix.live_logs import (
    LOG_BYTES, PhaseLogMirror, bounded_tail_command, phase_log_target,
)


def test_log_mapping_accepts_only_current_job_logs():
    root = '/workspace/splatfix-jobs/run_123'
    target, relative = phase_log_target(root, '/dss/jobs/run_123', root + '/results/run/reconstruct.log')
    assert target == '/dss/jobs/run_123/results/run/reconstruct.log'
    assert relative == 'results/run/reconstruct.log'
    for raw in ('/etc/secrets.log', root + '/../other/file.log', root + 'extra/file.log',
                'relative.log', root + '/result.json', root + '/bad\x00.log', None):
        with pytest.raises(ValueError):
            phase_log_target(root, '/dss/jobs/run_123', raw)


def test_mirror_reads_bounded_tail_no_more_than_once_per_15_seconds(tmp_path):
    commands = []
    clock = [0.0]
    def ssh(command):
        commands.append(command)
        return 'x' * (LOG_BYTES + 100)
    mirror = PhaseLogMirror(tmp_path, '/workspace/job', '/dss/job', ssh, clock=lambda: clock[0])
    state = {'phase': 'reconstruct', 'log': '/workspace/job/results/reconstruct.log'}
    assert mirror.poll(state)
    path = tmp_path / 'gpu/live-stage.log'
    assert path.stat().st_size == LOG_BYTES
    assert f'tail -c {LOG_BYTES} --' in commands[0]
    assert 'realpath /dss/job/results/reconstruct.log' in commands[0]
    assert 'case "$target" in "$root"/*)' in commands[0]
    clock[0] = 14.9
    assert not mirror.poll({'phase': 'plus', 'log': '/workspace/job/results/plus.log'})
    assert len(commands) == 1
    clock[0] = 15.0
    assert mirror.poll(state)
    assert len(commands) == 2
    metadata = json.loads((tmp_path / 'gpu/live-stage.json').read_text())
    assert metadata['phase'] == 'reconstruct'
    assert metadata['relative_path'] == 'results/reconstruct.log'


def test_invalid_log_paths_never_reach_ssh_and_timeout_does_not_fail_job(tmp_path):
    def no_ssh(command):
        pytest.fail('Invalid path must not reach transport')
    mirror = PhaseLogMirror(tmp_path, '/workspace/job', '/dss/job', no_ssh)
    assert not mirror.poll({'log': '/workspace/another-job/secret.log'})
    def timeout(command):
        raise subprocess.TimeoutExpired(command, 30)
    mirror = PhaseLogMirror(tmp_path, '/workspace/job', '/dss/job', timeout)
    assert not mirror.poll({'log': '/workspace/job/reconstruct.log'})
    assert not (tmp_path / 'gpu/live-stage.log').exists()


def test_tail_shell_blocks_symlink_escape_and_quotes_paths(tmp_path):
    root = tmp_path / "job with ' quote"
    root.mkdir()
    inside = root / "phase with ' quote.log"
    inside.write_bytes(b'prefix' + b'z' * LOG_BYTES)
    command = bounded_tail_command(str(root), str(inside))
    read = subprocess.run(['bash', '-c', command], capture_output=True)
    # Login workers are Linux; this check also exercises the available local realpath.
    assert read.returncode == 0, read.stderr
    assert read.stdout == b'z' * LOG_BYTES
    outside = tmp_path / 'private.log'
    outside.write_text('must not appear')
    link = root / 'escape.log'
    link.symlink_to(outside)
    blocked = subprocess.run(['bash', '-c', bounded_tail_command(str(root), str(link))], capture_output=True)
    assert blocked.returncode == 64
    assert not blocked.stdout


def test_tail_drops_partial_utf8_at_byte_boundary(tmp_path):
    root = tmp_path / 'job'
    root.mkdir()
    log = root / 'training.log'
    # A 64 KiB tail starts inside the emoji, like a progress-bar update.
    log.write_bytes('😀'.encode() + b'x' * (LOG_BYTES - 2))
    result = subprocess.run(['bash', '-c', bounded_tail_command(str(root), str(log))], capture_output=True)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout) <= LOG_BYTES
    assert result.stdout.decode('utf-8') == 'x' * (LOG_BYTES - 2)
