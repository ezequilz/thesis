"""Persist portable job artifacts before releasing remote scratch storage."""
from __future__ import annotations

import os
import re
import shlex
import uuid
from pathlib import Path, PurePosixPath

from .checkpoint import atomic_json
from .jobs import read_json

RECEIPT = 'remote-persistence.json'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.exr', '.npy', '.npz'}


def remote_path(cfg, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise ValueError('Invalid remote run ID')
    return str(cfg['workspace']).rstrip('/') + '/splatfix-jobs/' + run_id


def portable_links(root, run_id):
    """Relocate job-owned links; reject missing image bytes before cleanup.

    Shared executable/runtime overlays are dependencies, not result images.
    Never dereference those into the archive or follow directory symlinks.
    """
    root = Path(root).resolve()
    gpu = root / 'gpu'
    prefix = PurePosixPath('/workspace/splatfix-jobs') / run_id
    repaired = 0
    images = []
    for directory, dirs, files in os.walk(gpu, followlinks=False):
        for name in dirs + files:
            link = Path(directory) / name
            if not link.is_symlink():
                continue
            relative = link.relative_to(gpu)
            temporary_cache_input = (('captions' in relative.parts and 'source' in relative.parts)
                                     or ('metric_alignment' in relative.parts and 'images' in relative.parts))
            if 'corrected-runtime' in relative.parts or temporary_cache_input:
                continue
            old = os.readlink(link)
            target = link.parent / old
            raw = PurePosixPath(old)
            if raw.is_absolute() and raw.is_relative_to(prefix):
                tail = raw.relative_to(prefix)
                if '..' in tail.parts:
                    raise ValueError('Image target escapes remote job')
                target = gpu / str(tail)
                # Older ArtiFixer emitted nested checkpoint paths although
                # the staged checkpoint lives at the job root.
                if not target.exists() and 'checkpoint' in tail.parts:
                    i = tail.parts.index('checkpoint')
                    target = gpu.joinpath(*tail.parts[i:])
            resolved = target.resolve()
            image = link.suffix.lower() in IMAGE_SUFFIXES
            if not resolved.is_relative_to(root) or not resolved.exists():
                if image:
                    raise ValueError(f'Result image target is not preserved locally: {relative} -> {old}')
                continue
            if image and not resolved.is_file():
                raise ValueError(f'Result image target is not a file: {relative}')
            new = os.path.relpath(resolved, link.parent)
            if old != new:
                temporary = link.with_name('.portable-' + uuid.uuid4().hex)
                try:
                    temporary.symlink_to(new, target_is_directory=resolved.is_dir())
                    os.replace(temporary, link)
                finally:
                    temporary.unlink(missing_ok=True)
                repaired += 1
            if image:
                images.append({'link': str(link.relative_to(root)),
                               'target': str(resolved.relative_to(root)),
                               'bytes': resolved.stat().st_size})
    for item in images:
        if not (root / item['link']).is_file():
            raise ValueError('Portable result image verification failed: ' + item['link'])
    return {'repaired_links': repaired, 'verified_image_links': images}


def _ssh(lrz, cfg, command):
    reply = lrz._ssh_run(cfg, command, timeout=60)
    if reply.returncode:
        raise RuntimeError(reply.stderr or reply.stdout or 'Remote persistence command failed')
    return reply.stdout.strip()


def wait_for_exit(lrz, cfg, remote):
    # Terminal worker status can precede subprocess/launcher shutdown.
    script = """import pathlib, sys, time
marker = pathlib.Path(sys.argv[1]) / 'launcher-exit'
until = time.monotonic() + 30
while not marker.is_file() and time.monotonic() < until:
    time.sleep(1)
print('exited' if marker.is_file() else 'pending')
"""
    if _ssh(lrz, cfg, shlex.join(['python3', '-c', script, remote])) != 'exited':
        raise RuntimeError('Remote launcher has not exited; retaining remote artifacts')


_DELETE = '''import pathlib, shutil, sys
p = pathlib.Path(sys.argv[1])
if p.is_symlink():
    raise RuntimeError('Refusing symlink job directory')
if not p.exists():
    print('removed')
elif (p / 'launcher-exit').is_file():
    shutil.rmtree(p)
    print('removed')
else:
    print('pending')
'''


def cleanup_persisted(lrz, cfg, root, run_id):
    root = Path(root)
    receipt = read_json(root / RECEIPT, {})
    remote = remote_path(cfg, run_id)
    if receipt.get('status') != 'persisted' or receipt.get('remote_dir') != remote:
        return
    if receipt.get('remote_cleanup') == 'removed':
        return
    try:
        result = _ssh(lrz, cfg, shlex.join(['python3', '-c', _DELETE, remote]))
        receipt['remote_cleanup'] = 'removed' if result == 'removed' else 'pending'
        receipt.pop('cleanup_error', None)
    except Exception as exc:
        receipt.update(remote_cleanup='pending', cleanup_error=str(exc))
    atomic_json(root / RECEIPT, receipt)


def persist_remote_run(lrz, cfg, root, run_id, remote, *, excludes=()):
    """A successful checksum transfer plus portable targets authorizes cleanup."""
    root = Path(root)
    if remote != remote_path(cfg, run_id):
        raise ValueError('Remote job is outside the configured workspace')
    # Invalidate an older receipt before any transfer starts.
    atomic_json(root / RECEIPT, {'status': 'downloading', 'remote_dir': remote})
    wait_for_exit(lrz, cfg, remote)
    (root / 'gpu').mkdir(exist_ok=True)
    # Reuse the immutable WorldExplorer RGBs already staged locally. rsync's
    # checksum comparison skips unchanged inputs and replaces changed files
    # atomically, so a remote change cannot overwrite the shared source inode.
    from .worldexplorer.source import link_source_image
    staging = read_json(root / 'oraclegs-staging.json', {})
    manifest = staging.get('input_manifest')
    bundles = [(root / 'worldexplorer-source', root / 'gpu/worldexplorer-source')]
    if manifest:
        inputs = (root / manifest).resolve().parent
        if inputs.is_relative_to(root.resolve()):
            bundles.append((inputs / 'worldexplorer-source', root / 'gpu/inputs/worldexplorer-source'))
    for source, destination in bundles:
        if not source.is_dir():
            continue
        # Camera/trajectory metadata is immutable too, and shares the same lifetime.
        for item in [source / 'transforms.json', *(source / 'images').glob('*')]:
            target = destination / item.relative_to(source)
            if item.is_file() and not item.is_symlink() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                link_source_image(item, target)
    argv = ['rsync', '-azc', '-e', lrz.rsync_ssh_cmd(cfg)]
    argv += ['--exclude=' + pattern for pattern in excludes]
    argv += [f"{cfg['user']}@{cfg['host']}:{remote}/", str(root / 'gpu') + '/']
    lrz._mux_run(argv)
    verification = portable_links(root, run_id)
    from .deduplication import deduplicate_artifacts
    deduplication = deduplicate_artifacts(root)
    atomic_json(root / RECEIPT, {'status': 'persisted', 'remote_dir': remote,
                'excluded_patterns': list(excludes), 'remote_cleanup': 'pending',
                'deduplication': deduplication, **verification})
    cleanup_persisted(lrz, cfg, root, run_id)


def retry_cleanup(lrz, cfg, root):
    """Only receipts from successful future persistence authorize retries."""
    for receipt in Path(root).glob('*/' + RECEIPT):
        cleanup_persisted(lrz, cfg, receipt.parent, receipt.parent.name)
