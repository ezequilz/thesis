"""Share verified immutable artifact bytes within a finished run archive.

Hard links keep archives portable and survive deletion of either path. Archived
artifacts must be replaced atomically when updated (as rsync does), never edited
in place. Mutable metadata, logs and runtime dependencies are not candidates.
"""
from __future__ import annotations

import errno
import hashlib
import os
import stat
import uuid
from collections import defaultdict
from pathlib import Path

SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.exr',
            '.npy', '.npz', '.bin', '.h5', '.ply', '.pt', '.pth', '.safetensors'}


def _signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Artifact is no longer a regular file: {path}')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_mode, info.st_uid, info.st_gid)


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.digest()


def deduplicate_artifacts(root):
    """Share verified artifacts and invalidate the dashboard's size snapshot."""
    from ..scene_runs.store import OUTPUT_SIZE_NAME
    try:
        return _deduplicate_artifacts(root)
    finally:
        # Also invalidate after a partial failure: some replacements may already
        # have succeeded. The next dashboard read counts unique inodes again.
        (Path(root) / OUTPUT_SIZE_NAME).unlink(missing_ok=True)


def _deduplicate_artifacts(root):
    """Deduplicate a quiescent GPU archive; never follow symlinks or cross runs.

    Call only after the worker exits and its final download is verified. Size
    and permissions filter candidates before hashing. Replacement is atomic;
    unsupported hard links leave the original intact and are reported.
    """
    gpu = Path(root) / 'gpu'
    if gpu.is_symlink():
        raise ValueError('Refusing symlink GPU archive')
    groups = defaultdict(list)
    for directory, dirs, files in os.walk(gpu, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != 'corrected-runtime'
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in SUFFIXES or path.is_symlink():
                continue
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_size:
                groups[(info.st_dev, info.st_size, info.st_mode,
                        info.st_uid, info.st_gid)].append(path)
    result = {'version': 1, 'method': 'sha256-hardlink', 'linked_files': 0,
              'reclaimed_bytes': 0, 'skipped_files': 0}
    for paths in groups.values():
        if len(paths) < 2:
            continue
        verified = {}
        inodes = {}
        for path in paths:
            before = _signature(path)
            inode = before[:2]
            digest = inodes.get(inode)
            if digest is None:
                digest = _digest(path)
                if _signature(path) != before:
                    raise RuntimeError(f'Artifact changed during verification: {path}')
                inodes[inode] = digest
            prior = verified.get(digest)
            if prior is None:
                verified[digest] = (path, before)
                continue
            source, source_signature = prior
            if source_signature[:2] == inode:
                continue
            temporary = path.with_name('.dedup-' + uuid.uuid4().hex)
            try:
                os.link(source, temporary, follow_symlinks=False)
                if (_signature(source) != source_signature
                        or _signature(temporary) != source_signature
                        or _signature(path) != before):
                    raise RuntimeError(f'Artifact changed during deduplication: {path}')
                # An existing external/internal hard link still owns these bytes.
                reclaimed = before[2] if path.stat().st_nlink == 1 else 0
                os.replace(temporary, path)
                result['linked_files'] += 1
                result['reclaimed_bytes'] += reclaimed
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES,
                                     errno.ENOTSUP, errno.EMLINK):
                    raise
                result['skipped_files'] += 1
            finally:
                temporary.unlink(missing_ok=True)
    return result


def deduplicate_persisted_run(root):
    """Explicit backfill for archives with a successful persistence receipt."""
    from .checkpoint import atomic_json
    from .jobs import read_json
    from .persistence import RECEIPT
    root = Path(root)
    receipt = read_json(root / RECEIPT, {})
    if receipt.get('status') != 'persisted' or receipt.get('remote_cleanup') != 'removed':
        raise ValueError('Run must be fully persisted with remote cleanup complete')
    result = deduplicate_artifacts(root)
    receipt['deduplication'] = result
    atomic_json(root / RECEIPT, receipt)
    return result


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(deduplicate_persisted_run(parser.parse_args().run), indent=2))
