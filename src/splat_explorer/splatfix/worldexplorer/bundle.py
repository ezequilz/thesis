"""Standalone bundle reader for isolated GPU Python environments."""
import hashlib
import json
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_staged(directory):
    root = Path(directory).resolve()
    data = json.loads((root / 'transforms.json').read_text())
    for frame in data['frames']:
        path = (root / frame['file_path']).resolve()
        if not path.is_relative_to(root) or digest_file(path) != frame['sha256']:
            raise ValueError('WorldExplorer supervision changed after staging')
        frame['file_path'] = str(path)
    return data

