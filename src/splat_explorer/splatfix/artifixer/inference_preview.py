"""Publish completed first-pass images while the GPU continues reconstruction."""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path, PurePosixPath

from ..checkpoint import atomic_json


def publish_preview(root, predictions, indices, reference_count, **metadata):
    root, predictions = Path(root), Path(predictions)
    request = root / 'request.json'
    if request.is_file():
        trajectory = json.loads(request.read_text()).get('trajectory')
        if trajectory and Path(trajectory).is_file():
            shutil.copyfile(trajectory, root / 'preview-trajectory.json')
    atomic_json(root / 'inference-preview.json', {
        **metadata, 'inference_ready': True,
        'prediction_frames': str(predictions.relative_to(root)),
        'reconstruction_inputs': sorted(indices), 'reference_count': reference_count,
    })


class InferencePreviewMirror:
    """One successful, restart-safe transfer per result; failures are best effort."""

    def __init__(self, root, container_root, remote_root, endpoint, ssh_command, transfer):
        self.root = Path(root)
        self.container_root = PurePosixPath(container_root)
        self.remote_root, self.endpoint = remote_root, endpoint
        self.ssh_command, self.transfer = ssh_command, transfer
        self.thread = None
        self.retry_after = 0

    def poll(self, state):
        raw = state.get('inference_preview')
        if not raw or (self.thread is not None and self.thread.is_alive()) or time.monotonic() < self.retry_after:
            return
        try:
            path = PurePosixPath(raw)
            relative = path.relative_to(self.container_root)
            if '..' in path.parts or len(relative.parts) != 3 or relative.parts[0] != 'results' or path.name != 'inference-preview.json':
                return
        except (ValueError, TypeError):
            return
        if (self.root / 'gpu' / str(relative)).is_file():
            return
        self.retry_after = time.monotonic() + 30
        self.thread = threading.Thread(target=self._download, args=(relative,), daemon=True)
        self.thread.start()

    def _download(self, relative):
        staging = self.root / '.inference-preview-transfer'
        staging.mkdir(parents=True, exist_ok=True)
        try:
            # No GPU process or checkpoint access. Bound DSS/network bandwidth;
            # timeout also bounds the join before the final ordinary download.
            self.transfer(['rsync', '-az', '--timeout=60', '--bwlimit=10240',
                '--include=/inference/', '--include=/inference/**/', '--include=/inference-preview.json',
                '--include=/inference/**/pred/*.png',
                '--include=/prepared/', '--include=/prepared/**/', '--include=/prepared/**/*.json',
                '--include=/request.json', '--include=/preview-trajectory.json',
                '--include=/benchmark-run.json', '--exclude=*',
                '-e', self.ssh_command,
                self.endpoint + ':' + str(PurePosixPath(self.remote_root) / relative.parent) + '/', str(staging) + '/'])
            data = json.loads((staging / relative.name).read_text())
            data.setdefault('output_dir', str(self.container_root / relative.parent))
            if any(p.is_symlink() for p in staging.rglob('*')):
                raise ValueError('Preview transfer must contain regular local files')
            prediction = Path(data['prediction_frames'])
            if prediction.is_absolute() or '..' in prediction.parts:
                raise ValueError('Preview predictions must be relative to the result')
            for index in data['reconstruction_inputs']:
                if not (staging / prediction / f'{int(index):05d}.png').is_file():
                    raise ValueError('Preview transfer is missing a reconstruction input')
            destination = self.root / 'gpu' / str(relative.parent)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging, destination, dirs_exist_ok=True,
                            ignore=lambda directory, names: [relative.name] if Path(directory) == staging else [])
            # Publish last: the catalog never observes a partially transferred gallery.
            atomic_json(destination / relative.name, data)
            try:
                from .evaluation_chart import write_evaluation_chart
                chart = write_evaluation_chart(destination)
                atomic_json(destination / 'preview-evaluation-status.json', {'status': 'complete', 'chart': chart})
            except Exception as exc:
                atomic_json(destination / 'preview-evaluation-status.json', {'status': 'error', 'message': str(exc)})
        except Exception as exc:
            (self.root / 'inference-preview-transfer.log').write_text(f'{type(exc).__name__}: {exc}\n')
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def finish(self):
        if self.thread is not None:
            self.thread.join()
