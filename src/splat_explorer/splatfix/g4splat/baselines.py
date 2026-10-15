"""Published demo selections; never substitute incomplete demos for benchmarks."""
from pathlib import Path


def baseline_catalog(output_root, file_url):
    root = Path(output_root) / 'g4splat-published-baseline'
    scenes = [
        ('scene1', 'Room · 1 input view',
         'Original input, generated Stable Virtual Camera views/cameras, and exact generation/training settings are missing. No held-out ground truth established.'),
        ('scene4', 'Facade · 5 input views',
         'Original five images, camera calibration/poses, published split and exact scene recipe are missing. Held-out ground truth is not established.'),
    ]
    result = []
    for ident, label, reason in scenes:
        item = {'id': ident, 'label': label, 'ready': False,
                'status': 'Awaiting original author assets', 'reason': reason,
                'reference_url': 'https://dali-jack.github.io/g4splat-web/',
                'protocol': 'Untouched author inputs/configuration, native rendering, and held-out evaluation only when an authenticated split and ground truth exist.',
                'thumbnail_url': None, 'video_url': None, 'audit_url': None}
        for key, relative in [('thumbnail_url', f'evidence/{ident}_thumbnail.png'),
                              ('video_url', f'evidence/{ident}_ours.mp4'),
                              ('audit_url', 'manifest.json')]:
            path = root / relative
            if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                item[key] = file_url(path)
        result.append(item)
    return result
