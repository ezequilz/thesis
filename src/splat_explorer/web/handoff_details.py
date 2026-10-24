"""Read local handoff artifacts while GPU downloads and repairs are in progress."""
from ..splatfix.jobs import read_json


def handoff_details(studio, root):
    worker = read_json(root / 'worker-request.json', {})
    folders = list((root / 'handoff').glob('*')) + list((root / 'gpu/handoff').glob('*'))
    folders += list((root / 'gpu/results').glob('*/handoffs/*/*'))
    records = {}
    for folder in sorted(folders):
        if not folder.is_dir() or not folder.resolve().is_relative_to(root.resolve()):
            continue
        def safe(name):
            path = folder / name
            return path if path.is_file() and path.resolve().is_relative_to(root.resolve()) else None
        request_path = safe('camera.json') or safe('request.json')
        request = read_json(request_path, {}) if request_path else {}
        response_path = safe('response.json')
        response = read_json(response_path, {}) if response_path else {}
        ident = request.get('id') or response.get('id') or folder.name
        record = records.setdefault(ident, {'id': ident, 'images': {}})
        record.update({k: v for k, v in request.items() if k in ('phase', 'frame_index', 'camera')})
        for name, label in (('before.png', 'Downloaded boundary'), ('response-original.png', 'Original GPT response'),
                            ('reference.png', 'Next repaired reference (context)'), ('repaired.png', 'Repaired handoff')):
            path = safe(name)
            if path:
                record['images'][name] = {'label': label, 'url': studio.file_url(path),
                                         'modified': path.stat().st_mtime_ns}
        if response_path:
            record.update(error=response.get('error'), model=response.get('model'),
                          metadata_url=studio.file_url(response_path))
        if request_path:
            record['camera_url'] = studio.file_url(request_path)
    for record in records.values():
        record['status'] = ('failed' if record.get('error') else 'repaired' if 'repaired.png' in record['images']
                            else 'waiting for repair' if record['images'] else 'waiting for download')
    return {'enabled': bool(worker.get('runtime', {}).get('handoff_generation')) or bool(records),
            'items': sorted(records.values(), key=lambda r: (r.get('phase', ''), r.get('frame_index', -1), r['id']))}
