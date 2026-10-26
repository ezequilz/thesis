"""Small, path-checked view of connected-trajectory boundary diagnostics."""
from ..splatfix.jobs import read_json


def connected_details(studio, manifest):
    result = {'enabled': False, 'passes': []}
    if manifest is None:
        return result
    root = manifest.parent.resolve()
    for phase in ('inference', 'plus'):
        path = root / f'{phase}-connected.json'
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        data = read_json(path, {})
        result['enabled'] = True
        entry = {'phase': phase, 'complete': data.get('complete', False),
                 'metadata_url': studio.file_url(path), 'legs': []}
        for leg in data.get('legs', []):
            item = {key: leg.get(key) for key in ('leg_index', 'status', 'skip_reason',
                    'generation_mode', 'fallback_reason', 'boundary_mae', 'seeding')}
            item.update(real_window_count=len(leg.get('target_indices', [])),
                        bridge_count=len(leg.get('keep_indices', [])), padding_count=leg.get('padding_count', 0))
            item['overlaps'] = []
            for row in leg.get('overlaps', []):
                image = root / row.get('comparison', '')
                if not image.is_file() or not image.resolve().is_relative_to(root):
                    continue
                item['overlaps'].append({key: row.get(key) for key in ('frame_index', 'side', 'mae', 'psnr_db', 'identical')}
                                       | {'url': studio.file_url(image)})
            entry['legs'].append(item)
        result['passes'].append(entry)
    return result
