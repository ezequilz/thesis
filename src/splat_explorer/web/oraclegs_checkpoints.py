"""Discover completed iteration-test PLYs within a downloaded OracleGS result."""
from ..splatfix.jobs import read_json


def iteration_checkpoints(row):
    manifest = row.get('_manifest')
    if manifest is None:
        return []
    root = manifest.parent.resolve()
    options = row.get('config', {}).get('splatfix', {})

    def safe(path):
        return path.resolve().is_relative_to(root) and path.is_file()

    recipe_path = root/'recipe.json'
    recipe = read_json(recipe_path, {}) if safe(recipe_path) else {}
    if not (options.get('oracle_iterations_testing') is True or
            recipe.get('runtime', {}).get('oracle_iterations_testing') is True):
        return []
    found = {}
    for marker in (root/'model/point_cloud').glob('iteration_*/saved-snapshot.json'):
        path = marker.parent/'point_cloud.ply'
        if not safe(marker) or not safe(path):
            continue
        record = read_json(marker, {})
        if not isinstance(record, dict):
            continue
        iteration = record.get('iteration')
        if (type(iteration) is int and iteration > 0
                and marker.parent.name == f'iteration_{iteration}'
                and path.stat().st_size > 0 and path.stat().st_size == record.get('size_bytes')):
            found[iteration] = path
    # Include an off-interval graceful stop or a final PLY retained by reduction.
    result = read_json(manifest, {}) if safe(manifest) else {}
    iteration = result.get('saved_iteration')
    path = row.get('_paths', {}).get('repaired')
    if type(iteration) is int and iteration > 0 and path and safe(path) and path.stat().st_size:
        found[iteration] = path
    return [{'iteration':iteration, 'path':found[iteration]} for iteration in sorted(found)]
