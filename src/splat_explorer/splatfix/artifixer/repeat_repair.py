"""Immutable inputs for another full pass over a saved reconstruction."""
from copy import deepcopy
from pathlib import Path
import shutil

from ..checkpoint import Checkpoint, atomic_json, source_fingerprint
from ..jobs import read_json


def repeat_inputs(studio, row):
    if not row or row['kind'] != 'ArtiFixer3D':
        raise ValueError('Choose an ArtiFixer result')
    manifest = row['_manifest']
    if (manifest is None or manifest.name not in ('result.json', 'partial-result.json')
            or not row['_paths']['repaired'] or not Path(row['_paths']['repaired']).is_file()):
        raise ValueError('A saved reconstruction PLY is required')
    request = read_json(manifest.parent / 'request.json', {})
    if request.get('trajectory_mode') != 'authors_orbit':
        raise ValueError('This result has no saved checkpoint orbit; use the benchmark repeat workflow for bicycle runs')
    trajectory = studio.result_artifact(manifest, request.get('trajectory'))
    checkpoint = studio.result_artifact(manifest, request.get('checkpoint_root'))
    if not trajectory or not checkpoint or not (checkpoint / 'checkpoint.json').is_file():
        raise ValueError('Saved run cameras or checkpoint are missing')
    saved = read_json(trajectory, {})
    orbit = saved.get('recipe', {}).get('orbit')
    if not orbit or not request.get('runtime') or not request.get('camera_scale'):
        raise ValueError('Exact orbit or effective runtime settings are unavailable')
    cp = Checkpoint.load(checkpoint)
    for view in cp.views:
        cp.image_path(view)
        if view.get('repaired_rgb'):
            cp.image_path(view, repaired=True)
    return cp, request, orbit


def snapshot_repeat(studio, row, destination):
    cp, request, orbit = repeat_inputs(studio, row)
    root = Path(destination)
    manifest = deepcopy(cp.manifest)
    # Copy only conditioning references, never old render/geometry/caption caches.
    for view in manifest['views']:
        for key in ('original_rgb', 'repaired_rgb'):
            if view.get(key):
                source = cp.image_path({'original_rgb': view[key]})
                relative = Path('views') / view['id'] / (key + '.png')
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                view[key] = str(relative)
        # The prepared, calibrated reference is the exact image used previously.
        for key in ('image_repair', 'manual_resized_rgb', 'full_repaired_rgb'):
            view.pop(key, None)
    source = root / 'original.ply'
    shutil.copy2(row['_paths']['repaired'], source)
    manifest.update(scene_path=str(source.resolve()), source_fingerprint=source_fingerprint(source))
    metadata = manifest.setdefault('metadata', {})
    metadata.pop('artifixer_input_cache', None)
    if request.get('worldexplorer_prompt'):
        metadata['worldexplorer_prompt'] = request['worldexplorer_prompt']
    if request.get('reconstruction_images') == 'all':
        bundle = studio.result_artifact(row['_manifest'], request.get('worldexplorer_source'))
        if not bundle:
            original = row['config'].get('splatfix', {}).get('checkpoint')
            from ..worldexplorer.source import stage_checkpoint_source
            bundle = stage_checkpoint_source(Checkpoint.load(original), root / 'worldexplorer-source') if original else None
        if not bundle:
            raise ValueError('Original reconstruction image bundle is unavailable')
        if Path(bundle).resolve() != (root / 'worldexplorer-source').resolve():
            shutil.copytree(bundle, root / 'worldexplorer-source')
        manifest['worldexplorer_source'] = 'worldexplorer-source'
    runtime = deepcopy(request['runtime'])
    for key in ('scene_path', 'source_points3d', 'worldexplorer_source', 'repeat_from'):
        runtime.pop(key, None)
    repeat = {'runtime': runtime, 'orbit': orbit, 'seed': request.get('seed', 42),
              'camera_scale': request['camera_scale'], 'parent_result': row['id'],
              'parent_run': row['run_id'], 'repair_pass': row.get('repair_pass', 1) + 1,
              'parent_incomplete': bool(row.get('incomplete')),
              'parent_splat': str(Path(row['_paths']['repaired']).relative_to(row['_manifest'].parent))}
    atomic_json(root / 'checkpoint.json', manifest)
    atomic_json(root / 'repeat-repair.json', repeat)
    return repeat
