"""Immutable inputs for another full pass over a saved reconstruction."""
from copy import deepcopy
from pathlib import Path
import shutil

from ..checkpoint import Checkpoint, atomic_json, source_fingerprint
from ..jobs import read_json



def resolve_repeat_prompt(studio, row, request, checkpoint):
    """Recover prompt before replacing the scene path with the repeat PLY."""
    from ..worldexplorer.source import scene_prompt
    def saved(value, source):
        return (value, source) if isinstance(value, str) and value.strip() else None
    found = saved(request.get('worldexplorer_prompt'), 'parent_request')
    if found:
        return found
    found = saved(checkpoint.manifest.get('metadata', {}).get('worldexplorer_prompt'), 'parent_checkpoint')
    if found:
        return found
    original = row.get('config', {}).get('splatfix', {}).get('checkpoint')
    if original:
        path = Path(original).resolve()
        if path.is_relative_to(Path(studio.cfg.output.dir).resolve()):
            body = read_json(path / 'checkpoint.json', {})
            found = saved(body.get('metadata', {}).get('worldexplorer_prompt'), 'original_checkpoint')
            if found:
                return found
            scene = body.get('scene_path')
            if scene:
                found = saved(scene_prompt(scene), 'original_scene_request')
                if found:
                    return found
    found = saved(scene_prompt(checkpoint.manifest['scene_path']), 'checkpoint_scene_request')
    if found:
        return found
    scene_id = row.get('scene') or row.get('config', {}).get('scene_id')
    if scene_id:
        for scene in studio.scenes():
            if scene.get('id') == scene_id and scene.get('path'):
                found = saved(scene_prompt(scene['path']), 'registered_scene_request')
                if found:
                    return found
    return None, None


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
    if not trajectory:
        trajectory = studio.result_artifact(manifest, 'preview-trajectory.json')
    saved = read_json(trajectory, {}) if trajectory else {}
    orbit = saved.get('recipe', {}).get('orbit')
    if not orbit:
        full_path = studio.result_artifact(manifest, 'orbit-full.json')
        full = read_json(full_path, {}) if full_path else {}
        if full.get('frames') and request.get('orbit_provenance'):
            import numpy as np
            poses = np.asarray([f['transform_matrix'] for f in full['frames']]) @ np.diag([1., -1., -1., 1.])
            orbit = {'poses': poses.tolist(), 'provenance': {
                key: value for key, value in request['orbit_provenance'].items() if key != 'source_transforms'}}
    checkpoint = studio.result_artifact(manifest, request.get('checkpoint_root'))
    source_checkpoint = not checkpoint or not (checkpoint / 'checkpoint.json').is_file()
    if source_checkpoint:
        raw = row.get('config', {}).get('splatfix', {}).get('checkpoint')
        checkpoint = Path(raw).resolve() if raw else None
        if (not checkpoint or not checkpoint.is_relative_to(Path(studio.cfg.output.dir).resolve())
                or not (checkpoint / 'checkpoint.json').is_file()):
            raise ValueError('Saved checkpoint poses and reference images are unavailable')
    if not request.get('runtime') or not request.get('camera_scale'):
        raise ValueError('Exact orbit or effective runtime settings are unavailable')
    if not orbit:
        from .repair import UPSTREAM_REVISION
        if not row.get('reduced') or request.get('upstream_revision') != UPSTREAM_REVISION:
            raise ValueError('Exact orbit unavailable; rebuilding requires a reduced run with the pinned trajectory revision')
        if not request.get('orbit_provenance'):
            raise ValueError('Saved trajectory interpolation settings are unavailable')
    from .checkpoint_policy import checkpoint_for_run
    cp = checkpoint_for_run(Checkpoint.load(checkpoint),
                            'endpoints' if request.get('repair_policy') == 'endpoints' else
                            request['runtime'].get('image_trajectory', 'checkpoint'))
    prompt, prompt_origin = resolve_repeat_prompt(studio, row, request, cp)
    if prompt:
        request['worldexplorer_prompt'] = prompt
        request['_repeat_prompt_origin'] = prompt_origin
    prompt_source = request['runtime'].get('prompt_source', row.get('config', {}).get('splatfix', {}).get('prompt_source', 'worldexplorer'))
    if prompt_source == 'worldexplorer' and not prompt:
        raise ValueError('Original WorldExplorer prompt is unavailable in the saved run, checkpoint or original scene')
    request['_repeat_source_checkpoint'] = source_checkpoint
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
        if request.get('_repeat_source_checkpoint'):
            # Let normal resolution preparation reproduce the original crop from
            # its full response/manual crop rather than resize a resized image.
            for key in ('manual_resized_rgb',):
                if view.get(key):
                    asset = cp.image_path({'original_rgb': view[key]})
                    target = Path('views') / view['id'] / (key + '.png')
                    shutil.copy2(asset, root / target)
                    view[key] = str(target)
            response = view.get('image_repair', {}).get('response_image')
            if response:
                asset = cp.image_path({'original_rgb': response})
                target = Path('views') / view['id'] / 'response.image'
                shutil.copy2(asset, root / target)
                view['image_repair']['response_image'] = str(target)
            view.pop('full_repaired_rgb', None)
        else:
            # Prepared references have already been calibrated.
            for key in ('image_repair', 'manual_resized_rgb', 'full_repaired_rgb'):
                view.pop(key, None)
    # Reduced runs can retain the exact calibrated GPT inputs even when the
    # source checkpoint's repair/crop has subsequently been changed.
    previews = read_json(row['_manifest'].parent / 'input-preview.json', {})
    references = sorted(previews.get('reference_images', []), key=lambda item: item['index'])
    reference_views = ([manifest['views'][0], manifest['views'][-1]] if cp.endpoint_only else manifest['views'])
    expected_hashes = request.get('reference_sha256', [])
    if request.get('mode') == 'edited' and len(references) == len(reference_views) == len(expected_hashes):
        from PIL import Image
        from .repair import digest_file
        for view, item, expected in zip(reference_views, references, expected_hashes):
            asset = studio.result_artifact(row['_manifest'], item['path'])
            if asset and digest_file(asset) == expected:
                with Image.open(asset) as image:
                    if image.size != (view['camera']['width'], view['camera']['height']):
                        continue
                relative = Path('views') / view['id'] / 'repaired_rgb.png'
                shutil.copy2(asset, root / relative)
                view['repaired_rgb'] = str(relative)
                for key in ('image_repair', 'manual_resized_rgb', 'full_repaired_rgb'):
                    view.pop(key, None)
    source = root / 'original.ply'
    shutil.copy2(row['_paths']['repaired'], source)
    manifest.update(scene_path=str(source.resolve()), source_fingerprint=source_fingerprint(source))
    metadata = manifest.setdefault('metadata', {})
    metadata.pop('artifixer_input_cache', None)
    if request.get('worldexplorer_prompt'):
        metadata['worldexplorer_prompt'] = request['worldexplorer_prompt']
        metadata['worldexplorer_prompt_source'] = request.get('_repeat_prompt_origin', 'parent_request')
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
    runtime.setdefault('connected_trajectory_generation', False)
    runtime.setdefault('connected_bridge_seeding', 'off')
    for key in ('scene_path', 'source_points3d', 'worldexplorer_source', 'repeat_from'):
        runtime.pop(key, None)
    anchor_path = studio.result_artifact(row['_manifest'], 'anchor-transforms.json')
    repeat = {'runtime': runtime, 'orbit': orbit,
              'trajectory_replay': 'saved_orbit' if orbit else 'rebuild_from_checkpoint',
              'expected_anchor_transforms': read_json(anchor_path, None) if anchor_path else None,
              'expected_reference_sha256': request.get('reference_sha256') if request.get('mode') == 'edited' else None,
              'expected_orbit_provenance': request.get('orbit_provenance'), 'seed': request.get('seed', 42),
              'camera_scale': request['camera_scale'], 'parent_result': row['id'],
              'parent_run': row['run_id'], 'repair_pass': row.get('repair_pass', 1) + 1,
              'parent_incomplete': bool(row.get('incomplete')),
              'parent_splat': str(Path(row['_paths']['repaired']).relative_to(row['_manifest'].parent))}
    atomic_json(root / 'checkpoint.json', manifest)
    atomic_json(root / 'repeat-repair.json', repeat)
    return repeat


def validate_rebuilt_provenance(expected, actual):
    """Do not silently run a different orbit when replaying historical inputs."""
    for key in ('implementation', 'split_mode', 'repair_policy', 'series_policy',
                'reference_frame_indices', 'frames', 'segments', 'metric_scale',
                'requested_spacing', 'effective_spacing', 'loop'):
        if key in (expected or {}) and actual.get(key) != expected[key]:
            raise ValueError(f'Rebuilt orbit differs from the saved run: {key}')
