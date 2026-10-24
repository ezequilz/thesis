"""Snapshot ArtiFixer RGBs and exact cameras for post-generation OracleGS."""
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from ..checkpoint import atomic_json
from ..jobs import read_json
from .backend import validate_options
from .source_geometry import digest, inspect_ply, stage_saved_pose_depth

# UI selections are separate from the worker's initialization algorithm.
INITIALIZATIONS = {
    'source_splat': ('original', 'source_splat'),
    'point_cloud': ('original', 'point_cloud'),
    'repaired_splat': ('repaired', 'source_splat'),
    'repaired_full': ('repaired', 'source_splat'),
    'repaired_point_cloud': ('repaired', 'point_cloud'),
    'sfm': (None, 'sfm'),
}


def initialization_source(studio, row, source_kind):
    if source_kind == 'original':
        source = row['_paths'].get('original')
        if source is not None and Path(source).is_file():
            return Path(source)
        raise ValueError('Original starting PLY is unavailable; select from scratch')
    # Completed/partial manifests retain nested and remote artifact paths.
    # Resolve all candidates through the existing containment boundary.
    manifest = row['_manifest']
    candidates = ['repaired.ply']
    for name in ('result.json', 'partial-result.json'):
        metadata = studio.result_artifact(manifest, name)
        if metadata is not None:
            candidates.append(read_json(metadata, {}).get('splat_path'))
    candidates.extend([row.get('_paths', {}).get('repaired'),
                       'artifixer3d.ply', 'interrupted.ply',
                       'distill.interrupted/interrupted.ply'])
    available = []
    for candidate in candidates:
        source = studio.result_artifact(manifest, candidate)
        if source is not None and source.is_file():
            available.append(source)
    # Never let a partial-result page override an available completed repair.
    for interrupted in (False, True):
        for source in available:
            if (source.name == 'interrupted.ply') == interrupted:
                return source
    raise ValueError('Repaired PLY is unavailable (no repaired.ply or interrupted.ply); select Original PLY or from scratch')


AXES = np.diag([1., -1., -1., 1.])


def source_assets(studio, row):
    manifest = row['_manifest']
    if row['kind'] != 'ArtiFixer3D' or manifest is None or row['mode'] != 'edited':
        raise ValueError('Requires an edited ArtiFixer run with saved inference images')
    result = read_json(manifest, {})
    local = lambda raw: studio.result_artifact(manifest, raw)
    def data(raw):
        path = local(raw)
        return read_json(path, {}) if path else {}
    request = data('request.json')
    trajectory = data('preview-trajectory.json')
    if not trajectory:
        trajectory = data(request.get('trajectory'))
    if trajectory.get('camera_convention') != 'opencv_c2w':
        raise ValueError('Saved ArtiFixer camera trajectory is unavailable')
    anchors = trajectory.get('anchors', [])
    refs = data('input-preview.json').get('reference_images', [])
    endpoint = (result.get('repair_policy') or request.get('repair_policy')
                or trajectory.get('recipe', {}).get('repair_policy')) == 'endpoints'
    reference_count = 2 if endpoint else 6
    if len(anchors) != reference_count or len(refs) != reference_count:
        raise ValueError(f'Requires all {reference_count} saved GPT repaired reference images')
    refs = sorted(refs, key=lambda item: item['index'])
    if [item['index'] for item in refs] != list(range(reference_count)):
        raise ValueError('Reference image mapping is invalid')
    references = [local(item['path']) for item in refs]
    if any(p is None or not p.is_file() for p in references):
        raise ValueError('Repaired reference images are missing')
    preview = data('inference-preview.json')
    prediction = local(result.get('prediction_frames') or preview.get('prediction_frames') or 'inference/splatfix/frames/batch_0000/pred')
    if prediction is None:
        raise ValueError('ArtiFixer inference images are unavailable')
    cameras = trajectory['transforms']['frames']
    if endpoint and [a['frame_index'] for a in anchors] != [0, len(cameras) - 1]:
        raise ValueError('Endpoint reference cameras must bound the saved trajectory')
    predictions = {int(p.stem): p for p in prediction.glob('*.png') if p.stem.isdigit() and p.resolve().is_relative_to(manifest.parent.resolve())}
    expected = preview.get('reconstruction_inputs')
    if not isinstance(expected, list) or not expected:
        raise ValueError('Inference input index manifest is unavailable')
    if not set(expected) <= set(predictions) or not set(predictions) <= set(range(len(cameras))):
        raise ValueError('Wait for all ArtiFixer inference images to download')
    if request.get('runtime', {}).get('handoff_generation', False):
        return handoff_assets(studio, row, trajectory, references, predictions, data)
    return trajectory, references, predictions



def handoff_assets(studio, row, trajectory, references, predictions, data):
    """Use completed inference handoffs at their exact cameras, like fitting does."""
    backward_enabled = data('request.json').get('runtime', {}).get('handoff_backward_pass', False)
    state = data('inference-references.json')
    series = data('inference-seeding.json').get('series', [])
    original = [a['frame_index'] for a in trajectory['anchors']]
    indices, paths, hashes = (state.get(key, []) for key in ('frame_indices', 'paths', 'sha256'))
    if (state.get('original_frame_indices') != original or
            not len(indices) == len(paths) == len(hashes) == len(original) or
            any(type(i) is not int for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError('Completed handoff reference set is missing or invalid; wait for inference artifacts')
    first, last = min(original), max(original)
    anchors, repaired, retained_slots = [], [], []
    for slot, (old, new, raw, expected) in enumerate(zip(original, indices, paths, hashes)):
        if not 0 <= new < len(trajectory['transforms']['frames']):
            raise ValueError('Handoff reference camera is outside trajectory')
        if old in (first, last):
            if new != old:
                raise ValueError('Handoff must preserve endpoint cameras')
            # Original paths can live outside the downloaded result directory;
            # input-preview contains a byte-for-byte copy of each original RGB.
            source = references[slot]
        else:
            if new in original:
                raise ValueError('Interior reference was not replaced by a generated handoff')
            source = studio.result_artifact(row['_manifest'], raw)
        if source is None or not source.is_file() or digest(source) != expected:
            raise ValueError('Handoff reference is missing or its checksum does not match')
        if old == last and not backward_enabled:
            continue  # Generation context only; excluded from trusted fitting references.
        anchors.append({**trajectory['anchors'][slot], 'frame_index': new})
        repaired.append(source)
        retained_slots.append(slot)
    # The worker runs nonempty single-split segments in order. Each later seed
    # is the preceding segment's final generated frame, repaired in place.
    legs = []
    for segment in trajectory.get('segments', []):
        generated = segment.get('indices')
        if generated is None:
            generated = list(range(segment['start'], segment['start'] + segment['count']))
        generated = [i for i in generated if i not in original]
        if generated:
            legs.append(generated)
    if not legs or len(series) != len(legs):
        raise ValueError('Completed handoff trajectory seeding metadata is missing or incomplete')
    retained_indices = {a['frame_index'] for a in anchors}
    seeds = []
    normalized_segments = []
    for leg_index, (generated, audit) in enumerate(zip(legs, series)):
        seed = audit.get('seed_global_index')
        expected_seed = first if leg_index == 0 else legs[leg_index - 1][-1]
        if (audit.get('output') != 'inference' or not audit.get('enabled') or
                seed != expected_seed or seed not in retained_indices):
            raise ValueError('Handoff trajectory seed disagrees with the completed repaired references')
        seeds.append(seed)
        normalized_segments.append({'indices': generated, 'seed_index': seed})
    forward_seeds = retained_indices - ({last} if backward_enabled else set())
    if len(seeds) != len(set(seeds)) or set(seeds) != forward_seeds:
        raise ValueError('Handoff trajectories do not cover the completed repaired reference set')
    if backward_enabled:
        from ..artifixer.official_worker import handoff_backward_jobs
        backward = data('inference-backward.json')
        expected_jobs = handoff_backward_jobs(trajectory, indices)
        records = backward.get('series', [])
        if backward.get('complete') is not True or len(records) != len(expected_jobs):
            raise ValueError('Completed backward inference metadata is missing or incomplete')
        replacements = set()
        reverse_segments = []
        for record, expected in zip(records, expected_jobs):
            if (any(record.get(key) != value for key, value in expected.items())
                    or record.get('output') != 'inference'
                    or record.get('reference_frame_indices') != indices):
                raise ValueError('Backward inference trajectory disagrees with repaired references')
            replacements.update(expected['export_global_indices'])
            reverse_segments.append({'indices': expected['export_global_indices'],
                                     'seed_index': expected['seed_global_index'], 'direction': 'reverse'})
        normalized_segments = [{**segment, 'indices': [i for i in segment['indices'] if i not in replacements]}
                               for segment in normalized_segments] + reverse_segments
    # Repaired boundaries are trusted references, never duplicate synthetic targets.
    excluded = set(original) | retained_indices
    predictions = {i: path for i, path in predictions.items() if i not in excluded}
    if not predictions:
        raise ValueError('Handoff run has no generated non-reference images to export')
    trajectory = {**trajectory, 'anchors': anchors, 'segments': normalized_segments,
                  'handoff_export': {'reference_frame_indices': [a['frame_index'] for a in anchors],
                                     'original_reference_slots': retained_slots,
                                     'excluded_original_reference_indices': sorted(set(original) - ({first, last} if backward_enabled else {first})),
                                     'policy': ('first/last original repairs and completed handoffs; actual forward/backward seeds'
                                                if backward_enabled else 'first original repair and completed handoffs; actual single-split generation seeds')}}
    return trajectory, repaired, predictions


def trajectory_references(studio, row, trajectory, predictions):
    """Map exported forward predictions to the trusted seed used for generation."""
    request_path = studio.result_artifact(row['_manifest'], 'request.json')
    request = read_json(request_path, {}) if request_path else {}
    anchors = {a['frame_index']: i for i, a in enumerate(trajectory['anchors'])}
    # Endpoint reconstruction consumes only front-to-back inference. The reverse
    # output is separate, even when its cameras coincide with these predictions.
    if not trajectory.get('handoff_export') and (len(anchors) == 2 or (request.get('trajectory_mode') == 'authors_orbit'
                            and request.get('runtime', {}).get('split_mode') == 'no-split')):
        segments = [{'indices': sorted(predictions), 'cache_seed_index': trajectory['anchors'][0]['frame_index']}]
    else:
        segments = trajectory.get('segments', [])
    assignments = {}
    for segment_index, segment in enumerate(segments):
        indices = segment.get('indices')
        if indices is None:
            indices = list(range(segment['start'], segment['start'] + segment['count']))
        seed = segment.get('seed_index', segment.get('cache_seed_index'))
        if seed is None and indices and indices[0] in anchors:
            seed = indices[0]  # Historical local loops begin at their repaired anchor.
        selected = set(indices) & set(predictions)
        if not selected:
            continue
        if seed not in anchors:
            raise ValueError('ArtiFixer trajectory has no saved repaired seed; select camera proximity for this historical run')
        for index in selected:
            if index in assignments:
                raise ValueError('ArtiFixer prediction belongs to multiple saved trajectories')
            assignments[index] = (anchors[seed], segment_index)
    if set(assignments) != set(predictions):
        raise ValueError('ArtiFixer trajectory reference mapping is incomplete; select camera proximity for this historical run')
    return assignments


def export_inputs(studio, row, destination, initialization, *, oracle_attention_batching='within_trajectory',
                  oracle_artifixer_input='new_only', oracle_artifixer_reference_assignment='trajectory'):
    if oracle_artifixer_reference_assignment not in ('trajectory', 'camera_proximity'):
        raise ValueError('oracle_artifixer_reference_assignment must be trajectory or camera_proximity')
    if oracle_artifixer_input not in ('new_only', 'original_and_new'):
        raise ValueError('oracle_artifixer_input must be new_only or original_and_new')
    if initialization not in INITIALIZATIONS:
        raise ValueError('Initialization must be one of: ' + ', '.join(INITIALIZATIONS))
    selection = initialization
    source_kind, initialization = INITIALIZATIONS[selection]
    options = validate_options({'oracle_reference_mode':'all', 'oracle_adapter':'saved_poses',
                                'oracle_initialization':initialization,
                                'oracle_attention_batching':oracle_attention_batching,
                                'oracle_depth_source':'source_splat' if initialization in ('source_splat', 'point_cloud') else 'predicted'})
    trajectory, references, predictions = source_assets(studio, row)
    assignments = (trajectory_references(studio, row, trajectory, predictions)
                   if oracle_artifixer_reference_assignment == 'trajectory' else None)
    root = Path(destination)
    (root/'scene/images').mkdir(parents=True)
    calibration = trajectory['transforms']
    anchors = trajectory['anchors']
    frames, views, targets = [], [], []
    with Image.open(next(iter(predictions.values()))) as image:
        size = image.size
    def stage(source, camera_index, name):
        with Image.open(source) as image:
            # Both sources cover the recorded full camera FOV; only resolution changes.
            if abs(image.width/image.height-calibration['w']/calibration['h']) > .01:
                raise ValueError('ArtiFixer RGB aspect ratio disagrees with saved cameras')
            image.convert('RGB').resize(size, Image.Resampling.LANCZOS).save(root/'scene/images'/name)
        camera = calibration['frames'][camera_index]
        pose = np.asarray(camera['transform_matrix'], dtype=float)
        if pose.shape != (4,4) or not np.isfinite(pose).all():
            raise ValueError('Invalid ArtiFixer camera pose')
        return {'file_path':'images/'+name, 'w':size[0], 'h':size[1],
                **{k:float(camera.get(k, calibration[k]))*size[axis]/calibration['wh'[axis]]
                   for k,axis in [('fl_x',0),('fl_y',1),('cx',0),('cy',1)]},
                'transform_matrix':(pose@AXES).tolist()}
    for i, source in enumerate(references):
        name = f'ref_{i:06d}.png'
        frames.append(stage(source, anchors[i]['frame_index'], name))
        views.append({'image':'scene/images/'+name, 'view_id':anchors[i]['view_id'], 'source_sha256':digest(source)})
    for index, source in sorted(predictions.items()):
        name = f'target_{index:06d}.png'
        frames.append(stage(source, index, name))
        if assignments is not None:
            reference, segment_index = assignments[index]
        else:
            reference = min(range(len(references)), key=lambda i: np.linalg.norm(np.asarray(frames[i]['transform_matrix'])[:3,3]-np.asarray(frames[-1]['transform_matrix'])[:3,3]))
            segment_index = reference
        targets.append({'target_index':index, 'image_name':name, 'reference_index':reference,
                        'segment_index':segment_index, 'alpha':0.0, 'source_sha256':digest(source)})
    manifest = {**options, 'oracle_artifixer_reference_assignment':oracle_artifixer_reference_assignment, 'oracle_artifixer_input':oracle_artifixer_input, 'mode':'edited', 'views':views, 'resolution':list(size),
                'seva_internal_resolution':list(size), 'seva_task':'img2trajvid',
                'artifixer_source':{'run_id':row['run_id'], 'result_id':row['id'], 'generated_count':len(targets)},
                'trajectory':{'frames':targets, 'segments':[
                    {'index':i, 'kind':'artifixer_trajectory' if assignments is not None else 'artifixer_reference_group',
                     'reference_index':next(t['reference_index'] for t in targets if t['segment_index'] == i),
                     'target_indices':[t['target_index'] for t in targets if t['segment_index'] == i]}
                    for i in sorted({t['segment_index'] for t in targets})]}, 'camera_convention':'opengl_c2w'}
    if trajectory.get('handoff_export'):
        manifest['artifixer_source']['handoff'] = trajectory['handoff_export']
    if initialization in ('source_splat', 'point_cloud'):
        source = initialization_source(studio, row, source_kind)
        count, degree = inspect_ply(source)
        shutil.copyfile(source, root/'source-initialization.ply')
        manifest['artifixer_source']['initialization'] = selection
        manifest['artifixer_source']['ply_source'] = source_kind
        manifest['artifixer_source']['ply_filename'] = source.name
        manifest['source_initialization'] = {'path':'source-initialization.ply', 'sha256':digest(root/'source-initialization.ply'), 'gaussian_count':count, 'sh_degree':degree}
        manifest['source_splat_used_for_initialization'] = True
        manifest['source_depth'] = stage_saved_pose_depth(root, frames[:len(views)])
    if oracle_artifixer_input == 'original_and_new':
        from ..worldexplorer.source import stage_source, stage_checkpoint_source
        # A saved original.ply may be a copy; the checkpoint retains archive provenance.
        checkpoint = row.get('config', {}).get('splatfix', {}).get('checkpoint')
        checkpoint_path = Path(checkpoint) if checkpoint else None
        saved = read_json(checkpoint_path if checkpoint_path.is_file() else checkpoint_path/'checkpoint.json', {}) if checkpoint_path else {}
        if saved:
            from ..checkpoint import Checkpoint
            cp_root = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
            staged = stage_checkpoint_source(Checkpoint(cp_root, saved), root/'worldexplorer-source')
        else:
            source = row['_paths']['original']
            staged = stage_source(source, root/'worldexplorer-source') if source else None
        if not staged:
            raise ValueError('Original WorldExplorer images are unavailable for this run; select new images only')
        manifest['worldexplorer_source'] = 'worldexplorer-source'
    atomic_json(root/'scene/transforms.json', {'frames':frames})
    atomic_json(root/'inputs.json', manifest)
    return manifest


def publish_generated(inputs, output, manifest, *, should_stop=lambda:False):
    """Supply the existing assembly boundary without invoking SEVA."""
    destination = Path(output)/'seva'/manifest['seva_task']/'scene'
    source = Path(inputs)/'scene'
    destination.mkdir(parents=True)
    for path in sorted(source.rglob('*')):
        if should_stop():
            raise InterruptedError('Cancelled while staging ArtiFixer images')
        target = destination/path.relative_to(source)
        if path.is_dir():
            target.mkdir()
        else:
            shutil.copy2(path, target)
