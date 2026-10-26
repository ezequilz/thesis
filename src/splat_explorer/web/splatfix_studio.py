"""Dedicated dashboard adapter over the shared scene-run store and manager."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import quote
from ..splatfix.checkpoint import Checkpoint, camera_from_record, atomic_json
from ..splatfix.jobs import read_json, validate_job, registered_benchmark
from .scene_run_studio import SceneRunStudio
from ..splatfix.methods import DEFAULT_METHOD, available_methods, get_method


RESTARTABLE_STATUSES = frozenset(('error', 'stopped', 'gpu_expired', 'stopping', 'completed'))


class SplatfixStudio(SceneRunStudio):
    def __init__(self, app, store=None):
        super().__init__(app, store, pipeline='splatfix')
        self.checkpoint_root = Path(self.cfg.output.dir).resolve() / 'splatfix'
        from .splatfix_results import ResultCatalog
        self.results = ResultCatalog(self)
        self.benchmark_root = Path(self.cfg.output.dir).resolve() / 'benchmarks'
        from .custom_selection_viser import CustomSelectionViser
        self.custom_visor = CustomSelectionViser(self.cfg)

    def defaults(self):
        from ..splatfix.resolution import profile_size
        defaults = super().defaults()
        defaults['width'], defaults['height'] = profile_size(
            self.cfg.get('splatfix', {}).get('resolution_profile', '720p'))
        return defaults

    def checkpoints(self):
        manifests = list(self.checkpoint_root.glob('*/checkpoint.json'))
        manifests += list(self.root.glob('run_*/checkpoints/*/checkpoint.json'))
        result = []
        for path in sorted(manifests, reverse=True):
            try:
                cp = Checkpoint.load(path)
                views = [{**v, 'original_url': self.file_url(cp.image_path(v)) if v.get('original_rgb') else None,
                          'repaired_url': self.file_url(cp.image_path(v, True)) if v.get('repaired_rgb') else None}
                         for v in cp.views]
                for view in views:
                    from ..splatfix.oraclegs.captures import capture_77
                    from ..splatfix.pose_correction import correction_status
                    view['pose_correction_status'] = correction_status(cp, view)
                    record = view.get('pose_correction') or {}
                    if record.get('artifacts'):
                        candidate = cp.root / record['artifacts'] / 'candidate.png'
                        if candidate.is_file():
                            view['pose_candidate_url'] = self.file_url(candidate)
                    if view['pose_correction_status'] == 'accepted':
                        view['pose_corrected_url'] = view.get('pose_candidate_url')
                    elif view['pose_correction_status'] == 'rejected':
                        original = cp.root / record.get('artifacts', '') / 'original.png'
                        view['pose_corrected_url'] = self.file_url(original) if original.is_file() else view['original_url']
                    else:
                        view['pose_corrected_url'] = None
                    wide = capture_77(view)
                    view['fov77_original_url'] = self.file_url(cp.image_path(wide)) if wide.get('original_rgb') else None
                    view['fov77_repaired_url'] = self.file_url(cp.image_path(wide, True)) if wide.get('repaired_rgb') else None
                    capture = view.get('seva_capture') or {}
                    view['seva_original_url'] = self.file_url(cp.image_path(capture)) if capture.get('original_rgb') else None
                    view['seva_response_url'] = self.file_url(cp.image_path({'original_rgb':capture['original_response_rgb']})) if capture.get('original_response_rgb') else None
                    view['seva_repaired_url'] = self.file_url(cp.image_path(capture, True)) if capture.get('repaired_rgb') else None
                distinct = len({tuple(camera_from_record(view).c2w.ravel()) for view in cp.views})
                method_readiness = {method['id']: get_method(method['id']).readiness(cp)
                                    for method in available_methods()}
                reconstruction_ready = method_readiness[DEFAULT_METHOD]['ready']
                reconstruction_reason = method_readiness[DEFAULT_METHOD]['reason']
                result.append({'id': str(cp.root), 'name': self.checkpoint_name(cp.root), 'storage_name': cp.root.name, 'scene': Path(cp.manifest['scene_path']).name,
                               'repair_target_views': len(cp.repair_views),
                               'required_edited': sum(bool(v.get('repaired_rgb')) for v in cp.repair_views),
                               'seva_captured': sum(bool(v.get('seva_original_url')) for v in views),
                               'seva_edited': sum(bool(v.get('seva_repaired_url')) for v in views),
                               'rgb_renderer': cp.manifest.get('metadata', {}).get('renderer', {}).get('backend'),
                               'scene_path': cp.manifest['scene_path'],
                               'created_at': cp.manifest.get('created_at'), 'views': views, 'target_views': cp.target_views,
                               'reconstruction_methods': method_readiness,
                               'complete': cp.complete, 'reconstruction_ready': reconstruction_ready,
                               'reconstruction_readiness': reconstruction_reason, 'distinct_camera_poses': distinct, 'edited': sum(bool(v.get('repaired_rgb')) for v in views),
                               'trajectory_caches': len(list((cp.root / 'trajectories').glob('*/trajectory.json'))),
                               'metadata': cp.manifest.get('metadata', {})})
            except (OSError, ValueError, KeyError):
                continue
        return result

    @staticmethod
    def checkpoint_name(root):
        # Keep UI labels separate from manifests written by running stages.
        label = read_json(root / 'display-name.json', {})
        name = label.get('name') if isinstance(label, dict) else None
        return name if isinstance(name, str) and name.strip() else root.name

    def crop_checkpoint(self, body):
        from PIL import Image
        from ..splatfix.manual_crop import crop_source, save_crop
        if not isinstance(body, dict) or not isinstance(body.get('checkpoint'), str):
            raise ValueError('Choose an existing checkpoint')
        root = Path(body['checkpoint']).resolve()
        manifests = list(self.checkpoint_root.glob('*/checkpoint.json'))
        manifests += list(self.root.glob('run_*/checkpoints/*/checkpoint.json'))
        if (not any(root.is_relative_to(base.resolve()) for base in (self.checkpoint_root, self.root))
                or not any(p.parent.resolve() == root for p in manifests)):
            raise ValueError('Choose an existing checkpoint')
        cp = Checkpoint.load(root)
        view = next((v for v in cp.views if v['id'] == body.get('view_id')), None)
        if view is None:
            raise ValueError('Unknown saved view')
        image_set = body.get('image_set', 'saved')
        if image_set == 'seva':
            view = view.get('seva_capture') or {}
        elif image_set == '77':
            from ..splatfix.oraclegs.captures import capture_77
            view = capture_77(view)
        elif image_set != 'saved':
            raise ValueError('Unknown image set')
        source, token = crop_source(cp, view)
        if 'box' in body:
            if any(run.config.splatfix.get('stage') == 'pose_correction'
                   and run.config.splatfix.get('checkpoint') == str(root)
                   and run.state.status.value in ('starting', 'running', 'stopping')
                   for run in self.store.list_runs()):
                raise ValueError('Wait for pose correction to finish before changing a crop')
            save_crop(cp, view, body['box'], body.get('source_token'))
            return {'ok': True, 'checkpoint': next(c for c in self.checkpoints() if c['id'] == str(root))}
        with Image.open(source) as raw:
            image = raw.convert('RGB')
        # PNG preview for response.image payloads, whose original bytes stay untouched.
        preview = source.parent / ('crop-source-' + token + '.png')
        if not preview.exists():
            image.save(preview)
        return {'url': self.file_url(preview), 'width': image.width, 'height': image.height,
                'source_token': token, 'box': (view.get('manual_crop') or {}).get('box_xywh')}

    def rename_checkpoint(self, body):
        if not isinstance(body, dict):
            raise ValueError('Request must be an object')
        name, ident = body.get('name'), body.get('checkpoint')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
            raise ValueError('Enter a checkpoint name between 1 and 120 characters')
        if not isinstance(ident, str):
            raise ValueError('Choose an existing splatfix checkpoint')
        root = Path(ident).resolve()
        manifests = list(self.checkpoint_root.glob('*/checkpoint.json'))
        manifests += list(self.root.glob('run_*/checkpoints/*/checkpoint.json'))
        roots = (self.checkpoint_root.resolve(), self.root.resolve())
        if (not any(root.is_relative_to(base) for base in roots)
                or not any(p.parent.resolve() == root for p in manifests)):
            raise ValueError('Choose an existing splatfix checkpoint')
        Checkpoint.load(root)
        atomic_json(root / 'display-name.json', {'name': name.strip()})
        return {'ok': True, 'checkpoint': str(root), 'name': name.strip()}

    def delete_checkpoint(self, body):
        import shutil
        if not isinstance(body, dict) or not isinstance(body.get('checkpoint'), str):
            raise ValueError('Choose an existing splatfix checkpoint')
        raw = Path(body['checkpoint'])
        root = raw.resolve()
        # Only delete whole registered checkpoint directories, never a run or an alias.
        standalone = root.parent == self.checkpoint_root.resolve()
        nested = (root.parent.name == 'checkpoints'
                  and root.parent.parent.name.startswith('run_')
                  and root.parent.parent.parent == self.root.resolve())
        if (raw.is_symlink() or not (standalone or nested)
                or not (root / 'checkpoint.json').is_file()
                or (root / 'checkpoint.json').is_symlink()):
            raise ValueError('Choose an existing splatfix checkpoint')
        cp = Checkpoint.load(root)
        for run in self.store.list_runs():
            from ..splatfix.downloads import active as download_active
            if run.config.pipeline != 'splatfix' or (run.state.status.value in ('completed', 'error', 'stopped') and not download_active(run.path)):
                continue
            refs = [run.config.splatfix.get('checkpoint'), run.state.details.get('checkpoint')]
            if (any(ref and Path(ref).resolve() == root for ref in refs)
                    or root.is_relative_to(Path(run.path).resolve())):
                raise ValueError('This checkpoint is used by an active or queued run. Cancel or finish that run before deleting it.')
        repaired = sum(bool(view.get('repaired_rgb')) + bool((view.get('seva_capture') or {}).get('repaired_rgb')) for view in cp.views)
        if repaired and body.get('confirm_repaired') is not True:
            return {'ok': False, 'requires_confirmation': True, 'repaired': repaired,
                    'checkpoint': str(root), 'name': self.checkpoint_name(root)}
        shutil.rmtree(root)
        return {'ok': True, 'checkpoint': str(root)}

    def benchmarks(self):
        rows = []
        for manifest in sorted(self.benchmark_root.glob('*/input/benchmark.json')):
            if (manifest.is_symlink() or manifest.parent.is_symlink() or manifest.parent.parent.is_symlink()
                    or not manifest.resolve().is_relative_to(self.benchmark_root.resolve())):
                continue
            metadata = read_json(manifest, {})
            if not isinstance(metadata, dict):
                metadata = {}
            try:
                source, metadata = registered_benchmark(self.benchmark_root, manifest.parent)
                ready, reason = True, 'Published dataset inputs ready'
            except ValueError as exc:
                source, ready, reason = manifest.parent, False, str(exc)
            rows.append({'id': str(source), 'name': metadata.get('label') or metadata.get('name') or source.parent.name,
                         'ready': ready, 'status': reason, 'metadata': metadata,
                         'provenance_note': 'Published dataset pipeline test; exact website orbit and settings are not confirmed.'})
        return rows

    def allowed_file(self, raw):
        path = Path(raw).resolve()
        roots = (self.root.resolve(), self.checkpoint_root.resolve(), self.benchmark_root.resolve())
        if path.is_relative_to(self.benchmark_root.resolve()) and path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.json'):
            raise ValueError('Benchmark artifacts are limited to images and JSON provenance')
        if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
            raise ValueError('Artifact is outside the splatfix output roots or missing')
        # Never expose credentials or runtime configuration as artifacts.
        model_checkpoint = (path.suffix.lower() in ('.pt', '.pth', '.ckpt')
                            and path.is_relative_to(self.root.resolve())
                            and 'gpu' in path.relative_to(self.root.resolve()).parts)
        if not model_checkpoint and path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.ply', '.json', '.log', '.jsonl', '.mp4'):
            raise ValueError('Unsupported artifact type')
        return path

    def file_url(self, path):
        from ..splatfix.deduplication import canonical_artifact
        path = canonical_artifact(path, self.root)
        return '/api/splatfix/file?path=' + quote(str(Path(path).resolve()), safe='')

    def jobs(self):
        queue = self.store.ordered_queue()
        positions = {row['run_id']: index + 1 for index, row in enumerate(queue)}
        movable = [row['run_id'] for row in queue
                   if row['config'].get('pipeline') == 'splatfix'
                   and not row['state'].get('details', {}).get('remote_dir')]
        rows = []
        for run in self.store.list_runs():
            if run.config.pipeline != 'splatfix':
                continue
            row = run.to_dict()
            from ..splatfix.downloads import status as download_status
            row['download'] = download_status(run.path)
            row['queue_position'] = positions.get(run.run_id)
            row['queue_move_up'] = run.run_id in movable and movable.index(run.run_id) > 0
            row['queue_move_down'] = run.run_id in movable and movable.index(run.run_id) < len(movable) - 1
            row['restart_supported'] = run.state.status.value in RESTARTABLE_STATUSES
            from ..splatfix.artifixer.resume import resume_available, preparation_available
            row['resume_supported'] = resume_available(run)
            row['preparation_supported'] = preparation_available(run)
            row['state']['status'] = {'stopped': 'cancelled', 'error': 'failed'}.get(run.state.status.value, run.state.status.value)
            row['log_url'] = self.file_url(Path(run.path) / 'stage.log') if (Path(run.path) / 'stage.log').exists() else None
            if (Path(run.path) / 'gpu' / 'stage.log').exists():
                row['log_url'] = self.file_url(Path(run.path) / 'gpu' / 'stage.log')
            log_snapshot = Path(run.path) / 'gpu' / 'live-stage.log'
            log_metadata = read_json(log_snapshot.with_suffix('.json'), {})
            if log_snapshot.is_file() and isinstance(log_metadata, dict):
                relative = str(log_metadata.get('relative_path') or '')
                gpu_root = (Path(run.path) / 'gpu').resolve()
                full_log = (gpu_root / relative).resolve()
                full_available = (run.state.status.value in ('completed', 'error', 'stopped')
                                  and full_log.is_relative_to(gpu_root) and full_log.is_file() and full_log.suffix == '.log')
                row['log_url'] = self.file_url(full_log if full_available else log_snapshot)
                row['log_label'] = str(log_metadata.get('phase') or 'Current phase') + (' log' if full_available else ' log · latest 64 KiB')
                row['log_updated_at'] = log_metadata.get('updated_at')
            transfer_log = Path(run.path) / 'artifact-transfer.log'
            row['transfer_log_url'] = self.file_url(transfer_log) if transfer_log.is_file() else None
            row['results'] = []
            for result_file in sorted((Path(run.path) / 'gpu' / 'results').rglob('result.json')):
                result = read_json(result_file, {})
                ply = self.result_artifact(result_file, result.get('splat_path'))
                if ply is None:
                    ply = result_file.parent / 'artifixer3d.ply'
                row['results'].append({'metadata': result, 'provenance_url': self.file_url(result_file.parent / 'request.json' if (result_file.parent / 'request.json').exists() else result_file),
                                       'ply_url': self.file_url(ply) if ply.exists() else None,
                                       'plus_frames': self.plus_previews(result_file, result)})
            rows.append(row)
        rows.sort(key=lambda row: (
            0 if row['state']['status'] in ('starting', 'running', 'stopping') else
            1 if row['queue_position'] is not None else 2,
            row['queue_position'] or 0,
        ))
        return rows

    def result_artifact(self, result_file, raw):
        if not raw:
            return None
        remote = Path(str(raw))
        if remote.is_absolute():
            if result_file.parent.name not in remote.parts:
                return None
            suffix = remote.parts[remote.parts.index(result_file.parent.name) + 1:]
            local = result_file.parent.joinpath(*suffix).resolve()
        else:
            local = (result_file.parent / remote).resolve()
        if not local.is_relative_to(result_file.parent.resolve()):
            return None
        return local if local.exists() else None

    def plus_previews(self, result_file, result):
        local = self.result_artifact(result_file, result.get('plus_frames'))
        if local is None or not local.is_dir():
            return []
        return [self.file_url(p) for p in sorted(local.glob('*.png'))[:6]]

    def snapshot(self):
        from ..splatfix.g4splat.baselines import baseline_catalog
        manager = read_json(self.root / 'manager.json', {})
        from ..scene_runs.store import _pid_alive
        alive = _pid_alive(int(manager.get("pid") or 0))
        from .. import repair_lrz
        from ..splatfix.resolution import PROFILES, PROFILE_LABELS
        return {'g4splat_baselines': baseline_catalog(self.cfg.output.dir, self.file_url), 'reconstruction_methods': available_methods(), 'scenes': self.scenes(), 'checkpoints': self.checkpoints(), 'jobs': self.jobs(), 'benchmarks': self.benchmarks(),
                'manager': {'active': manager.get('status') == 'running' and alive, 'updated_at': manager.get('updated_at')},
                'compute': {'configured': repair_lrz.lrz_configured(), 'label': 'Configured LRZ allocation'},
                'capture_viewer_port': self.cfg.get('viewer', {}).get('port', 8080),
                'image_repair_modes': ['sequential', 'shared_six', 'shared_six_native'],
                'resolution_profiles': [{'id': key, 'label': PROFILE_LABELS[key], 'width': size[0], 'height': size[1]} for key, size in PROFILES.items()],
                'defaults': {'reconstruction_method': DEFAULT_METHOD, 'resolution_profile': self.cfg.get('splatfix', {}).get('resolution_profile', '720p'), 'views': 6, 'frames': 25, 'span_fraction': .04,
                             'width': self.defaults()['width'], 'height': self.defaults()['height']}}

    def start_custom(self, body):
        """Create a checkpoint and load its scene in the isolated Custom viewer."""
        import time
        import uuid
        from ..scene.catalog import SceneSpec
        from ..splatfix.resolution import profile_size
        if not isinstance(body, dict):
            raise ValueError('Request must be an object')
        options = validate_job({
            'resolution_profile': self.cfg.get('splatfix', {}).get('resolution_profile', '720p'),
            **body, 'stage': 'select'})
        scene = next((s for s in self.scenes() if s['id'] == body.get('scene_id')), None)
        if scene is None or not scene.get('path'):
            raise ValueError('Choose an available scene')
        capture_set = body.get('capture_set', 'saved')
        if capture_set not in ('saved', 'seva', '77'):
            raise ValueError('Unknown Custom capture FOV')
        if options.get('repair_policy') == 'endpoints' and capture_set == 'seva':
            raise ValueError('Endpoint checkpoints require original or 77° captures')
        width, height = (1024, 576) if capture_set == 'seva' else profile_size(options['resolution_profile'])
        ident, generation = 'splatfix-manual-' + uuid.uuid4().hex, time.time_ns() // 1000000
        spec = SceneSpec(ident, scene['label'], Path(scene['path']),
                         up_axis=scene.get('up_axis', '+y'), lod_level=scene.get('lod_level', 0))
        cp = Checkpoint.create(self.checkpoint_root, spec.path, target_views=options['views'], metadata={
            'selection': 'manual', 'capture_set': capture_set,
            'repair_policy': options.get('repair_policy', 'all'),
            'horizontal_fov': 77 if capture_set == '77' else options.get('horizontal_fov'), 'renderer': {'backend': 'viser'},
            'width': width, 'height': height, 'resolution_profile': options['resolution_profile'],
            'up_axis': spec.up_axis,
            'scene_load': {'lod_level': spec.lod_level,
                           'min_opacity': self.cfg.get('scene', {}).get('min_opacity', 0.0)},
            'manual_scene': {'id': ident, 'generation': generation}})
        return self.custom_visor.start(cp)

    def create(self, body):
        if isinstance(body, dict) and 'oracle_artifixer_inputs' in body:
            raise ValueError('Use the result reconstruction action to create an input snapshot')
        if not isinstance(body, dict):
            raise ValueError('Request must be an object')
        settings_snapshot = {'submitted': {k: v for k, v in body.items() if k != 'dashboard_snapshot'},
                             'dashboard': body.get('dashboard_snapshot')}
        if body.get('stage') == 'repair' and body.get('reconstruction_method') == 'oraclegs':
            body = {'oracle_capture_source':'seva', **body}
        options = validate_job({'resolution_profile': self.cfg.get('splatfix', {}).get('resolution_profile', '720p'), **body})
        scene_id = str(body.get('scene_id') or '')
        if options['stage'] == 'select':
            scenes = {s['id']: s for s in self.scenes()}
            if scene_id not in scenes:
                raise ValueError('Choose an available scene')
        elif options['stage'] == 'benchmark':
            source, metadata = registered_benchmark(self.benchmark_root, options['source'])
            options['source'] = str(source)
            if options.get('resume_from'):
                from ..splatfix.artifixer.resume import validate_resume
                validate_resume(self.store, options)
            if options.get('preparation_from'):
                from ..splatfix.artifixer.resume import validate_preparation
                validate_preparation(self.store, options)
            if options.get('repeat_from'):
                from ..splatfix.artifixer.resume import validate_preparation
                from ..splatfix.artifixer.repeat_benchmark import inherited_inference_command
                prior = validate_preparation(self.store, {**options, 'preparation_from': options['repeat_from']})
                if prior.state.status.value != 'completed' or prior.config.splatfix['model'] != options['model']:
                    raise ValueError('Repeat requires a completed benchmark with the same model')
                manifests = list((Path(prior.path) / 'gpu/results').glob('*/benchmark-run.json'))
                if len(manifests) != 1:
                    raise ValueError('Repeat requires one downloaded original benchmark manifest')
                original = read_json(manifests[0])
                for phase in ('inference', 'plus'):
                    inherited_inference_command(original, phase, 'new-split', 'new-output')
            scene_id = 'benchmark-' + source.parent.name
        else:
            checkpoints = {cp['id']: cp for cp in self.checkpoints()}
            checkpoint = str(Path(options['checkpoint']).resolve())
            if checkpoint not in checkpoints:
                raise ValueError('Choose an existing splatfix checkpoint')
            cp = checkpoints[checkpoint]
            if options['stage'] == 'edit' and options.get('repair_mode') == 'auto':
                options['repair_mode'] = ('shared_six_native' if len(cp['views']) == 6
                    and cp['metadata'].get('repair_policy') != 'endpoints' else 'sequential')
            if not cp['complete']:
                raise ValueError('Finish selecting all views before starting this stage')
            if (options['stage'] == 'edit' and options.get('repair_mode') in ('shared_six', 'shared_six_native')
                    and (len(cp['views']) != 6 or cp['metadata'].get('repair_policy') == 'endpoints')):
                raise ValueError('Shared Thinking repair requires exactly six views with all-view repair enabled')
            endpoint_run = options['stage'] == 'repair' and (cp['metadata'].get('repair_policy') == 'endpoints'
                            or options.get('image_trajectory') == 'endpoints')
            if endpoint_run:
                if len(cp['views']) < 2:
                    raise ValueError('Endpoint image trajectory requires a completed checkpoint with at least two views')
                if options['reconstruction_method'] != 'artifixer' or options['mode'] != 'edited':
                    raise ValueError('Endpoint checkpoints require GPT-repaired ArtiFixer runs')
                options['split_mode'] = 'no-split'
                options['trajectory_order'] = 'picked'
                options['connected_trajectory_generation'] = False
            if options['stage'] == 'edit' and options.get('image_set') == '77':
                from ..splatfix.oraclegs.captures import capture_77
                if any(not capture_77(v).get('original_rgb') or
                       (capture_77(v).get('rgb_renderer') or cp['rgb_renderer']) != 'viser'
                       for v in cp['views']):
                    raise ValueError('Complete the 77° capture set in Step 2 first')
            seva_set = options.get('oracle_capture_source') == 'seva' or options.get('image_set') == 'seva'
            if seva_set:
                from ..splatfix.oraclegs.captures import select_checkpoint
                select_checkpoint(Checkpoint.load(checkpoint), 'seva', edited=options.get('mode') == 'edited')
            if options['stage'] == 'pose_correction':
                if cp['rgb_renderer'] != 'viser' or cp['required_edited'] != cp['repair_target_views']:
                    raise ValueError('Complete original-FOV Viser image repairs in Step 3 before pose correction')
                for existing in self.store.list_runs():
                    opts = existing.config.splatfix
                    if (opts.get('checkpoint') == checkpoint and opts.get('stage') == 'pose_correction'
                            and existing.state.status.value in ('queued', 'waiting_gpu', 'starting', 'running', 'stopping')):
                        raise ValueError('Pose correction is already queued or active for this checkpoint')
            if options['stage'] == 'seva_capture':
                if options.get('view_id') is not None and options['view_id'] not in {v['id'] for v in cp['views']}:
                    raise ValueError('Unknown saved view')
                for existing in self.store.list_runs():
                    opts = existing.config.splatfix
                    if (existing.config.pipeline == 'splatfix' and opts.get('stage') == 'seva_capture'
                            and opts.get('checkpoint') == checkpoint
                            and existing.state.status.value in ('queued', 'waiting_gpu', 'starting', 'running', 'stopping')):
                        raise ValueError('A SEVA capture job is already queued or active for this checkpoint')
            separate_77 = options['stage'] == 'edit' and options.get('image_set') == '77'
            if not seva_set and not separate_77 and options['stage'] in ('edit', 'repair') and any(not v.get('original_rgb') for v in cp['views']):
                raise ValueError('This checkpoint has no original-FOV images; select SEVA captures')
            if not seva_set and not separate_77 and (options['stage'] == 'edit' or options.get('mode') == 'edited') and cp['rgb_renderer'] != 'viser':
                raise ValueError('Recapture these views in Viser before image editing or edited reconstruction')
            if options['stage'] == 'repair':
                readiness = cp['reconstruction_methods'][options['reconstruction_method']]
                if not readiness['ready']:
                    raise ValueError(readiness['reason'])
            if options['stage'] == 'repair' and options['mode'] == 'edited' and not seva_set and (any(not v.get('repaired_rgb') for v in [cp['views'][0], cp['views'][-1]])
                         if endpoint_run else cp['required_edited'] != cp['repair_target_views']):
                raise ValueError('Repair all checkpoint images before GPT-image improved reconstruction')
            if options.get('correct_generated_pose'):
                from ..splatfix.pose_correction import corrected_checkpoint
                from ..splatfix.artifixer.checkpoint_policy import checkpoint_for_run
                corrected_checkpoint(checkpoint_for_run(Checkpoint.load(checkpoint), options.get('image_trajectory', 'checkpoint')))
            options['checkpoint'] = checkpoint
            # The saved camera/RGB source determines provenance, irrespective
            # of the scene currently selected for a future view-finding run.
            source_path = Path(cp['scene_path']).resolve()
            scene_id = next((str(scene['id']) for scene in self.scenes()
                             if scene.get('path') and Path(scene['path']).resolve() == source_path), cp['scene'])
        from ..splatfix.resolution import profile_size
        width, height = profile_size(options['resolution_profile'])
        run = self.store.create_run({'pipeline': 'splatfix', 'splatfix': options,
                                     'width': width, 'height': height,
                                     'scene_id': scene_id, 'duration_seconds': 24 * 3600},
                                    settings_snapshot=settings_snapshot)
        return run.to_dict()

    def restart(self, run_id):
        """Queue the saved configuration anew, leaving the original run untouched."""
        from copy import deepcopy
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('Choose a run to restart')
        with self.results._run_storage_lock:
            prior = self.store.get_run(run_id)
            if prior.config.pipeline != 'splatfix':
                raise ValueError('Not a splatfix job')
            if prior.state.status.value not in RESTARTABLE_STATUSES:
                raise ValueError('Only failed, cancelled, GPU-expired, stopping or completed runs can be restarted')
            # Repeated clicks / retried HTTP requests share a pending restart.
            for existing in self.store.list_runs():
                if existing.state.status.value not in ('queued', 'waiting_gpu', 'starting', 'running', 'stopping'):
                    continue
                snapshot = read_json(Path(existing.path) / 'settings.json', {})
                if (snapshot.get('submitted') or {}).get('restart_from') == run_id:
                    return existing.to_dict()
            config = deepcopy(prior.config.to_dict())
            options = config['splatfix']
            options.pop('scheduled_at', None)
            if options['stage'] == 'select':
                if config['scene_id'] not in {scene['id'] for scene in self.scenes()}:
                    raise ValueError('The original scene is no longer available')
            elif options['stage'] == 'benchmark':
                registered_benchmark(self.benchmark_root, options['source'])
            else:
                snapshot_path = options.get('oracle_artifixer_inputs')
                raw = snapshot_path or options.get('checkpoint')
                path = Path(raw).resolve() if raw else None
                if not path or not path.is_relative_to(Path(self.cfg.output.dir).resolve()):
                    raise ValueError('Saved restart inputs are unavailable')
                if snapshot_path:
                    if not (path / 'inputs.json').is_file():
                        raise ValueError('Saved OracleGS reconstruction inputs have been removed')
                else:
                    if not (path / 'checkpoint.json').is_file():
                        raise ValueError('Saved checkpoint has been removed; restore it before restarting')
                    Checkpoint.load(path)  # Validate saved images and cameras too.
            saved_settings = read_json(Path(prior.path) / 'settings.json', {})
            run = self.store.create_run(config, settings_snapshot={
                'submitted': {'restart_from': run_id, 'original_submission': saved_settings.get('submitted')},
                'dashboard': saved_settings.get('dashboard')})
            return run.to_dict()

    def cancel(self, run_id):
        run = self.store.get_run(run_id)
        if run.config.pipeline != 'splatfix':
            raise ValueError('Not a splatfix job')
        if run.state.status.value not in ('queued', 'waiting_gpu', 'starting', 'running', 'stopping'):
            return run.to_dict()
        self.store.request_stop(run_id)
        if run.state.status.value in ('queued', 'waiting_gpu'):
            if run.state.details.get('remote_dir') and not run.state.details.get('remote_finished'):
                self.store.update_status(run_id, status='queued', message='Reattaching to cancel the existing GPU job')
            else:
                self.store.update_status(run_id, status='stopped', message='Cancelled while queued')
        else:
            self.store.update_status(run_id, status='stopping', message='Cancellation requested')
        return self.store.get_run(run_id).to_dict()
