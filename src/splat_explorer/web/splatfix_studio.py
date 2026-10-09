"""Dedicated dashboard adapter over the shared scene-run store and manager."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import quote
from ..splatfix.checkpoint import Checkpoint, camera_from_record, atomic_json
from ..splatfix.jobs import read_json, validate_job, registered_benchmark
from .scene_run_studio import SceneRunStudio
from ..splatfix.methods import DEFAULT_METHOD, available_methods, get_method


class SplatfixStudio(SceneRunStudio):
    def __init__(self, app, store=None):
        super().__init__(app, store, pipeline='splatfix')
        self.checkpoint_root = Path(self.cfg.output.dir).resolve() / 'splatfix'
        from .splatfix_results import ResultCatalog
        self.results = ResultCatalog(self)
        self.benchmark_root = Path(self.cfg.output.dir).resolve() / 'benchmarks'

    def defaults(self):
        from ..splatfix.resolution import profile_size
        defaults = super().defaults()
        defaults['width'], defaults['height'] = profile_size(
            self.cfg.get('splatfix', {}).get('resolution_profile', 'training'))
        return defaults

    def checkpoints(self):
        manifests = list(self.checkpoint_root.glob('*/checkpoint.json'))
        manifests += list(self.root.glob('run_*/checkpoints/*/checkpoint.json'))
        result = []
        for path in sorted(manifests, reverse=True):
            try:
                cp = Checkpoint.load(path)
                views = [{**v, 'original_url': self.file_url(cp.image_path(v)),
                          'repaired_url': self.file_url(cp.image_path(v, True)) if v.get('repaired_rgb') else None}
                         for v in cp.views]
                distinct = len({tuple(camera_from_record(view).c2w.ravel()) for view in cp.views})
                method_readiness = {method['id']: get_method(method['id']).readiness(cp)
                                    for method in available_methods()}
                reconstruction_ready = method_readiness[DEFAULT_METHOD]['ready']
                reconstruction_reason = method_readiness[DEFAULT_METHOD]['reason']
                result.append({'id': str(cp.root), 'name': self.checkpoint_name(cp.root), 'storage_name': cp.root.name, 'scene': Path(cp.manifest['scene_path']).name,
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
            if run.config.pipeline != 'splatfix' or run.state.status.value in ('completed', 'error', 'stopped'):
                continue
            refs = [run.config.splatfix.get('checkpoint'), run.state.details.get('checkpoint')]
            if (any(ref and Path(ref).resolve() == root for ref in refs)
                    or root.is_relative_to(Path(run.path).resolve())):
                raise ValueError('This checkpoint is used by an active or queued run. Cancel or finish that run before deleting it.')
        repaired = sum(bool(view.get('repaired_rgb')) for view in cp.views)
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
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.ply', '.json', '.log', '.jsonl'):
            raise ValueError('Unsupported artifact type')
        return path

    def file_url(self, path):
        return '/api/splatfix/file?path=' + quote(str(Path(path).resolve()), safe='')

    def jobs(self):
        rows = []
        for run in self.store.list_runs():
            if run.config.pipeline != 'splatfix':
                continue
            row = run.to_dict()
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
        manager = read_json(self.root / 'manager.json', {})
        from ..scene_runs.store import _pid_alive
        alive = _pid_alive(int(manager.get("pid") or 0))
        from .. import repair_lrz
        from ..splatfix.resolution import PROFILES, PROFILE_LABELS
        return {'reconstruction_methods': available_methods(), 'scenes': self.scenes(), 'checkpoints': self.checkpoints(), 'jobs': self.jobs(), 'benchmarks': self.benchmarks(),
                'manager': {'active': manager.get('status') == 'running' and alive, 'updated_at': manager.get('updated_at')},
                'compute': {'configured': repair_lrz.lrz_configured(), 'label': 'Configured LRZ allocation'},
                'capture_viewer_port': self.cfg.get('viewer', {}).get('port', 8080),
                'resolution_profiles': [{'id': key, 'label': PROFILE_LABELS[key], 'width': size[0], 'height': size[1]} for key, size in PROFILES.items()],
                'defaults': {'reconstruction_method': DEFAULT_METHOD, 'resolution_profile': self.cfg.get('splatfix', {}).get('resolution_profile', 'training'), 'views': 6, 'frames': 25, 'span_fraction': .04,
                             'width': self.defaults()['width'], 'height': self.defaults()['height']}}

    def start_custom(self, body):
        """Create a normal checkpoint and install controls in the shared Viser."""
        import json
        import time
        import uuid
        from ..rendering.viser_renderer import ViserCaptureRenderer, ViserCaptureError
        from ..scene.catalog import SceneSpec, publish_live_scene, portable_scene_path
        from ..splatfix.resolution import profile_size
        if not isinstance(body, dict):
            raise ValueError('Request must be an object')
        options = validate_job({
            'resolution_profile': self.cfg.get('splatfix', {}).get('resolution_profile', 'training'),
            **body, 'stage': 'select'})
        scene = next((s for s in self.scenes() if s['id'] == body.get('scene_id')), None)
        if scene is None or not scene.get('path'):
            raise ValueError('Choose an available scene')
        width, height = profile_size(options['resolution_profile'])
        ident, generation = 'splatfix-manual-' + uuid.uuid4().hex, time.time_ns() // 1000000
        spec = SceneSpec(ident, scene['label'], Path(scene['path']),
                         up_axis=scene.get('up_axis', '+y'), lod_level=scene.get('lod_level', 0))
        cp = Checkpoint.create(self.checkpoint_root, spec.path, target_views=options['views'], metadata={
            'selection': 'manual', 'renderer': {'backend': 'viser'},
            'width': width, 'height': height, 'resolution_profile': options['resolution_profile'],
            'up_axis': spec.up_axis,
            'scene_load': {'lod_level': spec.lod_level,
                           'min_opacity': self.cfg.get('scene', {}).get('min_opacity', 0.0)},
            'manual_scene': {'id': ident, 'generation': generation}})
        renderer = ViserCaptureRenderer(None, url='http://localhost:' + str(
            self.cfg.get('viewer', {}).get('render_port', 8081)))
        try:
            renderer._request('POST', '/manual-selection',
                              body=json.dumps({'checkpoint': portable_scene_path(cp.root)}).encode(),
                              content_type='application/json', timeout=10)
        except ViserCaptureError as exc:
            # Keep the empty checkpoint: a timeout may have installed the controls.
            raise ValueError('Could not start Custom selection. Ensure the updated Viser is running. ' + str(exc)) from exc
        publish_live_scene(spec, generation, reload=True, catalog_id=scene['id'])
        return {'ok': True, 'checkpoint': str(cp.root)}

    def create(self, body):
        if not isinstance(body, dict):
            raise ValueError('Request must be an object')
        options = validate_job({'resolution_profile': self.cfg.get('splatfix', {}).get('resolution_profile', 'training'), **body})
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
            if not cp['complete']:
                raise ValueError('Finish selecting all views before starting this stage')
            if (options['stage'] == 'edit' or options.get('mode') == 'edited') and cp['rgb_renderer'] != 'viser':
                raise ValueError('Recapture these views in Viser before image editing or edited reconstruction')
            if options['stage'] == 'repair':
                readiness = cp['reconstruction_methods'][options['reconstruction_method']]
                if not readiness['ready']:
                    raise ValueError(readiness['reason'])
            if options['stage'] == 'repair' and options['mode'] == 'edited' and cp['edited'] != cp['target_views']:
                raise ValueError('Repair all checkpoint images before GPT-image improved reconstruction')
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
                                     'scene_id': scene_id, 'duration_seconds': 24 * 3600})
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
