"""Dedicated dashboard adapter over the shared scene-run store and manager."""
from __future__ import annotations
from pathlib import Path
from urllib.parse import quote
from ..splatfix.checkpoint import Checkpoint
from ..splatfix.jobs import read_json, validate_job, registered_benchmark
from .scene_run_studio import SceneRunStudio


class SplatfixStudio(SceneRunStudio):
    def __init__(self, app, store=None):
        super().__init__(app, store, pipeline='splatfix')
        self.checkpoint_root = Path(self.cfg.output.dir).resolve() / 'splatfix'
        self.benchmark_root = Path(self.cfg.output.dir).resolve() / 'benchmarks'

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
                result.append({'id': str(cp.root), 'name': cp.root.name, 'scene': Path(cp.manifest['scene_path']).name,
                               'scene_path': cp.manifest['scene_path'],
                               'created_at': cp.manifest.get('created_at'), 'views': views, 'target_views': cp.target_views,
                               'complete': cp.complete, 'edited': sum(bool(v.get('repaired_rgb')) for v in views),
                               'trajectory_caches': len(list((cp.root / 'trajectories').glob('*/trajectory.json'))),
                               'metadata': cp.manifest.get('metadata', {})})
            except (OSError, ValueError, KeyError):
                continue
        return result

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
            from ..splatfix.resume import resume_available, preparation_available
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
        return {'scenes': self.scenes(), 'checkpoints': self.checkpoints(), 'jobs': self.jobs(), 'benchmarks': self.benchmarks(),
                'manager': {'active': manager.get('status') == 'running' and alive, 'updated_at': manager.get('updated_at')},
                'compute': {'configured': repair_lrz.lrz_configured(), 'label': 'Configured LRZ allocation'},
                'defaults': {'views': 6, 'frames': 25, 'span_fraction': .04}}

    def create(self, body):
        if not isinstance(body, dict):
            raise ValueError('Request must be an object')
        options = validate_job(body)
        scene_id = str(body.get('scene_id') or '')
        if options['stage'] == 'select':
            scenes = {s['id']: s for s in self.scenes()}
            if scene_id not in scenes:
                raise ValueError('Choose an available scene')
        elif options['stage'] == 'benchmark':
            source, metadata = registered_benchmark(self.benchmark_root, options['source'])
            options['source'] = str(source)
            if options.get('resume_from'):
                from ..splatfix.resume import validate_resume
                validate_resume(self.store, options)
            if options.get('preparation_from'):
                from ..splatfix.resume import validate_preparation
                validate_preparation(self.store, options)
            scene_id = 'benchmark-' + source.parent.name
        else:
            checkpoints = {cp['id']: cp for cp in self.checkpoints()}
            checkpoint = str(Path(options['checkpoint']).resolve())
            if checkpoint not in checkpoints:
                raise ValueError('Choose an existing splatfix checkpoint')
            cp = checkpoints[checkpoint]
            if not cp['complete']:
                raise ValueError('Finish selecting all views before starting this stage')
            if options['stage'] == 'repair' and options['mode'] == 'edited' and cp['edited'] != cp['target_views']:
                raise ValueError('Repair all checkpoint images before GPT-image improved reconstruction')
            options['checkpoint'] = checkpoint
            # The saved camera/RGB source determines provenance, irrespective
            # of the scene currently selected for a future view-finding run.
            source_path = Path(cp['scene_path']).resolve()
            scene_id = next((str(scene['id']) for scene in self.scenes()
                             if scene.get('path') and Path(scene['path']).resolve() == source_path), cp['scene'])
        run = self.store.create_run({'pipeline': 'splatfix', 'splatfix': options,
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
