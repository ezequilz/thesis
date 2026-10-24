"""Catalog of reconstruction jobs, live inputs and completed outputs."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import shutil
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from ..splatfix.jobs import read_json
from ..scene_runs.store import _atomic_write_json
from .scene_run_viser import SceneRunViser


class ResultCatalog:
    def __init__(self, studio):
        self.studio = studio
        self.cfg = studio.cfg
        self._visor = None
        self._visibility_lock = threading.Lock()
        self._run_storage_lock = threading.Lock()
        self._catalog_lock = threading.Lock()
        self._catalog_rows = None
        self._catalog_time = 0.0
        self._catalog_revision = 0
        self._catalog_refreshing = False
        self._catalog_error = False

    def invalidate_catalog(self):
        """Discard snapshots after a mutation, including any in-flight old result."""
        with self._catalog_lock:
            self._catalog_revision += 1
            self._catalog_rows = None
            self._catalog_time = 0.0

    def catalog_snapshot(self, *, refresh=False):
        """Serve cached cards immediately; one worker updates expensive file sizes.

        A cold request reads metadata and existing size snapshots only. Completed
        snapshots live for 15 seconds; stale cards remain usable during refresh.
        """
        with self._catalog_lock:
            if self._catalog_rows is None:
                self._catalog_rows = self.entries(cached_sizes_only=True)
                self._catalog_time = 0.0
            stale = refresh or not self._catalog_time or time.monotonic() - self._catalog_time >= 15
            if stale and not self._catalog_refreshing:
                self._catalog_refreshing = True
                self._catalog_error = False
                threading.Thread(target=self._refresh_catalog,
                                 args=(self._catalog_revision,), daemon=True,
                                 name='results-catalog').start()
            return {'results': deepcopy(self._catalog_rows),
                    'refreshing': self._catalog_refreshing,
                    'refresh_error': self._catalog_error}

    def _refresh_catalog(self, revision):
        try:
            rows = self.entries()
        except Exception:
            logging.getLogger(__name__).exception('Could not refresh results catalog')
            # Preserve the last successful cards and retry on the next poll.
            with self._catalog_lock:
                self._catalog_refreshing = False
                self._catalog_error = True
            return
        with self._catalog_lock:
            if revision == self._catalog_revision:
                self._catalog_rows = rows
                self._catalog_time = time.monotonic()
            self._catalog_refreshing = False

    def reconstruct_artifixer(self, key, initialization='repaired_point_cloud', *, oracle_attention_batching='within_trajectory',
                              oracle_artifixer_input='new_only', oracle_artifixer_reference_assignment='trajectory'):
        """Snapshot inputs, then enqueue through the normal manager/GPU lease path."""
        import tempfile
        from ..splatfix.oraclegs.artifixer_inputs import export_inputs
        from ..splatfix.oraclegs.backend import validate_options
        with self._run_storage_lock:
            row = self.detail(key)
            if row is None:
                raise ValueError('Result not found')
            snapshots = Path(self.cfg.output.dir) / 'oraclegs-reuse'
            snapshots.mkdir(exist_ok=True, parents=True)
            destination = Path(tempfile.mkdtemp(prefix='artifixer-', dir=snapshots))
            try:
                manifest = export_inputs(self.studio, row, destination, initialization,
                                         oracle_attention_batching=oracle_attention_batching,
                                         oracle_artifixer_input=oracle_artifixer_input,
                                         oracle_artifixer_reference_assignment=oracle_artifixer_reference_assignment)
                options = {'stage':'repair', 'reconstruction_method':'oraclegs', 'mode':'edited',
                           'checkpoint':row['config'].get('splatfix', {}).get('checkpoint', ''),
                           'resolution_profile':'training', 'oracle_artifixer_inputs':str(destination.resolve()),
                           **validate_options(manifest)}
                run = self.studio.store.create_run({'pipeline':'splatfix', 'splatfix':options,
                    'scene_id':row['scene'], 'width':manifest['resolution'][0],
                    'height':manifest['resolution'][1], 'duration_seconds':24*3600})
            except Exception:
                shutil.rmtree(destination)
                raise
            return run.to_dict()

    def repeat_artifixer(self, key):
        import tempfile
        from copy import deepcopy
        from ..splatfix.artifixer.repeat_repair import snapshot_repeat
        with self._run_storage_lock:
            row = self.detail(key)
            if row is None:
                raise ValueError('Result not found')
            self.studio.checkpoint_root.mkdir(parents=True, exist_ok=True)
            destination = Path(tempfile.mkdtemp(prefix='repeat-', dir=self.studio.checkpoint_root))
            try:
                repeat = snapshot_repeat(self.studio, row, destination)
                config = deepcopy(row['config'])
                options = config['splatfix']
                options.update(stage='repair', reconstruction_method='artifixer', checkpoint=str(destination.resolve()))
                options.pop('scheduled_at', None)
                parent_settings = read_json(self.studio.root / row['run_id'] / 'settings.json', {})
                run = self.studio.store.create_run(config, settings_snapshot={
                    'submitted': {'repeat_repair': repeat, 'parent_settings': parent_settings},
                    'dashboard': parent_settings.get('dashboard')})
            except Exception:
                shutil.rmtree(destination)
                raise
            return run.to_dict()

    def _comments_path(self, run_id):
        run = self.studio.store.get_run(run_id)
        path = Path(run.path) / 'comments.md'
        if path.is_symlink():
            raise ValueError('Unsafe comments path')
        return path

    def read_comments(self, run_id):
        with self._run_storage_lock:
            path = self._comments_path(run_id)
            try:
                return path.read_text(encoding='utf-8')
            except FileNotFoundError:
                path.touch(exist_ok=False)
                return ''

    def save_comments(self, run_id, comments):
        if not isinstance(comments, str):
            raise ValueError('comments must be a string')
        with self._run_storage_lock:
            path = self._comments_path(run_id)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                                 prefix='.comments-', delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(comments)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return {'run_id': run_id}

    COLORS = ('', 'red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink')

    def read_color(self, run_id):
        path = self.studio.store.run_path(run_id) / 'color-label.json'
        if path.is_symlink():
            return ''
        saved = read_json(path, {})
        color = saved.get('color', '') if isinstance(saved, dict) else ''
        return color if color in self.COLORS else ''

    def save_color(self, run_id, color):
        if not isinstance(color, str) or color not in self.COLORS:
            raise ValueError('Unknown run color')
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            path = Path(run.path) / 'color-label.json'
            if path.is_symlink():
                raise ValueError('Unsafe color label path')
            _atomic_write_json(path, {'color': color})
        return {'run_id': run_id, 'color': color}

    def read_name(self, run_id):
        path = self.studio.store.run_path(run_id) / 'run-name.json'
        if path.is_symlink():
            return ''
        saved = read_json(path, {})
        name = saved.get('name', '') if isinstance(saved, dict) else ''
        return name if isinstance(name, str) else ''

    def save_name(self, run_id, name):
        if not isinstance(name, str) or len(name.strip()) > 200:
            raise ValueError('Run name must be text with at most 200 characters')
        name = name.strip()
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            path = Path(run.path) / 'run-name.json'
            if path.is_symlink():
                raise ValueError('Unsafe run name path')
            _atomic_write_json(path, {'name': name})
        return {'run_id': run_id, 'custom_name': name}

    def delete_run(self, run_id):
        """Permanently remove only the validated inactive run directory."""
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            if run.state.status.value not in ('completed', 'error', 'stopped', 'gpu_expired'):
                raise ValueError('Only inactive runs can be deleted')
            shutil.rmtree(run.path)
            return {'run_id': run.run_id}

    def delete_hidden_runs(self, run_ids):
        """Delete the confirmed batch, rechecking visibility and lifecycle."""
        if not isinstance(run_ids, list) or any(not isinstance(run_id, str) for run_id in run_ids):
            raise ValueError('run_ids must be a list of run IDs')
        deleted, skipped, failed = [], [], []
        with self._visibility_lock:
            visibility = self.visibility_records()
            for run_id in dict.fromkeys(run_ids):
                if not visibility.get(run_id, {}).get('hidden', False):
                    skipped.append(run_id)
                    continue
                try:
                    self.delete_run(run_id)
                except (ValueError, FileNotFoundError):
                    skipped.append(run_id)
                except OSError:
                    failed.append(run_id)
                else:
                    deleted.append(run_id)
        return {'deleted': deleted, 'skipped': skipped, 'failed': failed}

    def archive_run(self, run_id):
        """Move a whole inactive run into local history for manual backup."""
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            if run.state.status.value not in ('completed', 'error', 'stopped', 'gpu_expired'):
                raise ValueError('Only inactive runs can be moved to history')
            history = Path(self.cfg.output.dir) / 'run_history'
            history.mkdir(parents=True, exist_ok=True)
            destination = history / run.run_id
            if destination.exists() or destination.is_symlink():
                raise ValueError('A run with this ID already exists in run_history')
            # Same-filesystem rename keeps the entire folder intact and avoids
            # partial copies. Failure leaves the original run in place.
            Path(run.path).rename(destination)
            return {'run_id': run.run_id, 'path': str(destination.resolve())}

    def reduce_run(self, run_id, *, backup=False, preserve_reconstruction=False):
        if type(backup) is not bool:
            raise ValueError('backup must be a boolean')
        if type(preserve_reconstruction) is not bool:
            raise ValueError('preserve_reconstruction must be a boolean')
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            if run.state.status.value not in ('completed', 'error', 'stopped', 'gpu_expired'):
                raise ValueError('Only inactive runs can be reduced')
            from .run_reduction import reduce_oraclegs, reduce_artifixer
            before = self.studio.store.output_size_bytes(run.run_id, refresh=True)
            history = Path(self.cfg.output.dir) / 'run_history' if backup else None
            rows = [row for row in self.entries(private=True) if row['run_id'] == run_id]
            if any(row['kind'] == 'OracleGS' for row in rows):
                if preserve_reconstruction:
                    raise ValueError('Keeping reconstruction inputs supports ArtiFixer runs only')
                report = reduce_oraclegs(Path(run.path), history, before_size_bytes=before)
            elif rows and all(row['kind'] == 'ArtiFixer3D' for row in rows):
                plans = self._artifixer_plans(rows, preserve_reconstruction=preserve_reconstruction)
                report = reduce_artifixer(Path(run.path), plans, history, before_size_bytes=before,
                                         preserve_reconstruction=preserve_reconstruction)
            else:
                raise ValueError('Reduction supports OracleGS and ArtiFixer runs only')
            after = self.studio.store.output_size_bytes(run.run_id, refresh=True)
            return {'run_id': run.run_id, **report, 'output_size_bytes': after,
                    'reduction_saved_bytes': max(0, before - after) if before is not None and after is not None else None}

    def _artifixer_plans(self, rows, *, preserve_reconstruction=False):
        plans = []
        for row in rows:
            manifest = row['_manifest']
            if manifest is None:
                continue
            result = read_json(manifest, {})
            copies = [(path, manifest.parent / 'original.ply' if name == 'original' else path)
                      for name, path in row['_paths'].items() if path is not None]
            native = self.studio.result_artifact(manifest, result.get('native_splat_path'))
            if native and native.is_file():
                copies.append((native, native))
            preview = read_json(manifest.parent / 'inference-preview.json', {})
            predictions = [self.studio.result_artifact(manifest, result.get(key) or preview.get(key))
                           for key in ('prediction_frames', 'reverse_inference_frames', 'plus_frames')]
            plan = {'root': manifest.parent, 'ply_copies': copies,
                    'prediction_dirs': [p for p in predictions if p and p.is_dir()],
                    'plus_frames_dir': predictions[-1]}
            if preserve_reconstruction:
                from ..splatfix.oraclegs.artifixer_inputs import source_assets
                trajectory, references, _ = source_assets(self.studio, row)
                plan['trajectory'] = trajectory
                plan['reconstruction_files'] = references
                if trajectory.get('handoff_export'):
                    # Preserve the original generation segments for validating the
                    # completed handoff metadata again after storage reduction.
                    request = read_json(manifest.parent / 'request.json', {})
                    original = (self.studio.result_artifact(manifest, 'preview-trajectory.json')
                                or self.studio.result_artifact(manifest, request.get('trajectory')))
                    plan['trajectory'] = read_json(original, {})
                    preview = read_json(manifest.parent / 'input-preview.json', {})
                    for item in preview.get('reference_images', []):
                        plan['reconstruction_files'].append(self.studio.result_artifact(manifest, item['path']))
                    for name in ('inference-references.json', 'inference-seeding.json', 'inference-backward.json'):
                        plan['reconstruction_files'].append(manifest.parent / name)
                # Preserve any additional saved input/output camera exports too.
                for pattern in ('*trajectory*.json', '*transforms*.json', '*poses*.json', '*cameras*.json'):
                    plan['reconstruction_files'].extend(manifest.parent.rglob(pattern))
                for name in ('inference-directions.json', 'plus-directions.json',
                             'inference-insertion-audit.json', 'plus-insertion-audit.json'):
                    path = manifest.parent / name
                    if path.is_file():
                        plan['reconstruction_files'].append(path)
            plans.append(plan)
        return plans

    def save_inference_videos(self, run_id):
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            if run.state.status.value not in ('completed', 'error', 'stopped', 'gpu_expired'):
                raise ValueError('Wait for the run to finish before saving inference videos')
            rows = [row for row in self.entries(private=True) if row['run_id'] == run_id]
            if not rows or any(row['kind'] != 'ArtiFixer3D' for row in rows):
                raise ValueError('Inference video export supports ArtiFixer runs only')
            from .run_reduction import reduce_artifixer
            try:
                report = reduce_artifixer(Path(run.path), self._artifixer_plans(rows), export_only=True)
            finally:
                size = self.studio.store.output_size_bytes(run_id, refresh=True)
            return {'run_id': run_id, **report, 'output_size_bytes': size}

    def visibility_records(self):
        path = self.studio.root / 'result-visibility.json'
        try:
            records = json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(records, dict) or any(
                not isinstance(record, dict) or type(record.get('hidden')) is not bool
                for record in records.values()):
            raise ValueError('Invalid result visibility records')
        return records

    def set_hidden(self, run_id, hidden):
        if not isinstance(run_id, str) or type(hidden) is not bool:
            raise ValueError('run_id and a boolean hidden value are required')
        if (not any(run.run_id == run_id for run in self.studio.store.list_runs())
                and not any(row['run_id'] == run_id for row in self.entries())):
            raise ValueError('Run not found')
        with self._visibility_lock:
            records = self.visibility_records()
            records[run_id] = {'hidden': hidden, 'updated_at': datetime.now(timezone.utc).isoformat()}
            _atomic_write_json(self.studio.root / 'result-visibility.json', records)
        return {'run_id': run_id, **records[run_id]}

    @property
    def visor(self):
        if self._visor is None:
            self._visor = ResultViser(self)
        return self._visor

    def entries(self, *, private=False, review_key=None, detail_key=None, cached_sizes_only=False):
        rows = []
        for run in self.studio.store.list_runs():
            root = Path(run.path)
            candidates = sorted((root / 'gpu/results').glob('*/result.json'))
            candidates += sorted(p for p in (root / 'gpu/results').glob('*/partial-result.json')
                                 if not (p.parent / 'result.json').is_file())
            candidates += sorted(p for p in (root / 'gpu/results').glob('*/inference-preview.json')
                                 if not (p.parent / 'result.json').is_file()
                                 and not (p.parent / 'partial-result.json').is_file())
            candidates += sorted(p for p in (root / 'gpu/results').glob('*/input-preview.json')
                                 if not any((p.parent / name).is_file() for name in
                                            ('result.json', 'partial-result.json', 'inference-preview.json')))
            if detail_key and detail_key != run.run_id:
                if not any(p.resolve().is_relative_to(self.studio.root.resolve()) and any(
                    hashlib.sha256(str(p.with_name(name).resolve().relative_to(self.studio.root.resolve())).encode()).hexdigest()[:20] == detail_key
                    for name in ('result.json', 'partial-result.json')) for p in candidates):
                    continue
            if review_key and review_key != run.run_id:
                candidates = [p for p in candidates
                    if p.resolve().is_relative_to(self.studio.root.resolve()) and any(
                    hashlib.sha256(str(p.with_name(name).resolve().relative_to(self.studio.root.resolve())).encode()).hexdigest()[:20] == review_key
                    for name in ('result.json', 'partial-result.json'))]
                if not candidates:
                    continue
            record = run.to_dict()
            record['custom_name'] = self.read_name(run.run_id)
            record['color'] = self.read_color(run.run_id)
            record['output_size_bytes'] = None if review_key or detail_key else self.studio.store.output_size_bytes(run.run_id, cached_only=cached_sizes_only)
            comments_path = root / 'comments.md'
            try:
                record['has_notes'] = not comments_path.is_symlink() and bool(comments_path.read_text(encoding='utf-8').strip())
            except (OSError, UnicodeError):
                record['has_notes'] = False
            config = record.get('config', {})
            first_row = len(rows)
            for manifest in candidates:
                if not manifest.resolve().is_relative_to(self.studio.root.resolve()):
                    continue
                result = read_json(manifest, {})
                if not isinstance(result, dict):
                    continue
                repaired = self.studio.result_artifact(manifest, result.get('splat_path'))
                if repaired is None:
                    repaired = manifest.parent / 'artifixer3d.ply'
                if not repaired.resolve().is_relative_to(manifest.parent.resolve()) or not repaired.is_file():
                    if config.get('splatfix', {}).get('stage') not in ('repair', 'benchmark') and not ((manifest.name == 'partial-result.json' and result.get('incomplete') is True)
                            or (manifest.name == 'inference-preview.json' and result.get('inference_ready') is True)
                            or (manifest.name == 'input-preview.json' and result.get('inputs_ready') is True)):
                        continue
                    repaired = None
                original = manifest.parent / 'original.ply'
                options = config.get('splatfix', {})
                if not original.resolve().is_relative_to(manifest.parent.resolve()) or not original.is_file():
                    original = None
                    cp = options.get('checkpoint')
                    if cp:
                        cp_path = Path(cp)
                        if cp_path.resolve().is_relative_to(Path(self.cfg.output.dir).resolve()):
                            saved = read_json(cp_path / 'checkpoint.json', {})
                            source = Path(saved.get('scene_path', ''))
                            # Only registered source scenes can be opened, never arbitrary JSON paths.
                            allowed = {str(Path(s['path']).resolve()) for s in self.studio.scenes() if s.get('path')}
                            if str(source.resolve()) in allowed and source.exists():
                                original = source
                key = hashlib.sha256(str(manifest.with_name('result.json').resolve().relative_to(self.studio.root.resolve())).encode()).hexdigest()[:20]
                if review_key:
                    rows.append({'id': key, 'run_id': run.run_id,
                                 'scene': config.get('scene_id', ''), 'config': config,
                                 'kind': 'reconstruction', '_manifest': manifest,
                                 '_paths': {'original': original, 'repaired': repaired}})
                    continue
                metrics_file = manifest.parent / 'published-test-metrics.json'
                metrics = read_json(metrics_file, {})
                if not isinstance(metrics, dict):
                    metrics = {}
                # Older evaluations may live outside the run. Match their recorded root,
                # never attach an evaluation merely because the scene name matches.
                if not metrics:
                    for candidate in self.studio.benchmark_root.glob('*/published-test-metrics*.json'):
                        data = read_json(candidate, {})
                        provenance_root = Path(str(data.get('benchmark_provenance', {}).get('root', ''))) if isinstance(data, dict) else Path('.')
                        if provenance_root.name == manifest.parent.name and record['run_id'] in provenance_root.parts:
                            metrics, metrics_file = data, candidate
                            break
                rows.append(self._row(key, record, original, repaired, result, metrics, metrics_file, manifest))
            if review_key and len(rows) > first_row:
                return rows
            if len(rows) == first_row and config.get('pipeline') == 'splatfix' and config.get('splatfix', {}).get('stage') in ('benchmark', 'repair'):
                rows.append(self._row(record['run_id'], record, None, None, {}, {}, None, None))
            job_rows = rows[first_row:]
            if config.get('splatfix', {}).get('stage') == 'benchmark':
                job_rows.sort(key=lambda row: (row['repair_pass'], row['result_name']))
                rows[first_row:] = job_rows
                total = max(config.get('splatfix', {}).get('benchmark_runs', 1), len(job_rows))
                chain = [{'id': item['id'], 'number': index, 'detail_url': item['detail_url'],
                          'status': item['status']} for index, item in enumerate(job_rows, 1)]
                for index, item in enumerate(job_rows, 1):
                    item.update(chain_runs=chain, chain_position=index, chain_total=total)
            original, repaired = root / 'scene_original.ply', root / 'scene_repaired.ply'
            if any(p.is_file() and not p.resolve().is_relative_to(self.studio.root.resolve()) for p in (original, repaired)):
                continue
            if original.is_file() or repaired.is_file():
                rows.append(self._row(record['run_id'], record, original if original.is_file() else None,
                    repaired if repaired.is_file() else None, {}, {}, None, None))
        from .worldexplorer_details import entries as worldexplorer_entries
        if not detail_key or detail_key.startswith('we_'):
            rows.extend(worldexplorer_entries(self, key=detail_key))
        visibility = self.visibility_records()
        for row in rows:
            row['hidden'] = visibility.get(row['run_id'], {}).get('hidden', False)
        if not private:
            for row in rows:
                row.pop('_paths', None)
                row.pop('_manifest', None)
        return rows

    def _row(self, key, record, original, repaired, result, metrics, metrics_file, manifest):
        config = record.get('config', {})
        options = config.get('splatfix', {})
        reduction = read_json(self.studio.root / record['run_id'] / 'reduction.json', {})
        before = reduction.get('before_size_bytes')
        backup_path = reduction.get('backup_path')
        backup_path = backup_path if isinstance(backup_path, str) and backup_path.strip() else None
        after = record.get('output_size_bytes')
        saved_bytes = max(0, before - after) if type(before) is int and type(after) is int else None
        preview = []
        preview_label = '3D+ image preview'
        if manifest:
            preview = self.studio.plus_previews(manifest, result)
            if not preview:
                gallery, _ = self._reconstruction_gallery(manifest, result, limit=1)
                preview = [frame['url'] for frame in gallery['frames'][:1]]
                preview_label = 'img 1 preview'
            if not preview:
                summary = self.studio.result_artifact(manifest, 'stage-comparison.jpg')
                if summary and summary.is_file():
                    preview = [self.studio.file_url(summary)]
                    preview_label = 'Stage comparison'
        benchmark = read_json(manifest.parent / 'benchmark-run.json', {}) if manifest else {}
        repair_pass = (result.get('repeat_repair') or benchmark.get('repeat_repair') or {}).get('repair_pass', 1)
        status = record.get('state', {}).get('status')
        if manifest and manifest.name == 'result.json':
            status = 'completed'
        return {'repair_pass': repair_pass, 'job_status': record.get('state', {}).get('status'), 'id': key, 'run_id': record['run_id'], 'scene': config.get('scene_id', ''),
                'can_stop': not record.get('state', {}).get('details', {}).get('remote_finished') and config.get('pipeline') == 'splatfix' and record.get('state', {}).get('status') in ('queued', 'waiting_gpu', 'starting', 'running'),
                'message': record.get('state', {}).get('message'),
                'custom_name': record.get('custom_name', ''),
                'color': record.get('color', ''),
                'has_notes': record.get('has_notes', False),
                'incomplete': result.get('incomplete', False),
                'inference_ready': result.get('inference_ready', False),
                'phase': record.get('state', {}).get('details', {}).get('phase'),
                'interruption': result.get('interruption'),
                'diagnostics_url': self.studio.file_url(manifest.parent / 'benchmark-run.json') if manifest and (manifest.parent / 'benchmark-run.json').is_file() else None,
                'created_at': record.get('state', {}).get('created_at'), 'status': status,
                'output_size_bytes': record.get('output_size_bytes'),
                'reduction_saved_bytes': saved_bytes,
                'backup_path': backup_path,
                'backed_up': backup_path is not None,
                'reduced': (self.studio.root / record['run_id'] / 'reduction.json').is_file(),
                'can_reduce': (options.get('reconstruction_method') in ('oraclegs', 'artifixer') or result.get('reconstruction_method') in ('oraclegs', 'artifixer') or (manifest is not None and not options.get('reconstruction_method') and not result.get('reconstruction_method'))) and not (self.studio.root / record['run_id'] / 'reduction.json').exists(),
                'can_archive': record.get('state', {}).get('status') in ('completed', 'error', 'stopped', 'gpu_expired'),
                'mode': options.get('mode') or config.get('repair_type', ''),
                'model': result.get('model_variant') or options.get('model'),
                'trajectory': result.get('trajectory_mode'), 'frames': result.get('frame_count'),
                'kind': 'OracleGS' if options.get('reconstruction_method') == 'oraclegs' or result.get('reconstruction_method') == 'oraclegs' else 'G4Splat · viewer approximation' if options.get('reconstruction_method') == 'g4splat' else 'ArtiFixer3D' if manifest or options.get('stage') in ('repair', 'benchmark') else 'Scene repair',
                'result_name': manifest.parent.name if manifest else record['run_id'],
                'original': original is not None, 'repaired': repaired is not None,
                'viser_url': '/splatfix/results/viser?id=' + quote(key),
                'detail_url': '/splatfix/results/run?id=' + quote(key),
                'ply_url': self.studio.file_url(repaired) if repaired else None,
                'native_ply_url': self.studio.file_url(native) if manifest and (native := self.studio.result_artifact(manifest, result.get('native_splat_path'))) else None,
                'provenance_url': self.studio.file_url(manifest) if manifest else None,
                'preview_url': preview[0] if preview else None,
                'preview_label': preview_label if preview else None,
                'metrics': metrics.get('aggregate', {}), 'test_count': metrics.get('published_test_count'),
                'metrics_scope': metrics.get('scope'), 'metrics_url': self.studio.file_url(metrics_file) if metrics else None,
                'limitation': result.get('limitation'), 'config': config,
                '_manifest': manifest, '_paths': {'original': original, 'repaired': repaired}}

    def _reconstruction_gallery(self, manifest, result, *, limit=None):
        """Share input selection and ordering between gallery and card preview."""
        root = manifest.parent
        gallery = {'frames': [], 'reference_count': None, 'verified_inputs': False,
                   'stage': 'ArtiFixer · reconstruction inputs'}
        def local(raw):
            return self.studio.result_artifact(manifest, raw)
        prediction = local(result.get('prediction_frames'))
        if prediction is None:
            # Saved-view worker uses this fixed first-pass location (not plus/).
            prediction = local('inference/splatfix/frames/batch_0000/pred')
        allowed = None
        preview = result if result.get('inference_ready') else read_json(root / 'inference-preview.json', {})
        if preview.get('inference_ready') and isinstance(preview.get('reconstruction_inputs'), list):
            allowed = set(preview['reconstruction_inputs'])
            gallery['reference_count'] = preview.get('reference_count')
            gallery['verified_inputs'] = True
        supervision = local(result.get('supervision_manifest') or 'supervision.json')
        groups = read_json(supervision, {}).get('groups', []) if supervision else []
        if groups:
            allowed = {int(g['source_index']) for g in groups if not g.get('reference')}
            gallery['reference_count'] = sum(bool(g.get('reference')) for g in groups)
            gallery['verified_inputs'] = True
        if allowed is None:
            split = local(result.get('inference_split') or 'prepared/bicycle/split.json')
            scenes = read_json(split, {}).get('test', {}) if split else {}
            if len(scenes) == 1:
                scene = next(iter(scenes.values()))
                def split_asset(value):
                    return local(str(split.parent.relative_to(root) / value)) if value else None
                selected = split_asset(scene.get('selected_indices_path'))
                targets = split_asset(scene.get('target_indices_path'))
                refs = read_json(selected, []) if selected else []
                target_ids = read_json(targets, []) if targets else None
                if isinstance(refs, list) and refs:
                    gallery['reference_count'] = len(refs)
                    if isinstance(target_ids, list):
                        allowed = set(target_ids)
                    elif result.get('frame_count') is not None:
                        allowed = set(range(int(result['frame_count'])))
                    if allowed is not None:
                        allowed -= set(refs)
                        gallery['verified_inputs'] = True
        if prediction and prediction.is_dir():
            frames = [p for p in prediction.glob('*.png') if p.stem.isdigit()]
            safe_root = root.resolve()
            for frame in sorted(frames, key=lambda p: int(p.stem)):
                index = int(frame.stem)
                if (allowed is None or index in allowed) and frame.resolve().is_relative_to(safe_root):
                    gallery['frames'].append({'index': index, 'name': frame.name,
                                              'url': self.studio.file_url(frame)})
                    if limit is not None and len(gallery['frames']) >= limit:
                        break
        gallery['count'] = len(gallery['frames'])
        gallery['expected_count'] = len(allowed) if allowed is not None else None
        return gallery, allowed

    def run_detail(self, key, *, deferred=False):
        """List first-pass inputs without reading pixel data or trusting remote paths."""
        row = self.detail(key)
        if row is None:
            return None
        if row['kind'] == 'WorldExplorer':
            from .worldexplorer_details import run_detail
            return run_detail(self, row, deferred=deferred)
        row['comments'] = self.read_comments(row['run_id'])
        settings_path = self.studio.root / row['run_id'] / 'settings.json'
        row['settings_snapshot'] = read_json(settings_path, None)
        row['settings_url'] = self.studio.file_url(settings_path) if settings_path.is_file() else None
        run_root = (self.studio.root / row['run_id']).resolve()
        from .handoff_details import handoff_details
        row['handoff_debug'] = handoff_details(self.studio, run_root)
        manifest = row.get('_manifest')
        json_paths = [run_root / 'worker-request.json']
        if manifest:
            json_paths += [manifest.parent / name for name in (
                'orbit-provenance.json', 'inference-directions.json')]
        row['json_artifacts'] = []
        for path in json_paths:
            if not path.is_file() or not path.resolve().is_relative_to(run_root):
                continue
            row['json_artifacts'].append({
                'name': path.name, 'path': str(path.relative_to(run_root)),
                'url': self.studio.file_url(path), 'snapshot': read_json(path, None),
            })
        if row['kind'] == 'ArtiFixer3D':
            from ..splatfix.artifixer.repeat_repair import repeat_inputs
            try:
                _, _, orbit = repeat_inputs(self.studio, row)
                row['artifixer_repeat'] = {'ready': True, 'note': (
                    ('Uses the saved intermediate PLY from this interrupted run. ' if row.get('incomplete') else '') +
                    ('Rebuilds the camera trajectory from saved checkpoint poses and recorded interpolation settings.' if not orbit else ''))}
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row['artifixer_repeat'] = {'ready': False, 'reason': str(exc)}
        if deferred:
            row['deferred_sections'] = ['training', 'artifacts', 'previews', 'anchor']
        else:
            for section in ('training', 'artifacts', 'previews', 'anchor'):
                row.update(self._detail_section(row, section))
        source_row = dict(row)
        manifest = row.pop('_manifest')
        row.pop('_paths')
        gallery = {'frames': [], 'reference_count': None, 'verified_inputs': False,
                   'stage': 'ArtiFixer · reconstruction inputs'}
        row['gallery'] = gallery
        row['plus_gallery'] = {'frames': []}
        reduction = read_json(self.studio.root / row['run_id'] / 'reduction.json', {})
        row['inference_videos'] = []
        saved_videos = reduction if row['reduced'] else read_json(self.studio.root / row['run_id'] / 'inference-videos.json', {})
        for video in saved_videos.get('inference_videos', []):
            path = self.studio.root / row['run_id'] / video['path']
            if manifest and path.is_file() and path.resolve().is_relative_to(manifest.parent.resolve()):
                row['inference_videos'].append({**video, 'url': self.studio.file_url(path)})
        row['reference_images'] = []
        row['original_rgb_images'] = []
        row['stage_comparison'] = None
        row['comparison_gallery'] = []
        if row['kind'] == 'OracleGS' or (manifest and read_json(manifest, {}).get('reconstruction_method') == 'oraclegs'):
            return row
        if manifest is None:
            return row
        result = read_json(manifest, {})
        root = manifest.parent
        if result.get('reconstruction_method') == 'g4splat' or row['kind'].startswith('G4Splat'):
            return row
        gallery, allowed = self._reconstruction_gallery(manifest, result)
        row['gallery'] = gallery
        preview = read_json(root / 'inference-preview.json', {})
        plus = self.studio.result_artifact(manifest, result.get('plus_frames') or preview.get('plus_frames'))
        if plus is None:
            plus = self.studio.result_artifact(manifest, 'plus/splatfix/frames/batch_0000/pred')
        if plus and plus.is_dir():
            images = [p for p in plus.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.stem.isdigit() and p.is_file() and p.stat().st_size > 0
                      and p.resolve().is_relative_to(root.resolve())]
            row['plus_gallery']['frames'] = [
                {'index': int(p.stem), 'name': p.name, 'url': self.studio.file_url(p)}
                for p in sorted(images, key=lambda p: (int(p.stem), p.name))]
        from ..splatfix.oraclegs.artifixer_inputs import source_assets
        try:
            source_assets(self.studio, source_row)
            row['artifixer_oraclegs'] = {'ready':True, 'original_available':row['original']}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row['artifixer_oraclegs'] = {'ready':False, 'reason':str(exc)}
        from ..splatfix.resolution import PROFILE_LABELS
        recorded_request = read_json(root / 'request.json', {})
        profile = (result.get('resolution_policy') or {}).get('profile') or recorded_request.get('resolution_profile')
        row['resolution_setting'] = PROFILE_LABELS.get(profile, 'Early original resolution · unversioned run')
        def local(raw):
            return self.studio.result_artifact(manifest, raw)
        comparison = local('stage-comparison.jpg')
        if comparison and comparison.is_file():
            metadata = read_json(root / 'stage-comparison.json', {})
            row['stage_comparison'] = {'url': self.studio.file_url(comparison),
                                       'caption': metadata.get('caption', 'Same-camera comparison across stages. Open the image for full resolution.')}
            row['comparison_gallery'].append({**row['stage_comparison'], 'title': 'Stage comparison'})
        chart = local('trajectory-quality.png')
        if chart and chart.is_file():
            metadata = read_json(root / 'trajectory-quality.json', {})
            row['comparison_gallery'].append({'url': self.studio.file_url(chart),
                'title': 'Trajectory and image quality',
                'caption': metadata.get('caption', 'Camera motion and available per-view evaluation scores.')})
        if row['reduced'] and reduction.get('method') == 'artifixer':
            return row
        row['preview_evaluation_status'] = read_json(root / 'preview-evaluation-status.json', {})
        row['evaluation_status'] = read_json(root / 'evaluation-status.json', {})
        def add_reference(path, index, name=None):
            if path is None or path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                return
            try:
                path = self.studio.allowed_file(str(path))
            except ValueError:
                return
            row['reference_images'].append({'index': index, 'name': name or path.name,
                                             'url': self.studio.file_url(path)})
        inputs = read_json(root / 'input-preview.json', {})
        for key in ('reference_images', 'original_rgb_images'):
            for item in inputs.get(key, []):
                path = local(item.get('path'))
                if path and path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    row[key].append({'index': item['index'], 'name': item['name'],
                                     'url': self.studio.file_url(path)})
        snapshot_references = list(row['reference_images'])
        supervision = local(result.get('supervision_manifest') or 'supervision.json')
        if supervision and supervision.is_file():
            data = read_json(supervision, {})
            groups = data.get('groups', [])
            if groups:
                # These are materialized copies of the actual conditioning
                # references, including edited RGB, rather than viewer previews.
                reference_index = 0
                request = read_json(root / 'request.json', {})
                reference_hashes = dict(zip(request.get('references', []), request.get('reference_sha256', [])))
                for group in groups:
                    if group.get('reference'):
                        path = local(f'source_colmap/images/anchor_{reference_index:05d}.png')
                        if path is None and reference_index < len(request.get('references', [])):
                            path = local(request['references'][reference_index])
                        # Downloaded source_colmap links may still point into
                        # /workspace. Recover only the recorded checkpoint file
                        # with the exact hash used for this run.
                        if path is None:
                            checkpoint = row['config'].get('splatfix', {}).get('checkpoint')
                            remote_checkpoint = request.get('checkpoint_root')
                            expected = reference_hashes.get(group['reference'])
                            if checkpoint and remote_checkpoint and expected:
                                try:
                                    relative = Path(group['reference']).relative_to(remote_checkpoint)
                                    candidate = self.studio.allowed_file(str(Path(checkpoint) / relative))
                                    if candidate.is_relative_to(Path(checkpoint).resolve()) and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected:
                                        path = candidate
                                except ValueError:
                                    pass
                        add_reference(path, group['source_index'], f'Reference {reference_index + 1}')
                        reference_index += 1
        if allowed is None or not row['reference_images']:
            split = local(result.get('inference_split') or 'prepared/bicycle/split.json')
            if split and split.is_file():
                scenes = read_json(split, {}).get('test', {})
                if len(scenes) == 1:
                    scene = next(iter(scenes.values()))
                    def split_asset(value):
                        if not value:
                            return None
                        return local(str(split.parent.relative_to(root) / value))
                    selected = split_asset(scene.get('selected_indices_path'))
                    refs = read_json(selected, []) if selected else []
                    if isinstance(refs, list) and refs:
                        gallery['reference_count'] = len(refs)
                        transforms = split_asset(scene.get('transforms_path'))
                        image_root = split_asset(scene.get('image_root'))
                        cameras = read_json(transforms, {}).get('frames', []) if transforms else []
                        if image_root:
                            for index in refs:
                                if type(index) is int and 0 <= index < len(cameras):
                                    raw = cameras[index].get('file_path')
                                    if raw:
                                        path = local(str(image_root / raw))
                                        if path is None and result.get('resolution_policy'):
                                            path = local(str(Path('conditioning-colmap/images') / Path(raw).name))
                                        add_reference(path, index, Path(raw).name)
        if not row['original_rgb_images']:
            trajectory_path = local(recorded_request.get('trajectory'))
            checkpoint = local(recorded_request.get('checkpoint_root'))
            if trajectory_path and checkpoint:
                for index, frame in enumerate(read_json(trajectory_path, {}).get('frames', [])):
                    path = local(str(checkpoint / frame['rgb'])) if frame.get('rgb') else None
                    if path and path.is_file():
                        row['original_rgb_images'].append({'index': index, 'name': path.name,
                                                          'url': self.studio.file_url(path)})
            split = local(result.get('inference_split') or 'prepared/bicycle/split.json')
            scenes = read_json(split, {}).get('test', {}) if split else {}
            if len(scenes) == 1:
                entry = next(iter(scenes.values()))
                render_root = local(str(split.parent / entry['render_dir'])) if entry.get('render_dir') else None
                if render_root:
                    for path in sorted(render_root.glob('*.png')):
                        safe = local(str(path))
                        if safe and safe.is_file() and path.stem.isdigit():
                            row['original_rgb_images'].append({'index': int(path.stem), 'name': path.name,
                                                              'url': self.studio.file_url(safe)})
        if snapshot_references:
            row['reference_images'] = snapshot_references
            gallery['reference_count'] = len(snapshot_references)
        if not row['reference_images']:
            # References are already known before supervision is written.
            for index, raw in enumerate(recorded_request.get('references', [])):
                add_reference(local(raw), index, f'Reference {index + 1}')
            if row['reference_images']:
                gallery['reference_count'] = len(row['reference_images'])
        return row

    DETAIL_SECTIONS = frozenset(('training', 'artifacts', 'previews', 'anchor', 'media'))

    def detail_section(self, key, section):
        """Load one independent expensive section without rebuilding other sections."""
        if section not in self.DETAIL_SECTIONS:
            raise ValueError('Unknown detail section')
        row = self.detail(key)
        return self._detail_section(row, section) if row else None

    def _detail_section(self, source, section):
        if source['kind'] == 'WorldExplorer':
            from .worldexplorer_details import run_detail
            return run_detail(self, dict(source), section=section) if section in ('media', 'training') else {}
        row = {}
        manifest = source.get('_manifest')
        run_root = (self.studio.root / source['run_id']).resolve()
        if section == 'artifacts':
            # Checkpoint downloads can precede every result/preview manifest.
            # Keep these separate from verified reconstruction or inference outputs.
            row['saved_artifacts'] = {'image_groups': [], 'checkpoints': []}
            saved = row['saved_artifacts']
            groups = {}
            seen_artifacts = set()
            gpu = run_root / 'gpu'
            for path in sorted(gpu.rglob('*')):
                if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.ckpt', '.ply', '.json'):
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(run_root) or not resolved.is_file():
                    continue
                relative = path.relative_to(gpu)
                url = self.studio.file_url(path)
                info = resolved.stat()
                identity = (info.st_dev, info.st_ino)
                if identity in seen_artifacts:
                    continue
                seen_artifacts.add(identity)
                if path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.ply') or path.name == 'checkpoint.json':
                    saved['checkpoints'].append({'name': str(relative), 'url': url})
                elif path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    name = str(relative.parent)
                    groups.setdefault(name, []).append({'name': path.name, 'url': url})
            saved['image_groups'] = [{'name': name, 'images': images} for name, images in groups.items()]
        elif section == 'training':
            row.update(training_metrics=None, training_metrics_error=None)
            candidates = list(run_root.glob('*.log')) + list((run_root / 'gpu').glob('*.log'))
            decoded_names = {}
            if manifest:
                result_root = manifest.parent.absolute()
                candidates += [p for p in result_root.rglob('*.log')
                               if '.training-metrics' not in p.parts and p.resolve().is_relative_to(result_root.resolve())]
                from .training_logs import export_training_logs
                try:
                    exports, chart, error = export_training_logs(result_root)
                except OSError as exc:
                    exports, chart, error = [], None, f'Training metric export unavailable: {exc}'
                for source, exported in exports:
                    candidates.append(exported)
                    decoded_names[exported] = str(source.relative_to(run_root)) + ' (decoded scalars)'
                if chart:
                    row['training_metrics'] = {
                        'url': self.studio.file_url(chart),
                        'modified': chart.stat().st_mtime_ns,
                        'name': 'ArtiFixer · TensorBoard training scalars',
                        'format': 'scalars',
                    }
                if error:
                    row['training_metrics_error'] = error
            else:
                candidates += list((run_root / 'gpu/results').glob('*/*.log'))
            row['logs'] = []
            seen = set()
            for path in sorted(candidates, key=lambda p: (p.name == 'live-stage.log', str(p))):
                resolved = path.resolve()
                if resolved in seen or not resolved.is_relative_to(run_root) or not resolved.is_file():
                    continue
                seen.add(resolved)
                stat = resolved.stat()
                row['logs'].append({
                    'name': decoded_names.get(path, str(path.relative_to(run_root))),
                    'url': self.studio.file_url(resolved),
                    'partial': path.name == 'live-stage.log',
                    'size': stat.st_size,
                    'modified': stat.st_mtime_ns,
                })
        elif section == 'anchor' and manifest and source['kind'].startswith('ArtiFixer'):
            from .artifixer_anchor_chart import anchor_chart
            try:
                row['anchor_chart'] = anchor_chart(self.studio, manifest, source['run_id'])
            except (OSError, ValueError, KeyError, TypeError):
                # Partial downloads and reduced archives may not contain the arrays.
                row['anchor_chart'] = None
        elif section == 'previews':
            result = read_json(manifest, {}) if manifest else {}
            method = result.get('reconstruction_method')
            if source['kind'] == 'OracleGS' or method == 'oraclegs':
                from .oraclegs_details import build_details
                pending = manifest or run_root / 'gpu/results/oraclegs/result.json'
                row['oraclegs'] = build_details(self.studio, pending, result)
            elif source['kind'].startswith('G4Splat') or method == 'g4splat':
                from .g4splat_details import build_details
                pending = manifest or run_root / 'gpu/results/g4splat/result.json'
                row['g4splat'] = build_details(self.studio, pending, result)
        return row

    def review_views(self, key, *, row=None):
        from .review_cameras import checkpoint_views, benchmark_views, historical_views
        row = row if row is not None else self.detail(key)
        if not row:
            return []
        if row['kind'] == 'WorldExplorer':
            from .worldexplorer_details import review_views
            return review_views(self, row)
        if row['_manifest']:
            derivative = row['_manifest'].parent / 'checkpoint'
            if (derivative / 'checkpoint.json').is_file():
                views = checkpoint_views(derivative)
                if views:
                    return views
            checkpoint = row['config'].get('splatfix', {}).get('checkpoint')
            if checkpoint and Path(checkpoint).resolve().is_relative_to(Path(self.cfg.output.dir).resolve()):
                views = checkpoint_views(checkpoint)
                if views:
                    return views
            return benchmark_views(row['_manifest'])
        return historical_views(self.studio.root / row['run_id'])

    def detail(self, key, *, review=False):
        for row in self.entries(private=True, review_key=key if review else None, detail_key=None if review else key):
            if row['id'] == key or row['run_id'] == key:
                return row
            manifest = row['_manifest']
            if manifest:
                legacy = manifest.with_name('partial-result.json').resolve().relative_to(self.studio.root.resolve())
                if hashlib.sha256(str(legacy).encode()).hexdigest()[:20] == key:
                    return row
        return None


class ResultViser(SceneRunViser):
    port_offset = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rows = {}

    def detail(self, run_id):
        # Resolve once per review session, never from polling or under _lock.
        if run_id not in self._rows:
            row = self.studio.detail(run_id, review=True)
            if row is not None:
                self._rows[run_id] = row
            return row
        return self._rows[run_id]

    def review_views(self, run_id):
        return self.studio.review_views(run_id, row=self._rows[run_id])

    def _stop_locked(self):
        super()._stop_locked()
        self._rows.clear()

    def ply_paths(self, run_id):
        row = self._rows.get(run_id)
        return row['_paths'] if row else {'original': None, 'repaired': None}

    def highlight_supported(self, run_id):
        return False

    def _snapshot_locked(self, run_id, paths, *, showing):
        result = super()._snapshot_locked(run_id, paths, showing=showing)
        result['viser_path'] = '/splatfix/results/viser?id=' + quote(run_id)
        row = self._rows.get(run_id)
        if row:
            result['label'] = f"{row['scene']} · {row['run_id']}"
        return result
