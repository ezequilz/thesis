"""Catalog of reconstruction jobs, live inputs and completed outputs."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
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

    def reduce_run(self, run_id, *, backup=False):
        if type(backup) is not bool:
            raise ValueError('backup must be a boolean')
        with self._run_storage_lock:
            run = self.studio.store.get_run(run_id)
            if run.state.status.value not in ('completed', 'error', 'stopped', 'gpu_expired'):
                raise ValueError('Only inactive runs can be reduced')
            from .run_reduction import reduce_oraclegs, reduce_artifixer
            before = self.studio.store.output_size_bytes(run.run_id, refresh=True)
            history = Path(self.cfg.output.dir) / 'run_history' if backup else None
            rows = [row for row in self.entries(private=True) if row['run_id'] == run_id]
            if any(row['kind'] == 'OracleGS' for row in rows):
                report = reduce_oraclegs(Path(run.path), history, before_size_bytes=before)
            elif rows and all(row['kind'] == 'ArtiFixer3D' for row in rows):
                plans = self._artifixer_plans(rows)
                report = reduce_artifixer(Path(run.path), plans, history, before_size_bytes=before)
            else:
                raise ValueError('Reduction supports OracleGS and ArtiFixer runs only')
            after = self.studio.store.output_size_bytes(run.run_id, refresh=True)
            return {'run_id': run.run_id, **report, 'output_size_bytes': after,
                    'reduction_saved_bytes': max(0, before - after) if before is not None and after is not None else None}

    def _artifixer_plans(self, rows):
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
            predictions = [self.studio.result_artifact(manifest, result.get(key))
                           for key in ('prediction_frames', 'plus_frames')]
            plans.append({'root': manifest.parent, 'ply_copies': copies,
                          'prediction_dirs': [p for p in predictions if p and p.is_dir()]})
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

    def entries(self, *, private=False):
        rows = []
        for run in self.studio.store.list_runs():
            record = run.to_dict()
            record['output_size_bytes'] = self.studio.store.output_size_bytes(run.run_id)
            root = Path(run.path)
            config = record.get('config', {})
            candidates = sorted((root / 'gpu/results').glob('*/result.json'))
            candidates += sorted(p for p in (root / 'gpu/results').glob('*/partial-result.json')
                                 if not (p.parent / 'result.json').is_file())
            candidates += sorted(p for p in (root / 'gpu/results').glob('*/inference-preview.json')
                                 if not (p.parent / 'result.json').is_file()
                                 and not (p.parent / 'partial-result.json').is_file())
            candidates += sorted(p for p in (root / 'gpu/results').glob('*/input-preview.json')
                                 if not any((p.parent / name).is_file() for name in
                                            ('result.json', 'partial-result.json', 'inference-preview.json')))
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
        rows.extend(worldexplorer_entries(self))
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
        before = read_json(self.studio.root / record['run_id'] / 'reduction.json', {}).get('before_size_bytes')
        after = record.get('output_size_bytes')
        saved_bytes = max(0, before - after) if type(before) is int and type(after) is int else None
        preview = []
        preview_label = '3D+ image preview'
        if manifest:
            preview = self.studio.plus_previews(manifest, result)
            if not preview:
                gallery, _ = self._reconstruction_gallery(manifest, result)
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
                'incomplete': result.get('incomplete', False),
                'inference_ready': result.get('inference_ready', False),
                'phase': record.get('state', {}).get('details', {}).get('phase'),
                'interruption': result.get('interruption'),
                'diagnostics_url': self.studio.file_url(manifest.parent / 'benchmark-run.json') if manifest and (manifest.parent / 'benchmark-run.json').is_file() else None,
                'created_at': record.get('state', {}).get('created_at'), 'status': status,
                'output_size_bytes': record.get('output_size_bytes'),
                'reduction_saved_bytes': saved_bytes,
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

    def _reconstruction_gallery(self, manifest, result):
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
            frames = [p for p in prediction.glob('*.png') if p.stem.isdigit()
                      and p.resolve().is_relative_to(root.resolve())]
            for frame in sorted(frames, key=lambda p: int(p.stem)):
                index = int(frame.stem)
                if allowed is None or index in allowed:
                    gallery['frames'].append({'index': index, 'name': frame.name,
                                              'url': self.studio.file_url(frame)})
        gallery['count'] = len(gallery['frames'])
        gallery['expected_count'] = len(allowed) if allowed is not None else None
        return gallery, allowed

    def run_detail(self, key):
        """List first-pass inputs without reading pixel data or trusting remote paths."""
        row = self.detail(key)
        if row is None:
            return None
        if row['kind'] == 'WorldExplorer':
            from .worldexplorer_details import run_detail
            return run_detail(self, row)
        manifest = row.pop('_manifest')
        row.pop('_paths')
        gallery = {'frames': [], 'reference_count': None, 'verified_inputs': False,
                   'stage': 'ArtiFixer · reconstruction inputs'}
        row['gallery'] = gallery
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
        # Keep full logs separate from the bounded live-worker snapshot.
        run_root = (self.studio.root / row['run_id']).resolve()
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
        if row['kind'] == 'OracleGS' or (manifest and read_json(manifest, {}).get('reconstruction_method') == 'oraclegs'):
            from .oraclegs_details import build_details
            pending = manifest or self.studio.root / row['run_id'] / 'gpu/results/oraclegs/result.json'
            row['oraclegs'] = build_details(self.studio, pending, read_json(manifest, {}) if manifest else {})
            return row
        if manifest is None:
            if row['kind'].startswith('G4Splat'):
                from .g4splat_details import build_details
                pending = self.studio.root / row['run_id'] / 'gpu/results/g4splat/result.json'
                row['g4splat'] = build_details(self.studio, pending, {})
            return row
        result = read_json(manifest, {})
        root = manifest.parent
        if result.get('reconstruction_method') == 'g4splat' or row['kind'].startswith('G4Splat'):
            from .g4splat_details import build_details
            row['g4splat'] = build_details(self.studio, manifest, result)
            return row
        gallery, allowed = self._reconstruction_gallery(manifest, result)
        row['gallery'] = gallery
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

    def review_views(self, key):
        from .review_cameras import checkpoint_views, benchmark_views, historical_views
        row = self.detail(key)
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

    def detail(self, key):
        for row in self.entries(private=True):
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

    def ply_paths(self, run_id):
        row = self.studio.detail(run_id)
        return row['_paths'] if row else {'original': None, 'repaired': None}

    def highlight_supported(self, run_id):
        return False

    def _snapshot_locked(self, run_id, paths, *, showing):
        result = super()._snapshot_locked(run_id, paths, showing=showing)
        result['viser_path'] = '/splatfix/results/viser?id=' + quote(run_id)
        row = self.studio.detail(run_id)
        if row:
            result['label'] = f"{row['scene']} · {row['run_id']}"
        return result
