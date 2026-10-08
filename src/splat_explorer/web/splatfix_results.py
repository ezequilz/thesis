"""Catalog of reconstruction jobs, live inputs and completed outputs."""
from __future__ import annotations

import hashlib
import json
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
        if not any(run.run_id == run_id for run in self.studio.store.list_runs()):
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
            original, repaired = root / 'scene_original.ply', root / 'scene_repaired.ply'
            if any(p.is_file() and not p.resolve().is_relative_to(self.studio.root.resolve()) for p in (original, repaired)):
                continue
            if original.is_file() or repaired.is_file():
                rows.append(self._row(record['run_id'], record, original if original.is_file() else None,
                    repaired if repaired.is_file() else None, {}, {}, None, None))
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
        preview = []
        if manifest:
            preview = self.studio.plus_previews(manifest, result)
        return {'id': key, 'run_id': record['run_id'], 'scene': config.get('scene_id', ''),
                'can_stop': config.get('pipeline') == 'splatfix' and record.get('state', {}).get('status') in ('queued', 'waiting_gpu', 'starting', 'running'),
                'message': record.get('state', {}).get('message'),
                'incomplete': result.get('incomplete', False),
                'inference_ready': result.get('inference_ready', False),
                'phase': record.get('state', {}).get('details', {}).get('phase'),
                'interruption': result.get('interruption'),
                'diagnostics_url': self.studio.file_url(manifest.parent / 'benchmark-run.json') if manifest and (manifest.parent / 'benchmark-run.json').is_file() else None,
                'created_at': record.get('state', {}).get('created_at'), 'status': record.get('state', {}).get('status'),
                'mode': options.get('mode') or config.get('repair_type', ''),
                'model': result.get('model_variant') or options.get('model'),
                'trajectory': result.get('trajectory_mode'), 'frames': result.get('frame_count'),
                'kind': 'G4Splat · viewer approximation' if options.get('reconstruction_method') == 'g4splat' else 'ArtiFixer3D' if manifest or options.get('stage') in ('repair', 'benchmark') else 'Scene repair',
                'result_name': manifest.parent.name if manifest else record['run_id'],
                'original': original is not None, 'repaired': repaired is not None,
                'viser_url': '/splatfix/results/viser?id=' + quote(key),
                'detail_url': '/splatfix/results/run?id=' + quote(key),
                'ply_url': self.studio.file_url(repaired) if repaired else None,
                'native_ply_url': self.studio.file_url(native) if manifest and (native := self.studio.result_artifact(manifest, result.get('native_splat_path'))) else None,
                'provenance_url': self.studio.file_url(manifest) if manifest else None,
                'preview_url': preview[0] if preview else None,
                'metrics': metrics.get('aggregate', {}), 'test_count': metrics.get('published_test_count'),
                'metrics_scope': metrics.get('scope'), 'metrics_url': self.studio.file_url(metrics_file) if metrics else None,
                'limitation': result.get('limitation'), 'config': config,
                '_manifest': manifest, '_paths': {'original': original, 'repaired': repaired}}

    def run_detail(self, key):
        """List first-pass inputs without reading pixel data or trusting remote paths."""
        row = self.detail(key)
        if row is None:
            return None
        manifest = row.pop('_manifest')
        row.pop('_paths')
        gallery = {'frames': [], 'reference_count': None, 'verified_inputs': False,
                   'stage': 'ArtiFixer · reconstruction inputs'}
        row['gallery'] = gallery
        row['reference_images'] = []
        row['original_rgb_images'] = []
        row['stage_comparison'] = None
        row['comparison_gallery'] = []
        if manifest is None:
            return row
        result = read_json(manifest, {})
        root = manifest.parent
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
        prediction = local(result.get('prediction_frames'))
        if prediction is None:
            # Saved-view worker uses this fixed first-pass location (not plus/).
            prediction = local('inference/splatfix/frames/batch_0000/pred')
        allowed = None
        if result.get('inference_ready') and isinstance(result.get('reconstruction_inputs'), list):
            allowed = set(result['reconstruction_inputs'])
            gallery['reference_count'] = result.get('reference_count')
            gallery['verified_inputs'] = True
        supervision = local(result.get('supervision_manifest') or 'supervision.json')
        if supervision and supervision.is_file():
            data = read_json(supervision, {})
            groups = data.get('groups', [])
            if groups:
                allowed = {int(g['source_index']) for g in groups if not g.get('reference')}
                gallery['reference_count'] = sum(bool(g.get('reference')) for g in groups)
                gallery['verified_inputs'] = True
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
                    targets = split_asset(scene.get('target_indices_path'))
                    refs = read_json(selected, []) if selected else []
                    target_ids = read_json(targets, []) if targets else None
                    if isinstance(refs, list) and refs:
                        gallery['reference_count'] = len(refs)
                        # Loading references must not replace indices already verified
                        # by the inference preview or supervision manifest.
                        if allowed is None:
                            if isinstance(target_ids, list):
                                allowed = set(target_ids)
                            elif result.get('frame_count') is not None:
                                allowed = set(range(int(result['frame_count'])))
                            if allowed is not None:
                                allowed -= set(refs)
                                gallery['verified_inputs'] = True
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
        return row

    def review_views(self, key):
        from .review_cameras import checkpoint_views, benchmark_views, historical_views
        row = self.detail(key)
        if not row:
            return []
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
