"""Read-only catalog of saved reconstructions and their evaluation records."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

from ..splatfix.jobs import read_json
from .scene_run_viser import SceneRunViser


class ResultCatalog:
    def __init__(self, studio):
        self.studio = studio
        self.cfg = studio.cfg
        self._visor = None

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
                    continue
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
                key = hashlib.sha256(str(manifest.resolve().relative_to(self.studio.root.resolve())).encode()).hexdigest()[:20]
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
            original, repaired = root / 'scene_original.ply', root / 'scene_repaired.ply'
            if any(p.is_file() and not p.resolve().is_relative_to(self.studio.root.resolve()) for p in (original, repaired)):
                continue
            if original.is_file() or repaired.is_file():
                rows.append(self._row(record['run_id'], record, original if original.is_file() else None,
                    repaired if repaired.is_file() else None, {}, {}, None, None))
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
                'created_at': record.get('created_at'), 'status': record.get('state', {}).get('status'),
                'mode': options.get('mode') or config.get('repair_type', ''),
                'model': result.get('model_variant') or options.get('model'),
                'trajectory': result.get('trajectory_mode'), 'frames': result.get('frame_count'),
                'kind': 'ArtiFixer3D' if manifest else 'Scene repair',
                'result_name': manifest.parent.name if manifest else record['run_id'],
                'original': original is not None, 'repaired': repaired is not None,
                'viser_url': '/splatfix/results/viser?id=' + quote(key),
                'ply_url': self.studio.file_url(repaired) if repaired else None,
                'provenance_url': self.studio.file_url(manifest) if manifest else None,
                'preview_url': preview[0] if preview else None,
                'metrics': metrics.get('aggregate', {}), 'test_count': metrics.get('published_test_count'),
                'metrics_scope': metrics.get('scope'), 'metrics_url': self.studio.file_url(metrics_file) if metrics else None,
                'limitation': result.get('limitation'), 'config': config,
                '_manifest': manifest, '_paths': {'original': original, 'repaired': repaired}}

    def review_views(self, key):
        from .review_cameras import checkpoint_views, benchmark_views, historical_views
        row = self.detail(key)
        if not row:
            return []
        if row['_manifest']:
            checkpoint = row['config'].get('splatfix', {}).get('checkpoint')
            if checkpoint and Path(checkpoint).resolve().is_relative_to(Path(self.cfg.output.dir).resolve()):
                return checkpoint_views(checkpoint)
            return benchmark_views(row['_manifest'])
        return historical_views(self.studio.root / row['run_id'])

    def detail(self, key):
        return next((r for r in self.entries(private=True) if r['id'] == key), None)


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
