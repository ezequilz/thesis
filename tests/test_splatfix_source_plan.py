import json
from pathlib import Path

import numpy as np
import pytest

from splat_explorer.splatfix.checkpoint import atomic_json, camera_from_record
from splat_explorer.splatfix.artifixer import repair, worker
from test_splatfix_repair import make_checkpoint, mock_scene, FakeRenderer


def test_planning_uses_original_depth_opacity_scale_and_orbit_only(tmp_path, monkeypatch):
    cp = make_checkpoint(tmp_path)
    camera = camera_from_record(cp.views[0])
    from dataclasses import replace
    cp.manifest['target_views'] = 2
    cp.add_view(np.full((32, 32, 3), 90, np.uint8), replace(camera, position=camera.position + [1, 0, 0]))
    mock_scene(monkeypatch)
    monkeypatch.setattr('splat_explorer.splatfix.rendering.BundleRenderer', FakeRenderer)
    monkeypatch.setattr(repair, 'validate_runtime', lambda runtime: {**repair.DEFAULT_RUNTIME, **runtime})
    phases, rendered_orbits = [], []
    original = repair.prepare_saved_path
    def geometry(*args, **kw):
        rendered_orbits.append(kw.get('orbit'))
        return original(*args, **kw)
    monkeypatch.setattr(repair, 'prepare_saved_path', geometry)
    first, last = [camera_from_record(view).c2w for view in cp.views]
    middle = first.copy(); middle[0, 3] += .5
    orbit = {'poses': [first.tolist(), middle.tolist(), last.tolist()],
             'provenance': {'segments': [{'full_frame_indices': [1]}]}}
    def run(command, **kw):
        if '--full-output' in command:
            phases.append('orbit')
            assert command[command.index('--metric-scale') + 1] == '2.5'
            atomic_json(Path(command[command.index('--full-output') + 1]), {'frames': [
                {'transform_matrix': (np.asarray(p) @ np.diag([1., -1., -1., 1.])).tolist()} for p in orbit['poses']]})
            atomic_json(Path(command[command.index('--provenance') + 1]), orbit['provenance'])
        else:
            assert command[-1] == 'scale', 'Planning must never caption, infer, distill or run plus'
            assert command[1].endswith('official_worker.py')
            phases.append('scale')
            request_path = Path(command[command.index('--request') + 1])
            request = json.loads(request_path.read_text())
            trajectory = json.loads(Path(request['trajectory']).read_text())
            assert len(trajectory['frames']) == 2
            for anchor in trajectory['anchors']:
                assert (cp.root / anchor['depth']).is_file() and (cp.root / anchor['opacity']).is_file()
            atomic_json(request_path.parent / 'scale-result.json', {'metric_scale': 2.5, 'official': True})
    monkeypatch.setattr(repair, 'run_worker', run)
    result = repair.run_repair(cp.root, tmp_path / 'planning', mode='baseline', prepare_source_only=True,
                              runtime={'prompt_source': 'original_anchors'})
    plan = json.loads(Path(result['source_plan']).read_text())
    assert phases == ['scale', 'orbit'] and rendered_orbits == [None]
    assert plan['orbit'] == orbit and plan['camera_scale'] == .025
    assert plan['scale_estimate'] == {'metric_scale': 2.5, 'official': True}
    monkeypatch.setattr(repair, 'run_worker', lambda *a, **kw: pytest.fail('No re-estimation after planning'))
    def stop_before_models(event):
        if event['phase'] == 'caption':
            raise InterruptedError('Verified exact planned geometry')
    with pytest.raises(InterruptedError, match='exact planned geometry'):
        repair.run_repair(cp.root, tmp_path / 'full', mode='baseline', source_plan=plan,
                          runtime={'prompt_source': 'original_anchors'}, on_progress=stop_before_models)
    assert rendered_orbits == [None, orbit]
    request_path, = (tmp_path / 'full').glob('*/request.json')
    request = json.loads(request_path.read_text())
    assert request['source_plan_sha256'] == plan['sha256']
    assert request['camera_scale'] == .025 and request['scale_estimate']['official']


@pytest.mark.parametrize('outcome', ['ready', 'capture_error', 'cancel', 'wrong_plan'])
def test_worker_waits_for_full_staging_or_stops_without_reconstruction(tmp_path, monkeypatch, outcome):
    request = {'checkpoint': str(tmp_path / 'checkpoint'), 'output': str(tmp_path / 'results'),
               'mode': 'baseline', 'frames': 25, 'span_fraction': .04, 'runtime': {}, 'prepare_source_first': True}
    atomic_json(tmp_path / 'worker-request.json', request)
    plan = {'sha256': 'exact-plan'}
    plan_path = tmp_path / 'prepared-plan.json'; atomic_json(plan_path, plan)
    calls = []
    def repair_run(*args, **kw):
        calls.append(kw)
        if kw.get('prepare_source_only'):
            return {'source_plan': str(plan_path)}
        assert (tmp_path / 'full-staging.json').is_file()
        assert kw['source_plan'] == plan
        return {'output_dir': str(tmp_path / 'results')}
    monkeypatch.setattr(worker, 'run_repair', repair_run)
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.run_evaluation.finalize_run_evaluation', lambda *a, **kw: None)
    monkeypatch.setattr('splat_explorer.splatfix.artifixer.interrupted.preserve_interrupted_results', lambda *a: None)
    def tick(_):
        state = json.loads((tmp_path / 'worker-status.json').read_text())
        assert state['phase'] == 'awaiting_source_capture' and len(calls) == 1
        if outcome == 'cancel':
            (tmp_path / 'STOP').touch()
        elif outcome == 'capture_error':
            atomic_json(tmp_path / 'source-capture-error.json', {'error': 'Viser disconnected'})
        else:
            atomic_json(tmp_path / 'full-staging.json', {'source_plan_sha256': 'exact-plan' if outcome == 'ready' else 'wrong'})
    monkeypatch.setattr(worker.time, 'sleep', tick)
    worker.execute(tmp_path)
    state = json.loads((tmp_path / 'worker-status.json').read_text())
    assert state['status'] == ('completed' if outcome == 'ready' else 'stopped' if outcome == 'cancel' else 'error')
    assert len(calls) == (2 if outcome == 'ready' else 1)
