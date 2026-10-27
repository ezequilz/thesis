import json
from pathlib import Path
from types import SimpleNamespace

from splat_explorer.config import Config
from splat_explorer.scene_runs.store import SceneRunStore
from splat_explorer.web.splatfix_studio import SplatfixStudio


def replay(tmp_path):
    studio = SplatfixStudio(SimpleNamespace(cfg=Config({'output': {'dir': str(tmp_path)}})),
                            SceneRunStore(tmp_path / 'scene-runs'))
    studio.scenes = lambda: []
    root = tmp_path / 'artifixer-teacher-replay/run_20261026_212233'
    def write(relative, body):
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body))
    write('provenance.json', {'source_run': root.name, 'inference_only': True,
                             'expected_frame_indices': [1, 2], 'created_at': '2026-10-27T00:00:00Z'})
    write('status.json', {'status': 'running', 'completed_frames': 1, 'completed_clips': 1,
                         'total_generated_frames': 2, 'total_clips': 2, 'current_clip': 2, 'denoising_step': 3})
    write('results/request.json', {'runtime': {'inference_model': '14b-teacher'},
                                  'checkpoint_root': '/remote/inputs', 'trajectory': '/remote/inputs/trajectory.json',
                                  'references': ['/remote/inputs/ref.png']})
    write('inputs/trajectory.json', {'frames': [{'rgb': 'original.png'}]})
    write('results/inference-teacher.json', {'complete': False, 'clips': [
        {'frame_indices': [1], 'output': 'inference'}]})
    pred = root / 'results/inference/splatfix/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    for name in ('00001.png', '00002.png'):
        (pred / name).write_bytes(b'png')
    (root / 'inputs/ref.png').write_bytes(b'png')
    (root / 'inputs/original.png').write_bytes(b'png')
    return studio, root


def test_live_teacher_replay_reviews_only_completed_clips(tmp_path):
    studio, root = replay(tmp_path)
    rows = studio.results.entries()
    json.dumps(rows)  # No private Path objects in API responses.
    row, = rows
    assert row['kind'] == 'ArtiFixer inference' and row['status'] == 'running' and row['inference_only']
    assert not row['can_stop'] and not row['can_archive'] and not row['can_reduce']
    detail = studio.results.run_detail(row['id'], deferred=True)
    assert [f['index'] for f in detail['gallery']['frames']] == [1]
    assert detail['gallery']['expected_count'] == 2
    assert len(detail['reference_images']) == len(detail['original_rgb_images']) == 1
    assert detail['inference_only'] and 'step 3/50' in detail['teacher_summary']
    for p in (root / 'inputs/ref.png', root / 'results/request.json'):
        assert studio.allowed_file(str(p)) == p.resolve()
    # A replay cannot replace or impersonate a normal run's catalog entry.
    assert studio.results.detail('run_20261026_212233') is None


def test_teacher_review_rejects_symlinks_and_private_source_files(tmp_path):
    import pytest
    studio, root = replay(tmp_path)
    outside = tmp_path / 'secret.png'; outside.write_bytes(b'secret')
    (root / 'inputs/ref.png').unlink(); (root / 'inputs/ref.png').symlink_to(outside)
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert not detail['reference_images']
    with pytest.raises(ValueError):
        studio.allowed_file(str(root / 'inputs/ref.png'))
    (root / 'src').mkdir(); (root / 'src/private.json').write_text('{}')
    with pytest.raises(ValueError):
        studio.allowed_file(str(root / 'src/private.json'))


def test_teacher_completed_state_exposes_validation_report(tmp_path):
    # The UI exposes the controller's state, while retaining its validation artifact.
    studio, root = replay(tmp_path)
    (root / 'status.json').write_text(json.dumps({'status': 'completed', 'completed_frames': 2}))
    (root / 'results/validation.json').write_text(json.dumps({'passed': True}))
    detail = studio.results.run_detail(studio.results.entries()[0]['id'])
    assert detail['status'] == 'completed'
    assert any(a['name'] == 'validation.json' for a in detail['json_artifacts'])
