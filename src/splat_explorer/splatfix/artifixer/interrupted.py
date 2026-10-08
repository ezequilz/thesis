"""Publish explicitly incomplete outputs after a controlled worker stop."""
from pathlib import Path
from ..checkpoint import atomic_json
from ..jobs import read_json


def preserve_interrupted_results(request, reason):
    output = Path(request.get('output', '/nonexistent'))
    for root in output.iterdir() if output.is_dir() else []:
        if not root.is_dir() or (root / 'result.json').exists():
            continue
        benchmark = read_json(root / 'benchmark-run.json', {})
        saved = [(p, read_json(p, {})) for p in root.glob('*.interrupted/stop-state.json')]
        # A simple base reconstruction is not a repaired result.
        repaired = [(p, s) for p, s in saved if p.parent.name in ('artifixer3d.interrupted', 'distill.interrupted')]
        snapshot = repaired[-1][1] if repaired else {}
        def artifact(raw):
            if not raw:
                return None
            path = Path(raw).resolve()
            return str(path) if path.is_relative_to(root.resolve()) and path.is_file() else None
        predictions = list((root / 'inference').glob('**/batch_0000/pred'))
        split = root / 'prepared/bicycle/split_trajectory.json'
        if not split.exists():
            split = root / 'prepared/bicycle/split.json'
        frame_count = None
        if split.exists():
            scenes = read_json(split, {}).get('test', {})
            if len(scenes) == 1:
                entry = next(iter(scenes.values()))
                frame_count = len(read_json(split.parent / entry.get('transforms_path', ''), {}).get('frames', []))
        atomic_json(root / 'partial-result.json', {
            'incomplete': True, 'status': 'interrupted', 'splat_path': artifact(snapshot.get('splat_path')),
            'reconstruction_checkpoint': artifact(snapshot.get('checkpoint')), 'plus_frames': None,
            'model_variant': request.get('runtime', {}).get('model_variant'),
            'trajectory_mode': benchmark.get('trajectory'), 'frame_count': frame_count,
            'resolution_policy': benchmark.get('resolution_policy'),
            'prediction_frames': str(predictions[0]) if len(predictions) == 1 else None,
            'inference_split': str(split.relative_to(root)) if split.exists() else None,
            'interruption': {'reason': reason, 'last_logged_training_step': snapshot.get('step'),
                             'save_status': snapshot.get('status', 'no_training_snapshot'),
                             'saved_states': [str(p.relative_to(root)) for p, _ in saved]},
            'limitation': 'Interrupted reconstruction. Available splat is an intermediate model, not a completed repair.'})
