"""Read bounded progress metadata on the login host; no GPU or model imports."""
import json
from pathlib import Path
import sys

LIMIT = 64 * 1024


def snapshot(root, container):
    root = Path(root).resolve()
    container = Path(container)

    def local(raw):
        try:
            path = (root / Path(raw).relative_to(container)).resolve()
            return path if path.is_relative_to(root) else None
        except (TypeError, ValueError):
            return None

    def read(path, *, tail=False):
        if path is None or not path.resolve().is_relative_to(root) or not path.is_file():
            return '' if tail else {}
        try:
            if tail:
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - LIMIT))
                    return stream.read(LIMIT).decode('utf-8', errors='replace')
            if path.stat().st_size > 8 * 1024 * 1024:
                return {}
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return '' if tail else {}

    # Read state after the exit sentinel to avoid declaring a launcher race.
    exited = read(root / 'launcher-exit', tail=True).strip()
    state = read(root / 'worker-status.json')
    capture = read(root / 'viser-capture/request.json')
    ident = capture.get('id', '')
    pending = (len(ident) == 32 and all(c in '0123456789abcdef' for c in ident)
               and not (root / 'viser-capture' / ident / 'response.json').is_file())
    result = {'state': state, 'exit': exited, 'capture_pending': pending,
              'logs': {}, 'teachers': {}, 'reconstruction_complete': False}
    output = local(state.get('output_dir') or state.get('result', {}).get('output_dir'))
    if output is None:
        candidates = sorted((root / 'results').glob('*/request.json'), key=lambda p: p.stat().st_mtime)
        output = candidates[-1].parent if candidates else None
    if output is None:
        return result
    request = read(output / 'request.json')
    trajectory = read(local(request.get('trajectory')))
    result['plan'] = {'runtime': request.get('runtime', {}),
                      'repair_policy': request.get('repair_policy'),
                      'trajectory_mode': request.get('trajectory_mode'),
                      'trajectory': {key: trajectory.get(key, []) for key in ('anchors', 'segments')},
                      'frame_count': len(trajectory.get('frames', []))}
    for phase, filename in (('inference', 'infer'), ('reconstruction', 'distill'), ('plus', 'plus')):
        result['logs'][phase] = read(output / (filename + '.log'), tail=True)
    for phase in ('inference', 'plus'):
        teacher = read(output / (phase + '-teacher.json'))
        if teacher:
            result['teachers'][phase] = teacher
    result['reconstruction_complete'] = bool(read(output / 'distillation.json'))
    result['training'] = read(output / 'reconstruction-progress.json')
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(sys.argv[1], sys.argv[2])))
