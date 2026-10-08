"""Dispatch durable GPU requests to their selected reconstruction method."""
import argparse
from pathlib import Path
from .checkpoint import atomic_json
from .jobs import read_json
from .methods import get_method


def execute(root):
    root = Path(root)
    try:
        request = read_json(root / 'worker-request.json')
        if not isinstance(request, dict):
            raise ValueError('Missing or invalid worker request')
        method = get_method(request.get('reconstruction_method'), stage=request.get('stage', 'repair'))
    except ValueError as exc:
        atomic_json(root / 'worker-status.json', {'status': 'error', 'phase': 'finished', 'message': str(exc)})
        return
    return method.execute_worker(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    execute(parser.parse_args().run_dir)
