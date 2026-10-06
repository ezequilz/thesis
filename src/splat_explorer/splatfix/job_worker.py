"""Small LRZ job entry point; cancellation uses the queue's STOP-file contract."""
from __future__ import annotations
import argparse
from pathlib import Path
from .checkpoint import atomic_json
from .jobs import read_json
from .repair import run_repair


def execute(root):
    root = Path(root)
    request = read_json(root / 'worker-request.json')
    def publish(**body):
        atomic_json(root / 'worker-status.json', body)
    stop = lambda: (root / 'STOP').exists()
    try:
        if stop():
            raise InterruptedError('Cancelled before reconstruction')
        publish(status='running', phase='validation')
        progress = lambda event: publish(status='running', **event)
        if request.get('stage') == 'benchmark':
            if request['runtime'].get('repeat_from'):
                from .repeat_benchmark import run_repeat_benchmark
                result = run_repeat_benchmark(request['runtime']['repeat_from'], request['output'],
                                              resolution_profile=request['runtime'].get('resolution_profile', 'training'),
                                              should_stop=stop, on_progress=progress)
            else:
                from .benchmark import run_benchmark
                result = run_benchmark(request['source'], request['output'], runtime=request['runtime'],
                                       should_stop=stop, on_progress=progress)
        else:
            result = run_repair(request['checkpoint'], request['output'], mode=request['mode'],
                                frames=request['frames'], span_fraction=request['span_fraction'],
                                runtime=request['runtime'], should_stop=stop, on_progress=progress)
        from .run_evaluation import finalize_run_evaluation
        finalize_run_evaluation(result['output_dir'], request['runtime'], should_stop=stop, on_progress=progress)
        publish(status='completed', phase='finished', result=result)
    except InterruptedError as exc:
        publish(status='stopped', phase='finished', message=str(exc))
    except Exception as exc:
        publish(status='error', phase='finished', message=f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    execute(parser.parse_args().run_dir)
