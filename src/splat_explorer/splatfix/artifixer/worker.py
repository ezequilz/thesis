"""Small LRZ job entry point; cancellation uses the queue's STOP-file contract."""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from ..checkpoint import atomic_json
from ..jobs import read_json
from .repair import run_repair


def execute(root):
    root = Path(root)
    request = read_json(root / 'worker-request.json')
    if request.get('stage') != 'benchmark':
        request.setdefault('runtime', {}).setdefault('artifixer_pipeline', 'legacy_gsplat')
    def publish(**body):
        atomic_json(root / 'worker-status.json', body)
    def stop():
        deadline = request.get('stop_at_epoch')
        if deadline is not None and time.time() >= float(deadline):
            (root / 'STOP').touch()
            return True
        return (root / 'STOP').exists()
    previous_timeout = os.environ.get('SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS')
    previous_mailbox = os.environ.get('SPLATFIX_VISER_MAILBOX')
    if request.get('stage') != 'benchmark':
        os.environ['SPLATFIX_VISER_MAILBOX'] = str(root / 'viser-capture')
    os.environ['SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS'] = str(request.get('save_stop_timeout_seconds', 240))
    try:
        if stop():
            raise InterruptedError('Cancelled before reconstruction')
        publish(status='running', phase='validation')
        preview = {}
        completed_results = []
        def progress(event):
            if event.get('output_dir'):
                for name in ('input', 'inference'):
                    marker = Path(event['output_dir']) / f'{name}-preview.json'
                    if marker.is_file():
                        preview[f'{name}_preview'] = str(marker)
            publish(status='running', completed_results=completed_results, **preview, **event)
        from .run_evaluation import finalize_run_evaluation
        if request.get('stage') == 'benchmark':
            from .benchmark import run_benchmark
            from .repeat_benchmark import run_repeat_benchmark
            runs = request.get('benchmark_runs', 1)
            if type(runs) is not int or runs < 1:
                raise ValueError('benchmark_runs must be a positive integer')
            prior = request['runtime'].get('repeat_from')
            for pass_number in range(1, runs + 1):
                if stop():
                    raise InterruptedError('Cancelled before benchmark pass')
                preview.clear()
                def pass_progress(event):
                    progress({**event, 'benchmark_pass': pass_number, 'benchmark_runs': runs,
                              'message': f"Benchmark pass {pass_number}/{runs}: {event.get('phase', 'running')}"})
                pass_progress({'phase': 'validation'})
                if prior:
                    result = run_repeat_benchmark(prior, request['output'],
                        resolution_profile=request['runtime'].get('resolution_profile', '720p'),
                        regularization_profile=request['runtime'].get('regularization_profile', 'artifixer'),
                        should_stop=stop, on_progress=pass_progress)
                else:
                    result = run_benchmark(request['source'], request['output'], runtime=request['runtime'],
                                           should_stop=stop, on_progress=pass_progress)
                finalize_run_evaluation(result['output_dir'], request['runtime'],
                                        should_stop=stop, on_progress=pass_progress)
                prior = result['output_dir']
                completed_results.append(str(Path(prior) / 'result.json'))
                pass_progress({'phase': 'pass_complete', 'output_dir': prior})
        else:
            result = run_repair(request['checkpoint'], request['output'], mode=request['mode'],
                                frames=request['frames'], span_fraction=request['span_fraction'],
                                runtime=request['runtime'], should_stop=stop, on_progress=progress)
            finalize_run_evaluation(result['output_dir'], request['runtime'], should_stop=stop, on_progress=progress)
        publish(status='completed', phase='finished', result=result, completed_results=completed_results)
    except InterruptedError as exc:
        from .interrupted import preserve_interrupted_results
        try:
            preserve_interrupted_results(request, str(exc))
        except Exception as preservation_error:
            publish(status='stopped', phase='finished', message=str(exc),
                    preservation_error=f'{type(preservation_error).__name__}: {preservation_error}')
        else:
            publish(status='stopped', phase='finished', message=str(exc))
    except Exception as exc:
        publish(status='error', phase='finished', message=f'{type(exc).__name__}: {exc}')
    finally:
        if previous_mailbox is None:
            os.environ.pop('SPLATFIX_VISER_MAILBOX', None)
        else:
            os.environ['SPLATFIX_VISER_MAILBOX'] = previous_mailbox
        if previous_timeout is None:
            os.environ.pop('SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS', None)
        else:
            os.environ['SPLATFIX_SAVE_STOP_TIMEOUT_SECONDS'] = previous_timeout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    execute(parser.parse_args().run_dir)
