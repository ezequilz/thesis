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
        def progress(event):
            if event.get('output_dir'):
                for name in ('input', 'inference'):
                    marker = Path(event['output_dir']) / f'{name}-preview.json'
                    if marker.is_file():
                        preview[f'{name}_preview'] = str(marker)
            publish(status='running', **preview, **event)
        if request.get('stage') == 'benchmark':
            if request['runtime'].get('repeat_from'):
                from .repeat_benchmark import run_repeat_benchmark
                result = run_repeat_benchmark(request['runtime']['repeat_from'], request['output'],
                                              resolution_profile=request['runtime'].get('resolution_profile', 'training'),
                                              regularization_profile=request['runtime'].get('regularization_profile', 'artifixer'),
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
