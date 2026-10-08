"""G4Splat worker status, deadline and cooperative subprocess cancellation."""
import time
from pathlib import Path
from ..checkpoint import atomic_json
from ..jobs import read_json
from .repair import run_prepared


def execute(root):
    root = Path(root)
    request = read_json(root / 'worker-request.json')
    def publish(**body):
        atomic_json(root / 'worker-status.json', body)
    def stop():
        deadline = request.get('stop_at_epoch')
        return (root / 'STOP').exists() or (deadline is not None and time.time() >= float(deadline))
    try:
        if stop():
            raise InterruptedError('Cancelled before G4Splat')
        result = run_prepared(request['inputs'], request['output'], runtime=request['runtime'],
                              should_stop=stop, on_progress=lambda event: publish(status='running', **event))
        publish(status='completed', phase='finished', result=result)
    except InterruptedError as exc:
        publish(status='stopped', phase='finished', message=str(exc))
    except Exception as exc:
        publish(status='error', phase='finished', message=f'{type(exc).__name__}: {exc}')
