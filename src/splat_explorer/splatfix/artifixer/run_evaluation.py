"""Complete dashboard-run evaluation without discarding a successful reconstruction."""
from pathlib import Path
import os

from ..checkpoint import atomic_json
from .evaluation_chart import write_evaluation_chart
from .repair import run_worker, runtime_environment


def finalize_run_evaluation(root, runtime, *, should_stop=lambda: False, on_progress=lambda event: None):
    root = Path(root)
    if should_stop():
        raise InterruptedError('Cancelled before evaluation')
    on_progress({'phase': 'evaluation', 'output_dir': str(root), 'log': str(root / 'evaluation.log')})
    status = {'status': 'complete', 'errors': []}
    if (root / 'benchmark-run.json').is_file() and not (root / 'published-test-metrics.json').is_file():
        env = runtime_environment(runtime)
        env['PYTHONPATH'] += os.pathsep + str(Path(__file__).resolve().parents[3])
        try:
            run_worker([runtime['python'], '-m', 'splat_explorer.splatfix.artifixer.benchmark_evaluation',
                        str(root), '--repo', runtime['repo'], '--device', 'cuda'],
                       cwd=runtime['repo'], env=env, log_path=root / 'evaluation.log', should_stop=should_stop)
        except InterruptedError:
            raise
        except Exception as exc:
            status['errors'].append(f'Photographic metrics: {exc}')
    try:
        status['chart'] = write_evaluation_chart(root)
    except Exception as exc:
        status['errors'].append(f'Trajectory chart: {exc}')
    status['photographic_metrics'] = 'available' if (root / 'published-test-metrics.json').is_file() else 'unavailable'
    if status['errors']:
        status['status'] = 'partial'
    atomic_json(root / 'evaluation-status.json', status)
    return status
