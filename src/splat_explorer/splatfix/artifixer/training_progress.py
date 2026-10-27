"""Standalone, scoped reporting of real upstream optimizer iterations."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


@contextmanager
def report_training_progress(path, total, *, enabled=True):
    if not enabled:
        yield
        return
    from data_processing import threedgrut_training as training
    original = training.Trainer3DGRUT
    path = Path(path)
    last_write, iteration = 0., 0

    def publish(status, *, force=False):
        nonlocal last_write
        now = time.monotonic()
        if not force and now - last_write < 5:
            return
        last_write = now
        temporary = path.with_suffix('.json.tmp')
        try:
            temporary.write_text(json.dumps({'iteration': iteration, 'total': total,
                'status': status, 'updated_at': time.time()}))
            os.replace(temporary, path)
        except OSError:
            pass  # Observability must not interrupt optimization.

    class ReportingTrainer(original):
        def log_training_iter(self, *args, **kwargs):
            nonlocal iteration
            result = super().log_training_iter(*args, **kwargs)
            iteration = int(self.global_step)
            publish('training', force=iteration >= total)
            return result

    training.Trainer3DGRUT = ReportingTrainer
    publish('initializing', force=True)
    try:
        yield
    except BaseException:
        publish('interrupted', force=True)
        raise
    else:
        iteration = total
        publish('trained', force=True)
    finally:
        training.Trainer3DGRUT = original
