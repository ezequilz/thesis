"""Low-frequency remote observation with responsive local cancellation."""
import time

POLL_SECONDS = 10


def wait_for_remote_poll(should_stop):
    """Wake early for a new stop; keep an already-stopping worker throttled."""
    stopping = should_stop()
    for _ in range(POLL_SECONDS):
        time.sleep(1)
        if not stopping and should_stop():
            break
