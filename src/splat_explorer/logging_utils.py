"""Per-run file logging.

`log_to_file` attaches a temporary FileHandler to the root logger, so every
log line emitted while it is active — from any module and any thread,
including tracebacks from `logger.exception` — also lands in the given file.
The episode loop uses it to leave an `episode.log` next to the frames of each
run, so broken runs can be diagnosed after the fact without docker logs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@contextmanager
def log_to_file(path: Path, level: int = logging.INFO) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(level)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
