"""Bounded read-only cache observer for workers launched before live previews existed.

Usage: python -m splat_explorer.splatfix.oraclegs.preview_cache
       --run-dir /local/run --remote-dir /remote/workspace/splatfix-jobs/run
No SSH status polling, job launch, cancellation, model execution or lease changes.
"""
import argparse
from pathlib import Path
import time

from ..jobs import read_json
from .execution import mirror_previews


def observe(root, remote, *, max_seconds=43200):
    from ... import repair_lrz as lrz
    cfg = lrz.load_lrz_config()
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Run directory does not exist')
    endpoint = f"{cfg['user']}@{cfg['host']}"
    until = time.monotonic() + max_seconds
    while time.monotonic() < until:
        state = read_json(root / 'worker-status.json', {})
        terminal = state.get('status') in ('completed','error','stopped')
        phase = state.get('phase')
        if phase and phase != 'setup':
            try:
                mirror_previews(lrz, cfg, endpoint, remote, root, phase=phase)
            except Exception:
                pass  # Failure is recorded in the local cache status for retry.
        if terminal:
            return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--remote-dir', required=True)
    parser.add_argument('--max-seconds', type=int, default=43200)
    args = parser.parse_args()
    observe(args.run_dir, args.remote_dir, max_seconds=args.max_seconds)
