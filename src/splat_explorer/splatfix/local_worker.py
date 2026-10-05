"""Queue-owned local CLI stage that exits if its manager process disappears."""
from __future__ import annotations
import argparse
import os
import signal
import sys
import threading


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--owner-pid', required=True, type=int)
    args, remaining = parser.parse_known_args()
    if os.getpgrp() != os.getpid():
        raise RuntimeError('Local queue worker requires its own process group')
    def watch_parent():
        while os.getppid() == args.owner_pid:
            threading.Event().wait(.5)
        os.killpg(os.getpgrp(), signal.SIGTERM)
    threading.Thread(target=watch_parent, daemon=True).start()
    sys.argv = [sys.argv[0], *remaining]
    from .cli import main as cli_main
    cli_main()


if __name__ == '__main__':
    main()
