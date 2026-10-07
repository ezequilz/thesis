"""Process-local lifecycle hook for saving a stopped upstream trainer.

No optimizer, loss, iteration count or scheduled checkpoint changes. The hook
runs after a complete iteration, before the next iteration, outside a signal
handler. The authors' checkout stays untouched.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

STOP_EXIT_CODE = 75


def atomic_json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(body, indent=2) + '\n')
    os.replace(temporary, path)


def save_state(trainer, destination):
    import torch
    from threedgrut.export.ply_exporter import PLYExporter
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / 'stop-state.json'
    status = {'status': 'saving', 'incomplete': True, 'step': int(trainer.global_step),
              'started_at_epoch': time.time(), 'checkpoint': None, 'splat_path': None,
              'checkpoint_policy': 'On stop only; scheduled checkpoints unchanged',
              'resume_limitation': 'Authors checkpoint format; exact optimizer/RNG continuation is not guaranteed'}
    atomic_json(status_path, status)
    try:
        torch.cuda.synchronize()
        trainer.save_checkpoint()
        checkpoint = Path(trainer.tracking.output_dir) / f'ours_{trainer.global_step}' / f'ckpt_{trainer.global_step}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Upstream trainer did not write {checkpoint}')
        status['checkpoint'] = str(checkpoint.resolve())
        atomic_json(status_path, status)
        temporary = destination / 'interrupted.tmp.ply'
        PLYExporter().export(trainer.model, temporary)
        if not temporary.is_file():
            raise FileNotFoundError('PLY exporter did not write the interrupted splat')
        final = destination / 'interrupted.ply'
        os.replace(temporary, final)
        status.update(status='saved', splat_path=str(final.resolve()), finished_at_epoch=time.time())
    except Exception as exc:
        status.update(status='save_failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        writer = getattr(trainer.tracking, 'writer', None)
        if writer is not None:
            try:
                writer.flush()
            except Exception as exc:
                status['log_flush_error'] = str(exc)
        atomic_json(status_path, status)
    print('SPLATFIX_SAVE_ON_STOP: ' + json.dumps(status), flush=True)
    return status


def install_hook(trainer_class, control, destination, *, saver=save_state):
    original = trainer_class.render_gui
    def iteration_boundary(trainer, *args, **kwargs):
        # Pinned trainer calls render_gui after optimizer/scheduler updates,
        # logging and its normal scheduled-checkpoint check on every iteration.
        if Path(control).exists():
            saver(trainer, destination)
            raise SystemExit(STOP_EXIT_CODE)
        return original(trainer, *args, **kwargs)
    trainer_class.render_gui = iteration_boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if command[:1] == ['-u']:
        command = command[1:]
    sys.path.insert(0, os.getcwd())
    from threedgrut.trainer import Trainer3DGRUT
    install_hook(Trainer3DGRUT, args.control, args.snapshot)
    print('SPLATFIX_SAVE_ON_STOP: iteration-boundary hook installed', flush=True)
    if command[:1] == ['-m']:
        sys.argv = command[1:]
        runpy.run_module(command[1], run_name='__main__', alter_sys=True)
    else:
        sys.argv = command
        sys.path.insert(0, str(Path(command[0]).resolve().parent))
        runpy.run_path(command[0], run_name='__main__')


if __name__ == '__main__':
    main()
