"""Persistent remote worker. Only the generation child imports GPU libraries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import uuid


def atomic(path, body):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(body, indent=2))
    temp.replace(path)


def model_environment(base):
    # Exactly the account cache used by OracleGS, even with redirected HOME.
    import pwd
    home = Path(pwd.getpwuid(os.getuid()).pw_dir) / '.cache/huggingface'
    prefix = base / 'worldexplorer-env'
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'TORCH_CUDA_ARCH_LIST'):
        env.pop(key, None)
    cuda = base / 'worldexplorer-cuda'
    if not (cuda / 'bin/nvcc').exists():
        cuda = Path('/usr/local/cuda')
    env.update(HF_HOME=str(home), HF_HUB_CACHE=str(home/'hub'),
               HUGGINGFACE_HUB_CACHE=str(home/'hub'), HF_TOKEN_PATH=str(home/'token'),
               PYTHONNOUSERSITE='1', CUDA_HOME=str(cuda),
               PATH=f'{prefix}/bin:{cuda}/bin:' + env['PATH'],
               LD_LIBRARY_PATH=f'{prefix}/lib:{cuda}/lib:{cuda}/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
               CPATH=f'{cuda}/targets/x86_64-linux/include:{cuda}/include',
               LIBRARY_PATH=f'{cuda}/targets/x86_64-linux/lib:{cuda}/lib',
               TORCH_EXTENSIONS_DIR=str(base/'worldexplorer-cache/extensions'),
               MAX_JOBS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    return env


def inventory(run):
    files = []
    for folder in ('panoramas', 'scenes', 'nerfstudio_output'):
        for path in sorted((run/folder).rglob('*')):
            if path.is_file() and not path.is_symlink():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                        digest.update(chunk)
                files.append({'path':str(path.relative_to(run)), 'size':path.stat().st_size,
                              'sha256':digest.hexdigest()})
    return files


def validate_splat(path):
    with path.open('rb') as stream:
        raw = stream.read(65536)
    header, separator, _ = raw.partition(b'end_header')
    text = header.decode('ascii', errors='replace')
    if not separator or not text.startswith('ply\n') or not re.search(r'element vertex [1-9][0-9]*\s', text):
        raise RuntimeError(f'Invalid or empty Gaussian PLY export: {path.name}')
    fields = {line.split()[-1] for line in text.splitlines() if line.startswith('property ')}
    required = {'x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
    if not required.issubset(fields):
        raise RuntimeError(f'PLY export is missing Gaussian properties: {path.name}')


def manual_candidates(run):
    finals = list((run/'panoramas').glob('*/*/final'))
    if len(finals) != 1:
        raise RuntimeError('Expected exactly one manual panorama scaffold')
    folder = finals[0].parent
    originals = {f'{i:03}':str((folder/'final'/f'{i:03}.png').relative_to(run)) for i in (0,2,4,6)}
    groups = {f'{i:03}':[str(p.relative_to(run)) for p in sorted(folder.glob(f'{i:03}_blend_*.png'))]
              for i in (1,3,5,7)}
    if not all((run/p).is_file() for p in originals.values()) or not all(groups.values()):
        raise RuntimeError('Manual scaffold is incomplete; inspect generation log')
    return {'originals':originals, 'groups':groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('/workspace'))
    args = parser.parse_args()
    run, base = args.run_dir.resolve(), args.base.resolve()
    request = json.loads((run/'request.json').read_text())
    state = {'status':'running', 'phase':request['operation'], 'started_at':time.time()}
    atomic(run/'status.json', state)
    stop_heartbeat = threading.Event()
    def heartbeat():
        while not stop_heartbeat.is_set():
            atomic(run/'heartbeat.json', {'updated_at':time.time()})
            stop_heartbeat.wait(30)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if request['operation'] == 'setup':
            subprocess.run(['bash', str(Path(__file__).with_name('provision.sh')), str(base)], check=True)
            state.update(status='completed', phase='ready')
        else:
            ready = base/'worldexplorer-ready.json'
            if not ready.exists():
                raise RuntimeError('Prepare WorldExplorer runtime first.')
            shutil.copy2(ready, run/'runtime.json')
            repo = base/'third_party/worldexplorer'
            for name in ('model', 'predefined_trajectories'):
                (run/name).symlink_to(repo/name, target_is_directory=True)
            # Snapshot entrypoint/adapter so later local edits cannot change this job.
            shutil.copy2(repo/'worldexplorer.py', run/'worldexplorer.py')
            shutil.copy2(Path(__file__).with_name('runner.py'), run/'runner.py')
            subprocess.run([str(base/'worldexplorer-env/bin/python'), '-u', str(run/'runner.py')],
                           cwd=run, env=model_environment(base), stdin=subprocess.DEVNULL, check=True)
            files = inventory(run)
            if request['operation'] == 'generate' and request['mode'] == 'manual':
                candidates = manual_candidates(run)
                atomic(run/'candidates.json', candidates)
                atomic(run/'artifacts.json', {'files':files, 'splats':[]})
                state.update(status='awaiting_selection', phase='Choose one candidate for 001, 003, 005 and 007')
            else:
                splats = [f['path'] for f in files if f['path'].endswith('/splat.ply') and f['size'] > 100]
                if not splats:
                    raise RuntimeError('Generation exited without an exported splat; retained all diagnostics.')
                for splat in splats:
                    validate_splat(run/splat)
                atomic(run/'artifacts.json', {'files':files, 'splats':splats})
                state.update(status='completed', phase='complete', splats=splats)
    except Exception as exc:
        state.update(status='failed', error=str(exc))
    finally:
        stop_heartbeat.set()
        thread.join()
        state['finished_at'] = time.time()
        atomic(run/'status.json', state)
    return 0 if state['status'] in ('completed', 'awaiting_selection') else 1

if __name__ == '__main__':
    sys.exit(main())
