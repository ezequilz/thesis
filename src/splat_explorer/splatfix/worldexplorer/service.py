"""Laptop-side durable job catalog and detached DSS execution."""
from __future__ import annotations
import hashlib
import functools
import threading
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import time
import sys
import uuid
from . import DEFAULT_PROMPT, REVISION
from .worker import atomic
from ... import repair_lrz as lrz


def rsync_binary():
    # macOS's openrsync dispatcher can misforward -e with checksum/partial flags.
    legacy = Path('/usr/libexec/rsync/rsync.samba')
    return str(legacy) if sys.platform == 'darwin' and legacy.is_file() else 'rsync'


def serialized(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


def validate(body):
    if not isinstance(body, dict):
        raise ValueError('Expected an object')
    operation = body.get('operation', 'generate')
    if operation not in ('generate', 'setup', 'expand'):
        raise ValueError('Unknown operation')
    prompt = body.get('prompt', DEFAULT_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError('Prompt must contain 1–2000 characters')
    if any(ord(c) < 32 for c in prompt):
        raise ValueError('Prompt must be a single line')
    mode = body.get('mode', 'automatic')
    if mode not in ('automatic', 'fast', 'manual'):
        raise ValueError('Mode must be automatic, fast or manual')
    scale = float(body.get('translation_scaling', 3.0))
    if not math.isfinite(scale) or not 0 < scale <= 20:
        raise ValueError('Translation scaling must be > 0 and <= 20')
    images = int(body.get('num_images_for_vggt', 40))
    if not 32 <= images <= 100:
        raise ValueError('VGGT image budget must be 32–100 to sample all 32 trajectories')
    return dict(operation=operation, prompt=prompt.strip(), mode=mode,
                translation_scaling=scale, num_images_for_vggt=images, revision=REVISION)


def launch_command(cfg, run_id):
    remote = f"{cfg['workspace']}/worldexplorer-runs/{run_id}"
    worker = (f"python -u /workspace/worldexplorer-runs/{run_id}/control/worker.py "
              f"--run-dir /workspace/worldexplorer-runs/{run_id}")
    # Separate container name; reuse the squashfs and persistent DSS mount.
    isolated = {**cfg, 'container_name':'worldexplorer'}
    srun = lrz.container_srun_prefix(isolated) + 'bash -lc ' + shlex.quote(worker)
    lock = shlex.quote(f"{cfg['workspace']}/worldexplorer-runs/gpu-{cfg['job_id']}.lock")
    script = (f'flock -n {lock} {srun}; code=$?; '
              f"printf '{{\"returncode\":%s}}' \"$code\" > {shlex.quote(remote+'/exit.tmp')}; "
              f"mv {shlex.quote(remote+'/exit.tmp')} {shlex.quote(remote+'/exit.json')}")
    return (f'nohup bash -c {shlex.quote(script)} > {shlex.quote(remote+"/worker.log")} 2>&1 < /dev/null & '
            f'echo $! > {shlex.quote(remote+"/launcher.pid")}')


class WorldExplorerService:
    def __init__(self, root, *, auto_sync=False):
        self._lock = threading.RLock()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._reconcile_guard = threading.Lock()
        self._reconcile_again = False
        self._last_reconcile = 0.0
        if auto_sync:
            self.request_reconcile(force=True)

    def request_reconcile(self, *, force=False):
        if not force and time.monotonic()-self._last_reconcile < 60:
            return {'started':False}
        if not self._reconcile_guard.acquire(blocking=False):
            if force:
                self._reconcile_again = True
            return {'started':False}
        self._last_reconcile = time.monotonic()
        def execute():
            try:
                self.reconcile()
            finally:
                self._reconcile_guard.release()
                if self._reconcile_again:
                    self._reconcile_again = False
                    self.request_reconcile(force=True)
        threading.Thread(target=execute, daemon=True, name='worldexplorer-results').start()
        return {'started':True}

    def reconcile(self):
        for job in self.jobs():
            if job.get('remote_cleaned_at'):
                continue
            try:
                job = self.refresh(job['id'])
                if not job.get('connection_error') and job['status'] in ('completed','failed','awaiting_selection'):
                    self.download(job['id'])
            except Exception as exc:
                path = self.directory(job['id'])
                with self._lock:
                    current = json.loads((path/'job.json').read_text())
                    current['persistence_error'] = str(exc)
                    self._save(path, current)


    def directory(self, run_id):
        if not re.fullmatch(r'we_[0-9]{8}_[0-9]{6}_[a-f0-9]{12}', str(run_id)):
            raise ValueError('Invalid WorldExplorer run ID')
        path = self.root/run_id
        if not (path/'job.json').is_file():
            raise FileNotFoundError('Unknown WorldExplorer job')
        return path

    def jobs(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob('we_*/job.json'), reverse=True)]

    def _save(self, path, job):
        atomic(path/'job.json', job)
        return job

    def _ssh(self, cfg, command, *, timeout=30):
        result = lrz._ssh_run(cfg, command, timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'LRZ command failed')
        return result.stdout

    @serialized
    def create(self, body, *, scaffold=None, selection=None):
        request = validate(body)
        if request['operation'] == 'expand' and scaffold is None:
            raise ValueError('Expand requires a completed manual selection')
        cfg = lrz.load_lrz_config()
        if not str(cfg['job_id']).isdigit():
            raise ValueError('Select a reserved GPU on /repair/gpu first')
        if not lrz.lrz_session_alive(cfg):
            raise RuntimeError(lrz.session_required_message())
        # One scheduler check at submission only, never in dashboard polling.
        row = self._ssh(cfg, f"squeue --me --job={cfg['job_id']} -h -o '%t %L'").strip().split()
        if len(row) != 2 or row[0] != 'R':
            raise RuntimeError('Selected allocation is not running')
        if request['operation'] != 'setup':
            try:
                self._ssh(cfg, 'test -f '+shlex.quote(cfg['workspace']+'/worldexplorer-ready.json'))
            except RuntimeError as exc:
                raise RuntimeError('WorldExplorer runtime is not ready. Prepare it and inspect the setup log first.') from exc
        run_id = time.strftime('we_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:12]
        path = self.root/run_id
        path.mkdir()
        remote = f"{cfg['workspace']}/worldexplorer-runs/{run_id}"
        job = dict(id=run_id, request=request, remote=remote, connection=cfg,
                   status='submitting', created_at=time.time())
        self._save(path, job)
        atomic(path/'request.json', request)
        if scaffold is not None:
            inputs = path/'input_scaffold'
            inputs.mkdir()
            for name, source in scaffold.items():
                shutil.copy2(source, inputs/(name+'.png'))
            atomic(path/'selection.json', selection)
            job['source_run'] = selection['source_run']
            self._save(path, job)
        try:
            self._ssh(cfg, 'mkdir -p '+shlex.quote(remote+'/control'))
            lrz._mux_run([rsync_binary(),'-az','--exclude=__pycache__','-e',lrz.rsync_ssh_cmd(cfg),
                          str(Path(__file__).parent)+'/',
                          f"{cfg['user']}@{cfg['host']}:{remote}/control/"])
            lrz._mux_run([rsync_binary(),'-az','-e',lrz.rsync_ssh_cmd(cfg),str(path/'request.json'),
                          f"{cfg['user']}@{cfg['host']}:{remote}/request.json"])
            if scaffold is not None:
                lrz._mux_run([rsync_binary(),'-az','-e',lrz.rsync_ssh_cmd(cfg),
                              str(path/'input_scaffold'), str(path/'selection.json'),
                              f"{cfg['user']}@{cfg['host']}:{remote}/"])
            # After this point a timeout has an uncertain outcome; do not resubmit.
            job['status'] = 'launch_unknown'
            self._save(path, job)
            self._ssh(cfg, launch_command(cfg, run_id))
            job['status'] = 'submitted'
        except Exception as exc:
            if job['status'] == 'submitting':
                job['status'] = 'failed'
            job['error'] = str(exc)
        result = self._save(path, job)
        self.request_reconcile(force=True)
        return result

    @serialized
    def refresh(self, run_id):
        path = self.directory(run_id)
        job = json.loads((path/'job.json').read_text())
        if job.get('remote_cleaned_at'):
            return job
        cfg, remote = job['connection'], job['remote']
        # Read filesystem state only; no squeue/sinfo loop.
        script = """import json,pathlib,sys,os
p=pathlib.Path(sys.argv[1]); result={'missing':not p.exists()}
for name in ('status.json','exit.json','heartbeat.json'):
 if (p/name).exists(): result[name]=json.loads((p/name).read_text())
if (p/'launcher.pid').exists():
 try: os.kill(int((p/'launcher.pid').read_text()),0); result['alive']=True
 except ProcessLookupError: result['alive']=False
log=p/'worker.log'
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-24000)); result['log']=f.read().decode(errors='replace')
print(json.dumps(result))
"""
        try:
            result = json.loads(self._ssh(cfg, 'python3 -c '+shlex.quote(script)+' '+shlex.quote(remote)))
            if result.get('missing') and job.get('cleanup_requested_at') and job.get('downloaded_at'):
                job['remote_cleaned_at'] = time.time()
                job['persistence'] = 'Verified locally · remote run files removed'
                job['connection_error'] = None
                job.pop('persistence_error', None)
                return self._save(path, job)
            job.update(result.get('status.json', {}))
            if result.get('status.json', {}).get('status') in ('running','completed','awaiting_selection'):
                job.pop('error', None)
            heartbeat = result.get('heartbeat.json', {}).get('updated_at', 0)
            if 'exit.json' in result and (result['exit.json']['returncode'] or job['status'] not in ('completed','failed','awaiting_selection')):
                job.update(status='failed', error=job.get('error') or f"Remote launcher exited ({result['exit.json']['returncode']}); inspect log.")
            elif result.get('alive') is False and time.time()-heartbeat > 180 and job['status'] not in ('completed','failed','awaiting_selection'):
                job.update(status='interrupted', error='Launcher exited without completion; outputs remain on DSS.')
            job['log'] = result.get('log', '')
            job['connection_error'] = None
            job['checked_at'] = time.time()
        except Exception as exc:
            # A disconnected laptop is not a failed remote generation.
            job['connection_error'] = str(exc)
        return self._save(path, job)

    @serialized
    def download(self, run_id):
        path = self.directory(run_id)
        job = self.refresh(run_id)
        if job.get('remote_cleaned_at'):
            return job
        if job.get('connection_error'):
            raise RuntimeError(job['connection_error'])
        if job['status'] not in ('completed','failed','interrupted','awaiting_selection'):
            raise ValueError('Wait for the remote job to finish before downloading')
        cfg = job['connection']
        target = path/'results'
        target.mkdir(exist_ok=True)
        job['persistence'] = 'Preparing verified archive'
        self._save(path, job)
        from .archive import MANIFEST_SCRIPT, DELETE_SCRIPT
        digest = self._ssh(cfg, 'python3 -c '+shlex.quote(MANIFEST_SCRIPT)+' '+shlex.quote(job['remote']), timeout=3600).strip()
        job['persistence'] = 'Downloading results'
        self._save(path, job)
        # Never dereference model/trajectory/runtime links into the result archive.
        lrz._mux_run([rsync_binary(),'-azc','--partial','-e',lrz.rsync_ssh_cmd(cfg),
                      '--exclude=/model','--exclude=/predefined_trajectories',
                      f"{cfg['user']}@{cfg['host']}:{job['remote']}/",str(target)+'/'])
        archive = target/'.archive-manifest.json'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Archive manifest transfer mismatch; remote files retained')
        job['persistence'] = 'Verifying local results'
        self._save(path, job)
        for record in json.loads(archive.read_text())['files']:
            file = target/record['path']
            if not file.resolve().is_relative_to(target.resolve()):
                raise ValueError('Archive file escapes result directory')
            if 'link' in record:
                import os
                if not file.is_symlink() or os.readlink(file) != record['link']:
                    raise RuntimeError('Archive symlink verification failed')
            else:
                checksum = hashlib.sha256()
                with file.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                        checksum.update(chunk)
                if file.stat().st_size != record['size'] or checksum.hexdigest() != record['sha256']:
                    raise RuntimeError(f'Artifact verification failed: {record["path"]}')
        manifest = target/'artifacts.json'
        if job['request']['operation'] != 'setup' and job['status'] in ('completed','awaiting_selection'):
            archive_records = {r['path']:r for r in json.loads(archive.read_text())['files']}
            for record in json.loads(manifest.read_text())['files']:
                archived = archive_records.get(record['path'], {})
                if archived.get('sha256') != record['sha256'] or archived.get('size') != record['size']:
                    raise RuntimeError(f'Generated artifact changed before archive: {record["path"]}')
        if job['status'] == 'awaiting_selection':
            job['candidates'] = json.loads((target/'candidates.json').read_text())
        job['downloaded_at'] = time.time()
        job['local_results'] = str(target)
        job['files'] = [str(p.relative_to(target)) for p in target.rglob('*')
                        if p.is_file() and not p.is_symlink() and p.suffix in ('.ply','.json','.log','.png','.mp4','.yml')]
        job['archive_sha256'] = digest
        job['cleanup_requested_at'] = time.time()
        job['persistence'] = 'Verified locally; cleaning remote run files'
        job.pop('persistence_error', None)
        self._save(path, job)
        self._ssh(cfg, 'python3 -c '+shlex.quote(DELETE_SCRIPT)+' '+shlex.quote(job['remote'])+' '+shlex.quote(digest), timeout=3600)
        job['remote_cleaned_at'] = time.time()
        job['persistence'] = 'Verified locally · remote run files removed'
        return self._save(path, job)

    @serialized
    def expand(self, run_id, selections):
        path = self.directory(run_id)
        job = json.loads((path/'job.json').read_text())
        if job['status'] != 'awaiting_selection' or not job.get('candidates'):
            raise ValueError('Download a finished manual scaffold before selecting candidates')
        if not isinstance(selections, dict) or set(selections) != {'001','003','005','007'}:
            raise ValueError('Choose exactly one image for 001, 003, 005 and 007')
        candidates = job['candidates']
        for index, relative in selections.items():
            if relative not in candidates['groups'][index]:
                raise ValueError(f'Image is not a candidate for {index}')
        selected = {**candidates['originals'], **selections}
        records = {r['path']:r for r in json.loads((path/'results/artifacts.json').read_text())['files']}
        scaffold = {}
        for index, relative in selected.items():
            source = self.file(run_id, relative)
            if hashlib.sha256(source.read_bytes()).hexdigest() != records[relative]['sha256']:
                raise ValueError('Selected image changed since generation; download again')
            scaffold[index] = source
        return self.create({**job['request'], 'operation':'expand'}, scaffold=scaffold,
                           selection={'source_run':run_id, 'images':selected,
                                      'sha256':{k:records[v]['sha256'] for k,v in selected.items()}})

    def file(self, run_id, relative):
        root = self.directory(run_id)/'results'
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid result file')
        return path
