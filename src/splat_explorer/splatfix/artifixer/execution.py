"""ArtiFixer LRZ staging, execution, reattachment and cache publication."""
import json
import os
import shlex
import time
from pathlib import Path
from ..checkpoint import Checkpoint, atomic_json


def persist_checkpoint_caches(downloaded, checkpoint_root):
    """Publish reusable GPU caches without dereferencing remote source links."""
    import shutil
    import tempfile
    from .repair import digest_file
    downloaded, checkpoint_root = Path(downloaded), Path(checkpoint_root)
    trajectories = downloaded / 'trajectories'
    if trajectories.is_dir():
        # Metric alignment lives inside each trajectory. Its measurement image
        # links are temporary inputs, not replay data. Omit them rather than
        # dereferencing links to the old container or preserving dangling links.
        shutil.copytree(trajectories, checkpoint_root / 'trajectories',
                        dirs_exist_ok=True, symlinks=True,
                        ignore=lambda directory, names: [name for name in names
                            if (Path(directory) / name).is_symlink()])
    captions = downloaded / 'captions'
    if not captions.is_dir():
        return
    destination = checkpoint_root / 'captions'
    destination.mkdir(exist_ok=True)
    for entry in sorted(captions.iterdir()):
        if entry.name.startswith('.') or entry.is_symlink() or not entry.is_dir():
            continue
        manifest, artifact = entry / 'manifest.json', entry / 'caption.h5'
        if not manifest.is_file() or not artifact.is_file():
            raise ValueError('Committed caption cache is missing its manifest or HDF5')
        if manifest.is_symlink() or artifact.is_symlink():
            raise ValueError('Caption cache manifest and HDF5 must be regular files')
        metadata = json.loads(manifest.read_text())
        if metadata.get('signature') != entry.name or digest_file(artifact) != metadata.get('sha256'):
            raise ValueError('Downloaded caption cache failed signature/checksum validation')
        target = destination / entry.name
        if target.exists():
            if target.is_symlink() or (target / 'caption.h5').is_symlink() or digest_file(target / 'caption.h5') != metadata['sha256']:
                raise ValueError('Existing canonical caption cache conflicts with downloaded artifact')
            existing = json.loads((target / 'manifest.json').read_text())
            if any(existing.get(key) != metadata.get(key) for key in ('signature', 'sha256', 'recipe')):
                raise ValueError('Existing canonical caption manifest conflicts with downloaded artifact')
            continue
        temporary = Path(tempfile.mkdtemp(prefix='.' + entry.name + '.', dir=destination))
        try:
            # source/ contains only temporary input links. The HDF5 and recipe
            # manifest are sufficient for cache replay in a different job.
            shutil.copy2(manifest, temporary / 'manifest.json')
            shutil.copy2(artifact, temporary / 'caption.h5')
            if digest_file(temporary / 'caption.h5') != metadata['sha256']:
                raise ValueError('Caption HDF5 changed during cache publication')
            os.rename(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def execute_remote(executor, run_id, options, root, stop, update):
    from ... import repair_lrz as lrz
    from ...scene_runs.lrz_transport import LrzSceneRunTransport
    benchmark = options['stage'] == 'benchmark'
    cp = None if benchmark else Checkpoint.load(options['checkpoint'])
    if cp is not None and options['mode'] == 'edited':
        cp.require_viser_images()
    if cp is not None and (not cp.complete or (options['mode'] == 'edited' and any(not v.get('repaired_rgb') for v in cp.views))):
        raise ValueError('Checkpoint is not ready for the requested reconstruction mode')
    cfg = lrz.load_lrz_config()
    saved = executor.store.get_run(run_id).state.details
    # Reattachment only needs DSS/SSH, even if the original allocation expired.
    if not saved.get('remote_dir'):
        if options.get('repeat_from'):
            marker = lrz.read_remote_setup_marker(cfg)
            if not marker.get('ok') or str(marker.get('job_id')) != str(cfg['job_id']):
                update(phase='gpu_setup', message='Preparing the new GPU allocation for repeat repair')
                lrz.setup_lrz_gpu({**cfg, 'artifixer_required': True, 'artifixer_model': options['model']},
                                  qwen_required=False)
        LrzSceneRunTransport(executor.cfg, run_id, root).validate()
    remote_dir = saved.get('remote_dir') or str(cfg['workspace']).rstrip('/') + '/splatfix-jobs/' + run_id
    container_dir = '/workspace/splatfix-jobs/' + run_id
    endpoint = f"{cfg['user']}@{cfg['host']}"
    def ssh(command):
        answer = lrz._ssh_run(cfg, command, timeout=30)
        if answer.returncode:
            raise RuntimeError((answer.stderr or answer.stdout or 'Remote command failed').strip())
        return answer.stdout
    def sync(source, destination):
        lrz._mux_run(['rsync', '-az', '-e', lrz.rsync_ssh_cmd(cfg), source, destination])
    resuming = bool(executor.store.get_run(run_id).state.details.get('remote_dir'))
    if resuming:
        resuming = bool(ssh('test ! -d ' + shlex.quote(remote_dir + '/launched') + ' || printf launched').strip())
    if not resuming:
        update(message='Staging checkpoint and source on the configured LRZ allocation', phase='staging')
        lrz.sync_code_to_dss(cfg)
        from .repair import DEFAULT_RUNTIME
        runtime = {**DEFAULT_RUNTIME, **dict(executor.cfg.get('splatfix', {}).get('runtime', {}))}
        runtime['resolution_profile'] = options['resolution_profile']
        runtime['regularization_profile'] = options.get('regularization_profile', 'artifixer')
        runtime['split_mode'] = options.get('split_mode', 'double-split')
        if benchmark:
            from ..jobs import registered_benchmark
            source, _ = registered_benchmark(Path(executor.cfg.output.dir) / 'benchmarks', options['source'])
            from ...scene_runs_ext.config import model_runtime
            runtime = model_runtime(options['model'], runtime)
            if options.get('repeat_from'):
                from .resume import resolve_remote_preparation
                prior = executor.store.get_run(options['repeat_from'])
                if prior.state.status.value != 'completed' or prior.config.splatfix['model'] != options['model']:
                    raise ValueError('Repeat requires a completed benchmark with the same model')
                runtime['repeat_from'] = resolve_remote_preparation(
                    executor.store, {**options, 'preparation_from': options['repeat_from']}, cfg, ssh)
            if options.get('resume_from'):
                from .resume import resolve_remote_resume
                runtime['resume_from'] = resolve_remote_resume(executor.store, options, cfg, ssh)
            if options.get('preparation_from'):
                from .resume import resolve_remote_preparation
                runtime['preparation_from'] = resolve_remote_preparation(executor.store, options, cfg, ssh)
            ssh('mkdir -p ' + shlex.quote(remote_dir + '/benchmark-input'))
            sync(str(source) + '/', endpoint + ':' + remote_dir + '/benchmark-input/')
            request = {'stage': 'benchmark', 'source': container_dir + '/benchmark-input',
                       'output': container_dir + '/results', 'runtime': runtime}
        else:
            ssh('mkdir -p ' + shlex.quote(remote_dir + '/checkpoint') + ' ' + shlex.quote(remote_dir + '/source'))
            sync(str(cp.root) + '/', endpoint + ':' + remote_dir + '/checkpoint/')
            source = Path(cp.manifest['scene_path'])
            from ...scene.catalog import openable_scene_path
            source = openable_scene_path(source)
            if not source.exists() and not list((cp.root / 'trajectories').glob('*/trajectory.json')):
                raise FileNotFoundError('Source scene is unavailable for GPU trajectory preparation')
            directory = source if source.is_dir() else source.parent if source.name in ('meta.json', 'lod-meta.json') else None
            if directory:
                sync(str(directory) + '/', endpoint + ':' + remote_dir + '/source/')
                remote_scene = container_dir + '/source/' + (source.name if source.is_file() else '')
            else:
                if source.exists():
                    sync(str(source), endpoint + ':' + remote_dir + '/source/')
                remote_scene = container_dir + '/source/' + source.name
            runtime['scene_path'] = remote_scene
            request = {'checkpoint': container_dir + '/checkpoint', 'output': container_dir + '/results',
                       'runtime': runtime, **{k: options[k] for k in ('mode', 'frames', 'span_fraction')}}
        request['reconstruction_method'] = 'artifixer'
        request['stop_at_epoch'] = executor.store.get_run(run_id).state.details.get('effective_deadline')
        safety = executor.cfg.get('splatfix', {})
        buffer = float(safety.get('stop_before_gpu_end_seconds', 300))
        timeout = float(safety.get('save_stop_timeout_seconds', 240))
        if not 0 < timeout < buffer:
            raise ValueError('save_stop_timeout_seconds must be positive and below stop_before_gpu_end_seconds')
        request['save_stop_timeout_seconds'] = timeout
        atomic_json(root / 'worker-request.json', request)
        sync(str(root / 'worker-request.json'), endpoint + ':' + remote_dir + '/worker-request.json')
        inner = lrz._remote_pythonpath_exports(cfg) + shlex.join(['python', '-u', '-m', 'splat_explorer.splatfix.job_worker', '--run-dir', container_dir])
        command = lrz.container_srun_prefix(cfg) + 'bash -lc ' + shlex.quote(inner)
        # Stable job directory and exclusive marker prevent a restarted manager from replaying GPU work.
        wrapped = command + "; printf '%s\\n' $? > " + shlex.quote(remote_dir + '/launcher-exit')
        launcher = ('mkdir ' + shlex.quote(remote_dir + '/launched') + ' && (nohup bash -lc ' + shlex.quote(wrapped)
                    + ' > ' + shlex.quote(remote_dir + '/stage.log') + ' 2>&1 < /dev/null &)')
        if stop():
            raise InterruptedError('Cancelled before GPU launch')
        # Save location before launch so a manager restart can reattach safely.
        update(phase='reconstruction', message='Authors’ fresh reconstruction running on LRZ', remote_dir=remote_dir)
        ssh(launcher)
    from .live_logs import PhaseLogMirror
    log_mirror = PhaseLogMirror(root, container_dir, remote_dir, ssh)
    from .inference_preview import InferencePreviewMirror
    preview_mirror = InferencePreviewMirror(root, container_dir, remote_dir, endpoint,
                                            lrz.rsync_ssh_cmd(cfg), lrz._mux_run)
    last = None
    started = time.monotonic()
    from ..viser_capture import CaptureMirror
    def capture_stop():
        deadline = executor.store.get_run(run_id).state.details.get('effective_deadline')
        if deadline and time.time() >= float(deadline):
            executor.store.request_stop(run_id)
            return True
        return stop()
    capture_mirror = None if benchmark else CaptureMirror(
        root, remote_dir, container_dir, cp, endpoint, ssh, sync,
        url=executor.cfg.get('renderer', {}).get('viser_url') or None, should_stop=capture_stop, progress=update)
    try:
        while True:
            deadline = executor.store.get_run(run_id).state.details.get('effective_deadline')
            if deadline and time.time() >= float(deadline) and not stop():
                executor.store.request_stop(run_id)
                update(status='stopping', message='GPU deadline approaching; saving current reconstruction before stopping',
                       stop_reason='gpu_deadline')
            if stop():
                ssh('touch ' + shlex.quote(remote_dir + '/STOP'))
            elif capture_mirror is not None:
                capture_mirror.poll()
            raw = ssh('if [ -f ' + shlex.quote(remote_dir + '/worker-status.json') + ' ]; then cat ' + shlex.quote(remote_dir + '/worker-status.json') + '; fi')
            state = json.loads(raw) if raw.strip() else {}
            log_mirror.poll(state)
            preview_mirror.poll(state)
            if state != last and state:
                last = state
                update(phase=state.get('phase', 'reconstruction'), message=state.get('message') or state.get('phase', 'Running'))
            if state.get('status') in ('completed', 'error', 'stopped'):
                update(remote_finished=True)
                break
            exited = ssh('if [ -f ' + shlex.quote(remote_dir + '/launcher-exit') + ' ]; then cat ' + shlex.quote(remote_dir + '/launcher-exit') + '; fi')
            if exited.strip():
                # The worker can publish its terminal status between the
                # first status read and this exit-sentinel read. Its final
                # status is authoritative; exit code zero alone is not.
                raw = ssh('if [ -f ' + shlex.quote(remote_dir + '/worker-status.json') + ' ]; then cat ' + shlex.quote(remote_dir + '/worker-status.json') + '; fi')
                state = json.loads(raw) if raw.strip() else {}
                log_mirror.poll(state)
                if state != last and state:
                    last = state
                    update(phase=state.get('phase', 'reconstruction'), message=state.get('message') or state.get('phase', 'Running'))
                update(remote_finished=True)
                if state.get('status') in ('completed', 'error', 'stopped'):
                    break
                raise RuntimeError('GPU launcher exited before stage completion (code ' + exited.strip() + '); inspect the GPU stage log')
            if time.monotonic() - started > 24 * 3600:
                raise TimeoutError('GPU stage exceeded 24 hours; inspect remote stage.log')
            time.sleep(2)
    except BaseException:
        ssh('touch ' + shlex.quote(remote_dir + '/STOP'))
        raise
    finally:
        preview_mirror.finish()
        # Pull partial results/logs as well; checkpoint trajectories are reusable by both arms.
        try:
            # Dense benchmark depth is a regenerable rendering diagnostic,
            # not inference input or a reconstruction deliverable. Keep it
            # on LRZ rather than duplicating gigabytes on the desktop.
            excludes = ['--exclude=/results/*/prepared/**/depth/'] if benchmark else []
            atomic_json(root / 'artifact-download-policy.json', {
                'remote_dir': remote_dir, 'excluded_patterns': excludes,
                'retained': 'RGB renders, opacity, model checkpoints, PLY, evaluation/provenance and logs',
                'excluded_artifacts_remain_on_remote': True})
            lrz._mux_run(['rsync', '-az', '--exclude=source/', '--exclude=benchmark-input/', *excludes, '-e', lrz.rsync_ssh_cmd(cfg),
                          endpoint + ':' + remote_dir + '/', str(root / 'gpu') + '/'])
        except lrz.RemoteCommandError as exc:
            # Keep the initial SSH error as well as the rsync tail; the UI
            # summary intentionally truncates diagnostics and can hide it.
            diagnostic = root / 'artifact-transfer.log'
            temporary = diagnostic.with_suffix('.log.tmp')
            temporary.write_text(f'Artifact transfer exited with code {exc.returncode}\n'
                                 f'\nSTDERR\n{exc.stderr}\nSTDOUT\n{exc.stdout}')
            os.replace(temporary, diagnostic)
            raise
        if cp is not None:
            persist_checkpoint_caches(root / 'gpu' / 'checkpoint', cp.root)
    if state['status'] == 'stopped' or stop():
        raise InterruptedError('GPU reconstruction cancelled')
    if state['status'] != 'completed':
        raise RuntimeError(state.get('message', 'GPU reconstruction failed'))
    return {'reconstruction_method': 'artifixer', 'results': str(root / 'gpu' / 'results'), 'mode': options['mode'],
            **({'source': options['source'], 'model': options['model']} if benchmark else {'checkpoint': str(cp.root)})}
