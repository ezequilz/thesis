"""ArtiFixer LRZ staging, execution, reattachment and cache publication."""
import json
import os
import shlex
import time
from pathlib import Path
from ..checkpoint import Checkpoint, atomic_json
from ..remote_poll import wait_for_remote_poll


def persist_checkpoint_caches(downloaded, checkpoint_root):
    """Publish reusable GPU caches without dereferencing remote source links."""
    import shutil
    import tempfile
    from .repair import digest_file
    downloaded, checkpoint_root = Path(downloaded), Path(checkpoint_root)
    inputs = downloaded / 'artifixer-inputs'
    if inputs.is_dir():
        for entry in inputs.iterdir():
            if entry.is_dir() and not entry.is_symlink() and (entry / 'captures.json').is_file():
                target = checkpoint_root / 'artifixer-inputs' / entry.name
                target.mkdir(parents=True, exist_ok=True)
                metadata = json.loads((entry / 'captures.json').read_text())
                for index, expected in enumerate(metadata['sha256']):
                    image = entry / f'{index:05d}.png'
                    if image.is_symlink() or digest_file(image) != expected:
                        raise ValueError('Saved ArtiFixer capture checksum mismatch')
                    shutil.copy2(image, target / image.name)
                persist_checkpoint_caches(entry, target)
                atomic_json(target / 'captures.json', metadata)
    trajectories = downloaded / 'trajectories'
    if trajectories.is_dir():
        # Metric alignment lives inside each trajectory. Its measurement image
        # links are temporary inputs, not replay data. Omit them rather than
        # dereferencing links to the old container or preserving dangling links.
        shutil.copytree(trajectories, checkpoint_root / 'trajectories',
                        dirs_exist_ok=True, symlinks=True,
                        ignore=lambda directory, names: [name for name in names
                            if (Path(directory) / name).is_symlink()])
    plans = downloaded / 'source-plans'
    if plans.is_dir():
        from .source_plan import signature
        for path in plans.glob('*.json'):
            plan = json.loads(path.read_text())
            if (path.is_symlink() or path.stem != signature(plan['identity'])
                    or plan.get('sha256') != signature({k: v for k, v in plan.items() if k != 'sha256'})):
                raise ValueError('Invalid saved source trajectory plan')
            destination = checkpoint_root / 'source-plans' / path.name
            destination.parent.mkdir(exist_ok=True)
            if destination.is_file() and json.loads(destination.read_text()) != plan:
                raise ValueError('Conflicting saved source trajectory plans')
            atomic_json(destination, plan)
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
    from .checkpoint_policy import checkpoint_for_run
    cp = None if benchmark else checkpoint_for_run(Checkpoint.load(options['checkpoint']), options.get('image_trajectory', 'checkpoint'))
    if cp is not None and cp.endpoint_only and options.get('handoff_generation', False):
        raise ValueError('Handoff generation requires non-endpoint checkpoint trajectories')
    if cp is not None and cp.endpoint_only and options['mode'] != 'edited':
        raise ValueError('Endpoint checkpoints require GPT-repaired references')
    if cp is not None and options['mode'] == 'edited':
        cp.require_viser_images()
    if cp is not None and (not cp.complete or (options['mode'] == 'edited' and any(not v.get('repaired_rgb') for v in cp.repair_views))):
        raise ValueError('Checkpoint is not ready for the requested reconstruction mode')
    if cp is not None and options.get('correct_generated_pose') and not (cp.root / 'repeat-repair.json').is_file():
        # Revalidate after queue wait, before reserving remote work or uploading inputs.
        from ..pose_correction import corrected_checkpoint
        corrected_checkpoint(cp)
    if cp is not None and options.get('prompt_source') == 'worldexplorer':
        from ..worldexplorer.source import checkpoint_prompt
        if not checkpoint_prompt(cp):
            raise ValueError('Original WorldExplorer prompt is unavailable for this scene')
    if cp is not None and options.get('prompt_source') == 'original_and_repaired' and options['mode'] != 'edited':
        raise ValueError('Original + repaired captioning requires edited input images')
    cfg = lrz.load_lrz_config()
    saved = executor.store.get_run(run_id).state.details
    def capture_stop():
        deadline = executor.store.get_run(run_id).state.details.get('effective_deadline')
        if deadline and time.time() >= float(deadline):
            executor.store.request_stop(run_id)
            return True
        return stop()
    remote_dir = saved.get('remote_dir') or str(cfg['workspace']).rstrip('/') + '/splatfix-jobs/' + run_id
    container_dir = '/workspace/splatfix-jobs/' + run_id
    endpoint = f"{cfg['user']}@{cfg['host']}"
    def ssh(command):
        answer = lrz._ssh_run(cfg, command, timeout=30)
        if answer.returncode:
            raise RuntimeError((answer.stderr or answer.stdout or 'Remote command failed').strip())
        return answer.stdout
    def sync(source, destination, *, exclude=()):
        lrz._mux_run(['rsync', '-az', *['--exclude='+p for p in exclude],
                      '-e', lrz.rsync_ssh_cmd(cfg), source, destination])
    resuming = bool(executor.store.get_run(run_id).state.details.get('remote_dir'))
    if resuming:
        resuming = bool(ssh('test ! -d ' + shlex.quote(remote_dir + '/launched') + ' || printf launched').strip())
    if not resuming:
        from .repair import DEFAULT_RUNTIME
        runtime = {**DEFAULT_RUNTIME, **dict(executor.cfg.get('splatfix', {}).get('runtime', {}))}
        for key in ('teacher_inference', 'inference_model', 'teacher_steps', 'teacher_max_frames', 'teacher_resolution', 'teacher_clip_layout'):
            if key in options:
                runtime[key] = options[key]
        runtime['resolution_profile'] = options['resolution_profile']
        runtime['regularization_profile'] = options.get('regularization_profile', 'artifixer')
        runtime['split_mode'] = 'no-split' if cp is not None and cp.endpoint_only else options.get('split_mode', 'double-split')
        # New validated jobs explicitly carry picked order. Missing fields in
        # historical jobs retain the authors' orbit policy.
        runtime['trajectory_order'] = 'authors' if benchmark else options.get('trajectory_order', 'authors')
        runtime['image_trajectory'] = options.get('image_trajectory', 'checkpoint')
        runtime['endpoint_reference_mode'] = options.get('endpoint_reference_mode', 'both')
        if not benchmark:
            from .pipeline_profile import apply_pipeline
            # New validated jobs always carry the field. Missing means a saved
            # historical job, whose renderer and custom recipe must survive.
            options = apply_pipeline({'artifixer_pipeline': 'legacy_gsplat', **options})
            runtime['artifixer_pipeline'] = options['artifixer_pipeline']
            runtime['regularization_profile'] = options['regularization_profile']
            from .official_worker import validate_reconstruction_options
            runtime.update(validate_reconstruction_options(options))
            # A configured historical bundle must not leak into distilled-only runs.
            if runtime['reconstruction_images'] == 'distilled':
                runtime.pop('worldexplorer_source', None)
        runtime['gpt_image_context'] = options.get('gpt_image_context', 'single_last_frame')
        from .handoff_noise import noise_options
        runtime.update(noise_options(options))
        runtime['handoff_type'] = options.get('handoff_type', 'boundary') if not benchmark else 'boundary'
        runtime['handoff_backward_pass'] = options.get('handoff_backward_pass', False) if not benchmark else False
        runtime['handoff_generation'] = options.get('handoff_generation', False) if not benchmark else False
        runtime['connected_trajectory_generation'] = (options.get('connected_trajectory_generation', False)
            if not benchmark and not (cp is not None and cp.endpoint_only) else False)
        runtime['connected_bridge_seeding'] = options.get('connected_bridge_seeding', 'off') if not benchmark else 'off'
        runtime['generated_context_references'] = options.get('generated_context_references', False) if not benchmark else False
        runtime['initial_image_seeding'] = options.get('initial_image_seeding') if not benchmark else None
        runtime['image_cache_insertion'] = options.get('image_cache_insertion', False) if not benchmark else False
        runtime['correct_generated_pose'] = options.get('correct_generated_pose', False) if not benchmark else False
        if cp is not None and (cp.root / 'repeat-repair.json').is_file():
            # Recorded effective runtime wins over current dashboard/server defaults.
            repeat = json.loads((cp.root / 'repeat-repair.json').read_text())
            runtime = dict(repeat['runtime'])
            runtime.setdefault('trajectory_order', 'authors')
        from .inference_policy import model_runtime as inference_runtime
        if not benchmark:
            runtime = inference_runtime(runtime)
        # Capture completion belongs to this run/source, not to a repeated
        # reconstruction's inherited model settings.
        for key in ('source_rgb_prepared', 'source_anchor_capture_signature'):
            runtime.pop(key, None)
        preflight = None
        if cp is not None:
            from .preflight import prepare_source_captures
            preflight = prepare_source_captures(cp, runtime, root, should_stop=capture_stop,
                progress=update, url=executor.cfg.get('renderer', {}).get('viser_url') or None)
        if capture_stop():
            raise InterruptedError('Cancelled before GPU staging')
        lean = bool(preflight and not preflight.get('source_plan') and not preflight.get('known_repeat_orbit'))
        # No source/code upload or allocation setup before the Viser preflight.
        # Reattachment only needs DSS/SSH, even if the allocation has expired.
        if not saved.get('remote_dir'):
            if options.get('repeat_from'):
                marker = lrz.read_remote_setup_marker(cfg)
                if not marker.get('ok') or str(marker.get('job_id')) != str(cfg['job_id']):
                    update(phase='gpu_setup', message='Preparing the new GPU allocation for repeat repair')
                    lrz.setup_lrz_gpu({**cfg, 'artifixer_required': True, 'artifixer_model': options['model']},
                                      qwen_required=False)
            LrzSceneRunTransport(executor.cfg, run_id, root).validate()
        update(message='Staging six-view inputs and source for GPU depth, opacity, MoGe alignment and trajectory planning' if lean else
               'Source captures ready; staging checkpoint and source on LRZ' if preflight else
               'Staging checkpoint and source on the configured LRZ allocation', phase='source_staging' if lean else 'staging')
        lrz.sync_code_to_dss(cfg)
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
                       'output': container_dir + '/results', 'runtime': runtime,
                       'benchmark_runs': options.get('benchmark_runs', 1)}
        else:
            ssh('mkdir -p ' + shlex.quote(remote_dir + '/checkpoint') + ' ' + shlex.quote(remote_dir + '/source'))
            from ..persistence import checkpoint_cache_lock
            with checkpoint_cache_lock(cp.root):
                if preflight:
                    from .staging_cache import digest
                    if digest(cp.root / 'checkpoint.json') != preflight['checkpoint_sha256']:
                        raise ValueError('Checkpoint changed after Viser preflight; start a new run with the updated views')
                staged_checkpoint = cp.root
                if lean:
                    from .source_staging import lean_checkpoint
                    staged_checkpoint = lean_checkpoint(cp, preflight, root / 'lean-checkpoint')
                sync(str(staged_checkpoint) + '/', endpoint + ':' + remote_dir + '/checkpoint/',
                     exclude=('worldexplorer-source/',))
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
            from ..worldexplorer.source import stage_checkpoint_source
            originals = (stage_checkpoint_source(cp, root / 'worldexplorer-source')
                         if not lean and runtime['reconstruction_images'] == 'all' else None)
            if originals:
                sync(originals + '/', endpoint + ':' + remote_dir + '/worldexplorer-source/')
                runtime['worldexplorer_source'] = container_dir + '/worldexplorer-source'
            runtime['scene_path'] = remote_scene
            request = {'checkpoint': container_dir + '/checkpoint', 'output': container_dir + '/results',
                       'runtime': runtime, **{k: options[k] for k in ('mode', 'frames', 'span_fraction')}}
            if preflight:
                runtime['source_rgb_prepared'] = True
                runtime['source_anchor_capture_signature'] = Path(preflight['input_cache']).name
                request['source_checkpoint_sha256'] = preflight['checkpoint_sha256']
                request['prepare_source_first'] = lean
                if preflight.get('source_plan'):
                    request['source_plan'] = container_dir + '/checkpoint/' + str(Path(preflight['source_plan']).relative_to(cp.root))
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
        update(phase='source_preparation' if lean else 'reconstruction',
               message='GPU computing source depth, opacity, MoGe scale and exact trajectory poses' if lean else
               'Authors’ fresh reconstruction running on LRZ', remote_dir=remote_dir)
        ssh(launcher)
    from .live_logs import PhaseLogMirror
    log_mirror = PhaseLogMirror(root, container_dir, remote_dir, ssh)
    from .inference_preview import InferencePreviewMirror
    preview_mirror = InferencePreviewMirror(root, container_dir, remote_dir, endpoint,
                                            lrz.rsync_ssh_cmd(cfg), lrz._mux_run)
    last = None
    started = time.monotonic()
    from ..viser_capture import CaptureMirror
    capture_pipeline = options.get('artifixer_pipeline')
    if cp is not None and (cp.root / 'repeat-repair.json').is_file():
        capture_pipeline = json.loads((cp.root / 'repeat-repair.json').read_text())['runtime'].get('artifixer_pipeline')
    native = capture_pipeline == 'authors_3dgut'
    capture_mirror = None if benchmark or native else CaptureMirror(
        root, remote_dir, container_dir, cp, endpoint, ssh, sync,
        url=executor.cfg.get('renderer', {}).get('viser_url') or None, should_stop=capture_stop, progress=update)
    from .source_staging import SourceStagingMirror
    source_staging = None if benchmark or native else SourceStagingMirror(
        root, remote_dir, container_dir, cp, endpoint, ssh, sync,
        url=executor.cfg.get('renderer', {}).get('viser_url') or None, should_stop=capture_stop, progress=update)
    handoff_service = None
    if not benchmark and (options.get('handoff_generation', False) or
            (cp is not None and (cp.root / 'repeat-repair.json').is_file() and
             json.loads((cp.root / 'repeat-repair.json').read_text())['runtime'].get('handoff_generation', False))):
        from .handoff import HandoffService
        handoff_service = HandoffService(root / 'handoff', cfg=executor.cfg,
            remote=remote_dir + '/handoff', endpoint=endpoint, ssh=ssh, sync=sync)
    state = {}
    from .progress import ProgressObserver, large_model
    observer = ProgressObserver(root, container_dir, remote_dir) if large_model(options) else None
    primary_error = None
    try:
        while True:
            if observer is not None:
                observer.wait(capture_stop)
            deadline = executor.store.get_run(run_id).state.details.get('effective_deadline')
            if deadline and time.time() >= float(deadline) and not stop():
                executor.store.request_stop(run_id)
                update(status='stopping', message='GPU deadline approaching; saving current reconstruction before stopping',
                       stop_reason='gpu_deadline')
            if stop():
                ssh('touch ' + shlex.quote(remote_dir + '/STOP'))
            elif capture_mirror is not None and observer is None:
                capture_mirror.poll()
            if not stop() and handoff_service is not None:
                handoff_service.poll()
            if observer is not None:
                packet = observer.poll(ssh)
                state = packet.get('state', {})
                if not stop() and packet.get('capture_pending') and capture_mirror is not None:
                    capture_mirror.poll()
            else:
                raw = ssh('if [ -f ' + shlex.quote(remote_dir + '/worker-status.json') + ' ]; then cat ' + shlex.quote(remote_dir + '/worker-status.json') + '; fi')
                state = json.loads(raw) if raw.strip() else {}
            if state:
                atomic_json(root / 'worker-status.json', state)
            if observer is None:
                log_mirror.poll(state)
            preview_mirror.poll(state)
            if state != last and state:
                last = state
                update(phase=state.get('phase', 'reconstruction'), message=state.get('message') or state.get('phase', 'Running'))
            if not stop() and source_staging is not None:
                source_staging.poll(state)
            if state.get('status') in ('completed', 'error', 'stopped'):
                break
            exited = packet.get('exit', '') if observer is not None else ssh('if [ -f ' + shlex.quote(remote_dir + '/launcher-exit') + ' ]; then cat ' + shlex.quote(remote_dir + '/launcher-exit') + '; fi')
            if exited.strip():
                # The worker can publish its terminal status between the
                # first status read and this exit-sentinel read. Its final
                # status is authoritative; exit code zero alone is not.
                if observer is None:
                    raw = ssh('if [ -f ' + shlex.quote(remote_dir + '/worker-status.json') + ' ]; then cat ' + shlex.quote(remote_dir + '/worker-status.json') + '; fi')
                    state = json.loads(raw) if raw.strip() else {}
                if state:
                    atomic_json(root / 'worker-status.json', state)
                if observer is None:
                    log_mirror.poll(state)
                if state != last and state:
                    last = state
                    update(phase=state.get('phase', 'reconstruction'), message=state.get('message') or state.get('phase', 'Running'))
                if state.get('status') in ('completed', 'error', 'stopped'):
                    break
                state = {'status': 'error', 'phase': 'finished',
                         'message': 'GPU launcher exited before stage completion (code ' + exited.strip() + '); inspect the GPU stage log'}
                atomic_json(root / 'worker-status.json', state)
                break
            if time.monotonic() - started > 24 * 3600:
                raise TimeoutError('GPU stage exceeded 24 hours; inspect remote stage.log')
            if observer is None:
                wait_for_remote_poll(stop)
    except BaseException as exc:
        primary_error = exc
        try:
            ssh('touch ' + shlex.quote(remote_dir + '/STOP'))
        except Exception as stop_error:
            update(stop_signal_error=str(stop_error))
        raise
    finally:
        if handoff_service is not None:
            handoff_service.close()
        if state.get('status') in ('completed', 'error', 'stopped') and primary_error is None:
            from ..persistence import wait_for_exit
            wait_for_exit(lrz, cfg, remote_dir)
            update(remote_finished=True)
            # Dense benchmark depth is a regenerable rendering diagnostic,
            # not inference input or a reconstruction deliverable. Keep it
            # on LRZ rather than duplicating gigabytes on the desktop.
            excludes = ['--exclude=/results/*/prepared/**/depth/'] if benchmark else []
            atomic_json(root / 'artifact-download-policy.json', {
                'remote_dir': remote_dir, 'excluded_patterns': excludes,
                'retained': 'RGB renders, opacity, model checkpoints, PLY, evaluation/provenance and logs',
                'excluded_artifacts_remain_on_remote': True})
            from ..downloads import enqueue
            enqueue(cfg, root, run_id, remote_dir, state,
                    excludes=('source/', 'benchmark-input/',
                              *(['/results/*/prepared/**/depth/'] if benchmark else [])),
                    checkpoint=cp.root if cp is not None else None,
                    diagnostics_only=(not benchmark and state.get('status') == 'error'
                        and (state.get('failed_phase', '').startswith('source_') or state.get('failed_phase') in
                        ('validation', 'anchor_geometry', 'trajectory', 'scale', 'orbit', 'caption', 'awaiting_source_capture'))))
    if state['status'] == 'stopped' or stop():
        raise InterruptedError('GPU reconstruction cancelled')
    if state['status'] != 'completed':
        raise RuntimeError(state.get('message', 'GPU reconstruction failed'))
    return {'reconstruction_method': 'artifixer', 'results': str(root / 'gpu' / 'results'), 'mode': options['mode'],
            **({'source': options['source'], 'model': options['model']} if benchmark else {'checkpoint': str(cp.root)})}
