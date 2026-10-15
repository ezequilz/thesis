"""Python 3.9-compatible, stop-only surfel export for the pinned trainer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

TRAINER_SHA256 = '7a815e6b50e9a6e5ee04f7aa580405288513d7a72fd115d4ab6d028b0a38e1e2'


def write_json(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(body, indent=2) + '\n')
    os.replace(temporary, path)


def patch_runtime(repo):
    try:
        from .source_depth_runtime import _local_file
    except ImportError:
        from source_depth_runtime import _local_file
    relative = '2d-gaussian-splatting/train_with_refine_depth.py'
    source = (Path(repo) / relative).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != TRAINER_SHA256:
        raise ValueError('Unexpected pinned G4Splat trainer; refusing stop hook')
    anchor = '        with torch.no_grad():        \n            if network_gui.conn == None:'
    assert source.count(anchor) == 1
    patched = source.replace(anchor, '''        from splatfix_graceful import iteration_boundary
        iteration_boundary(gaussians, iteration)

''' + anchor)
    compile(patched, relative, 'exec')
    _local_file(Path(repo), relative).write_text(patched)
    shutil.copyfile(__file__, Path(repo) / '2d-gaussian-splatting/splatfix_graceful.py')
    write_json(Path(repo).parent / 'graceful-stop.json', {
        'trainer_sha256': TRAINER_SHA256,
        'patched_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'policy': 'Export after optimizer updates only on stop; scheduled saves unchanged',
        'shared_checkout_modified': False})


def iteration_boundary(gaussians, iteration):
    control = os.environ.get('SPLATFIX_G4_STOP')
    if not control or not Path(control).exists():
        return
    import torch
    root = Path(control).parent
    state = {'status': 'saving', 'step': int(iteration), 'incomplete': True,
             'native_splat_path': None, 'checkpoint': None}
    write_json(root / 'stop-state.json', state)
    try:
        torch.cuda.synchronize()
        temporary = root / 'interrupted.tmp.ply'
        gaussians.save_ply(str(temporary))
        native = root / 'interrupted-native.ply'
        os.replace(temporary, native)
        state.update(status='saved', native_splat_path=str(native))
        # PLY is the primary deliverable; a checkpoint failure must not lose it.
        try:
            temporary = root / 'interrupted.tmp.pth'
            torch.save((gaussians.capture(), iteration), temporary)
            checkpoint = root / 'interrupted.pth'
            os.replace(temporary, checkpoint)
            state['checkpoint'] = str(checkpoint)
        except Exception as exc:
            state['checkpoint_error'] = str(exc)
    except Exception as exc:
        state.update(status='save_failed', error=str(exc))
    write_json(root / 'stop-state.json', state)
    raise SystemExit(75)


def stop_poll(output, should_stop, on_progress, *, timeout=240):
    """Delay process-group termination only while a surfel trainer can save."""
    output = Path(output)
    started = None
    def poll():
        nonlocal started
        if not should_stop():
            return False
        if started is None:
            started = time.monotonic()
            (output / 'SAVE_STOP').touch()
            on_progress({'phase': 'stopping', 'message': 'Saving G4Splat and retaining available outputs'})
        if not (output / 'training-active').exists():
            return True
        if time.monotonic() - started >= timeout:
            write_json(output / 'stop-timeout.json', {'status': 'save_timeout', 'seconds': timeout})
            return True
        return False
    return poll


def preserve_partial(output, provenance, reason):
    try:
        from .official_runner import export_viewer_ply
    except ImportError:
        from official_runner import export_viewer_ply
    output = Path(output)
    state_path = output / 'stop-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    native = output / 'interrupted-native.ply'
    fresh = native.is_file()
    if not fresh:
        candidates = list((output / 'official/free_gaussians/point_cloud').glob('iteration_*/point_cloud.ply'))
        native = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    result = {'reconstruction_method': 'g4splat', 'incomplete': True,
              'mode': provenance.get('mode'), 'output_dir': str(output),
              'native_representation': '2d_gaussian_surfels', 'viewer_is_approximation': True,
              'input_manifest': provenance.get('input_manifest'),
              'splat_path': None, 'native_splat_path': str(native) if native else None,
              'interruption': {'reason': reason, 'save_state': state,
                               'snapshot': 'on_stop' if fresh else 'last_saved' if native else 'unavailable'}}
    if native:
        try:
            temporary = output / 'interrupted-viewer.tmp.ply'
            export_viewer_ply(native, temporary)
            viewer = output / 'interrupted-viewer.ply'
            os.replace(temporary, viewer)
            result['splat_path'] = str(viewer)
        except Exception as exc:
            result['interruption']['viewer_export_error'] = str(exc)
    else:
        result['interruption']['reason'] += '; no Gaussian splat was available yet'
    write_json(output / 'partial-result.json', result)
    return result
