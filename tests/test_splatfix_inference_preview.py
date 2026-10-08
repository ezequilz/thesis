import json
import shutil
import subprocess
import threading
from pathlib import Path

from splat_explorer.splatfix.inference_preview import InferencePreviewMirror, publish_preview


def preview(root):
    pred = root / 'inference/model/frames/batch_0000/pred'
    pred.mkdir(parents=True)
    for i in (0, 1, 2):
        (pred / f'{i:05d}.png').write_bytes(b'png')
    publish_preview(root, pred, [0, 2], 1, frame_count=3)
    return pred


def test_background_preview_is_published_once_and_survives_reattachment(tmp_path):
    remote = tmp_path / 'remote/results/run'
    preview(remote)
    local = tmp_path / 'local'
    local.mkdir()
    entered, release = threading.Event(), threading.Event()
    calls = []
    def transfer(argv):
        calls.append(argv)
        entered.set()
        assert release.wait(5)
        shutil.copytree(remote, Path(argv[-1]), dirs_exist_ok=True)
    def mirror():
        return InferencePreviewMirror(local, '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    state = {'inference_preview': '/workspace/job/results/run/inference-preview.json'}
    worker = mirror()
    worker.poll(state)
    assert entered.wait(5)
    worker.poll(state)
    assert not (local / 'gpu/results/run/inference-preview.json').exists()
    release.set()
    worker.finish()
    assert (local / 'gpu/results/run/inference-preview.json').is_file()
    assert (local / 'gpu/results/run/inference/model/frames/batch_0000/pred/00002.png').is_file()
    mirror().poll(state)
    assert len(calls) == 1


def test_failed_preview_does_not_publish_or_raise(tmp_path):
    def transfer(argv):
        raise RuntimeError('network unavailable')
    worker = InferencePreviewMirror(tmp_path, '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    worker.poll({'inference_preview': '/workspace/job/results/run/inference-preview.json'})
    worker.finish()
    assert not list(tmp_path.glob('gpu/results/*/inference-preview.json'))
    assert 'network unavailable' in (tmp_path / 'inference-preview-transfer.log').read_text()


def test_real_transfer_filters_exclude_large_artifacts(tmp_path):
    remote = tmp_path / 'remote/results/run'
    pred = preview(remote)
    (remote / 'artifixer3d.ply').write_bytes(b'large')
    (pred.parent / 'input.png').write_bytes(b'input')
    (remote / 'prepared').mkdir()
    (remote / 'prepared/model.pt').write_bytes(b'large')
    local = tmp_path / 'local'
    local.mkdir()
    def transfer(argv):
        # Exercise the actual rsync include/exclude rules using a local source.
        command = argv[:]
        i = command.index('-e')
        del command[i:i+2]
        command[-2] = str(remote) + '/'
        subprocess.run(command, check=True, capture_output=True)
    worker = InferencePreviewMirror(local, '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    worker.poll({'inference_preview': '/workspace/job/results/run/inference-preview.json'})
    worker.finish()
    saved = local / 'gpu/results/run'
    assert (saved / 'inference-preview.json').exists()
    assert len(list(saved.rglob('*.png'))) == 3
    assert not list(saved.rglob('*.ply')) and not list(saved.rglob('*.pt'))


def test_preview_rejects_paths_outside_current_run(tmp_path):
    def transfer(argv):
        raise AssertionError('must not transfer')
    worker = InferencePreviewMirror(tmp_path, '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    for path in ('/workspace/other/results/run/inference-preview.json',
                 '/workspace/job/results/../inference-preview.json', '/etc/passwd'):
        worker.poll({'inference_preview': path})
    assert worker.thread is None


def test_worker_advertises_preview_through_later_phases(tmp_path, monkeypatch):
    from splat_explorer.splatfix import job_worker
    from splat_explorer.splatfix.artifixer import worker as artifixer_worker
    root = tmp_path / 'results/run'
    states = []
    def repair(*args, on_progress, **kwargs):
        on_progress({'phase': 'infer', 'output_dir': str(root)})
        states.append(json.loads((tmp_path / 'worker-status.json').read_text()))
        preview(root)
        on_progress({'phase': 'distill', 'output_dir': str(root)})
        states.append(json.loads((tmp_path / 'worker-status.json').read_text()))
        on_progress({'phase': 'plus'})
        states.append(json.loads((tmp_path / 'worker-status.json').read_text()))
        return {'output_dir': str(root)}
    monkeypatch.setattr(artifixer_worker, 'run_repair', repair)
    monkeypatch.setattr('splat_explorer.splatfix.run_evaluation.finalize_run_evaluation', lambda *a, **k: None)
    (tmp_path / 'worker-request.json').write_text(json.dumps({
        'checkpoint': 'checkpoint', 'output': str(root.parent), 'mode': 'baseline',
        'frames': 3, 'span_fraction': 1, 'runtime': {}}))
    job_worker.execute(tmp_path)
    assert 'inference_preview' not in states[0]
    assert states[1]['inference_preview'] == states[2]['inference_preview'] == str(root / 'inference-preview.json')
    assert states[1]['status'] == 'running'


def test_transient_transfer_failure_retries(tmp_path):
    remote = tmp_path / 'remote/results/run'
    preview(remote)
    calls = []
    def transfer(argv):
        calls.append(argv)
        if len(calls) == 1:
            raise RuntimeError('temporary disconnect')
        shutil.copytree(remote, Path(argv[-1]), dirs_exist_ok=True)
    worker = InferencePreviewMirror(tmp_path / 'local', '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    state = {'inference_preview': '/workspace/job/results/run/inference-preview.json'}
    worker.poll(state)
    worker.finish()
    worker.retry_after = 0
    worker.poll(state)
    worker.finish()
    assert len(calls) == 2
    assert (tmp_path / 'local/gpu/results/run/inference-preview.json').is_file()


def test_preview_syncs_camera_metadata_and_builds_chart_without_gpu(tmp_path):
    import numpy as np
    remote = tmp_path / 'remote/results/run'
    pred = preview(remote)
    prepared = remote / 'prepared/bicycle'
    prepared.mkdir(parents=True)
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, 1, 3] = [0, 1, 2]
    (prepared / 'poses.json').write_text(json.dumps({'frames': [{'transform_matrix': p.tolist()} for p in poses]}))
    (prepared / 'refs.json').write_text('[1]')
    (prepared / 'targets.json').write_text('[0,2]')
    (prepared / 'split.json').write_text(json.dumps({'test': {'bicycle': {
        'transforms_path': '/workspace/job/results/run/prepared/bicycle/poses.json',
        'selected_indices_path': 'refs.json', 'target_indices_path': 'targets.json'}}}))
    (prepared / 'depth').mkdir()
    (prepared / 'depth/large.npy').write_bytes(b'not needed')
    local = tmp_path / 'local'
    def transfer(argv):
        command = argv[:]
        i = command.index('-e')
        del command[i:i+2]
        command[-2] = str(remote) + '/'
        subprocess.run(command, check=True, capture_output=True)
    worker = InferencePreviewMirror(local, '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    worker.poll({'inference_preview': '/workspace/job/results/run/inference-preview.json'})
    worker.finish()
    saved = local / 'gpu/results/run'
    assert (saved / 'trajectory-quality.png').is_file()
    assert json.loads((saved / 'trajectory-quality.json').read_text())['target_count'] == 2
    assert not list(saved.rglob('*.npy'))


def test_inputs_transfer_before_inference_and_remain_with_predictions(tmp_path):
    from splat_explorer.splatfix.inference_preview import publish_inputs
    remote = tmp_path / 'remote/results/run'
    checkpoint = remote / 'checkpoint'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'reference.png').write_bytes(b'edited reference')
    (checkpoint / 'rgb.png').write_bytes(b'original capture')
    (checkpoint / 'trajectory.json').write_text(json.dumps({'frames': [{'rgb': 'rgb.png'}]}))
    (remote / 'request.json').write_text(json.dumps({'references': [str(checkpoint / 'reference.png')],
        'checkpoint_root': str(checkpoint), 'trajectory': str(checkpoint / 'trajectory.json')}))
    publish_inputs(remote)
    local = tmp_path / 'local'
    def transfer(argv):
        command = argv[:]
        i = command.index('-e')
        del command[i:i+2]
        command[-2] = str(remote) + '/'
        subprocess.run(command, check=True, capture_output=True)
    worker = InferencePreviewMirror(local, '/workspace/job', '/dss/job', 'host', 'ssh', transfer)
    state = {'input_preview': '/workspace/job/results/run/input-preview.json'}
    worker.poll(state)
    worker.finish()
    saved = local / 'gpu/results/run'
    assert (saved / 'input-images/reference_images/00000.png').read_bytes() == b'edited reference'
    assert (saved / 'input-images/original_rgb_images/00000.png').read_bytes() == b'original capture'
    assert not (saved / 'inference-preview.json').exists()
    preview(remote)
    state['inference_preview'] = '/workspace/job/results/run/inference-preview.json'
    worker.retry_after = 0
    worker.poll(state)
    worker.finish()
    assert (saved / 'inference-preview.json').is_file()
    assert (saved / 'input-preview.json').is_file()
