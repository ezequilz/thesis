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
    monkeypatch.setattr(job_worker, 'run_repair', repair)
    monkeypatch.setattr('splat_explorer.splatfix.run_evaluation.finalize_run_evaluation', lambda *a, **k: None)
    (tmp_path / 'worker-request.json').write_text(json.dumps({
        'checkpoint': 'checkpoint', 'output': str(root.parent), 'mode': 'baseline',
        'frames': 3, 'span_fraction': 1, 'runtime': {}}))
    job_worker.execute(tmp_path)
    assert 'inference_preview' not in states[0]
    assert states[1]['inference_preview'] == states[2]['inference_preview'] == str(root / 'inference-preview.json')
    assert states[1]['status'] == 'running'
