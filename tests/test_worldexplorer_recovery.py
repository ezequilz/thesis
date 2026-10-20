import hashlib
import json
from pathlib import Path

import pytest

from splat_explorer.splatfix.worldexplorer import recovery, service, worker


ID = 'we_20261020_025819_5d494b3a09a5'
SCENE = 'scenes/input_scaffold_3.0_test'


def archive(tmp_path):
    root = tmp_path/ID/'results'
    frames = []
    for i in range(8):
        for direction in ('in', 'left', 'right', 'up'):
            relative = f'{SCENE}/img2trajvid/{i:03}_{direction}/samples-rgb/000.png'
            frames.append({'file_path': f'/workspace/worldexplorer-runs/{ID}/{relative}'})
            path = root/relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b'generated')
    for i in range(8):
        relative = f'{SCENE}/img2trajvid/000_in/input/{i:03}.png'
        frames.append({'file_path': f'/workspace/worldexplorer-runs/{ID}/{relative}'})
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'input')
    (root/SCENE/'img2trajvid/transforms.json').write_text(json.dumps(
        {'frames': frames, 'ply_file_path': '/old/cloud.ply'}))
    manifest = root/'.archive-manifest.json'
    worker.atomic(manifest, {'files': worker.inventory(root)})
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return root, digest


def test_verified_payload_contains_only_required_files(tmp_path):
    root, digest = archive(tmp_path)
    (root/'unneeded-video.mp4').write_bytes(b'not uploaded')
    payload = recovery.verified_payload(root, ID, digest)
    assert len(payload['files']) == 41
    assert payload['scene'] == SCENE
    assert all(p.startswith(SCENE+'/img2trajvid/') for p in payload['files'])


@pytest.mark.parametrize('damage', ['missing', 'changed', 'manifest', 'incomplete'])
def test_reject_bad_archive(tmp_path, damage):
    root, digest = archive(tmp_path)
    image = next(root.glob('scenes/*/img2trajvid/*/samples-rgb/*.png'))
    if damage == 'missing':
        image.unlink()
    elif damage == 'changed':
        image.write_bytes(b'corrupt')
    elif damage == 'manifest':
        (root/'.archive-manifest.json').write_text('{}')
    else:
        path = root/SCENE/'img2trajvid/transforms.json'
        data = json.loads(path.read_text())
        data['frames'] = data['frames'][1:]
        path.write_text(json.dumps(data))
        worker.atomic(root/'.archive-manifest.json', {'files': worker.inventory(root)})
        digest = hashlib.sha256((root/'.archive-manifest.json').read_bytes()).hexdigest()
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.verified_payload(root, ID, digest)


def test_rebase_only_new_run_metadata(tmp_path):
    import shutil
    root, digest = archive(tmp_path)
    target = tmp_path/'new-run'
    shutil.copytree(root, target)
    original = (root/SCENE/'img2trajvid/transforms.json').read_bytes()
    work = recovery.rebase_transforms(target, SCENE, ID)
    data = json.loads((work/'img2trajvid/transforms.json').read_text())
    assert 'ply_file_path' not in data
    assert all(Path(f['file_path']).is_relative_to(target) for f in data['frames'])
    assert (root/SCENE/'img2trajvid/transforms.json').read_bytes() == original


class AuthorsRun:
    def run(self):
        raise AssertionError('Must not regenerate trajectories')
        select_images_for_vggt(self.work_dir, max_images=self.budget)
        run_vggt_and_align(self.work_dir)
        self.train_and_export()


def test_continuation_executes_original_suffix_in_order():
    from types import SimpleNamespace
    calls = []
    namespace = dict(
        select_images_for_vggt=lambda path, max_images: calls.append(('select', path, max_images)),
        run_vggt_and_align=lambda path: calls.append(('align', path)))
    resume = recovery.continuation_method(AuthorsRun.run, namespace)
    resume(SimpleNamespace(work_dir='saved', budget=40,
                           train_and_export=lambda: calls.append(('train/export',))))
    assert calls == [('select', 'saved', 40), ('align', 'saved'), ('train/export',)]


def test_unknown_upstream_boundary_is_rejected():
    with pytest.raises(RuntimeError, match='boundary changed'):
        recovery.continuation_method(test_unknown_upstream_boundary_is_rejected, {})


@pytest.mark.parametrize('budget', [40, 64, 100])
def test_service_resume_requires_verification_and_preserves_settings(tmp_path, monkeypatch, budget):
    root, digest = archive(tmp_path)
    api = service.WorldExplorerService(tmp_path)
    job = dict(id=ID, status='failed', request=service.validate({'operation': 'expand'}))
    job['request']['num_images_for_vggt'] = budget
    worker.atomic(root.parent/'job.json', job)
    with pytest.raises(ValueError, match='archive'):
        api.resume(ID)
    job.update(downloaded_at=1, archive_sha256=digest)
    worker.atomic(root.parent/'job.json', job)
    calls = []
    monkeypatch.setattr(api, 'create', lambda request, **kw: calls.append((request, kw)) or {'id': 'new'})
    assert api.resume(ID) == {'id': 'new'}
    request, kwargs = calls[0]
    assert request['operation'] == 'resume'
    assert request['translation_scaling'] == job['request']['translation_scaling']
    assert request['num_images_for_vggt'] == max(64, budget)
    assert kwargs['recovery']['source_run'] == ID
    assert json.loads((root.parent/'job.json').read_text())['status'] == 'failed'
    assert json.loads((root.parent/'job.json').read_text())['request']['num_images_for_vggt'] == budget


def test_resume_deduplicates_uncertain_submission(tmp_path, monkeypatch):
    root, digest = archive(tmp_path)
    worker.atomic(root.parent/'job.json', dict(id=ID, status='failed',
        request=service.validate({}), downloaded_at=1, archive_sha256=digest))
    child = tmp_path/'we_20261020_120000_abcdef123456'
    child.mkdir()
    active = dict(id=child.name, source_run=ID, status='launch_unknown',
                  request=service.validate({'operation':'resume'}))
    worker.atomic(child/'job.json', active)
    api = service.WorldExplorerService(tmp_path)
    monkeypatch.setattr(api, 'create', lambda *a, **kw: pytest.fail('Duplicate submission'))
    assert api.resume(ID) == active


def test_direct_resume_without_payload_rejected(tmp_path):
    with pytest.raises(ValueError, match='verified saved trajectories'):
        service.WorldExplorerService(tmp_path).create({'operation':'resume'})


def test_recovery_uploads_required_files_and_records_provenance(tmp_path, monkeypatch):
    root, digest = archive(tmp_path/'source')
    payload = recovery.verified_payload(root, ID, digest)
    api = service.WorldExplorerService(tmp_path/'jobs')
    cfg = dict(user='u', host='host', workspace='/dss/test', job_id='123',
               cpus=4, mem='64G', container='/dss/test/image.sqsh')
    monkeypatch.setattr(service.lrz, 'load_lrz_config', lambda: cfg)
    monkeypatch.setattr(service.lrz, 'lrz_session_alive', lambda _: True)
    monkeypatch.setattr(api, '_ssh', lambda cfg, command: 'R 04:00:00' if command.startswith('squeue') else '')
    monkeypatch.setattr(api, 'request_reconcile', lambda **kw: None)
    transfers = []
    monkeypatch.setattr(service.lrz, '_mux_run', lambda args: transfers.append(args))
    result = api.create({'operation':'resume'}, recovery=payload)
    assert result['status'] == 'submitted'
    assert result['source_run'] == ID
    request = json.loads((api.directory(result['id'])/'request.json').read_text())
    assert request['resume_scene'] == SCENE and request['source_run'] == ID
    upload = next(args for args in transfers if any(a.startswith('--files-from=') for a in args))
    file_list = Path(next(a.split('=', 1)[1] for a in upload if a.startswith('--files-from=')))
    assert file_list.read_text().splitlines() == payload['files']
    assert str(root.resolve())+'/' in upload


def test_http_resume_dispatch():
    from types import SimpleNamespace
    from splat_explorer.web.server import DashboardHandler
    handler = object.__new__(DashboardHandler)
    calls = []
    handler.command = 'POST'
    handler.app = SimpleNamespace(worldexplorer=SimpleNamespace(
        resume=lambda run_id: calls.append(run_id) or {'id':'recovered'}))
    replies = []
    handler._send_json = lambda *args: replies.append(args)
    handler._serve_worldexplorer('/api/splatfix/worldexplorer/resume', {'id':ID})
    assert calls == [ID]
    assert replies == [({'id':'recovered'},)]
