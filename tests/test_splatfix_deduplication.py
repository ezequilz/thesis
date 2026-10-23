import errno
import json
import os

import pytest

from splat_explorer.splatfix import deduplication as d
from splat_explorer.splatfix import persistence as p
from test_splatfix_persistence import transport


def artifacts(tmp_path):
    gpu = tmp_path / 'gpu'
    gpu.mkdir()
    first = gpu / 'a.npy'
    second = gpu / 'b.npy'
    first.write_bytes(b'array data')
    second.write_bytes(first.read_bytes())
    return first, second


def test_verified_copies_share_storage_and_survive_deletion(tmp_path):
    first, second = artifacts(tmp_path)
    different = first.with_name('different.npy')
    different.write_bytes(b'other data')  # Same size is not enough.
    result = d.deduplicate_artifacts(tmp_path)
    assert first.samefile(second)
    assert not first.samefile(different)
    assert result['linked_files'] == 1
    assert result['reclaimed_bytes'] == len(b'array data')
    assert d.deduplicate_artifacts(tmp_path)['linked_files'] == 0
    first.unlink()
    assert second.read_bytes() == b'array data'


def test_symlinks_metadata_runtime_and_external_links_are_preserved(tmp_path):
    first, second = artifacts(tmp_path)
    external = tmp_path / 'external.npy'
    external.write_bytes(first.read_bytes())
    (first.parent / 'link.npy').symlink_to(external)
    (first.parent / 'external-dir').symlink_to(tmp_path, target_is_directory=True)
    runtime = first.parent / 'corrected-runtime'
    runtime.mkdir()
    (runtime / 'cache.npy').write_bytes(first.read_bytes())
    for name in ('a.json', 'b.json'):
        (first.parent / name).write_bytes(first.read_bytes())
    d.deduplicate_artifacts(tmp_path)
    assert (first.parent / 'link.npy').is_symlink()
    assert not external.samefile(first)
    assert not (runtime / 'cache.npy').samefile(first)
    assert not (first.parent / 'a.json').samefile(first.parent / 'b.json')


def test_existing_hardlinks_are_not_counted_as_reclaimed_twice(tmp_path):
    first, second = artifacts(tmp_path)
    third = second.with_name('c.npy')
    os.link(second, third)
    result = d.deduplicate_artifacts(tmp_path)
    assert result['linked_files'] == 2
    assert result['reclaimed_bytes'] == len(b'array data')
    assert first.samefile(second) and first.samefile(third)


def test_atomic_replacement_keeps_shared_archive_unchanged(tmp_path):
    first, second = artifacts(tmp_path)
    d.deduplicate_artifacts(tmp_path)
    replacement = tmp_path / 'new.npy'
    replacement.write_bytes(b'new array')
    os.replace(replacement, second)
    assert first.read_bytes() == b'array data'
    assert second.read_bytes() == b'new array'


def test_unsupported_links_leave_originals_intact(tmp_path, monkeypatch):
    first, second = artifacts(tmp_path)
    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOTSUP, 'unsupported')
    monkeypatch.setattr(os, 'link', unsupported)
    result = d.deduplicate_artifacts(tmp_path)
    assert result['skipped_files'] == 1
    assert result['reclaimed_bytes'] == 0
    assert not first.samefile(second)
    assert first.read_bytes() == second.read_bytes() == b'array data'
    assert not list(first.parent.glob('.dedup-*'))


def test_changed_candidate_aborts_without_replacing_it(tmp_path, monkeypatch):
    first, second = artifacts(tmp_path)
    original = d._digest
    def changed(path):
        digest = original(path)
        if path == second:
            path.write_bytes(b'changed')
        return digest
    monkeypatch.setattr(d, '_digest', changed)
    with pytest.raises(RuntimeError, match='changed during verification'):
        d.deduplicate_artifacts(tmp_path)
    assert first.read_bytes() == b'array data'
    assert second.read_bytes() == b'changed'
    assert not first.samefile(second)


def test_persistence_deduplicates_before_remote_cleanup(tmp_path):
    first, second = artifacts(tmp_path)
    lrz, events = transport(tmp_path)
    original = lrz._ssh_run
    def ssh(cfg, command, timeout):
        if 'shutil.rmtree' in command:
            assert first.samefile(second)
        return original(cfg, command, timeout)
    lrz._ssh_run = ssh
    p.persist_remote_run(lrz, {'workspace': '/dss', 'user': 'u', 'host': 'h'},
                         tmp_path, 'run', '/dss/splatfix-jobs/run')
    receipt = json.loads((tmp_path / p.RECEIPT).read_text())
    assert receipt['deduplication']['linked_files'] == 1
    assert events == ['exit', 'download', 'delete']


def test_backfill_requires_completed_archive(tmp_path):
    first, second = artifacts(tmp_path)
    for receipt in ({}, {'status': 'downloading'},
                    {'status': 'persisted', 'remote_cleanup': 'pending'}):
        (tmp_path / p.RECEIPT).write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match='fully persisted'):
            d.deduplicate_persisted_run(tmp_path)
        assert not first.samefile(second)
    (tmp_path / p.RECEIPT).write_text(json.dumps(
        {'status': 'persisted', 'remote_cleanup': 'removed'}))
    assert d.deduplicate_persisted_run(tmp_path)['linked_files'] == 1
