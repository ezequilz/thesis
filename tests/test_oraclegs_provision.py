from pathlib import Path

import pytest

from splat_explorer.splatfix.oraclegs import provision


FILES = ('bin/nvcc', 'include/cuda.h', 'include/cuda_runtime.h', 'lib/libcudart.so')


def toolkit(root):
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')


def test_repairs_cuda_even_when_python_environment_already_exists(tmp_path, monkeypatch):
    python = tmp_path / 'oraclegs-env/bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    cuda = tmp_path / 'oraclegs-cuda-11.7'
    def run(*args):
        calls.append(args)
        toolkit(cuda)
    monkeypatch.setattr(provision.subprocess, 'check_output', lambda *a, **kw: 'Cuda compilation tools, release 11.7, V11.7.99')
    assert provision.ensure_cuda(tmp_path, 'mamba', run) == cuda
    assert calls[0][1] == 'create'
    assert 'cuda-nvcc=11.7.99' in calls[0]
    assert 'nvidia/label/cuda-11.7.1' in calls[0]
    assert provision.ensure_cuda(tmp_path, 'mamba', run) == cuda
    assert len(calls) == 1


def test_partial_cuda_environment_is_repaired_not_trusted(tmp_path, monkeypatch):
    cuda = tmp_path / 'oraclegs-cuda-11.7'
    history = cuda / 'conda-meta/history'
    history.parent.mkdir(parents=True)
    history.touch()
    calls = []
    def run(*args):
        calls.append(args)
        toolkit(cuda)
    monkeypatch.setattr(provision.subprocess, 'check_output', lambda *a, **kw: 'release 11.7,')
    provision.ensure_cuda(tmp_path, 'mamba', run)
    assert calls[0][1] == 'install'
    assert '--force-reinstall' in calls[0]


def test_installer_success_without_working_compiler_fails_early(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='refusing extension build'):
        provision.ensure_cuda(tmp_path, 'mamba', lambda *a: None)
    toolkit(tmp_path / 'oraclegs-cuda-11.7')
    monkeypatch.setattr(provision.subprocess, 'check_output', lambda *a, **kw: 'release 12.6,')
    with pytest.raises(RuntimeError, match='refusing extension build'):
        provision.ensure_cuda(tmp_path, 'mamba', lambda *a: None)


def test_runtime_checks_do_not_inherit_container_python_or_architectures(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_CUDA_ARCH_LIST', '8.7;9.0')
    monkeypatch.setenv('PYTHONPATH', '/container/newer-torch')
    monkeypatch.setenv('PYTHONHOME', '/container/python')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/container/lib')
    env = provision.runtime_environment(tmp_path)
    assert not {'TORCH_CUDA_ARCH_LIST', 'PYTHONPATH', 'PYTHONHOME'} & env.keys()
    assert env['LD_LIBRARY_PATH'].split(':')[0] == str(tmp_path / 'lib')
    assert '/container/lib' not in env['LD_LIBRARY_PATH']
    assert env['PYTHONNOUSERSITE'] == '1'
    assert provision.os.environ['TORCH_CUDA_ARCH_LIST'] == '8.7;9.0'


def test_broken_installed_extension_requests_setup_repair(monkeypatch):
    import subprocess
    from splat_explorer.splatfix.oraclegs import repair
    def broken(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output='ImportError: missing CUDA extension')
    monkeypatch.setattr(repair.subprocess, 'check_output', broken)
    with pytest.raises(FileNotFoundError, match='missing CUDA extension'):
        repair.probe_runtime(repair.DEFAULT_RUNTIME)


def test_runtime_probe_checks_both_isolated_environments(monkeypatch):
    from splat_explorer.splatfix.oraclegs import repair
    calls = []
    monkeypatch.setattr(repair.subprocess, 'check_output', lambda *args, **kw: calls.append((args, kw)))
    repair.probe_runtime(repair.DEFAULT_RUNTIME)
    assert len(calls) == 2
    assert 'diff_gaussian_rasterization' in calls[0][0][0][-1]
    assert 'seva.eval' in calls[1][0][0][-1]
    for args, kw in calls:
        assert kw['timeout'] == 60
        assert 'PYTHONPATH' not in kw['env']


def test_hf_paths_ignore_redirected_launcher_home(monkeypatch):
    import pwd
    from types import SimpleNamespace
    monkeypatch.setattr(pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_dir='/accounts/researcher'))
    monkeypatch.setenv('HOME', '/workspace/python/home')
    monkeypatch.setenv('XDG_CACHE_HOME', '/workspace/python/cache')
    monkeypatch.setenv('HF_HOME', '/wrong/hf')
    monkeypatch.setenv('HF_HUB_CACHE', '/wrong/hub')
    monkeypatch.setenv('HF_TOKEN_PATH', '/wrong/token')
    env=provision.runtime_environment('/workspace/oraclegs-model-env')
    assert env['HF_HOME']=='/accounts/researcher/.cache/huggingface'
    assert env['HF_TOKEN_PATH']==env['HF_HOME']+'/token'
    assert env['HF_HUB_CACHE']==env['HUGGINGFACE_HUB_CACHE']==env['HF_HOME']+'/hub'
