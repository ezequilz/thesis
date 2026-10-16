import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_script():
    path = Path(__file__).resolve().parents[1]/'scripts/lrz/setup_oraclegs_access.py'
    spec = importlib.util.spec_from_file_location('oracle_access_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_token_only_travels_over_stdin(monkeypatch, capsys):
    script = load_script()
    from splat_explorer import repair_lrz as lrz
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {})
    monkeypatch.setattr(lrz, 'lrz_session_alive', lambda cfg: True)
    monkeypatch.setattr(script, 'remote_command', lambda cfg: ['ssh', 'worker', 'command'])
    monkeypatch.setattr(script.sys, 'argv', ['setup_oraclegs_access.py'])
    monkeypatch.setattr(script.getpass, 'getpass', lambda prompt: 'hf_test_secret')
    calls=[]
    def run(argv, **kw):
        calls.append((argv,kw))
        return SimpleNamespace(returncode=0, stdout='ORACLEGS_ASSETS=["/cache/config.yaml", "/cache/modelv1.1.safetensors"]\n')
    monkeypatch.setattr(script.subprocess, 'run', run)
    script.main()
    assert calls[0][1]['input']=='hf_test_secret\n'
    assert 'hf_test_secret' not in str(calls[0][0])
    assert 'hf_test_secret' not in capsys.readouterr().out


def test_remote_program_downloads_only_required_files_without_echoing_secret(monkeypatch, capsys):
    script=load_script()
    import io,sys
    calls=[]
    from splat_explorer.splatfix.oraclegs import provision
    monkeypatch.setitem(sys.modules,'provision', provision)
    monkeypatch.setattr(provision, 'huggingface_environment', lambda: {})
    fake=SimpleNamespace(login=lambda **kw:calls.append(('login',kw)),
        hf_hub_download=lambda **kw: calls.append(('download',kw)) or '/cache/'+kw['filename'])
    monkeypatch.setitem(sys.modules,'huggingface_hub',fake)
    monkeypatch.setattr(sys,'stdin',io.StringIO('hf_test_secret\n'))
    monkeypatch.setattr(script.os if hasattr(script,'os') else __import__('os'),'umask',lambda n:None)
    exec(compile(script.REMOTE,'remote_setup','exec'),{})
    assert [kw['filename'] for action,kw in calls if action=='download']==['config.yaml','modelv1.1.safetensors']
    assert calls[0][1]['add_to_git_credential'] is False
    assert 'hf_test_secret' not in capsys.readouterr().out


def test_check_only_never_prompts_or_copies_locally(monkeypatch):
    script = load_script()
    from splat_explorer import repair_lrz as lrz
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {})
    monkeypatch.setattr(lrz, 'lrz_session_alive', lambda cfg: True)
    monkeypatch.setattr(script, 'remote_command', lambda cfg, **kw: ['ssh', str(kw['copy_only'])])
    monkeypatch.setattr(script.sys, 'argv', ['setup', '--check-only'])
    monkeypatch.setattr(script.getpass, 'getpass', lambda prompt: (_ for _ in ()).throw(AssertionError('prompted')))
    calls=[]
    def run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stdout='ORACLEGS_ASSETS=["/cache/config.yaml", "/cache/modelv1.1.safetensors"]\n')
    monkeypatch.setattr(script.subprocess, 'run', run)
    script.main()
    assert len(calls) == 1 and calls[0][0] == ['ssh', 'True']
    assert calls[0][1]['input'] == ''


def test_explicit_copy_works_with_old_rsync(monkeypatch, tmp_path):
    script = load_script()
    from splat_explorer import repair_lrz as lrz
    monkeypatch.setattr(lrz, 'load_lrz_config', lambda: {'user':'u','host':'host'})
    monkeypatch.setattr(lrz, 'lrz_session_alive', lambda cfg: True)
    monkeypatch.setattr(lrz, 'rsync_ssh_cmd', lambda cfg: 'ssh')
    monkeypatch.setattr(script, 'remote_command', lambda cfg, **kw: ['ssh'])
    monkeypatch.setattr(script.sys, 'argv', ['setup','--copy-only','--download-local',str(tmp_path)])
    calls=[]
    def run(argv, **kw):
        calls.append(argv)
        assert '--protect-args' not in argv
        return SimpleNamespace(returncode=0, stdout='ORACLEGS_ASSETS=["/cache/space dir/config.yaml", "/cache/modelv1.1.safetensors"]\n')
    monkeypatch.setattr(script.subprocess, 'run', run)
    script.main()
    assert len(calls)==3
    assert calls[1][-2] == "u@host:'/cache/space dir/config.yaml'"
