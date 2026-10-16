"""Prepared remote jobs must not import local-only trajectory dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_worker_starts_and_publishes_status_without_splines(tmp_path):
    (tmp_path/'STOP').touch()
    (tmp_path/'worker-request.json').write_text(json.dumps({
        'reconstruction_method':'oraclegs', 'stage':'repair',
        'output':str(tmp_path/'output')}))
    source = Path(__file__).resolve().parents[1]/'src'
    script = '''
import importlib.abc
import sys
class MissingSplines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'splines' or fullname.startswith('splines.'):
            raise ModuleNotFoundError("No module named 'splines'", name='splines')
sys.meta_path.insert(0, MissingSplines())
from splat_explorer.splatfix.oraclegs.inputs import export_inputs
from splat_explorer.splatfix.job_worker import execute
execute(sys.argv[1])
assert 'splines' not in sys.modules
'''
    completed = subprocess.run([sys.executable,'-c',script,str(tmp_path)],
        env={**os.environ,'PYTHONPATH':str(source)},capture_output=True,text=True,timeout=30)
    assert completed.returncode == 0, completed.stderr
    status = json.loads((tmp_path/'worker-status.json').read_text())
    assert status['status'] == 'stopped'
