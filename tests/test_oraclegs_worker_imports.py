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


def test_source_depth_imports_real_upstream_camera_modules_without_cycle(tmp_path):
    """Keep the actual scene/camera_utils import cycle; stub only CUDA dependencies."""
    upstream = Path(__file__).resolve().parents[1]/'src/splat_explorer/splatfix/oraclegs/upstream/oraclegs'
    script = '''
import sys, types
from splat_explorer.splatfix.oraclegs.source_geometry_runtime import install_source_geometry
sys.path.insert(0, sys.argv[1])
def stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
stub('torch')
stub('cv2')
stub('utils.system_utils', searchForMaxIteration=None)
stub('scene.dataset_readers', sceneLoadTypeCallbacks={})
stub('scene.gaussian_model', GaussianModel=type('GaussianModel', (), {}))
stub('scene.cameras', Camera=None)
stub('arguments', ModelParams=None)
stub('utils.general_utils', PILtoTorch=None)
stub('utils.graphics_utils', fov2focal=None)
install_source_geometry({'source_depth': sys.argv[2]})
from scene import Scene
from utils.camera_utils import cameraList_from_camInfos
assert Scene is not None and callable(cameraList_from_camInfos)
'''
    completed = subprocess.run([sys.executable, '-c', script, str(upstream), str(tmp_path)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
