"""Narrow runtime adapters for the pinned authors' pipeline and shared SEVA v1.1."""
import hashlib
import functools
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import types


def safe_theme(theme):
    slug = re.sub('[^a-zA-Z0-9]+', '_', theme).strip('_')[:64] or 'scene'
    return slug + '_' + hashlib.sha256(theme.encode()).hexdigest()[:8]


def generation_args(request):
    if request['operation'] == 'expand':
        return ['worldexplorer.py', 'expand', 'input_scaffold', '--translation-scaling',
                str(request['translation_scaling']), '--num-images-for-vggt', str(request['num_images_for_vggt'])]
    if request['mode'] == 'manual':
        return ['worldexplorer.py', 'scaffold', '--mode', 'manual', '--', request['prompt']]
    return ['worldexplorer.py', 'generate', '--mode', request['mode'],
            '--translation-scaling', str(request['translation_scaling']),
            '--num-images-for-vggt', str(request['num_images_for_vggt']), '--', request['prompt']]


def seva_command(command):
    return command.replace('python model/stable-virtual-camera/demo.py ',
                           'python runner.py --seva ', 1)


def main():
    request = json.loads(Path('request.json').read_text())
    # OracleGS already caches the compatible SEVA v1.1 checkpoint.
    # Install before pipeline imports bind their load_model references.
    import seva.utils
    seva.utils.load_model = functools.partial(seva.utils.load_model, weight_name='modelv1.1.safetensors')
    import model.scaffold_generation as scaffold
    import model.scene_expansion as expansion
    scaffold.create_clean_folder_name = safe_theme
    original = expansion.TrajectoryTransformer.run_command
    def checked(self, command):
        code = original(self, seva_command(command))
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code
    expansion.TrajectoryTransformer.run_command = checked
    def checked_run(command, **kwargs):
        result = subprocess.run(command, **kwargs)
        result.check_returncode()
        # Upstream trains splatfacto-big but searches for a splatfacto folder.
        if isinstance(command, list) and command[0] == 'ns-train':
            for folder in Path('nerfstudio_output').glob('*/splatfacto-big'):
                alias = folder.parent/'splatfacto'
                if not alias.exists():
                    alias.symlink_to(folder.name, target_is_directory=True)
        return result
    expansion.subprocess = types.SimpleNamespace(Popen=subprocess.Popen, PIPE=subprocess.PIPE,
                                                 STDOUT=subprocess.STDOUT, run=checked_run)
    sys.argv = generation_args(request)
    runpy.run_path('worldexplorer.py', run_name='__main__')

if __name__ == '__main__':
    if sys.argv[1:2] == ['--seva']:
        import seva.utils
        seva.utils.load_model = functools.partial(seva.utils.load_model, weight_name='modelv1.1.safetensors')
        demo = 'model/stable-virtual-camera/demo.py'
        sys.argv = [demo, *sys.argv[2:]]
        sys.path.insert(0, str(Path(demo).parent.resolve()))
        runpy.run_path(demo, run_name='__main__')
    else:
        main()
