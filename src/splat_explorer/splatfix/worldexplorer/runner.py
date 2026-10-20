"""Narrow runtime adapters for the pinned authors' pipeline and shared SEVA v1.1."""
import hashlib
import functools
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
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


def select_images_for_vggt(scene_root, max_images=100):
    """Authors' per-trajectory stride sampling, plus all scaffold inputs.

    max_images is an approximate generated-view budget, as in upstream.
    """
    if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 1:
        raise ValueError('VGGT generated-image budget must be a positive integer')
    root = Path(scene_root)
    frames = json.loads((root/'transforms.json').read_text())['frames']
    inputs, trajectories = [], {}
    for frame in frames:
        path = Path(frame['file_path'])
        # The pinned merger emits absolute paths; also accept scene-relative paths.
        if not path.is_absolute():
            path = root/path
        if path.parent.name == 'input':
            inputs.append(path)
        elif path.parent.name == 'samples-rgb':
            trajectories.setdefault(path.parent, []).append(path)
        else:
            raise ValueError(f'Unexpected WorldExplorer image folder: {path}')
    if not trajectories:
        raise ValueError(f'No generated trajectory frames in {root}/transforms.json')

    allowed_per_trajectory = max_images // len(trajectories)
    if allowed_per_trajectory == 0:
        raise ValueError(f'VGGT image budget must be at least {len(trajectories)} '
                         'to sample every trajectory')
    selected = list(inputs)
    for paths in trajectories.values():
        paths = sorted(paths, key=lambda p: int(p.stem))
        # Retain upstream's floor division and first-frame stride sampling.
        # A short trajectory can only contribute its available frames.
        stride = max(1, len(paths) // allowed_per_trajectory)
        selected.extend(paths[::stride])

    # Keep upstream's filename contract for find_pose_in_transforms.
    names = [f'{p.parent.parent.name}_{p.parent.name}_{p.name}' for p in selected]
    if len(set(names)) != len(names):
        raise ValueError('VGGT selected image filenames collide')
    output = root/'selected_for_vggt'
    if output.is_symlink():
        raise ValueError('VGGT selection directory must not be a symlink')
    # Prepare completely before replacing a previous selection.
    with tempfile.TemporaryDirectory(prefix='.vggt-selection-', dir=root) as temporary:
        staging = Path(temporary)/'images'
        staging.mkdir()
        for source, name in zip(selected, names):
            shutil.copy2(source, staging/name)
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    print(f'Selected {len(selected)-len(inputs)} generated views across '
          f'{len(trajectories)} trajectories, plus {len(inputs)} input views')


def main():
    request = json.loads(Path('request.json').read_text())
    # OracleGS already caches the compatible SEVA v1.1 checkpoint.
    # Install before pipeline imports bind their load_model references.
    import seva.utils
    seva.utils.load_model = functools.partial(seva.utils.load_model, weight_name='modelv1.1.safetensors')
    import model.scaffold_generation as scaffold
    import model.scene_expansion as expansion
    # Patch the bound reference, without modifying the shared upstream checkout.
    expansion.select_images_for_vggt = select_images_for_vggt
    from control.model_assets import verified_inpainting, inpainting_loader
    if request['operation'] != 'expand':
        scaffold.StableDiffusionInpaintPipeline = inpainting_loader(
            scaffold.StableDiffusionInpaintPipeline, verified_inpainting())
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
