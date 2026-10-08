"""Run the pinned official G4Splat recipe in its own CUDA environment."""
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
from ..checkpoint import Checkpoint, atomic_json
from ..executor import run_process
from ..resolution import prepare_repair_checkpoint
from .inputs import export_inputs

REVISION = 'ec0736126707a42bb2c26ed8ba2c314909edc7a9'
DEFAULT_RUNTIME = {'repo': '/workspace/third_party/G4Splat',
                   'python': '/workspace/g4splat-env/bin/python', 'resolution_profile': 'training'}


def validate_runtime(runtime):
    repo = Path(runtime['repo']).resolve()
    python = Path(runtime['python']).absolute()
    for path in (repo, python):
        # Upstream builds unquoted shell commands internally.
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(path)):
            raise ValueError('G4Splat upstream runtime paths must not contain spaces or shell metacharacters')
    if not (repo / 'train.py').is_file() or not python.is_file():
        raise FileNotFoundError('Provision the official G4Splat checkout and CUDA environment; see docs/g4splat.md')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != REVISION:
        raise ValueError(f'G4Splat requires pinned revision {REVISION}, found {head}')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('G4Splat tracked source must be unmodified')
    required = ['Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
                'mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
                'mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth',
                'mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl',
                'checkpoint/segment-anything/sam_vit_h_4b8939.pth', 'checkpoint/MVD_weights']
    missing = [p for p in required if not (repo / p).exists()]
    if missing:
        raise FileNotFoundError('Missing official G4Splat weights: ' + ', '.join(missing))
    return repo, python


def run_prepared(inputs, output_dir, *, runtime=None, should_stop=lambda: False, on_progress=lambda event: None):
    runtime = {**DEFAULT_RUNTIME, **(runtime or {})}
    repo, python = validate_runtime(runtime)
    source = Path(inputs).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep all paths consumed by upstream shell concatenation predictable.
    for path in (source, output):
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(path)):
            raise ValueError('G4Splat upstream input/output paths must not contain spaces or shell metacharacters')
    provenance = json.loads((source / 'inputs.json').read_text())
    recipe = {'reconstruction_method': 'g4splat', 'repository': 'https://github.com/DaLi-Jack/G4Splat',
              'revision': REVISION, 'sfm_config': 'posed', 'select_inpaint_num': 10,
              'tetra_downsample_ratio': .25, 'generation_rounds': 3,
              'held_out_evaluation': 'skipped: no held-out ground truth', 'inputs': provenance,
              'native_representation': '2d_gaussian_surfels',
              'viewer_conversion': {'normal_thickness_ratio': .01, 'approximation': True}}
    atomic_json(output / 'recipe.json', recipe)
    on_progress({'phase': 'g4splat', 'message': 'Official geometry, See3D and surfel reconstruction',
                 'output_dir': str(output)})
    runner = Path(__file__).with_name('official_runner.py')
    # Upstream invokes `python` for child stages. Use a shim to preserve the
    # configured interpreter even if its executable is named python3.9.
    shim = output / 'bin'
    shim.mkdir(exist_ok=True)
    (shim / 'python').symlink_to(python)
    command = ['env', '-u', 'PYTHONPATH', 'PATH=' + str(shim) + os.pathsep + os.environ.get('PATH', ''),
               str(python), '-u', str(runner), str(repo), str(source), str(output / 'official'),
               str(len(provenance['views']))]
    atomic_json(output / 'launch.json', {'argv': command})
    last_command = None
    def tick():
        nonlocal last_command
        trace = output / 'commands.jsonl'
        if trace.is_file():
            lines = trace.read_text().splitlines()
            if lines and lines[-1] != last_command:
                try:
                    entry = json.loads(lines[-1])
                except json.JSONDecodeError:
                    return  # Writer may still be appending this record.
                last_command = lines[-1]
                on_progress({'phase': 'g4splat', 'message': entry['command'], 'output_dir': str(output)})
    run_process(command, output / 'g4splat.log', should_stop, tick)
    if should_stop():
        raise InterruptedError('G4Splat stopped before publishing results')
    native = output / 'official/free_gaussians/point_cloud/iteration_7000/point_cloud.ply'
    viewer = output / 'g4splat-viewer.ply'
    if not native.is_file() or not viewer.is_file():
        raise RuntimeError('G4Splat did not produce its native and viewer PLY outputs')
    result = {'reconstruction_method': 'g4splat', 'mode': provenance['mode'],
              'output_dir': str(output), 'splat_path': str(viewer), 'native_splat_path': str(native),
              'native_representation': '2d_gaussian_surfels', 'viewer_is_approximation': True,
              'upstream_revision': REVISION, 'input_manifest': str(source / 'inputs.json'),
              'limitation': 'Native G4Splat output is 2D Gaussian surfels. Viewer PLY uses an approximate 1% normal thickness; evaluate with the native renderer.'}
    atomic_json(output / 'result.json', result)
    return result


def run_repair(checkpoint_dir, output_dir, *, mode='edited', runtime=None,
               should_stop=lambda: False, on_progress=lambda event: None, **unused):
    runtime = {**DEFAULT_RUNTIME, **(runtime or {})}
    validate_runtime(runtime)  # Fail before requesting any captures.
    checkpoint = Checkpoint.load(checkpoint_dir)
    from .backend import readiness
    if not readiness(checkpoint)['ready']:
        raise ValueError(readiness(checkpoint)['reason'])
    if mode == 'edited':
        checkpoint.require_viser_images()
        for view in checkpoint.views:
            checkpoint.image_path(view, repaired=True)
    output = Path(output_dir).resolve() / ('g4splat_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    on_progress({'phase': 'inputs', 'message': 'Recapturing calibrated reconstruction inputs'})
    prepared = prepare_repair_checkpoint(checkpoint, output / 'checkpoint', runtime['resolution_profile'],
                                         scene_path=runtime.get('scene_path'), should_stop=should_stop)
    export_inputs(prepared, output / 'inputs', mode)
    return run_prepared(output / 'inputs', output, runtime=runtime, should_stop=should_stop, on_progress=on_progress)
