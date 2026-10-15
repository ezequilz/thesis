"""Python 3.9-compatible wrapper, executed in the separate authors' environment.

Runs train.py with a versioned run-local calibration correction.
Every upstream command is retained in commands.jsonl. Native surfels are kept.
"""
import json
import os
from pathlib import Path
import runpy
import shlex
import sys

try:
    from .corrected_runtime import prepare_runtime
    from .geometry import validate_pointmaps
    from . import source_depth_runtime
except ImportError:  # Direct invocation in the separate Python 3.9 environment.
    from corrected_runtime import prepare_runtime
    from geometry import validate_pointmaps
    import source_depth_runtime


def export_viewer_ply(source, target):
    import numpy as np
    # The pinned writer emits only scalar float32 vertex properties. Read that
    # precise native contract without importing the model's CUDA dependencies.
    names, count = [], None
    with Path(source).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Expected PLY')
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete native PLY header')
            fields = line.decode('ascii').strip().split()
            if fields[:1] == ['format'] and fields[1] != 'binary_little_endian':
                raise ValueError('Expected little-endian native PLY')
            if fields[:1] == ['element']:
                if fields[1] != 'vertex':
                    raise ValueError('Expected only native surfel vertices')
                count = int(fields[2])
            if fields[:1] == ['property']:
                if fields[1] != 'float':
                    raise ValueError('Expected float32 native surfel properties')
                names.append(fields[2])
            if fields == ['end_header']:
                break
        if count is None or count <= 0:
            raise ValueError('Native surfel PLY is empty')
        vertex = np.fromfile(stream, dtype=[(name, '<f4') for name in names], count=count)
    if len(vertex) != count:
        raise ValueError('Truncated native surfel PLY')
    if 'scale_2' in names or not {'scale_0', 'scale_1'}.issubset(names):
        raise ValueError('Expected native G4Splat two-axis Gaussian surfels')
    converted = np.empty(len(vertex), dtype=vertex.dtype.descr + [('scale_2', '<f4')])
    for name in names:
        converted[name] = vertex[name]
    # Local z is the surfel normal. Explicit visualization approximation only.
    converted['scale_2'] = np.minimum(vertex['scale_0'], vertex['scale_1']) + np.log(.01)
    header = ['ply', 'format binary_little_endian 1.0', 'element vertex ' + str(count)]
    header += ['property float ' + name for name in converted.dtype.names]
    with Path(target).open('wb') as stream:
        stream.write(('\n'.join(header) + '\nend_header\n').encode('ascii'))
        converted.tofile(stream)


def main():
    repo, source, output, count = sys.argv[1:]
    output = Path(output)
    source = str(Path(source).resolve())
    input_manifest = json.loads((Path(source) / 'inputs.json').read_text())
    use_source_depth = input_manifest.get('source_depth', {}).get('enabled') is True
    repo = str(prepare_runtime(repo, output.parent / 'corrected-runtime'))
    if use_source_depth:
        source_depth_runtime.patch_source_depth_runtime(repo)
    original_system = os.system
    geometry_checked = False
    def command(value):
        nonlocal geometry_checked
        tokens = shlex.split(value)
        skipped = len(tokens) > 1 and tokens[1] == '2d-gaussian-splatting/eval/eval.py'
        with (output.parent / 'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'command': value, 'skipped': skipped,
                                     'reason': 'No held-out ground truth' if skipped else None}) + '\n')
        if skipped:
            return 0
        if len(tokens) > 1 and tokens[1] == 'scripts/align_charts.py':
            if use_source_depth:
                source_depth_runtime.apply_source_depth_pointmaps(
                    output / 'mast3r_sfm', output.parent / 'source-depth-check.json')
            validate_pointmaps(output / 'mast3r_sfm', output.parent / 'reprojection-check.json')
            geometry_checked = True
        return original_system(value)
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.system = command
    sys.argv = [str(Path(repo) / 'train.py'), '-s', source, '-o', str(output),
                '--sfm_config', 'posed', '--use_view_config', '--config_view_num', count,
                '--select_inpaint_num', '10', '--tetra_downsample_ratio', '0.25']
    extension_env = {'SPLATFIX_SOURCE_DEPTH_INPUTS': source,
                     'SPLATFIX_SOURCE_DEPTH_SCENE': str((output / 'mast3r_sfm').resolve())}
    previous_env = {key: os.environ.get(key) for key in extension_env}
    for key, value in extension_env.items():
        if use_source_depth:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        os.system = original_system
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not geometry_checked:
        raise RuntimeError('G4Splat did not pass the pre-alignment geometry gate')
    native = output / 'free_gaussians/point_cloud/iteration_7000/point_cloud.ply'
    if not native.is_file():
        raise RuntimeError('Official final 7000-iteration surfel output is missing')
    export_viewer_ply(native, output.parent / 'g4splat-viewer.ply')


if __name__ == '__main__':
    main()
