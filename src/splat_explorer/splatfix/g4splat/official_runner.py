"""Python 3.9-compatible wrapper, executed in the separate authors' environment.

Runs unmodified train.py. Only the held-out benchmark evaluator is skipped.
Every upstream command is retained in commands.jsonl. Native surfels are kept.
"""
import json
import os
from pathlib import Path
import runpy
import shlex
import sys


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
    original_system = os.system
    def command(value):
        tokens = shlex.split(value)
        skipped = len(tokens) > 1 and tokens[1] == '2d-gaussian-splatting/eval/eval.py'
        with (output.parent / 'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'command': value, 'skipped': skipped,
                                     'reason': 'No held-out ground truth' if skipped else None}) + '\n')
        if skipped:
            return 0
        return original_system(value)
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.system = command
    sys.argv = [str(Path(repo) / 'train.py'), '-s', source, '-o', str(output),
                '--sfm_config', 'posed', '--use_view_config', '--config_view_num', count,
                '--select_inpaint_num', '10', '--tetra_downsample_ratio', '0.25']
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        os.system = original_system
    native = output / 'free_gaussians/point_cloud/iteration_7000/point_cloud.ply'
    if not native.is_file():
        raise RuntimeError('Official final 7000-iteration surfel output is missing')
    export_viewer_ply(native, output.parent / 'g4splat-viewer.ply')


if __name__ == '__main__':
    main()
