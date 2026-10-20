"""Recover the completed trajectory stage without regenerating its images."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


def frame_relative(path, source_run):
    prefix = Path('/workspace/worldexplorer-runs')/source_run
    relative = Path(path).relative_to(prefix)
    if '..' in relative.parts:
        raise ValueError('Recovery frame escapes its source run')
    return relative


def verified_payload(root, source_run, archive_sha256):
    """Validate the saved archive and every file needed to resume training."""
    root = Path(root).resolve()
    raw = (root/'.archive-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != archive_sha256:
        raise ValueError('Recovery archive manifest changed; download again')
    records = {r['path']: r for r in json.loads(raw)['files']}

    def verify(relative):
        path = root/relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Recovery files must be inside the saved archive')
        record = records.get(str(relative), {})
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        if path.stat().st_size != record.get('size') or digest.hexdigest() != record.get('sha256'):
            raise ValueError(f'Recovery file changed or unverified: {relative}')
        return path

    transforms = list(root.glob('scenes/*/img2trajvid/transforms.json'))
    if len(transforms) != 1:
        raise ValueError('Recovery requires one completed trajectory scene')
    relative = transforms[0].relative_to(root)
    data = json.loads(verify(relative).read_text())
    files = {str(relative)}
    inputs, trajectories = set(), set()
    for frame in data['frames']:
        image = frame_relative(frame['file_path'], source_run)
        if not image.is_relative_to(relative.parent):
            raise ValueError('Recovery frame is outside the trajectory scene')
        verify(image)
        files.add(str(image))
        if image.parent.name == 'input':
            inputs.add(image.name)
        elif image.parent.name == 'samples-rgb':
            trajectories.add(image.parent.parent.name)
        else:
            raise ValueError('Unexpected recovery image directory')
    expected = {f'{i:03}_{direction}' for i in range(8) for direction in ('in','left','right','up')}
    if inputs != {f'{i:03}.png' for i in range(8)} or trajectories != expected:
        raise ValueError('Trajectory generation is incomplete; all 32 trajectories and 8 inputs are required')
    return dict(root=str(root), scene=str(relative.parent.parent), files=sorted(files),
                source_run=source_run, archive_sha256=archive_sha256)


def rebase_transforms(run, scene, source_run):
    """Relocate camera metadata to a new run; never mutate the archived source."""
    run = Path(run).resolve()
    relative = Path(scene)
    if len(relative.parts) != 2 or relative.parts[0] != 'scenes' or '..' in relative.parts:
        raise ValueError('Invalid recovery scene')
    path = run/relative/'img2trajvid/transforms.json'
    data = json.loads(path.read_text())
    for frame in data['frames']:
        image = frame_relative(frame['file_path'], source_run)
        target = run/image
        if not image.is_relative_to(relative/'img2trajvid') or not target.resolve().is_relative_to(run):
            raise ValueError('Recovery frame escapes its scene')
        if not target.is_file():
            raise ValueError(f'Missing recovery image: {image}')
        frame['file_path'] = str(target)
    # Alignment is rerun, so an earlier point-cloud path must not be reused.
    data.pop('ply_file_path', None)
    path.write_text(json.dumps(data, indent=2))
    return run/relative


def continuation_method(method, namespace):
    """Execute the pinned authors' unchanged run() suffix after generation.

    Extract at a checked AST boundary rather than copying training/export code
    or pretending that skipped generation subprocesses succeeded.
    """
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    function = tree.body[0]
    starts = [i for i, node in enumerate(function.body)
              if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == 'select_images_for_vggt']
    if len(starts) != 1:
        raise RuntimeError('Upstream recovery boundary changed; refusing to skip generation')
    function.body = function.body[starts[0]:]
    function.decorator_list = []
    function.name = '_resume_after_trajectories'
    compiled = compile(ast.fix_missing_locations(tree), inspect.getsourcefile(method), 'exec')
    scope = dict(namespace)
    exec(compiled, scope)
    return scope[function.name]
