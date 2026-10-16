"""Stop-only iteration hook for the pinned trainer; shared sources stay intact."""
import ast
import hashlib
import json
from pathlib import Path
import time


def write_json(path, body):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(body, indent=2))
    temporary.replace(path)


def iteration_boundary(scene, iteration, control):
    control = Path(control)
    if not control.exists():
        return
    import torch
    root = control.parent
    state = {'status':'saving', 'iteration':int(iteration)}
    write_json(root/'stop-state.json',state)
    try:
        torch.cuda.synchronize()
        temporary = root/'stop-snapshot.tmp.ply'
        scene.gaussians.save_ply(str(temporary))
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError('Trainer produced an empty stop snapshot')
        native = root/'stop-snapshot.ply'
        temporary.replace(native)
        write_json(root/'stop-state.json', {**state, 'status':'saved',
            'size_bytes':native.stat().st_size})
    except Exception as exc:
        write_json(root/'stop-state.json',{**state,'status':'save_failed','error':str(exc)})
        raise
    raise SystemExit(0)


def training_code(repo):
    """Insert a stop check after all optimizer/MCMC updates of each iteration."""
    path = Path(repo)/'train.py'
    source = path.read_bytes()
    lock = json.loads(Path(__file__).with_name('upstream-lock.json').read_text())
    if hashlib.sha256(source).hexdigest() != lock['oraclegs']['files']['train.py']:
        raise ValueError('Unexpected OracleGS trainer; refusing graceful-stop hook')
    tree = ast.parse(source,filename=str(path))
    training = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training')
    loops = [n for n in training.body if isinstance(n,ast.For)
             and isinstance(n.target,ast.Name) and n.target.id=='iteration']
    if len(loops) != 1:
        raise ValueError('OracleGS iteration boundary is ambiguous')
    loops[0].body.append(ast.parse('_splatfix_stop_boundary(scene, iteration)').body[0])
    return compile(ast.fix_missing_locations(tree),str(path),'exec')


def run_training(repo, control):
    code = training_code(repo)
    namespace = {'__name__':'__main__','__file__':str(Path(repo)/'train.py'),
                 '_splatfix_stop_boundary':lambda scene,iteration: iteration_boundary(scene,iteration,control)}
    exec(code,namespace)


def stop_poll(output, should_stop, on_progress, timeout=240):
    output = Path(output)
    started = None
    def poll():
        nonlocal started
        if started is None and not should_stop():
            return False
        if started is None:
            started = time.monotonic()
            (output/'SAVE_STOP').touch()
            on_progress({'phase':'stopping','message':'Saving OracleGS at the next completed iteration, then downloading the partial PLY'})
        if time.monotonic()-started >= timeout:
            write_json(output/'stop-timeout.json',{'seconds':timeout})
            return True
        return False
    return poll
