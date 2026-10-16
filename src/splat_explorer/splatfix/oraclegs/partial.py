"""Publish only a confirmed, fully saved upstream snapshot after interruption."""
from pathlib import Path
import shutil

from ..checkpoint import atomic_json
from ..jobs import read_json


def salvage(output, reason):
    output = Path(output)
    stopped = read_json(output/'stop-state.json', {})
    fresh = stopped.get('status') == 'saved'
    record = stopped if fresh else read_json(output/'model/saved-snapshot.json', {})
    iteration = record.get('iteration')
    if type(iteration) is not int or iteration <= 0:
        return None
    native = output/'stop-snapshot.ply' if fresh else output/f'model/point_cloud/iteration_{iteration}/point_cloud.ply'
    if not native.resolve().is_relative_to(output.resolve()) or not native.is_file():
        return None
    if native.stat().st_size != record.get('size_bytes') or not native.stat().st_size:
        return None
    ply = output/'oraclegs-partial.ply'
    temporary = ply.with_suffix('.tmp')
    shutil.copyfile(native,temporary)
    temporary.replace(ply)
    recipe = read_json(output/'recipe.json', {})
    result = {'reconstruction_method':'oraclegs', 'incomplete':True, 'splat_path':str(ply),
              'saved_iteration':iteration, 'saved_on_stop':fresh, 'method_variant':recipe.get('method_variant'),
              'interruption':{'reason':reason},
              'limitation':('Saved at a completed iteration after the stop request. Not a completed reconstruction.' if fresh else
                            'Latest fully saved upstream snapshot; may precede stop request. Not a completed reconstruction.')}
    atomic_json(output/'partial-result.json',result)
    return result
