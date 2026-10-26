"""Lean GPU planning followed by full local source capture and full staging."""
import json
import shlex
import shutil
from pathlib import Path

from ..checkpoint import atomic_json
from ..persistence import checkpoint_cache_lock
from .preflight import prepare_source_captures
from .staging_cache import digest


def lean_checkpoint(checkpoint, preflight, destination):
    """Six selected inputs plus their small capture/anchor-geometry caches only."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    paths = set()
    for view in checkpoint.views:
        paths.update(view[key] for key in ('original_rgb', 'repaired_rgb', 'manual_resized_rgb') if view.get(key))
        response = (view.get('image_repair') or {}).get('response_image')
        if response:
            paths.add(response)
    for relative in paths:
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Lean checkpoint inputs must use checkpoint-relative paths')
        source = checkpoint.image_path({'original_rgb': relative})
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_json(destination / 'checkpoint.json', checkpoint.manifest)
    repeat = checkpoint.root / 'repeat-repair.json'
    if repeat.is_file():
        shutil.copy2(repeat, destination / repeat.name)
    cache = Path(preflight['input_cache'])
    target = destination / cache.relative_to(checkpoint.root)
    target.mkdir(parents=True)
    saved = json.loads((cache / 'captures.json').read_text())
    for filename in ['captures.json', *(f'{index:05d}.png' for index in range(len(saved['sha256'])))]:
        shutil.copy2(cache / filename, target / filename)
    for marker in (cache / 'trajectories').glob('*/trajectory.json'):
        if json.loads(marker.read_text()).get('recipe', {}).get('kind') == 'scale_anchors':
            shutil.copytree(marker.parent, target / marker.parent.relative_to(cache),
                ignore=lambda directory, names: [name for name in names if (Path(directory) / name).is_symlink()])
    return destination


class SourceStagingMirror:
    """Resume the worker's durable planning/capture barrier after manager restarts."""
    def __init__(self, root, remote, container, checkpoint, endpoint, ssh, sync, *,
                 should_stop=lambda: False, progress=lambda **event: None, url=None):
        self.root, self.remote, self.container = Path(root), remote, container
        self.checkpoint, self.endpoint = checkpoint, endpoint
        self.ssh, self.sync, self.should_stop, self.progress, self.url = ssh, sync, should_stop, progress, url

    def poll(self, state):
        if state.get('phase') != 'awaiting_source_capture':
            return
        request = json.loads((self.root / 'worker-request.json').read_text())
        if not request.get('prepare_source_first'):
            raise ValueError('Unexpected source capture barrier for a historical run')
        ready = self.root / 'full-staging.json'
        failed = self.root / 'source-capture-error.json'
        if ready.is_file() or failed.is_file():
            marker = ready if ready.is_file() else failed
            self.sync(str(marker), self.endpoint + ':' + self.remote + '/' + marker.name)
            return
        if self.should_stop():
            raise InterruptedError('Stopped before trajectory capture')
        raw = self.ssh('cat ' + shlex.quote(self.remote + '/source-plan.json'))
        plan = json.loads(raw)
        atomic_json(self.root / 'source-plan.json', plan)
        try:
            if digest(self.checkpoint.root / 'checkpoint.json') != request['source_checkpoint_sha256']:
                raise ValueError('Checkpoint changed during source preparation; start a new run with the updated views')
            prepared = prepare_source_captures(self.checkpoint, request['runtime'], self.root,
                source_plan=plan, should_stop=self.should_stop, progress=self.progress, url=self.url)
        except InterruptedError:
            raise
        except Exception as exc:
            atomic_json(failed, {'error': f'Source capture before full staging failed: {type(exc).__name__}: {exc}'})
            self.sync(str(failed), self.endpoint + ':' + self.remote + '/' + failed.name)
            return
        if self.should_stop():
            raise InterruptedError('Stopped before full staging')
        self.progress(phase='staging', message='All source trajectory RGBs captured; staging full checkpoint on LRZ')
        with checkpoint_cache_lock(self.checkpoint.root):
            if digest(self.checkpoint.root / 'checkpoint.json') != prepared['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after trajectory capture')
            self.sync(str(self.checkpoint.root) + '/', self.endpoint + ':' + self.remote + '/checkpoint/',
                      exclude=('worldexplorer-source/',))
        staged = {'source_plan_sha256': plan['sha256'], 'trajectory_frames': prepared['trajectory_frames']}
        if request['runtime']['reconstruction_images'] == 'all':
            from ..worldexplorer.source import stage_checkpoint_source
            originals = stage_checkpoint_source(self.checkpoint, self.root / 'worldexplorer-source')
            if originals:
                self.sync(originals + '/', self.endpoint + ':' + self.remote + '/worldexplorer-source/')
                staged['worldexplorer_source'] = self.container + '/worldexplorer-source'
        if self.should_stop():
            raise InterruptedError('Stopped before releasing reconstruction')
        atomic_json(ready, staged)
        self.sync(str(ready), self.endpoint + ':' + self.remote + '/' + ready.name)
        self.progress(phase='reconstruction', message='Full staging complete; reconstruction uses the captured source trajectory')
