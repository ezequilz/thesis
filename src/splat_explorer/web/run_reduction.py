"""Minimal, self-contained OracleGS runs and verified pre-reduction backups."""
import hashlib
import os
import shutil
import uuid
from pathlib import Path

from ..splatfix.jobs import read_json
from ..scene_runs.store import _atomic_write_json


def files(root):
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                dirs.remove(name)
                yield path
        for name in names:
            yield Path(directory) / name


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.digest()


def backup_run(root, history):
    history.mkdir(parents=True, exist_ok=True)
    destination = history / root.name
    if destination.exists() or destination.is_symlink():
        raise ValueError('A backup already exists in run_history; move it before making another full backup')
    temporary = history / ('.' + root.name + '-' + uuid.uuid4().hex + '.partial')
    try:
        # Independent copies, never hard links back to files we will rewrite.
        shutil.copytree(root, temporary, symlinks=True)
        for source in files(root):
            copied = temporary / source.relative_to(root)
            if source.is_symlink():
                if not copied.is_symlink() or os.readlink(source) != os.readlink(copied):
                    raise OSError('Backup symlink verification failed')
            elif digest(source) != digest(copied):
                raise OSError('Backup checksum verification failed')
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return str(destination.resolve())


def reduce_oraclegs(root, history=None, *, before_size_bytes=None):
    result = root / 'gpu/results/oraclegs'
    manifest = read_json(result / 'result.json', {})
    config = read_json(root / 'config.json', {}).get('splatfix', {})
    inputs = read_json(root / 'gpu/inputs/inputs.json', {}) or read_json(result / 'recipe.json', {}).get('inputs', {})
    if 'oraclegs' not in (manifest.get('reconstruction_method'), config.get('reconstruction_method'), inputs.get('reconstruction_method')):
        raise ValueError('Reduction supports OracleGS runs only')
    if (root / 'reduction.json').exists():
        raise ValueError('This run has already been reduced')
    references = []
    missing_references = []
    for view in inputs.get('views', []):
        name = Path(view['image']).name
        candidates = [result / 'dataset/images' / name, root / 'gpu/inputs' / view['image']]
        source = next((p for p in candidates if p.is_file() and p.resolve().is_relative_to(root.resolve())), None)
        if source is None:
            missing_references.append(name)
            continue
        references.append((source, result / 'dataset/images' / name))
    original = result / 'original.ply'
    source_ply = original if original.is_file() else root / 'gpu/inputs/source-initialization.ply'
    copies = list(references)
    if source_ply.is_file():
        copies.append((source_ply, original))
    retained_plys = {p for p in (original, result / 'oraclegs.ply', root / 'scene_original.ply', root / 'scene_repaired.ply') if p.is_file()}
    retained_sources = [p for p, _ in copies] + list(retained_plys)
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in retained_sources):
        raise ValueError('Cannot reduce linked artifacts outside the run')
    if any(not target.resolve().is_relative_to(root.resolve()) for _, target in copies) or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Cannot reduce linked destinations outside the run')
    backup = backup_run(root, history) if history else None
    # Materialize references and original at the locations used by the viewer.
    for source, target in copies:
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    keep = {p for _, p in copies} | retained_plys
    keep.update(result / name for name in ('result.json', 'partial-result.json', 'input-preview.json', 'inference-preview.json') if (result / name).is_file())
    # Keep lifecycle/provenance records, not source datasets or camera exports.
    keep.update(p for p in root.glob('*.json') if p.name not in ('output-size.json', 'worker-request.json', 'oraclegs-staging.json'))
    keep.add(root / 'comments.md')
    keep.update(p for p in files(root) if not p.is_symlink() and (p.suffix in ('.log', '.jsonl') or p.name.startswith('events.out.tfevents')))
    keep.update(p for p in result.glob('**/samples-rgb.mp4') if 'first-pass' not in p.parts)
    recipe = read_json(result / 'recipe.json', {})
    minimal_inputs = {key: inputs[key] for key in ('seva_task', 'seva_segments', 'oracle_adapter', 'oracle_preset') if key in inputs}
    minimal_inputs['views'] = [{key: view[key] for key in ('view_id', 'image', 'sha256', 'source_kind') if key in view} for view in inputs.get('views', [])]
    # Segment labels need endpoints, but never their dense pose arrays.
    minimal_inputs['seva_segments'] = [{k: v for k, v in segment.items() if k in ('index', 'from_reference', 'to_reference', 'preset')} for segment in inputs.get('seva_segments', [])]
    recipe = {key: recipe[key] for key in ('commands', 'revisions', 'reconstruction_method') if key in recipe}
    recipe['inputs'] = minimal_inputs
    _atomic_write_json(result / 'recipe.json', recipe)
    keep.add(result / 'recipe.json')
    # Invalidate the saved size even if deletion fails halfway through.
    (root / 'output-size.json').unlink(missing_ok=True)
    for path in files(root):
        if path not in keep or path.is_symlink():
            path.unlink()
    for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
        if Path(directory) != root and not any(Path(directory).iterdir()):
            Path(directory).rmdir()
    report = {'backup_path': backup, 'reference_count': len(references), 'missing_references': missing_references, 'reduced': True}
    report['before_size_bytes'] = before_size_bytes
    _atomic_write_json(root / 'reduction.json', report)
    return report


def encode_inference_video(frames, destination):
    """Stream disk frames through H.264, then verify count and full decoding."""
    import json
    import subprocess
    import tempfile
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('Reduction needs ffmpeg and ffprobe; no images were removed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.encoding.mp4')
    def command(args):
        try:
            return subprocess.run(args, check=True, capture_output=True, text=True, timeout=1800).stdout
        except (subprocess.SubprocessError, OSError) as exc:
            raise ValueError('Inference video export or verification failed; source images were kept') from exc
    try:
        with tempfile.TemporaryDirectory(prefix='artifixer-video-') as staging:
            for index, frame in enumerate(frames):
                (Path(staging) / f'{index:08d}.png').symlink_to(frame.resolve())
            command(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
                     '-framerate', '12', '-i', str(Path(staging) / '%08d.png'),
                     '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264',
                     '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)])
        info = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                  '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(temporary)]))
        if not info.get('streams') or int(info['streams'][0].get('nb_read_frames', 0)) != len(frames):
            raise ValueError('Video frame count differs from source images; no images were removed')
        command(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(temporary), '-f', 'null', '-'])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def reduce_artifixer(root, plans, history=None, *, before_size_bytes=None, export_only=False,
                     preserve_reconstruction=False):
    """Encode every prediction batch before pruning any source artifacts."""
    if (root / 'reduction.json').exists():
        raise ValueError('This run has already been reduced')
    if not plans:
        raise ValueError('No ArtiFixer result directories were found')
    keep = {p for p in root.glob('*.json') if p.name not in ('output-size.json', 'worker-request.json')}
    keep.add(root / 'comments.md')
    keep.update(p for p in files(root) if not p.is_symlink() and
                (p.suffix in ('.log', '.jsonl') or p.name.startswith('events.out.tfevents')))
    sequences = []
    incomplete_frames = []
    copies = []
    for plan in plans:
        result = plan['root']
        if not result.resolve().is_relative_to(root.resolve()):
            raise ValueError('Result directory escapes run')
        copies.extend(plan['ply_copies'])
        # Keep 3D+ image outputs independently of video batch discovery, including
        # images in nested output folders. All three reduction modes share this rule.
        plus_frames = plan.get('plus_frames_dir')
        if plus_frames and plus_frames.is_dir():
            for path in files(plus_frames):
                if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Cannot preserve linked ArtiFixer3D+ images')
                if path.stat().st_size > 0:
                    keep.add(path)
        if preserve_reconstruction:
            if not plan.get('trajectory'):
                raise ValueError('Saved inference trajectory is required to keep reconstruction inputs')
            for path in plan.get('reconstruction_files', []):
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Cannot preserve linked or missing reconstruction inputs')
                keep.add(path)
            keep.add(result / 'preview-trajectory.json')
        for name in ('result.json', 'partial-result.json', 'inference-preview.json', 'input-preview.json',
                     'benchmark-run.json', 'published-test-metrics.json', 'request.json',
                     'failure.json', 'distill.lifecycle.json', 'anchor-transforms.json',
                     'orbit-full.json', 'orbit-provenance.json'):
            if (result / name).is_file():
                keep.add(result / name)
        for pattern in ('stage-comparison*.jpg', 'stage-comparison.json', 'trajectory-quality.png', 'trajectory-quality.json'):
            keep.update(p for p in result.glob(pattern) if p.is_file() and not p.is_symlink())
        keep.update(p for p in result.glob('**/stop-state.json') if p.is_file() and not p.is_symlink())
        folders = set(result.glob('**/frames/*/pred'))
        folders.update(plan.get('prediction_dirs', []))
        for folder in sorted(folders):
            if not folder.is_dir():
                continue
            frames = [p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.is_file()]
            if not frames:
                continue
            if any(not p.resolve().is_relative_to(root.resolve()) for p in frames):
                raise ValueError('Inference frames escape run')
            if any(not p.stem.isdigit() for p in frames):
                raise ValueError('Cannot establish numeric inference frame order; source images were kept')
            frames.sort(key=lambda p: (int(p.stem), p.name))
            # Interrupted inference may leave its final output opened but empty.
            # Skip them during encoding and let normal reduction pruning remove them.
            # Nonempty corrupt images still have to pass video verification.
            incomplete_frames.extend(str(p.relative_to(root)) for p in frames if p.stat().st_size == 0)
            frames = [p for p in frames if p.stat().st_size > 0]
            if not frames:
                continue
            relative = str(folder.relative_to(result))
            name = ('plus' if 'plus' in folder.relative_to(result).parts else 'inference') + '-' + hashlib.sha256(relative.encode()).hexdigest()[:12] + '.mp4'
            destination = result / 'inference-videos' / name
            if not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError('Video destination escapes run')
            sequences.append((frames, destination, relative))
    if not export_only:
        for source, destination in copies:
            if not source.exists() or not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError('Cannot preserve PLY artifact')
    backup = backup_run(root, history) if history else None
    videos = []
    for frames, destination, relative in sequences:
        encode_inference_video(frames, destination)
        keep.add(destination)
        keep.update(frames)
        videos.append({'path': str(destination.relative_to(root)), 'name': relative,
                       'frame_count': len(frames), 'fps': 12,
                       'frame_names': [p.name for p in frames]})
    if export_only:
        if not videos:
            raise ValueError('No inference output images are available to encode')
        report = {'inference_videos': videos, 'incomplete_inference_frames': incomplete_frames}
        _atomic_write_json(root / 'inference-videos.json', report)
        return report
    if preserve_reconstruction:
        for plan in plans:
            # Canonical snapshot also supports runs whose request referenced another location.
            _atomic_write_json(plan['root'] / 'preview-trajectory.json', plan['trajectory'])
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            if source.is_file() and source.suffix.lower() == '.ply':
                shutil.copy2(source, destination)
            else:
                # Catalog originals may be bundled or streamed SOG scenes.
                # Use the same unfiltered LOD-0 scene as the review viewer.
                from ..scene import load_scene, save_ply
                temporary = destination.with_name(destination.stem + '.exporting.ply')
                try:
                    save_ply(load_scene(source, min_opacity=0.0, lod_level=0), temporary)
                    temporary.replace(destination)
                except Exception as exc:
                    raise ValueError('Original scene could not be exported to PLY; source images were kept') from exc
                finally:
                    temporary.unlink(missing_ok=True)
        elif source.is_symlink():
            raise ValueError('Cannot retain a linked PLY; source files were kept')
        keep.add(destination)
    # Every sequence has been verified; retain predictions and prune other inputs/intermediates.
    (root / 'output-size.json').unlink(missing_ok=True)
    for path in files(root):
        if path not in keep or path.is_symlink():
            path.unlink()
    for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
        if Path(directory) != root and not any(Path(directory).iterdir()):
            Path(directory).rmdir()
    report = {'reduced': True, 'method': 'artifixer', 'backup_path': backup,
              'before_size_bytes': before_size_bytes, 'inference_videos': videos,
              'incomplete_inference_frames': incomplete_frames,
              'preserve_reconstruction': preserve_reconstruction}
    _atomic_write_json(root / 'reduction.json', report)
    return report
