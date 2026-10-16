"""Local-only OracleGS research diagnostics; never contacts a compute worker."""
from pathlib import Path
import hashlib

from .g4splat_details import _json, _safe, _LOCK
from PIL import Image, ImageDraw


def generation_task(root):
    return 'img2vid' if (root/'seva/img2vid/scene').is_dir() else 'img2trajvid'


def generation_overview(root):
    """Small cached contact sheet; filenames stay visible, no inferred camera pairing."""
    task = generation_task(root)
    groups = []
    for label, pattern in [("Sparse anchors", f"seva/{task}/scene/first-pass/samples-rgb/*.png"),
                           ("Full trajectory", f"seva/{task}/scene/samples-rgb/*.png")]:
        paths = [p for p in sorted(root.glob(pattern))[:1000] if _safe(root, str(p.relative_to(root)))]
        if paths:
            indices = sorted({round(i * (len(paths)-1) / min(23, max(1,len(paths)-1))) for i in range(min(24,len(paths)))})
            groups.append((label, [paths[i] for i in indices], len(paths)))
    if not groups:
        return None
    fingerprint = hashlib.sha256(repr([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for _,ps,_ in groups for p in ps]).encode()).hexdigest()[:20]
    relative = "debug-previews/generation-" + fingerprint + ".jpg"
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        return None
    with _LOCK:
        if target.is_file():
            return target
        height = sum(40 + ((len(ps)+5)//6)*130 for _,ps,_ in groups)
        sheet = Image.new("RGB", (1200,height), "#101416")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for label, paths, total in groups:
            draw.text((12,y+12), f"{label} | {len(paths)} of {total} exported frames (evenly sampled)", fill="white")
            y += 40
            for i,p in enumerate(paths):
                x,dy = (i%6)*200, y+(i//6)*130
                try:
                    with Image.open(p) as im:
                        if im.width*im.height > 8_000_000:
                            continue
                        im.thumbnail((194,105))
                        sheet.paste(im.convert("RGB"), (x+(200-im.width)//2,dy))
                    draw.text((x+5,dy+110),p.name[:28],fill="white")
                except (OSError,ValueError):
                    draw.text((x+5,dy+50),"Preview unavailable",fill="white")
            y += ((len(paths)+5)//6)*130
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_suffix(".tmp")
        if not temporary.resolve().is_relative_to(root.resolve()):
            return None
        with temporary.open("wb") as stream:
            sheet.save(stream,format="JPEG",quality=85)
        temporary.replace(target)
    return target


def build_details(studio, manifest, result):
    root = manifest.parent
    task = generation_task(root)
    run = root.parent.parent.parent
    def data(name, default=None):
        return _json(_safe(root, name), {} if default is None else default)
    def link(path):
        return studio.file_url(path) if path else None
    recipe = data('recipe.json')
    inputs = recipe.get('inputs') or {}
    staging = _json(_safe(run, 'oraclegs-staging.json'), {})
    input_path = _safe(run, staging.get('input_manifest', ''))
    if input_path and hashlib.sha256(input_path.read_bytes()).hexdigest() == staging.get('manifest_sha256'):
        local_inputs = _json(input_path, {})
        if not inputs or local_inputs == inputs:
            inputs = local_inputs
        else:
            input_path = None
            staging = {**staging, 'error': 'Local input manifest differs from GPU recipe.'}
    else:
        input_path = None
    if input_path is None:
        candidate = _safe(run, 'gpu/inputs/inputs.json')
        if candidate and (not inputs or _json(candidate, {}) == inputs):
            input_path = candidate
            inputs = inputs or _json(candidate, {})
    alignment = data('camera-alignment.json')
    state = _json(_safe(run, 'worker-status.json'), {})
    cache = _json(_safe(run, 'oraclegs-preview-sync.json'), {})
    def panels(pattern):
        return [{'name': p.name, 'url': link(safe)} for p in sorted(root.glob(pattern))[:1000]
                if (safe := _safe(root, str(p.relative_to(root))))]
    views = []
    for v in inputs.get('views', [])[:8]:
        name = Path(v.get('image', '')).name
        rgb = _safe(root, 'dataset/images/' + name)
        if rgb is None and input_path:
            rgb = _safe(input_path.parent, v.get('image', ''))
        views.append({**v, 'url': link(rgb), 'depth_url': link(_safe(root, 'dataset/depths/' + name)),
                      'alignment': next((a for a in alignment.get('references', []) if a.get('image') == name), None)})
    n = len(inputs.get('views', []))
    synthetic = panels(f'dataset/synthetic_{n}/*.png')
    for p in synthetic:
        p['attention_url'] = link(_safe(root, f'dataset/synthetic_{n}_attention_maps/' + p['name']) or
                                  _safe(root, 'attention-raw/synth_' + p['name']))
        p['role'] = 'Synthetic at saved pose' if p['name'].startswith('ref_') else 'Synthetic at intermediate pose'
    phases = ['setup','sfm_features','sfm_matching','sfm_triangulation','sfm_export','reference_depth',
              'depth_alignment','seva_generation','vggt_attention','reconstruction']
    commands = recipe.get('commands', [])
    attempted = [c.get('phase') for c in commands]
    current = state.get('phase')
    if current == 'finished' and state.get('status') in ('error','stopped'):
        live_stage = _json(_safe(run, 'gpu/live-stage.json'), {})
        current = attempted[-1] if attempted else Path(live_stage.get('relative_path', '')).stem
        state = {**state, 'phase':current or 'finished'}
    stages = []
    for name in phases:
        status = 'pending'
        if name in attempted:
            status = 'completed' if attempted.index(name) < len(attempted)-1 or state.get('status') == 'completed' else 'running'
        if name == current:
            status = state.get('status', 'running')
        elif status == 'running' and state.get('status') in ('error', 'stopped'):
            status = state['status']
        stages.append({'name': name, 'status': status})
    artifacts = []
    for name, label in [('recipe.json','Recipe, revisions and commands'), ('camera-alignment.json','Camera alignment gate'),
                        ('geometry-check.json','SfM and depth gate'),
                        ('seva-camera-audit.json','SEVA parsed camera audit'),
                        ('seva-conditioning-audit.json','SEVA diffusion translation and ray audit'), ('seva_generation-request.json','SEVA parameters'),
                        ('dataset/sparse/0/cameras.txt','COLMAP intrinsics'), ('dataset/sparse/0/images.txt','COLMAP camera poses'),
                        ('dataset/sparse/0/points3D.txt','Triangulated source points'), ('dataset/sparse/0/depth_params.json','Depth alignment scales'),
                        (f'seva/{task}/scene/transforms.json','Generated camera encoding'),
                        ('result.json','Final reconstruction provenance')]:
        p = _safe(root, name)
        if p:
            artifacts.append({'label':label,'url':link(p)})
    for p in sorted(root.glob('*.log')):
        safe = _safe(root, p.name)
        if safe:
            artifacts.append({'label':p.name,'url':link(safe)})
    for name, label in [('stage-comparison.jpg','Generation overview'), ('trajectory-quality.png','Trajectory overview')]:
        p = _safe(root, name)
        if p:
            artifacts.append({'label':label,'url':link(p)})
    overview = generation_overview(root)
    live = _safe(run, 'gpu/live-stage.log')
    if live:
        artifacts.insert(0, {'label':'Current worker log (cached)', 'url':link(live)})
    return {'saved_iteration':result.get('saved_iteration'), 'overview_url':link(overview), 'inputs':inputs, 'input_staging':staging, 'views':views, 'recipe':recipe,
            'state':state, 'cache':cache, 'stages':stages, 'alignment':alignment,
            'geometry':data('geometry-check.json'), 'depth_scales':data('dataset/sparse/0/depth_params.json'),
            'anchors':panels(f'seva/{task}/scene/first-pass/samples-rgb/*.png') or panels(f'seva/{task}/scene/first-pass/forward_*/samples-rgb/*.png'),
            'trajectory':panels(f'seva/{task}/scene/samples-rgb/*.png'),
            'synthetic':synthetic, 'artifacts':artifacts,
            'videos':[{'label':label,'url':link(p)} for name,label in [
                ('first-pass/samples-rgb.mp4','First pass · sparse anchors'),('samples-rgb.mp4','Full generated trajectory')]
                if (p := _safe(root, f'seva/{task}/scene/' + name))]}
