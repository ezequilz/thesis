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
        for folder in sorted(root.glob('seva-segments/[0-9][0-9][0-9]/seva/img2*/scene')):
            paths = [p for p in sorted(folder.glob('first-pass/samples-rgb/*.png'))
                     if _safe(root, str(p.relative_to(root)))]
            if paths:
                step = max(1,len(paths)//12)
                groups.append((f"Pair {int(folder.parents[2].name)+1} anchors", paths[::step][:12],len(paths)))
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


def attention_overlay(root, rgb, attention):
    """Cached legacy preview from quantized maps; native exports take priority."""
    import numpy as np
    from .plasma import PLASMA
    if not rgb or not attention:
        return None
    fingerprint = hashlib.sha256(repr([(str(p),p.stat().st_size,p.stat().st_mtime_ns)
        for p in (rgb,attention)]).encode()).hexdigest()[:20]
    target = root/'debug-previews'/('attention-overlay-'+fingerprint+'.png')
    if not target.resolve().is_relative_to(root.resolve()):
        return None
    with _LOCK:
        if target.is_file():
            return target
        try:
            with Image.open(rgb) as image, Image.open(attention) as raw:
                w,h = image.size
                nw,nh = (518,round(h*518/w/14)*14) if w>=h else (round(w*518/h/14)*14,518)
                canvas = Image.new('RGB',(518,518),'white')
                canvas.paste(image.convert('RGB').resize((nw,nh),Image.Resampling.BICUBIC),((518-nw)//2,(518-nh)//2))
                weights = np.asarray(raw.convert('F').resize((518,518),Image.Resampling.BILINEAR))/255
                color = (np.asarray(PLASMA)[np.minimum((weights*256).astype(int),255)]*255).astype(np.uint8)
                overlay = Image.fromarray(np.rint(.2*np.asarray(canvas)+.8*color).astype(np.uint8))
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                overlay.save(stream,format='PNG')
            temporary.replace(target)
            return target
        except (OSError,ValueError):
            return None


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
        items = []
        for p in sorted(root.glob(pattern))[:1000]:
            safe = _safe(root, str(p.relative_to(root)))
            if not safe:
                continue
            item = {'name':p.name, 'url':link(safe)}
            relative = p.relative_to(root)
            if relative.parts[0] == 'seva-segments' and p.stem.isdigit():
                base = '/'.join(relative.parts[:2])
                ordered = data(base+'/seva-ordered-pair.json')
                first_pass = 'first-pass' in relative.parts
                index = int(p.stem)
                key = 'anchor_index' if first_pass else 'frame_index'
                record = next((r for r in ordered.get('anchors' if first_pass else 'targets', [])
                               if r[key] == index), None)
                endpoints = ordered.get('endpoint_frame_indices', [])
                if not first_pass and endpoints and index < endpoints[0]:
                    continue  # Off-path context images are not trajectory frames.
                if record:
                    item['source_reference_index'] = record['source_reference_index']
                    item['name'] += ' · source reference '+str(record['source_reference_index']+1)
                elif not first_pass and index in endpoints:
                    source = ordered['endpoint_reference_indices'][endpoints.index(index)]
                    item['name'] += ' · saved endpoint reference '+str(source+1)
            elif relative.parts[0] == 'seva' and 'first-pass' not in relative.parts and p.stem.isdigit():
                target = next((f for f in inputs.get('trajectory', {}).get('frames', [])
                               if f['target_index'] == int(p.stem)), {})
                source = target.get('generation_source_reference_index')
                if source is not None:
                    item['source_reference_index'] = source
                    item['name'] += ' · source reference '+str(source+1)
            items.append(item)
        return items
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
    # Both the released extractor and our batching adapter call the authors'
    # save_attention_maps; expose its RGB/plasma grids without re-rendering them.
    attention_plan = data('attention-chunks.json')
    chunks = {f"chunk_{c['index']:03d}":c for c in attention_plan.get('chunks', [])}
    attention_grids = []
    for pattern in ('attention-raw/*_grid.png', 'attention-raw/chunk_*/*_grid.png'):
        for p in sorted(root.glob(pattern))[:1000]:
            safe = _safe(root, str(p.relative_to(root)))
            if safe is None:
                continue
            chunk = chunks.get(p.parent.name, {})
            attention_grids.append({'name':p.name, 'url':link(safe),
                'chunk_index':chunk.get('index'), 'source_image':chunk.get('source_image'),
                'segment_indices':chunk.get('segment_indices', []),
                'reference_count':len(chunk.get('reference_images', [])) if chunk else None,
                'context_target_count':len(chunk.get('context_target_images', [])) if chunk else None,
                'assigned_target_count':len(chunk.get('target_images', [])) if chunk else None})
    assignments = {name:chunk for chunk in attention_plan.get('chunks', [])
                   for name in chunk.get('target_images', [])}
    for item in synthetic:
        name = 'synth_'+item['name']
        chunk = assignments.get(name)
        native = (_safe(root, f"attention-raw/chunk_{chunk['index']:03d}/overlays/"+name)
                  if chunk else _safe(root,'attention-raw/overlays/'+name))
        item['overlay_reconstructed'] = native is None
        overlay = native or attention_overlay(root,
            _safe(root,f'dataset/synthetic_{n}/'+item['name']),
            _safe(root,f'dataset/synthetic_{n}_attention_maps/'+item['name']) or
            _safe(root,'attention-raw/'+name))
        item['overlay_url'] = link(overlay)
    phases = ['setup','sfm_features','sfm_matching','sfm_triangulation','sfm_export','reference_depth',
              'depth_alignment','seva_generation','vggt_attention','reconstruction']
    if inputs.get('oracle_depth_source') == 'source_splat':
        phases = [p for p in phases if p not in ('reference_depth', 'depth_alignment')]
    commands = recipe.get('commands', [])
    attempted = [('seva_generation' if (c.get('phase') or '').startswith('seva_generation_') else c.get('phase')) for c in commands]
    current = state.get('phase')
    if current == 'finished' and state.get('status') in ('error','stopped'):
        live_stage = _json(_safe(run, 'gpu/live-stage.json'), {})
        current = attempted[-1] if attempted else Path(live_stage.get('relative_path', '')).stem
        state = {**state, 'phase':current or 'finished'}
    if (current or '').startswith('seva_generation_'):
        current = 'seva_generation'
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
                        ('attention-chunks.json','Attention chunk and source assignments'),
                        ('seva-camera-audit.json','SEVA parsed camera audit'),
                        ('seva-conditioning-audit.json','SEVA diffusion translation and ray audit'), ('seva-geometry-guidance.json','SEVA source geometry guidance'), ('seva_generation-request.json','SEVA parameters'),
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
    segment_videos = []
    for segment in inputs.get('seva_segments', []):
        base = f"seva-segments/{segment['index']:03d}"
        label = f"Pair {segment['index']+1}: reference {segment['from_reference']+1} → {segment['to_reference']+1}"
        for relative, kind in [('first-pass/samples-rgb.mp4','sparse anchors'), ('samples-rgb.mp4','full sub-trajectory')]:
            path = _safe(root, base+'/seva/'+inputs.get('seva_task','img2trajvid')+'/scene/'+relative)
            if path:
                segment_videos.append({'label':label+' · '+kind, 'url':link(path)})
        for name in ('seva-camera-audit.json','seva-conditioning-audit.json','seva-geometry-guidance.json','seva-ordered-pair.json'):
            path = _safe(root, base+'/'+name)
            if path:
                artifacts.append({'label':label+' · '+name, 'url':link(path)})
    overview = generation_overview(root)
    live = _safe(run, 'gpu/live-stage.log')
    if live:
        artifacts.insert(0, {'label':'Current worker log (cached)', 'url':link(live)})
    return {'saved_iteration':result.get('saved_iteration'), 'overview_url':link(overview), 'inputs':inputs, 'input_staging':staging, 'views':views, 'recipe':recipe,
            'state':state, 'cache':cache, 'stages':stages, 'alignment':alignment,
            'geometry':data('geometry-check.json'), 'depth_scales':data('dataset/sparse/0/depth_params.json'),
            'anchors':panels(f'seva/{task}/scene/first-pass/samples-rgb/*.png') or panels(f'seva/{task}/scene/first-pass/forward_*/samples-rgb/*.png') or panels('seva-segments/*/seva/img2*/scene/first-pass/samples-rgb/*.png'),
            'trajectory':panels(f'seva/{task}/scene/samples-rgb/*.png') or panels('seva-segments/*/seva/img2*/scene/samples-rgb/*.png'),
            'synthetic':synthetic, 'attention_grids':attention_grids,
            'attention_batching':attention_plan.get('batching', recipe.get('attention_batching')), 'artifacts':artifacts,
            'videos':segment_videos+[{'label':label,'url':link(p)} for name,label in [
                ('first-pass/samples-rgb.mp4','First pass · sparse anchors'),('samples-rgb.mp4','Full generated trajectory')]
                if (p := _safe(root, f'seva/{task}/scene/' + name))]}
