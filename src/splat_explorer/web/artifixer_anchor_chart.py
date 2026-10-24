"""Cached same-camera RGB, opacity and depth charts from recorded ArtiFixer inputs."""
from pathlib import Path
import hashlib
import json
import os
import threading
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..splatfix.jobs import read_json

_LOCK = threading.Lock()


def anchor_chart(studio, manifest, run_id):
    """Only use the requested trajectory, never an arbitrary cached variant."""
    root = manifest.parent.resolve()
    request = read_json(root / 'request.json', {})
    trajectory = studio.result_artifact(manifest, request.get('trajectory'))
    checkpoint = studio.result_artifact(manifest, request.get('checkpoint_root') or 'checkpoint')
    if not trajectory or not checkpoint:
        return None
    data = read_json(trajectory, {})
    if data.get('depth_convention') != 'expected_camera_z':
        return None
    anchors = data.get('anchors', [])
    selected = [a for name in ('000', '003') for a in anchors if a.get('view_id') == name]
    if not selected:
        selected = anchors[:2]
    resolved = []
    inputs = [trajectory]
    for anchor in selected:
        item = dict(anchor)
        for key, target in [('original_rgb','rgb_path'), ('opacity','opacity_path'), ('depth','depth_path')]:
            raw = anchor.get(key)
            if not isinstance(raw, str):
                return None
            path = (checkpoint / raw).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                return None
            item[target] = path
            inputs.append(path)
        resolved.append(item)
    if not resolved:
        return None
    signature = hashlib.sha256(json.dumps([3, run_id, [(str(p), p.stat().st_size,
        p.stat().st_mtime_ns) for p in inputs]]).encode()).hexdigest()[:20]
    cache = root / '.anchor-chart'
    if cache.is_symlink():
        return None
    # Keep the cache revision in the directory, leaving a useful basename for
    # browser drag-and-drop and Save Image (also used by Content-Disposition).
    version = cache / signature
    if version.is_symlink():
        return None
    chart = version / (Path(run_id).name + '_intermediate_arr.png')
    with _LOCK:
        if not chart.is_file():
            image = render_anchors(resolved, run_id)
            version.mkdir(parents=True, exist_ok=True)
            temporary = cache / (uuid.uuid4().hex + '.tmp')
            try:
                image.save(temporary, format='PNG')
                os.replace(temporary, chart)
            finally:
                temporary.unlink(missing_ok=True)
    return {'url': studio.file_url(chart), 'anchors': [a['view_id'] for a in resolved],
            'caption': 'Same-camera RGB, opacity and camera-Z depth. White is opaque; purple is near and yellow far. Depth uses scene units and shared 2–98% colour limits across the displayed anchors; grey marks invalid depth or opacity ≤ 0.05.'}


def render_anchors(anchors, run_id):
    rows=[]
    for anchor in anchors:
        rgb=Image.open(anchor['rgb_path']).convert('RGB')
        alpha=np.load(anchor['opacity_path'],allow_pickle=False).squeeze()
        depth=np.load(anchor['depth_path'],allow_pickle=False).squeeze()
        if alpha.ndim != 2 or alpha.shape != depth.shape or rgb.size != (alpha.shape[1], alpha.shape[0]):
            raise ValueError('Anchor image and array dimensions disagree')
        valid=np.isfinite(depth)&(depth>0)&np.isfinite(alpha)&(alpha>0.05)
        rows.append((anchor,rgb,alpha,depth,valid))
    if not any(r[4].any() for r in rows):
        raise ValueError('No valid anchor depths')
    lo,hi=np.percentile(np.concatenate([r[3][r[4]] for r in rows]),[2,98])
    if hi <= lo:
        hi = lo + 1e-6
    # Sequential scientific colour scale, interpolated from viridis control colours.
    colours=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])
    def colour(t):
        return np.stack([np.interp(t,np.linspace(0,1,len(colours)),colours[:,c]) for c in range(3)],axis=-1).astype('uint8')
    def font(n):
        for name in ('DejaVuSans.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf'):
            try:
                return ImageFont.truetype(name,n)
            except OSError:
                pass
        return ImageFont.load_default(size=n)
    w=480; h=round(w*rows[0][2].shape[0]/rows[0][2].shape[1]);gap=28;margin=30
    W=3*w+2*gap+2*margin; H=122+len(rows)*(h+50)+94
    out=Image.new('RGB',(W,H),'#ffffff');draw=ImageDraw.Draw(out)
    views = '1 and 3' if [a['view_id'] for a in anchors] == ['000', '003'] else ' and '.join(str(a['view_id']) for a in anchors)
    draw.text((margin,20),f'{run_id} | saved anchor views {views}, intermediate arrays aligned by camera',font=font(20),fill='#52616b')
    for i,title in enumerate(['Original RGB','Opacity (alpha)','Depth (camera Z)']):
        draw.text((margin+i*(w+gap),68),title,font=font(24),fill='#17212a')
    for row,(anchor,rgb,alpha,depth,valid) in enumerate(rows):
        y=122+row*(h+50)
        gray=Image.fromarray((np.nan_to_num(alpha).clip(0,1)*255).astype('uint8')).convert('RGB')
        mapped=colour(np.nan_to_num((depth-lo)/(hi-lo)).clip(0,1));mapped[~valid]=[180,180,180]
        for i,im in enumerate([rgb,gray,Image.fromarray(mapped)]):
            out.paste(im.resize((w,h),Image.Resampling.LANCZOS),(margin+i*(w+gap),y))
        draw.text((margin,y+h+8),f"Anchor {anchor['view_id']} / trajectory frame {anchor['frame_index']}",font=font(18),fill='#52616b')
    y=122+len(rows)*(h+50)
    x=margin+w+gap
    bar=np.tile(np.linspace(0,255,w,dtype='uint8'),(14,1))
    out.paste(Image.fromarray(bar).convert('RGB'),(x,y))
    draw.text((x,y+23),'0: transparent',font=font(18),fill='#17212a')
    draw.text((x+w-112,y+23),'1: opaque',font=font(18),fill='#17212a')
    x=margin+2*(w+gap)
    out.paste(Image.fromarray(colour(np.tile(np.linspace(0,1,w),(14,1)))),(x,y))
    draw.text((x,y+23),f'{lo:.2f} near',font=font(18),fill='#17212a')
    draw.text((x+w-100,y+23),f'{hi:.2f} far',font=font(18),fill='#17212a')
    draw.text((margin,y+60),'Depth: source-scene units; shared 2–98% colour limits. Grey: invalid depth or alpha ≤ 0.05.',font=font(19),fill='#52616b')
    return out
