"""CPU data boundaries for official OracleGS; no model or loss replacements."""
import json
import shutil
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

AXES = np.diag([1., -1., -1., 1.])


def colmap_lines(frame, image_id, camera_id, name):
    pose = np.asarray(frame['transform_matrix']) @ AXES
    w2c = np.linalg.inv(pose)
    q = Rotation.from_matrix(w2c[:3,:3]).as_quat()[[3,0,1,2]]
    camera = f"{camera_id} PINHOLE {frame['w']} {frame['h']} " + ' '.join(str(frame[k]) for k in ('fl_x','fl_y','cx','cy'))
    image = f'{image_id} ' + ' '.join(format(float(x),'.17g') for x in [*q,*w2c[:3,3]]) + f' {camera_id} {name}\n'
    return camera, image


def known_model(inputs, output):
    """Match COLMAP-assigned IDs, retaining the checkpoint's exact cameras."""
    inputs, output = Path(inputs), Path(output)
    manifest = json.loads((inputs/'inputs.json').read_text())
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames'][:len(manifest['views'])]
    known = output/'known'
    known.mkdir(exist_ok=True)
    cameras, images = [], []
    with sqlite3.connect(output/'database.db') as db:
        for frame in frames:
            name = Path(frame['file_path']).name
            row = db.execute('SELECT image_id, camera_id FROM images WHERE name=?',(name,)).fetchone()
            if row is None:
                raise ValueError(f'COLMAP did not extract {name}')
            image_id, camera_id = row
            # Feature extraction uses single_camera_per_image, so no shared IDs.
            params = np.asarray([frame[k] for k in ('fl_x','fl_y','cx','cy')],dtype=np.float64)
            db.execute('UPDATE cameras SET model=1,width=?,height=?,params=?,prior_focal_length=1 WHERE camera_id=?',
                       (frame['w'],frame['h'],params.tobytes(),camera_id))
            c,i = colmap_lines(frame,image_id,camera_id,name)
            cameras.append(c); images.append(i)
    (known/'cameras.txt').write_text('\n'.join(cameras)+'\n')
    (known/'images.txt').write_text('\n'.join(images)+'\n')
    (known/'points3D.txt').write_text('')


def verify_exported_cameras(frames, exported, manifest):
    if len(frames) != len(exported):
        raise ValueError('SEVA output camera count does not match staged cameras')
    # Validate reference cameras too: native img2vid interleaves GT and targets.
    for frame, out_frame in zip(frames, exported):
        pose = np.asarray(out_frame['transform_matrix'])
        if pose.shape == (3,4):
            pose = np.vstack([pose,[0,0,0,1]])
        if not np.allclose(pose, frame['transform_matrix'],atol=1e-5):
            raise ValueError('SEVA changed target camera identity or order')
        width,height = manifest['resolution']
        internal = tuple(manifest.get('seva_internal_resolution', manifest['resolution']))
        top = (internal[1]-height)//2
        expected_k = [frame['fl_x'],frame['fl_y'],frame['cx'],frame['cy']+top]
        if (out_frame['w'],out_frame['h']) != internal or not np.allclose(
                [out_frame[k] for k in ('fl_x','fl_y','cx','cy')], expected_k, atol=1e-4):
            raise ValueError('SEVA changed target camera calibration')


def assemble(inputs, output):
    """Map SEVA output by its documented ordered target indices, never glob guesses."""
    inputs, output = Path(inputs), Path(output)
    manifest = json.loads((inputs/'inputs.json').read_text())
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames']
    n = len(manifest['views'])
    generated = output/'seva'/manifest.get('seva_task', 'img2trajvid')/'scene'
    exported = json.loads((generated/'transforms.json').read_text())['frames']
    if len(exported) != len(frames):
        raise ValueError('SEVA output camera count does not match staged references and target path')
    order = manifest.get('seva_frame_order', list(range(len(frames))))
    if sorted(order) != list(range(len(frames))):
        raise ValueError('Invalid SEVA frame mapping')
    exported = [exported[order.index(i)] for i in range(len(frames))]
    verify_exported_cameras(frames, exported, manifest)
    dataset = output/'dataset'
    synthetic = dataset/f'synthetic_{n}'
    synthetic.mkdir()
    combined = output/'combined'
    combined.mkdir()
    for frame in frames[:n]:
        name = Path(frame['file_path']).name
        shutil.copyfile(inputs/'scene'/frame['file_path'], combined/name)
    # model_converter has preserved SfM observations and point tracks in text.
    sparse = dataset/'sparse'/str(n)
    shutil.copytree(dataset/'sparse/0', sparse)
    camera_ids = [int(line.split()[0]) for line in (sparse/'cameras.txt').read_text().splitlines() if line and not line.startswith('#')]
    lines = [line for line in (sparse/'images.txt').read_text().splitlines() if not line.startswith('#')]
    image_ids = [int(lines[i].split()[0]) for i in range(0,len(lines),2) if lines[i].strip()]
    next_cam, next_img = max(camera_ids)+1, max(image_ids)+1
    with (sparse/'cameras.txt').open('a') as cams, (sparse/'images.txt').open('a') as imgs:
        for j, frame in enumerate(frames[n:]):
            out_frame = exported[n+j]
            raw = Path(out_frame['file_path'])
            path = (generated/raw).resolve()
            if raw.is_absolute() or not path.is_relative_to(generated.resolve()):
                raise ValueError('SEVA image escapes its output directory')
            name = Path(frame['file_path']).name
            with Image.open(path) as image:
                width,height = manifest['resolution']
                internal = tuple(manifest.get('seva_internal_resolution', manifest['resolution']))
                if image.size != internal:
                    raise ValueError('SEVA changed the requested dimensions')
                top = (internal[1]-height)//2
                image.convert('RGB').crop((0,top,width,top+height)).save(synthetic/name)
            # Upstream requires an RGB at every camera; the loader adapter below
            # excludes these copies from the GT camera set before sampling.
            if name not in {Path(f['file_path']).name for f in frames[:n]}:
                shutil.copyfile(synthetic/name, dataset/'images'/name)
                c,i = colmap_lines(frame,next_img+j,next_cam+j,name)
                cams.write(c+'\n'); imgs.write(i+'\n')
            # Keep both GT and synthetic images at coincident reference poses.
            shutil.copyfile(synthetic/name, combined/('synth_'+name))
    # Force text loading, including newly appended target cameras.
    for p in sparse.glob('*.bin'):
        p.unlink()
    (dataset/'reference_names.json').write_text(json.dumps([Path(f['file_path']).name for f in frames[:n]]))


def validate_geometry(output):
    output = Path(output)
    points = (output/'dataset/sparse/0/points3D.txt').read_text().splitlines()
    count = sum(bool(line.strip()) and not line.startswith('#') for line in points)
    scales = json.loads((output/'dataset/sparse/0/depth_params.json').read_text())
    reliable = sum(float(s['scale']) > 0 for s in scales.values())
    report = {'passed': bool(count and reliable), 'triangulated_points': count,
              'positive_depth_scales': reliable, 'depth_scale_count': len(scales),
              'scope': 'Nonempty triangulation and positive depth scales; not a source geometry accuracy measurement.'}
    from ..checkpoint import atomic_json
    atomic_json(output/'geometry-check.json', report)
    if not count:
        raise ValueError('Sparse reference triangulation produced no points; cannot run the OracleGS SfM baseline')
    if not reliable:
        raise ValueError('No reliable SfM-aligned reference depth; refusing to silently disable depth supervision')


def pad_seva_inputs(inputs, destination, stride=64):
    """Resolution-only adapter: preserve content/FOV and shift principal points."""
    scene = Path(inputs)/'scene'
    destination = Path(destination)
    shutil.copytree(scene,destination/'scene')
    meta = json.loads((scene/'transforms.json').read_text())
    sizes = {(f['w'],f['h']) for f in meta['frames']}
    if len(sizes) != 1:
        raise ValueError('SEVA staging requires uniform dimensions')
    width,height = sizes.pop()
    w = ((width+stride-1)//stride)*stride
    h = ((height+stride-1)//stride)*stride
    left,top = (w-width)//2,(h-height)//2
    for relative in {f['file_path'] for f in meta['frames']}:
        with Image.open(scene/relative) as image:
            padded = Image.new('RGB',(w,h))
            padded.paste(image,(left,top))
            padded.save(destination/'scene'/relative)
    for frame in meta['frames']:
        frame.update(w=w,h=h,cx=frame['cx']+left,cy=frame['cy']+top)
    (destination/'scene/transforms.json').write_text(json.dumps(meta,indent=2))
    return str(destination),w,h


def reference_only_ground_truth(cameras, references):
    """Drop dataset placeholders from the GT set, retain every synthetic camera."""
    references = set(references)
    cameras = [c for c in cameras if c.is_synthetic or c.image_name in references]
    if {c.image_name for c in cameras if not c.is_synthetic} != references:
        raise ValueError('OracleGS reference/synthetic split mismatch')
    if not any(c.is_synthetic for c in cameras):
        raise ValueError('OracleGS has no synthetic cameras')
    if any(c.is_synthetic and not c.attention_map_path for c in cameras):
        raise ValueError('Missing OracleGS synthetic attention map')
    return cameras


def verify_sfm_cameras(inputs, output):
    """Reject any pose/calibration drift between saved RGBs and triangulation."""
    inputs,output = Path(inputs),Path(output)
    manifest = json.loads((inputs/'inputs.json').read_text())
    frames = json.loads((inputs/'scene/transforms.json').read_text())['frames'][:len(manifest['views'])]
    sparse = output/'dataset/sparse/0'
    calibrations = {}
    for line in (sparse/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        if parts[1] != 'PINHOLE':
            raise ValueError('SfM changed the calibrated camera model')
        calibrations[int(parts[0])] = parts
    lines = [line for line in (sparse/'images.txt').read_text().splitlines() if not line.startswith('#')]
    poses = {line.split()[9]:line.split() for line in lines[::2] if line.strip()}
    if set(poses) != {Path(f['file_path']).name for f in frames}:
        raise ValueError('SfM reference image identities changed')
    errors=[]
    for frame in frames:
        name = Path(frame['file_path']).name
        parts = poses[name]
        camera = calibrations[int(parts[8])]
        params = np.array(camera[4:],float)
        expected = np.array([frame[k] for k in ('fl_x','fl_y','cx','cy')])
        if tuple(map(int,camera[2:4])) != (frame['w'],frame['h']) or not np.allclose(params,expected,rtol=0,atol=1e-5):
            raise ValueError(f'SfM calibration drift for {name}')
        q = np.array(parts[1:5],float)
        w2c = np.eye(4)
        w2c[:3,:3] = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        w2c[:3,3] = np.array(parts[5:8],float)
        c2w = np.asarray(frame['transform_matrix'])@AXES
        if not np.allclose(w2c@c2w,np.eye(4),rtol=0,atol=1e-5):
            raise ValueError(f'SfM camera pose drift for {name}')
        points = np.array([[.2,-.1,2,1],[-.3,.2,3,1],[0,0,4,1]]).T
        recovered = w2c@c2w@points
        k = np.array([[params[0],0,params[2]],[0,params[1],params[3]],[0,0,1.]])
        actual, wanted = k@recovered[:3], k@points[:3]
        error = float(np.max(np.abs(actual[:2]/actual[2]-wanted[:2]/wanted[2])))
        if error > 1e-3:
            raise ValueError(f'SfM reprojection drift for {name}: {error} pixels')
        errors.append({'image':name,'max_pixel_error':error})
    (output/'camera-alignment.json').write_text(json.dumps({'passed':True,'references':errors},indent=2))
