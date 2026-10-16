"""Python 3.9-compatible calibration corrections and mandatory SfM quality gate."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

CORRECTION_VERSION = 'camera-consistency-v1'
MAX_REPROJECTION_ERROR_PX = 0.5


def prepare_calibrated_image(source, destination, intrinsics, pointmap_wh):
    """Crop pixels and K together to the exact aspect used by MASt3R.

    SplatFix supplies centered square-pixel cameras. An exact integer aspect
    avoids MASt3R's additional center crop and later stretching of chart depths
    back over a different field of view. Keep the highest available resolution.
    """
    k = np.asarray(intrinsics, dtype=np.float64).copy()
    with Image.open(source) as opened:
        image = opened.convert('RGB')
    w, h = image.size
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
            or not np.allclose(k, [[k[0, 0], 0, w/2], [0, k[0, 0], h/2], [0, 0, 1]], atol=1e-4, rtol=0)):
        raise ValueError('G4Splat correction requires centered square-pixel calibration')
    pw, ph = map(int, pointmap_wh)
    if min(pw, ph) <= 0 or pw % 16 or ph % 16:
        raise ValueError('Invalid MASt3R pointmap dimensions')
    divisor = math.gcd(pw, ph)
    aw, ah = pw // divisor, ph // divisor
    # Even dimensions give an integer centered crop with no half-pixel offset.
    multiple = 2 * (min(w // aw, h // ah) // 2)
    if multiple <= 0:
        raise ValueError('Image is too small for calibrated MASt3R crop')
    tw, th = aw * multiple, ah * multiple
    if (w-tw) % 2 or (h-th) % 2:
        raise ValueError('Source dimensions require a fractional center crop')
    x, y = (w-tw)//2, (h-th)//2
    k[0, 2] -= x
    k[1, 2] -= y
    image.crop((x, y, x+tw, y+th)).save(destination)
    return k, {'source_wh': [w, h], 'output_wh': [tw, th], 'crop_xy': [x, y],
               'source_intrinsics': np.asarray(intrinsics).tolist(), 'intrinsics': k.tolist()}


def reanchor_points(points, previous_c2w, calibrated_c2w):
    """Preserve local coordinates while replacing a point map's owning camera."""
    old, new = np.asarray(previous_c2w), np.asarray(calibrated_c2w)
    points = np.asarray(points)
    local = (points - old[:3, 3]) @ np.linalg.inv(old[:3, :3]).T
    return local @ new[:3, :3].T + new[:3, 3]


def validate_pointmaps(scene, report_path):
    """Fail closed before chart/depth alignment; inspect every originating pixel.

    All points must be finite and in front of the owning camera. No confidence
    masking: upstream depth fitting consumes all point-map samples. Projection
    must match the originating pixel to within half a pixel, including tails.
    """
    scene, report_path = Path(scene), Path(report_path)
    report = {'version': CORRECTION_VERSION, 'passed': False,
              'max_reprojection_error_px': MAX_REPROJECTION_ERROR_PX, 'views': []}
    try:
        body = json.loads((scene / 'cameras.json').read_text())
        count = len(body['filepaths'])
        if count < 2 or any(len(body[key]) != count for key in ('cams2world', 'intrinsics', 'image_calibration')):
            raise ValueError('Missing or inconsistent corrected camera metadata')
        colmap = {}
        for line in (scene / 'sparse/0/cameras.txt').read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                fields = line.split()
                if fields[1] != 'PINHOLE':
                    raise ValueError('Expected PINHOLE camera')
                colmap[int(fields[0])] = np.asarray(fields[2:], dtype=float)
        failures = []
        for i, filename in enumerate(body['filepaths']):
            name = Path(filename).name
            pointmap = json.loads((scene / 'pointmaps' / (Path(name).stem + '.json')).read_text())
            h, w = np.asarray(pointmap['confs']).shape
            pts = np.asarray(pointmap['points'], dtype=float)
            pose = np.asarray(body['cams2world'][i], dtype=float)
            k = np.asarray(body['intrinsics'][i], dtype=float)
            if (pts.shape != (h*w, 3) or pose.shape != (4, 4) or k.shape != (3, 3)
                    or not np.isfinite(pose).all() or not np.isfinite(k).all()
                    or not np.allclose(pose[3], [0, 0, 0, 1])
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
                raise ValueError('Malformed geometry for ' + name)
            local = (pts - pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
            valid = np.isfinite(local).all(axis=1) & (local[:, 2] > 1e-6)
            yy, xx = np.mgrid[:h, :w]
            projected = local[valid] @ k.T
            uv = projected[:, :2] / projected[:, 2:3]
            error = np.linalg.norm(uv - np.column_stack([xx.ravel(), yy.ravel()])[valid], axis=1)
            finite_error = error[np.isfinite(error)]
            calibration = body['image_calibration'][i]
            full_k = np.asarray(calibration['intrinsics'], dtype=float)
            with Image.open(scene / 'images' / name) as im:
                wh = im.size
            expected = np.array([*wh, full_k[0, 0], full_k[1, 1], full_k[0, 2], full_k[1, 2]])
            # Both full-resolution and pointmap cameras must see the same rays.
            scaled = full_k.copy()
            scaled[0] *= w / wh[0]
            scaled[1] *= h / wh[1]
            calibration_ok = (list(wh) == calibration['output_wh']
                              and np.allclose(colmap[i+1], expected, atol=.01, rtol=0)
                              and np.allclose(k, scaled, atol=.01, rtol=0)
                              and np.allclose(full_k[:2, 2], np.array(wh)/2, atol=1e-4))
            row = {'image': name, 'pointmap_wh': [w, h], 'invalid_points': int((~valid).sum()),
                   'nonfinite_points': int((~np.isfinite(local).all(axis=1)).sum()),
                   'nonpositive_depth_points': int((np.isfinite(local).all(axis=1) & (local[:, 2] <= 1e-6)).sum()),
                   'nonfinite_projection_errors': int(len(error) - len(finite_error)),
                   'median_error_px': float(np.median(finite_error)) if len(finite_error) else None,
                   'p95_error_px': float(np.percentile(finite_error, 95)) if len(finite_error) else None,
                   'max_error_px': float(finite_error.max()) if len(finite_error) else None,
                   'calibration_matches_pixels': bool(calibration_ok)}
            row['passed'] = bool(valid.all() and len(error) and np.isfinite(error).all()
                                 and error.max() <= MAX_REPROJECTION_ERROR_PX and calibration_ok)
            report['views'].append(row)
            if not row['passed']:
                failures.append('{} (invalid_depth={}, max_error_px={}, calibration_ok={})'.format(
                    name, row['invalid_points'], row['max_error_px'], calibration_ok))
        if failures:
            raise ValueError('Camera/point-map reprojection failed: ' + ', '.join(failures))
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise ValueError('G4Splat geometry gate stopped depth alignment; see ' + str(report_path) + ': ' + str(exc)) from exc
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
