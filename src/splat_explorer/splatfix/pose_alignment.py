"""Conservative, fixed-intrinsics RGB-D registration for generated reference views.

OpenCV poses are world-to-camera. Here the fitted transform maps coordinates
in the *original camera* to the generated camera; no scene-scale assumption is
made. Flow and SIFT only propose correspondences, never warp the saved images.
"""
from dataclasses import replace

import numpy as np


def opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Pose correction requires: pip install -e ".[pose-correction]"') from exc
    return cv2


def correspondences(original, target):
    cv = opencv()
    h, w = original.shape[:2]
    scale = min(1., 832 / w)
    size = (round(w * scale), round(h * scale))
    def gray(image):
        return cv.cvtColor(cv.resize(image, size), cv.COLOR_RGB2GRAY)
    a, b = gray(original), gray(target)
    sift = cv.SIFT_create(nfeatures=4000, contrastThreshold=.015)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    pairs = []
    if da is not None and db is not None and len(db) >= 2:
        matcher = cv.BFMatcher()
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
        for neighbors in matcher.knnMatch(da, db, k=2):
            if len(neighbors) == 2:
                m, n = neighbors
                if m.distance < .8 * n.distance and reverse.get(m.trainIdx) == m.queryIdx:
                    pairs.append((ka[m.queryIdx].pt, kb[m.trainIdx].pt))
    # Coarse structural flow also matches blurred splats with newly sharp repairs.
    # Reject textureless and non-reversible flow instead of treating it as truth.
    if min(size) >= 64:
        a = cv.GaussianBlur(a, (0, 0), 1.5)
        b = cv.GaussianBlur(b, (0, 0), 1.5)
        forward = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a, b, None)
        backward = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(b, a, None)
        strength = cv.cornerMinEigenVal(a, 7)
        yy, xx = np.mgrid[16:size[1]-16:16, 16:size[0]-16:16]
        p = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        q = p + forward[yy, xx].reshape(-1, 2)
        qi = np.rint(q).astype(int)
        inside = (qi[:, 0] >= 0) & (qi[:, 0] < size[0]) & (qi[:, 1] >= 0) & (qi[:, 1] < size[1])
        qi[:, 0] = qi[:, 0].clip(0, size[0]-1)
        qi[:, 1] = qi[:, 1].clip(0, size[1]-1)
        reversible = np.linalg.norm(q + backward[qi[:, 1], qi[:, 0]] - p, axis=1) < 2
        valid = inside & reversible & (strength[yy, xx].ravel() > .0002)
        pairs.extend(zip(p[valid], q[valid]))
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2))
    points = np.asarray(pairs, dtype=np.float64) / [size[0]/w, size[1]/h]
    # SIFT's multiple orientations at one pixel must not count as independent evidence.
    _, unique = np.unique(np.rint(points[:, 0] / (4/scale)), axis=0, return_index=True)
    points = points[np.sort(unique)]
    return points[:, 0], points[:, 1]


def fit_pose(camera, source_pixels, target_pixels, depth, opacity=None):
    """Return a small candidate and diagnostics, or None for ambiguous geometry."""
    cv = opencv()
    from scipy.optimize import least_squares
    p, q = np.asarray(source_pixels), np.asarray(target_pixels)
    report = {'matches': len(p)}
    def reject(reason):
        return None, {**report, 'reason': reason}
    if len(p) < 24:
        return reject('Too few reliable correspondences')
    xy = np.rint(p).astype(int)
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < camera.width) &
              (xy[:, 1] >= 0) & (xy[:, 1] < camera.height))
    p, q, xy = p[inside], q[inside], xy[inside]
    z = depth[xy[:, 1], xy[:, 0]]
    valid = np.isfinite(z) & (z > 0) & np.isfinite(q).all(axis=1)
    if opacity is not None:
        valid &= opacity[xy[:, 1], xy[:, 0]] >= .7
    # Depth discontinuities make sub-pixel correspondences unreliable.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = depth[(xy[:, 1]+dy).clip(0, camera.height-1), (xy[:, 0]+dx).clip(0, camera.width-1)]
        valid &= np.isfinite(neighbor) & (np.abs(neighbor-z) < .1 * np.maximum(z, 1e-8))
    p, q, z = p[valid], q[valid], z[valid]
    report['depth_matches'] = len(p)
    if len(p) < 24:
        return reject('Too few opaque, stable-depth correspondences')
    scene_scale = float(np.median(z))
    k = np.asarray(camera.intrinsics, dtype=np.float64)
    xyz = (np.c_[p, np.ones(len(p))] @ np.linalg.inv(k).T) * (z / scene_scale)[:, None]
    # Spatially interleaved held-out points never enter RANSAC or refinement.
    held = (np.floor(p[:, 0]/16).astype(int) + 3*np.floor(p[:, 1]/16).astype(int)) % 5 == 0
    if held.sum() < 6 or (~held).sum() < 16:
        held = np.arange(len(p)) % 5 == 0
    threshold = 4 * camera.width / 832
    cv.setRNGSeed(42)
    ok, rv, tv, indices = cv.solvePnPRansac(
        xyz[~held], q[~held], k, None, iterationsCount=2000,
        reprojectionError=threshold, confidence=.999, flags=cv.SOLVEPNP_EPNP)
    if not ok or indices is None or len(indices) < 12:
        return reject('No well-supported rigid camera fit')
    initial = np.r_[rv.ravel(), tv.ravel()]
    limits = np.array([np.deg2rad(8)]*3 + [.12]*3)
    if np.any(np.abs(initial) >= limits):
        return reject('Proposed camera motion exceeds the small-correction bounds')
    training = np.flatnonzero(~held)[indices.ravel()]
    def project(x):
        return cv.projectPoints(xyz, x[:3], x[3:], k, None)[0][:, 0]
    def residual(x):
        return (project(x)[training] - q[training]).ravel()
    solution = least_squares(residual, initial, bounds=(-limits, limits),
                             loss='soft_l1', f_scale=threshold/2, max_nfev=100)
    x = solution.x
    rotation = cv.Rodrigues(x[:3])[0]
    offset = -rotation.T @ x[3:]
    angle = float(np.rad2deg(np.linalg.norm(x[:3])))
    report.update(rotation_degrees=angle, translation_fraction=float(np.linalg.norm(offset)),
                  median_scene_depth=scene_scale)
    if angle > 8 or np.linalg.norm(offset) > .12 or not solution.success:
        return reject('Camera motion is too large or optimization did not converge')
    before = np.linalg.norm(p-q, axis=1)
    after = np.linalg.norm(project(x)-q, axis=1)
    support = after < threshold
    coverage = np.ptp(p[support], axis=0) / [camera.width, camera.height] if support.any() else np.zeros(2)
    cells = np.floor(p[support] / [camera.width/4, camera.height/3]).astype(int)
    report.update(inliers=int(support.sum()), inlier_fraction=float(support.mean()),
                  coverage=coverage.tolist(), occupied_cells=len(np.unique(cells, axis=0)),
                  held_out_matches=int(held.sum()))
    # Some views have useful structure concentrated vertically; require both
    # two-dimensional spread and area, then let independent/render checks decide.
    if (support.sum() < 16 or support.mean() < .3 or min(coverage) < .35
            or np.prod(coverage) < .2 or report['occupied_cells'] < 5):
        return reject('Rigid fit is confined to too little of the image')
    # Evaluate all hold-out matches with capped error, not just chosen inliers.
    old = float(np.mean(np.minimum(before[held], 3*threshold)))
    new = float(np.mean(np.minimum(after[held], 3*threshold)))
    report.update(held_out_error_before=old, held_out_error_after=new)
    if new >= old * .9 or old-new < .5*camera.width/832:
        return reject('Camera fit does not improve independent correspondences enough')
    aligned = before < threshold
    if aligned.sum() >= 4 and np.median(after[aligned]-before[aligned]) > 2*camera.width/832:
        return reject('Camera fit would displace already aligned features')
    candidate = replace(camera, position=camera.position + camera.rotation @ (offset*scene_scale),
                        rotation=camera.rotation @ rotation.T)
    return candidate, {**report, 'reason': 'Candidate passed geometric checks'}


def structural_scores(render, target):
    """Equal-weight tile correlations; appearance changes cannot dominate by area."""
    cv = opencv()
    size = (416, max(64, round(render.shape[0]*416/render.shape[1])))
    def structure(image):
        gray = cv.cvtColor(cv.resize(image, size), cv.COLOR_RGB2GRAY).astype(np.float32)/255
        blur = cv.GaussianBlur(gray, (0, 0), 1.2)
        # Local contrast suppresses lighting/style differences and emphasizes layout.
        return blur - cv.GaussianBlur(blur, (0, 0), 5)
    a, b = structure(render), structure(target)
    scores = []
    for rows in np.array_split(np.arange(size[1]), 3):
        for cols in np.array_split(np.arange(size[0]), 4):
            x, y = a[np.ix_(rows, cols)].ravel(), b[np.ix_(rows, cols)].ravel()
            x, y = x-x.mean(), y-y.mean()
            norm = np.linalg.norm(x)*np.linalg.norm(y)
            scores.append(float(np.dot(x, y)/norm) if norm > 1e-5 else 0.)
    return np.asarray(scores)


def render_acceptance(before, after):
    delta = np.asarray(after)-before
    # A locally generated object must not pull an otherwise matching scene away.
    good = np.asarray(before) >= .45
    return (float(np.mean(delta)) >= .015 and int((delta > .01).sum()) >= 5
            and not np.any(delta[good] < -.08) and int((delta < -.08).sum()) <= 2)
