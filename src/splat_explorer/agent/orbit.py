"""Pixel-picked Gaussian pivots and gravity-aligned camera orbits."""
from __future__ import annotations

import numpy as np

from ..rendering.base import quats_to_covariances
from .orbit_collision import OrbitCollision, ORBIT_CLEARANCE


def pick_pivot(scene, camera, pixel_x, pixel_y):
    """Pick the Gaussian crossing 50% accumulated ray opacity, ignoring faint fog.

    Evaluate anisotropic Gaussian density at the closest point on the pixel ray.
    This is a CPU ray approximation, not the renderer's screen-space compositor.
    Work in chunks to bound memory even for multi-million-splat scenes.
    """
    ray = camera.rotation.astype(float) @ np.array([
        (pixel_x * (camera.width - 1) + .5 - camera.width / 2) / camera.fx,
        (pixel_y * (camera.height - 1) + .5 - camera.height / 2) / camera.fy, 1.])
    ray /= np.linalg.norm(ray)
    hits = []
    for start in range(0, len(scene.means), 65536):
        stop = start + 65536
        means = scene.means[start:stop].astype(float)
        scales = scene.scales[start:stop].astype(float)
        opacity = scene.opacities[start:stop].astype(float)
        quats = scene.quats[start:stop].astype(float)
        valid = (np.isfinite(means).all(axis=1) & np.isfinite(scales).all(axis=1)
                 & (scales > 0).all(axis=1) & np.isfinite(opacity) & (opacity > 0)
                 & np.isfinite(quats).all(axis=1) & (np.linalg.norm(quats, axis=1) > 0))
        ids = np.flatnonzero(valid) + start
        means, scales, opacity, quats = means[valid], scales[valid], opacity[valid], quats[valid]
        quats /= np.linalg.norm(quats, axis=1)[:, None]
        precision = quats_to_covariances(quats, 1 / np.maximum(scales, 1e-8)).astype(float)
        delta = means - camera.position
        pr = precision @ ray
        t = np.einsum('ni,ni->n', delta, pr) / (pr @ ray)
        residual = delta - t[:, None] * ray
        d2 = np.einsum('ni,nij,nj->n', residual, precision, residual)
        alpha = np.clip(opacity, 0, .999) * np.exp(-.5 * np.maximum(d2, 0))
        keep = (t > 0) & (d2 <= 9) & (alpha >= 1 / 255)
        hits.extend(zip(t[keep], ids[keep], alpha[keep]))
    transmission = 1.
    for distance, index, alpha in sorted(hits):
        transmission *= 1 - alpha
        if transmission <= .5:
            return scene.means[index].astype(float).copy(), int(index)
    return None


def apply_orbit(rig, action, ctx):
    outcome = {'kind': 'rotate_around'}
    try:
        px, py = (float(action.args[k]) for k in ('pixel_x', 'pixel_y'))
        azimuth = float(action.args.get('azimuth_pi', 0))
        elevation = action.args.get('elevation_pi')
        elevation = None if elevation is None else float(elevation)
        values = [px, py, azimuth] + ([] if elevation is None else [elevation])
        if not all(np.isfinite(values)) or not (0 <= px <= 1 and 0 <= py <= 1):
            raise ValueError
        if abs(azimuth) > 1 or (elevation is not None and abs(elevation) > .49):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return {**outcome, 'error': 'Use normalized pixels 0..1, azimuth_pi -1..1, elevation_pi -0.49..0.49.'}
    if ctx is None or ctx.camera is None or ctx.scene is None:
        return {**outcome, 'error': 'No Gaussian scene/camera available for pivot picking.'}
    picked = pick_pivot(ctx.scene, ctx.camera, px, py)
    if picked is None:
        return {**outcome, 'error': 'No substantial surface on this ray (accumulated opacity below 0.5). Pick an opaque part of the target.'}
    pivot, index = picked
    offset = rig.position - pivot
    radius = float(np.linalg.norm(offset))
    if radius < 1e-6:
        return {**outcome, 'error': 'Camera is at the selected pivot; choose a farther surface.'}
    height = float(np.dot(offset, rig.up))
    start_elevation = float(np.arcsin(np.clip(height / radius, -1, 1)))
    radial = offset - height * rig.up
    norm = np.linalg.norm(radial)
    radial = radial / norm if norm > 1e-8 else -rig.heading()
    right = np.cross(rig.up, radial)
    end_elevation = start_elevation if elevation is None else elevation * np.pi
    theta = azimuth * np.pi
    def position(fraction):
        angle = start_elevation + fraction * (end_elevation - start_elevation)
        horizontal = np.cos(theta * fraction) * radial + np.sin(theta * fraction) * right
        return pivot + radius * (np.cos(angle) * horizontal + np.sin(angle) * rig.up)
    # Check successive short chords along the arc, never the endpoint shortcut.
    arc_bound = radius * (abs(theta) + abs(end_elevation - start_elevation))
    steps = max(1, int(np.ceil(arc_bound / .02)), int(np.ceil(abs(theta) / np.radians(2))))
    start = rig.position.copy()
    collision = OrbitCollision(ctx.scene)
    # |p''(t)| <= radius * (|azimuth| + |elevation change|)^2.
    # Linear interpolation error on [a,b] is bounded by |p''| (b-a)^2 / 8.
    curvature_bound = radius * (abs(theta) + abs(end_elevation - start_elevation))**2
    def blocked_between(a, b):
        source, dest = position(a), position(b)
        if collision.intersects(source, dest, curvature_bound * (b - a)**2 / 8):
            return True
        delta = dest - source
        distance = float(np.linalg.norm(delta))
        return (ctx.world is not None and distance > 1e-10
                and ctx.world.clamp_motion(source, delta / distance, distance)[1])
    travelled = 0.
    blocked = False
    fraction = 0.
    for i in range(1, steps + 1):
        dest = position(i / steps)
        delta = dest - rig.position
        distance = float(np.linalg.norm(delta))
        if distance > 1e-10 and blocked_between(fraction, i / steps):
            blocked = True
            low, high = fraction, i / steps
            # Refine the first blocked interval; every accepted prefix is swept,
            # so even a thin obstacle between two clear endpoints stops motion.
            while (high - low) * arc_bound > 1e-5:
                middle = (low + high) / 2
                if blocked_between(fraction, middle):
                    high = middle
                else:
                    low = middle
            dest = position(low)
            travelled += float(np.linalg.norm(dest - rig.position))
            rig.position = dest
            fraction = low
            break
        rig.position = dest
        travelled += distance
        fraction = i / steps
    if travelled > 1e-8:
        rig.aim_at(pivot)
    return {**outcome, 'pivot': pivot.tolist(), 'gaussian_index': index,
            'pixel': [px, py], 'radius': radius, 'azimuth_pi': azimuth,
            'elevation_pi': end_elevation / np.pi, 'completed_fraction': fraction,
            'travelled': travelled, 'baseline': float(np.linalg.norm(rig.position - start)),
            'blocked': blocked, 'collision_clearance': ORBIT_CLEARANCE}
