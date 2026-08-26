"""Scene-aware navigation: ceiling removal, spawn-point search, collision.

Splat-extent-aware clearance
----------------------------
Both the spawn search and collision treat every solid gaussian as a sphere of
radius 2 sigma (its max scale) around the centroid, not as a point. This
matters for translucent sheet geometry like sheer curtains: those are
reconstructed as FEW, LARGE stretched gaussians, so center-distance metrics
see mostly empty space where a person would see (and the renderer draws) a
fabric wall. Clearance = min over the k nearest centers of
(center distance - splat radius), via one batched cKDTree k-NN query.

Spawn-point search
------------------
Cells of a 2D ground grid (interior height range found by ceiling/floor
percentiles) are scored by their 3D surface clearance at eye height minus a
pull toward the scene's center of mass, so winners sit in open space,
centrally (roughly equidistant from the surrounding geometry) rather than in
far corners; greedy non-max suppression spaces the picks apart. A cell only
qualifies if solid floor splats exist beneath it, which rejects open-space
maxima outside the room shell. Cost is one k-NN batch over grid cells —
fine for multi-million-splat scenes, never pairwise distances.

Collision
---------
CollisionWorld clamps every camera move by sampling the path and stopping
before surface clearance drops below clearance_radius, so the agent can no
longer end up inside walls, furniture, or curtains.

move_toward
-----------
The VLM picks a pixel in its current RGB view plus an amount in [0, 1]. The
pixel is unprojected through the depth map rendered for that exact view (the
depth value is the first splat surface hit along that ray), giving a world
target. Travel is then flattened onto the ground plane so the camera keeps
its eye height: a pixel on the floor walks you toward that spot, rather than
diving into it. amount = 1 means "walk up to the ground-projected surface",
capped a safety margin short and additionally collision-clamped along the path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .rendering.base import Camera, up_vector
from .scene import GaussianScene

logger = logging.getLogger(__name__)

# Cap on spawn-grid cells (each costs one k-NN query); cell size grows beyond.
_MAX_GRID_CELLS = 400_000


def ground_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (e0, e1) of the ground plane for a given up vector."""
    seed = np.array([0.0, 0.0, -1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e0 = seed - up * np.dot(seed, up)
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(up, e0)
    return e0, e1


def strip_ceiling(
    scene: GaussianScene, up_axis: str, percentile: float = 25.0
) -> tuple[GaussianScene, float]:
    """Remove the top `percentile` % of gaussians by centroid height along up.

    Returns (stripped scene, height cut along the up axis).
    """
    up = up_vector(up_axis)
    heights = scene.means @ up
    cut = float(np.percentile(heights, 100.0 - percentile))
    stripped = scene.filtered(heights <= cut)
    logger.info("Ceiling strip: removed top %.0f%% (height > %.2f), %d -> %d gaussians",
                percentile, cut, scene.num_gaussians, stripped.num_gaussians)
    return stripped, cut


@dataclass
class SpawnPoint:
    index: int
    position: np.ndarray   # (3,) world
    clearance: float       # surface distance to the nearest solid splat (scene units)
    view_distance: float   # median horizontal sight-line length (scene units)


def _view_distances(
    world: CollisionWorld,
    origins: np.ndarray,
    up: np.ndarray,
    num_rays: int = 16,
    max_dist: float = 6.0,
    max_iters: int = 24,
) -> np.ndarray:
    """Median horizontal sight-line length per origin, via sphere tracing.

    Casts num_rays evenly-spaced horizontal rays from each origin and marches
    each by the local clearance until it hits geometry or max_dist. Separates
    genuinely open vantage points from small enclosed pockets (e.g. inside a
    curtained bed nook) that plain clearance cannot distinguish.
    """
    e0, e1 = ground_basis(up)
    angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    dirs = np.cos(angles)[:, None] * e0[None, :] + np.sin(angles)[:, None] * e1[None, :]

    M = len(origins)
    P = np.repeat(origins, num_rays, axis=0)          # (M*R, 3)
    D = np.tile(dirs, (M, 1))                          # (M*R, 3)
    t = np.zeros(len(P))
    active = np.ones(len(P), dtype=bool)
    for _ in range(max_iters):
        if not active.any():
            break
        c = np.asarray(world.clearance(P[active] + t[active, None] * D[active]))
        t[active] += np.maximum(c, 0.0)
        still = (c > 0.05) & (t[active] < max_dist)
        active[np.flatnonzero(active)] = still
    return np.median(np.minimum(t, max_dist).reshape(M, num_rays), axis=1)


def find_spawn_points(
    scene: GaussianScene,
    up_axis: str,
    world: CollisionWorld | None = None,
    num_points: int = 5,
    ceiling_percentile: float = 25.0,
    grid_cell: float = 0.15,
    solid_opacity: float = 0.1,
    spawn_height_fraction: float = 0.5,
    centroid_pull: float = 0.35,
) -> list[SpawnPoint]:
    """Candidate start positions in open space with good vantage points.

    Pass an existing CollisionWorld to reuse its KD-tree; otherwise one is
    built here with the same solid_opacity.
    """
    up = up_vector(up_axis).astype(np.float64)
    e0, e1 = ground_basis(up)

    means = scene.means[scene.opacities >= solid_opacity].astype(np.float64)
    if len(means) < 100:
        logger.warning("Spawn search: too few solid gaussians (%d)", len(means))
        return []
    if world is None:
        world = CollisionWorld(scene, solid_opacity=solid_opacity)

    h = means @ up
    floor = float(np.percentile(h, 3.0))
    cut = float(np.percentile(h, 100.0 - ceiling_percentile))
    interior = max(cut - floor, 1e-6)

    x, y = means @ e0, means @ e1
    inside = (h >= floor) & (h <= cut)
    x0, x1 = np.percentile(x[inside], [1.0, 99.0])
    y0, y1 = np.percentile(y[inside], [1.0, 99.0])
    extent_x, extent_y = float(x1 - x0), float(y1 - y0)

    cell = float(grid_cell)
    while (extent_x / cell) * (extent_y / cell) > _MAX_GRID_CELLS:
        cell *= 1.5
    nx = max(int(np.ceil(extent_x / cell)), 4)
    ny = max(int(np.ceil(extent_y / cell)), 4)
    x_edges = np.linspace(x0, x1, nx + 1)
    y_edges = np.linspace(y0, y1, ny + 1)
    cx_cells = 0.5 * (x_edges[:-1] + x_edges[1:])
    cy_cells = 0.5 * (y_edges[:-1] + y_edges[1:])
    CX, CY = np.meshgrid(cx_cells, cy_cells, indexing="ij")

    # 3D surface clearance at eye height for every cell, splat extents
    # included — sheer curtains and other sparse-center geometry count.
    spawn_h = floor + spawn_height_fraction * interior
    cell_points = (CX.ravel()[:, None] * e0[None, :]
                   + CY.ravel()[:, None] * e1[None, :]
                   + spawn_h * up[None, :])
    clearance = np.asarray(world.clearance(cell_points)).reshape(CX.shape)

    # Cells only qualify above solid floor splats, so free space outside the
    # room shell never wins.
    body_lo = floor + 0.15 * interior
    floor_band = (h >= floor - 0.05 * interior) & (h < body_lo)
    floor_hist, _, _ = np.histogram2d(x[floor_band], y[floor_band], bins=(x_edges, y_edges))
    has_floor = ndimage.binary_dilation(floor_hist >= 1, iterations=2)

    dist_to_com = np.hypot(CX - float(x.mean()), CY - float(y.mean()))

    score = clearance - centroid_pull * dist_to_com
    valid = has_floor & (clearance >= max(2.0 * cell, 0.35))
    if not valid.any():
        logger.warning("Spawn search: no valid free cells found")
        return []
    score = np.where(valid, score, -np.inf)

    # Stage 1: shortlist spatially-separated candidates by clearance score
    # (greedy non-max suppression so they spread across the room).
    separation = max(4.0 * cell, 0.15 * min(extent_x, extent_y))
    num_candidates = max(3 * num_points, 12)
    shortlist: list[tuple[float, float, float, float]] = []  # px, py, clearance, dist_to_com
    for _ in range(num_candidates):
        flat = int(np.argmax(score))
        gi, gj = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[gi, gj]):
            break
        px, py = float(cx_cells[gi]), float(cy_cells[gj])
        shortlist.append((px, py, float(clearance[gi, gj]), float(dist_to_com[gi, gj])))
        score[np.hypot(CX - px, CY - py) < separation] = -np.inf
    if not shortlist:
        return []

    # Stage 2: rank the shortlist by actual visibility. Median sight-line
    # length separates open-floor vantage points from enclosed pockets
    # (curtained alcoves, gaps between furniture) that clearance alone scores
    # the same. The centroid pull is deliberately weakened here: visibility is
    # the better centrality signal, and a full pull would let a central but
    # enclosed pocket outrank open floor.
    positions = np.stack([px * e0 + py * e1 + spawn_h * up for px, py, _, _ in shortlist])
    views = _view_distances(world, positions, up)
    ranking = np.array([
        view + 0.5 * clr - 0.3 * centroid_pull * dcom
        for view, (_, _, clr, dcom) in zip(views, shortlist)
    ])
    order = np.argsort(-ranking)[:num_points]

    points = [
        SpawnPoint(index=i, position=positions[j].astype(np.float64),
                   clearance=shortlist[j][2], view_distance=float(views[j]))
        for i, j in enumerate(order)
    ]
    logger.info("Spawn search: %d candidate(s) on a %dx%d grid (cell %.2f): %s",
                len(points), nx, ny, cell,
                [f"#{p.index} clr={p.clearance:.2f} view={p.view_distance:.1f}" for p in points])
    return points


class CollisionWorld:
    """Splat-extent-aware clearance queries + motion clamping.

    Every solid gaussian counts as a sphere of radius 2 sigma (max scale,
    capped so room-sized fog splats can't block the whole scene) around its
    center. Clearance at a point is min_k(center_distance_k - radius_k) over
    the k nearest centers — one batched cKDTree query.
    """

    K_NEIGHBORS = 16
    # Cap on a single splat's effective radius: large low-opacity "fog"
    # gaussians would otherwise mark whole rooms as blocked.
    MAX_SPLAT_RADIUS = 0.75

    def __init__(self, scene: GaussianScene, solid_opacity: float = 0.1,
                 clearance_radius: float = 0.25):
        mask = scene.opacities >= solid_opacity
        self._tree = cKDTree(scene.means[mask].astype(np.float64))
        self._radii = np.minimum(
            2.0 * scene.scales[mask].max(axis=1).astype(np.float64),
            self.MAX_SPLAT_RADIUS,
        )
        self.clearance_radius = float(clearance_radius)
        logger.info("Collision world: %d solid gaussians, clearance radius %.2f",
                    int(mask.sum()), self.clearance_radius)

    def clearance(self, points: np.ndarray) -> np.ndarray | float:
        """Surface clearance for one (3,) point or a batch (M, 3)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        k = min(self.K_NEIGHBORS, self._tree.n)
        dists, idx = self._tree.query(pts, k=k)
        if k == 1:
            dists, idx = dists[:, None], idx[:, None]
        surface = np.maximum(dists - self._radii[idx], 0.0).min(axis=1)
        return float(surface[0]) if np.ndim(points) == 1 else surface

    def clamp_motion(
        self, start: np.ndarray, direction: np.ndarray, distance: float
    ) -> tuple[float, bool]:
        """Largest travel along unit `direction` keeping clearance_radius.

        Returns (allowed_distance, was_clamped). One batched KD-tree query
        over path samples, so cost is O(samples * k * log N).

        If the start pose is already inside the keep-out radius (e.g. after a
        previous dive toward the floor), motion that does not get *closer* to
        geometry is still allowed, so the agent can walk out horizontally.
        """
        distance = float(distance)
        if distance <= 0.0:
            return 0.0, False
        start = np.asarray(start, dtype=np.float64)
        start_c = float(self.clearance(start))
        if start_c >= self.clearance_radius:
            required = self.clearance_radius
        elif start_c > 1e-3:
            required = start_c
        else:
            required = self.clearance_radius
        step = self.clearance_radius * 0.5
        n = int(np.ceil(distance / step))
        ts = np.linspace(0.0, distance, n + 1)[1:]
        pts = start[None, :] + np.asarray(direction, dtype=np.float64)[None, :] * ts[:, None]
        bad = np.flatnonzero(self.clearance(pts) < required - 1e-6)
        if len(bad) == 0:
            return distance, False
        allowed = max(float(ts[bad[0]]) - step, 0.0)
        return allowed, True


@dataclass
class MotionContext:
    """What CameraRig.apply needs to resolve motion safely: the collision
    world plus the camera/depth the current observation was rendered from."""

    world: CollisionWorld | None = None
    camera: Camera | None = None
    depth: np.ndarray | None = None


def resolve_move_toward(
    camera: Camera,
    depth: np.ndarray,
    pixel_x: float,
    pixel_y: float,
    amount: float,
    world: CollisionWorld | None = None,
    up: np.ndarray | None = None,
) -> dict | None:
    """Resolve a move_toward action into a new camera position.

    The picked pixel is unprojected to a 3D surface point, then travel is
    flattened onto the ground plane (`up`) so the camera keeps its eye height.
    Returns None when there is no geometry anywhere near the picked pixel;
    otherwise a dict with new_position / target_distance / travelled / blocked
    (or an `error` key when the pixel is too steep to walk toward).
    """
    H, W = depth.shape
    px = int(np.clip(pixel_x, 0, W - 1))
    py = int(np.clip(pixel_y, 0, H - 1))

    z = float(depth[py, px])
    if not np.isfinite(z):
        # Background/hole picked: fall back to the median finite depth in a
        # small window so near-misses at object edges still work.
        win = depth[max(py - 10, 0):py + 11, max(px - 10, 0):px + 11]
        finite = win[np.isfinite(win)]
        if finite.size == 0:
            return None
        z = float(np.median(finite))

    # Unproject: ray through the pixel; depth is camera-z, so scaling the
    # z=1-normalized camera ray by z lands exactly on the rendered surface.
    ray_cam = np.array([
        (px + 0.5 - W / 2.0) / camera.fx,
        (py + 0.5 - H / 2.0) / camera.fy,
        1.0,
    ])
    position = np.asarray(camera.position, dtype=np.float64)
    target = position + camera.rotation.astype(np.float64) @ ray_cam * z

    walk = target - position
    # Walk on the ground plane: a floor pixel means "go there", not "dive".
    if up is not None:
        upn = np.asarray(up, dtype=np.float64)
        nrm = float(np.linalg.norm(upn))
        if nrm > 1e-12:
            upn = upn / nrm
            walk = walk - upn * np.dot(walk, upn)

    dist = float(np.linalg.norm(walk))
    if dist < 1e-3:
        return {
            "error": (
                "picked pixel is nearly vertical (under/above the camera); "
                "pick a point farther into the view"
            ),
        }
    unit = walk / dist

    margin = world.clearance_radius if world is not None else 0.25
    travel = float(np.clip(amount, 0.0, 1.0)) * dist
    travel = min(travel, max(dist - margin, 0.0))
    blocked = False
    if world is not None and travel > 0.0:
        travel, blocked = world.clamp_motion(position, unit, travel)

    new_position = position + unit * travel
    if up is not None:
        # Exact height lock (flattening already removes the up component, this
        # just kills leftover numerical drift).
        upn = np.asarray(up, dtype=np.float64)
        upn = upn / np.linalg.norm(upn)
        new_position = new_position + upn * np.dot(position - new_position, upn)

    return {
        "new_position": new_position,
        "target": target,
        "target_distance": round(dist, 3),
        "travelled": round(travel, 3),
        "blocked": blocked,
    }


@dataclass
class SpawnSelection:
    """Bird's-eye image with painted candidate start points, shown to the VLM
    in the initial prompt so it can choose where to begin exploring."""

    image: np.ndarray                     # (H, W, 3) uint8 annotated render
    points: list[SpawnPoint] = field(default_factory=list)

    def describe_points(self) -> str:
        lines = []
        for p in self.points:
            x, y, z = p.position
            lines.append(
                f"Point {p.index}: world coordinates ({x:.2f}, {y:.2f}, {z:.2f}), "
                f"~{p.clearance:.2f} units of open space around it, "
                f"median sight-line {p.view_distance:.1f} units"
            )
        return "\n".join(lines)


def prepare_spawn_selection(
    scene: GaussianScene, cfg, world: CollisionWorld | None = None
) -> SpawnSelection | None:
    """Spawn-point search + annotated bird's-eye render, from the full config.

    Returns None when the search finds nothing usable (caller falls back to
    the legacy centroid spawn). Pass the episode's CollisionWorld to reuse
    its KD-tree.
    """
    from .rendering.annotate import draw_spawn_markers
    from .rendering.birdseye import render_birdseye

    nav = cfg.navigation
    points = find_spawn_points(
        scene,
        up_axis=cfg.camera.up_axis,
        world=world,
        num_points=int(nav.num_spawn_points),
        ceiling_percentile=float(nav.ceiling_percentile),
        grid_cell=float(nav.grid_cell),
        solid_opacity=float(nav.solid_opacity),
        spawn_height_fraction=float(nav.spawn_height_fraction),
        centroid_pull=float(nav.centroid_pull),
    )
    if not points:
        return None

    stripped, _ = strip_ceiling(scene, cfg.camera.up_axis, float(nav.ceiling_percentile))
    image, camera = render_birdseye(
        stripped,
        up_axis=cfg.camera.up_axis,
        width=int(cfg.renderer.width),
        height=int(cfg.renderer.height),
        max_splat_radius_px=int(cfg.renderer.get("max_splat_radius_px", 120)),
    )
    image = draw_spawn_markers(image, camera, np.stack([p.position for p in points]))
    return SpawnSelection(image=image, points=points)
