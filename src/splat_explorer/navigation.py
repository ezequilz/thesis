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

Waypoint covering
-----------------
The same occupancy grid is reused for jump_to_waypoint vantages, but
selection maximises area coverage instead of centrality: clearance local
maxima (room / hallway / alcove centers, never inside objects) are
farthest-point sampled so one waypoint lands in each open region rather
than a cluster in the largest room. Valid cells must sit in a
neighborhood of reconstructed splat mass comparable to the indoor core,
so a large but sparse exterior halo (floaters outside the walls) cannot
win just because it is empty and far from the other picks.

Collision
---------
CollisionWorld can clamp every camera move by sampling the path and stopping
before surface clearance drops below a keep-out radius. Door frames and other
narrow openings are often reconstructed as large stretched gaussians, so the
full clamp treats them as walls and traps the agent in one room.

`navigation.collision` selects the policy (see CollisionWorld):
  full — original hard break (clearance_radius keep-out along the path)
  low  — same sampler, tiny keep-out so doorways stay passable
  off  — free-cam: no path clamp (default). move_toward still uses depth to
         walk toward a surface and stops a small margin short of that target.

move_toward
-----------
The VLM picks a pixel in its current RGB view plus an amount in [0, 1]. The
pixel is unprojected through the depth map rendered for that exact view (the
depth value is the first splat surface hit along that ray), giving a world
target. Travel is then flattened onto the ground plane so the camera keeps
its eye height: a pixel on the floor walks you toward that spot, rather than
diving into it. amount = 1 means "walk up to the ground-projected surface",
capped a safety margin short of the picked target. Path collision-clamping
is applied on top only when navigation.collision is full or low.
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

# Path-clamp policies for agent movement (navigation.collision). Spawn search
# always uses splat-extent clearance; these only affect move / move_toward.
COLLISION_FULL = "full"
COLLISION_LOW = "low"
COLLISION_OFF = "off"
COLLISION_MODES = (COLLISION_FULL, COLLISION_LOW, COLLISION_OFF)
# Keep-out used by collision: low. Full uses clearance_radius (default 0.25).
LOW_CLEARANCE = 0.05


def _normalize_collision(value) -> str:
    """Map config values onto COLLISION_MODES.

    Unquoted YAML `off`/`on` become booleans; treat those as off/full.
    """
    if isinstance(value, bool):
        return COLLISION_OFF if value is False else COLLISION_FULL
    mode = str(value).strip().lower()
    aliases = {
        "false": COLLISION_OFF, "none": COLLISION_OFF, "no": COLLISION_OFF,
        "true": COLLISION_FULL, "yes": COLLISION_FULL,
    }
    return aliases.get(mode, mode)


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


@dataclass
class Waypoint:
    """A precomputed open-space vantage the agent can jump to."""

    index: int
    position: np.ndarray   # (3,) world
    clearance: float
    view_distance: float


@dataclass
class OpenFloorGrid:
    """2D ground-plane occupancy shared by spawn search and waypoint covering."""

    world: CollisionWorld
    up: np.ndarray
    e0: np.ndarray
    e1: np.ndarray
    spawn_h: float
    cell: float
    nx: int
    ny: int
    CX: np.ndarray
    CY: np.ndarray
    cx_cells: np.ndarray
    cy_cells: np.ndarray
    clearance: np.ndarray
    valid: np.ndarray
    dist_to_com: np.ndarray
    extent_x: float
    extent_y: float

    def world_point(self, px: float, py: float) -> np.ndarray:
        return px * self.e0 + py * self.e1 + self.spawn_h * self.up


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


def _view_hit_fraction(
    world: CollisionWorld,
    origins: np.ndarray,
    up: np.ndarray,
    num_rays: int = 16,
    max_dist: float = 6.0,
    max_iters: int = 24,
    hit_dist: float = 5.4,
) -> np.ndarray:
    """Fraction of horizontal rays that hit geometry before `hit_dist`.

    Indoor vantages see walls/furniture on several sides. A void island next
    to a lone floater has almost every ray run out to max_dist.
    """
    e0, e1 = ground_basis(up)
    angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    dirs = np.cos(angles)[:, None] * e0[None, :] + np.sin(angles)[:, None] * e1[None, :]
    M = len(origins)
    P = np.repeat(origins, num_rays, axis=0)
    D = np.tile(dirs, (M, 1))
    t = np.zeros(len(P))
    active = np.ones(len(P), dtype=bool)
    for _ in range(max_iters):
        if not active.any():
            break
        c = np.asarray(world.clearance(P[active] + t[active, None] * D[active]))
        t[active] += np.maximum(c, 0.0)
        still = (c > 0.05) & (t[active] < max_dist)
        active[np.flatnonzero(active)] = still
    hits = np.minimum(t, max_dist).reshape(M, num_rays) < hit_dist
    return hits.mean(axis=1)


def _smoothed_density(hist: np.ndarray, cell: float, radius: float = 1.0) -> np.ndarray:
    """Mean splat count in a `radius`-metre window (uniform filter)."""
    win = max(3, int(round(radius / max(float(cell), 1e-6))))
    if win % 2 == 0:
        win += 1
    return ndimage.uniform_filter(np.asarray(hist, dtype=np.float64), size=win, mode="constant")


def _dense_floor_mask(
    floor_hist: np.ndarray,
    cell: float,
    body_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Keep floor cells that belong to a reconstructed interior.

    Binary occupancy (`>= 1` splat) plus dilation treats a large but sparse
    exterior floater cloud as a room — those cells then win waypoint covering
    because they are empty (high clearance) and far from the indoor picks.
    Score a ~1 m neighborhood of splat mass relative to the indoor core,
    close 1–2 cell floor gaps without expanding into the void, and drop
    tiny components.
    """
    floor_hist = np.asarray(floor_hist, dtype=np.float64)
    mass = floor_hist if body_hist is None else np.asarray(body_hist, dtype=np.float64)
    occupied = floor_hist >= 1.0
    if not occupied.any():
        return np.zeros_like(occupied)

    density = _smoothed_density(mass, cell, radius=1.0)
    core = density[occupied]
    ref = float(np.percentile(core, 60)) if core.size else 0.0
    # Hallways (floor + walls, little furniture) stay in; reconstruction
    # halo / sky floaters sit far below typical indoor mass.
    min_density = 0.22 * ref if ref > 0.0 else np.inf
    keep = occupied & (density >= min_density)

    keep = ndimage.binary_closing(keep, iterations=2)
    keep = ndimage.binary_fill_holes(keep)

    labeled, n_labels = ndimage.label(keep)
    # ~0.35 m²: small alcove / closet stays; 1–few cell floater clumps drop.
    min_cells = max(8, int(round(0.35 / max(cell * cell, 1e-6))))
    sizes = ndimage.sum(keep, labeled, index=np.arange(1, n_labels + 1))
    keep_ids = np.flatnonzero(np.asarray(sizes) >= min_cells) + 1
    if len(keep_ids) == 0:
        return np.zeros_like(occupied)
    return np.isin(labeled, keep_ids)


def _on_reconstructed_interior(
    image: np.ndarray,
    camera,
    positions: np.ndarray,
    frac: float = 0.40,
    radius_px: int = 8,
) -> np.ndarray:
    """True where a point projects onto reconstructed interior, not void / grey halo.

    `scene_mask` only rejects near-black pixels. Sparse exterior reconstruction
    renders as dark grey and would still pass that cutoff, so also require the
    patch luma to sit toward typical indoor brightness rather than the void.
    """
    from .rendering.annotate import project_to_pixels, scene_mask

    luma = np.asarray(image, dtype=np.float64).mean(axis=-1)
    mask = scene_mask(image)
    uv = project_to_pixels(camera, np.asarray(positions, dtype=np.float64))
    H, W = mask.shape
    keep = np.zeros(len(positions), dtype=bool)
    if not mask.any():
        return keep
    indoor_luma = float(np.percentile(luma[mask], 60))
    # Absolute floor catches grey reconstruction halo even when it covers a
    # lot of the frame (and would pull the percentile down). Relative term
    # scales with bright indoor parquet/walls. Dark bathrooms still sit above.
    min_luma = max(36.0, 0.50 * indoor_luma)
    for i, (u, v) in enumerate(uv):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < W and 0 <= y < H):
            continue
        y0, y1 = max(0, y - radius_px), min(H, y + radius_px + 1)
        x0, x1 = max(0, x - radius_px), min(W, x + radius_px + 1)
        patch = mask[y0:y1, x0:x1]
        patch_luma = luma[y0:y1, x0:x1]
        if (
            patch.size
            and float(patch.mean()) >= frac
            and float(patch_luma.mean()) >= min_luma
        ):
            keep[i] = True
    return keep


def _open_floor_grid(
    scene: GaussianScene,
    up_axis: str,
    world: CollisionWorld | None = None,
    ceiling_percentile: float = 25.0,
    grid_cell: float = 0.15,
    solid_opacity: float = 0.1,
    spawn_height_fraction: float = 0.5,
) -> OpenFloorGrid | None:
    """Eye-height clearance grid over cells that sit above solid floor."""
    up = up_vector(up_axis).astype(np.float64)
    e0, e1 = ground_basis(up)

    means = scene.means[scene.opacities >= solid_opacity].astype(np.float64)
    if len(means) < 100:
        logger.warning("Open-floor grid: too few solid gaussians (%d)", len(means))
        return None
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

    spawn_h = floor + spawn_height_fraction * interior
    cell_points = (CX.ravel()[:, None] * e0[None, :]
                   + CY.ravel()[:, None] * e1[None, :]
                   + spawn_h * up[None, :])
    clearance = np.asarray(world.clearance(cell_points)).reshape(CX.shape)

    body_lo = floor + 0.15 * interior
    floor_band = (h >= floor - 0.05 * interior) & (h < body_lo)
    floor_hist, _, _ = np.histogram2d(x[floor_band], y[floor_band], bins=(x_edges, y_edges))
    # All interior-height splats (floor + furniture + walls): rooms are dense,
    # reconstruction halo outside the shell is not.
    body_hist, _, _ = np.histogram2d(x[inside], y[inside], bins=(x_edges, y_edges))
    has_floor = _dense_floor_mask(floor_hist, cell, body_hist=body_hist)

    dist_to_com = np.hypot(CX - float(x.mean()), CY - float(y.mean()))
    valid = has_floor & (clearance >= max(2.0 * cell, 0.35))
    if not valid.any():
        logger.warning("Open-floor grid: no valid free cells found")
        return None
    return OpenFloorGrid(
        world=world, up=up, e0=e0, e1=e1, spawn_h=spawn_h, cell=cell,
        nx=nx, ny=ny, CX=CX, CY=CY, cx_cells=cx_cells, cy_cells=cy_cells,
        clearance=clearance, valid=valid, dist_to_com=dist_to_com,
        extent_x=extent_x, extent_y=extent_y,
    )


def _nms_clearance(
    grid: OpenFloorGrid, mask: np.ndarray, radius: float, limit: int,
) -> list[tuple[float, float, float]]:
    """Greedy non-max suppression on clearance; returns (px, py, clearance)."""
    score = np.where(mask, grid.clearance, -np.inf)
    out: list[tuple[float, float, float]] = []
    for _ in range(max(limit, 0)):
        flat = int(np.argmax(score))
        gi, gj = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[gi, gj]):
            break
        px, py = float(grid.cx_cells[gi]), float(grid.cy_cells[gj])
        out.append((px, py, float(grid.clearance[gi, gj])))
        score[np.hypot(grid.CX - px, grid.CY - py) < radius] = -np.inf
    return out


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
    grid: OpenFloorGrid | None = None,
) -> list[SpawnPoint]:
    """Candidate start positions in open space with good vantage points.

    Pass an existing CollisionWorld to reuse its KD-tree; otherwise one is
    built here with the same solid_opacity. Pass `grid` to reuse an occupancy
    grid already built for waypoint search.
    """
    if grid is None:
        grid = _open_floor_grid(
            scene, up_axis, world=world,
            ceiling_percentile=ceiling_percentile, grid_cell=grid_cell,
            solid_opacity=solid_opacity, spawn_height_fraction=spawn_height_fraction,
        )
    if grid is None:
        return []

    score = grid.clearance - centroid_pull * grid.dist_to_com
    score = np.where(grid.valid, score, -np.inf)

    # Stage 1: shortlist spatially-separated candidates by clearance score
    # (greedy non-max suppression so they spread across the room).
    separation = max(4.0 * grid.cell, 0.15 * min(grid.extent_x, grid.extent_y))
    num_candidates = max(3 * num_points, 12)
    shortlist: list[tuple[float, float, float, float]] = []  # px, py, clearance, dist_to_com
    for _ in range(num_candidates):
        flat = int(np.argmax(score))
        gi, gj = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[gi, gj]):
            break
        px, py = float(grid.cx_cells[gi]), float(grid.cy_cells[gj])
        shortlist.append((px, py, float(grid.clearance[gi, gj]),
                          float(grid.dist_to_com[gi, gj])))
        score[np.hypot(grid.CX - px, grid.CY - py) < separation] = -np.inf
    if not shortlist:
        return []

    # Stage 2: rank the shortlist by actual visibility. Median sight-line
    # length separates open-floor vantage points from enclosed pockets
    # (curtained alcoves, gaps between furniture) that clearance alone scores
    # the same. The centroid pull is deliberately weakened here: visibility is
    # the better centrality signal, and a full pull would let a central but
    # enclosed pocket outrank open floor.
    positions = np.stack([grid.world_point(px, py) for px, py, _, _ in shortlist])
    views = _view_distances(grid.world, positions, grid.up)
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
                len(points), grid.nx, grid.ny, grid.cell,
                [f"#{p.index} clr={p.clearance:.2f} view={p.view_distance:.1f}" for p in points])
    return points


def find_waypoints(
    scene: GaussianScene,
    up_axis: str,
    world: CollisionWorld | None = None,
    num_points: int = 8,
    ceiling_percentile: float = 25.0,
    grid_cell: float = 0.15,
    solid_opacity: float = 0.1,
    spawn_height_fraction: float = 0.5,
    grid: OpenFloorGrid | None = None,
    interior_image: np.ndarray | None = None,
    interior_camera=None,
) -> list[Waypoint]:
    """Spread vantage-point waypoints across reachable indoor floor.

    Same occupancy grid as spawn (open cells at eye height, never inside
    splat volumes) but the selection maximises *coverage*: local maxima of
    clearance (centers of open clusters / rooms) are farthest-point sampled
    so picks land in different rooms rather than clustering near the scene
    centroid. Candidates must sit in a dense reconstructed neighborhood
    (indoor floor / furniture / walls), not a sparse exterior halo. Void
    islands, grey bird's-eye haze, and enclosed pockets with short
    sight-lines are dropped.
    """
    if num_points <= 0:
        return []
    if grid is None:
        grid = _open_floor_grid(
            scene, up_axis, world=world,
            ceiling_percentile=ceiling_percentile, grid_cell=grid_cell,
            solid_opacity=solid_opacity, spawn_height_fraction=spawn_height_fraction,
        )
    if grid is None:
        return []

    # Local maxima of clearance ≈ room / hallway / alcove centers. A ~1.2 m
    # neighborhood keeps one peak per furniture-scale gap, not per grid cell.
    neigh = max(3, int(round(1.2 / grid.cell)))
    if neigh % 2 == 0:
        neigh += 1
    field = np.where(grid.valid, grid.clearance, -np.inf)
    local_max = (
        (field == ndimage.maximum_filter(field, size=neigh, mode="nearest"))
        & grid.valid
    )

    floor_area = float(grid.valid.sum()) * grid.cell ** 2
    nms_r = max(8.0 * grid.cell, 0.14 * min(grid.extent_x, grid.extent_y))
    min_sep = max(
        6.0 * grid.cell,
        min(
            0.22 * min(grid.extent_x, grid.extent_y),
            0.80 * float(np.sqrt(floor_area / max(num_points, 1))),
        ),
    )

    reps = _nms_clearance(grid, local_max, nms_r, limit=max(4 * num_points, 16))
    if len(reps) < num_points:
        for cand in _nms_clearance(grid, grid.valid, min_sep, limit=max(3 * num_points, 12)):
            if all(np.hypot(cand[0] - r[0], cand[1] - r[1]) >= min_sep for r in reps):
                reps.append(cand)
    if not reps:
        return []

    positions = np.stack([grid.world_point(px, py) for px, py, _ in reps])
    views = _view_distances(grid.world, positions, grid.up)
    hit_frac = _view_hit_fraction(grid.world, positions, grid.up)
    min_view = 0.7
    # Indoor: at least ~4 of 16 rays hit something. Void-next-to-a-floater
    # has almost every ray run out to max distance.
    min_hits = 0.25
    interior = np.ones(len(reps), dtype=bool)
    if interior_image is not None and interior_camera is not None:
        interior = _on_reconstructed_interior(
            interior_image, interior_camera, positions,
        )
    kept = [
        (px, py, clr, float(view), positions[i])
        for i, ((px, py, clr), view) in enumerate(zip(reps, views))
        if view >= min_view and hit_frac[i] >= min_hits and interior[i]
    ]
    if not kept:
        # Relax sight-line checks, never the interior/density mask: those are
        # what stop a sparse exterior halo from becoming a jump target.
        order = np.argsort(-views)[:num_points]
        kept = [
            (reps[i][0], reps[i][1], reps[i][2], float(views[i]), positions[i])
            for i in order
            if interior[i]
        ]
    if not kept:
        return []

    # Farthest-point sampling: first the most open vantage, then repeatedly
    # the remaining peak farthest from already picked ones. That covers
    # distant rooms before filling a second spot in the same open area.
    kept.sort(key=lambda t: t[2], reverse=True)
    picked = [kept[0]]
    rest = kept[1:]
    while len(picked) < num_points and rest:
        best_i = None
        best_d = -1.0
        for i, cand in enumerate(rest):
            d = min(np.hypot(cand[0] - p[0], cand[1] - p[1]) for p in picked)
            if d < min_sep:
                continue
            if d > best_d:
                best_d = d
                best_i = i
        if best_i is None:
            break
        picked.append(rest.pop(best_i))

    picked.sort(key=lambda t: (t[1], t[0]))
    waypoints = [
        Waypoint(
            index=i,
            position=np.asarray(pos, dtype=np.float64),
            clearance=clr,
            view_distance=view,
        )
        for i, (_px, _py, clr, view, pos) in enumerate(picked)
    ]
    logger.info(
        "Waypoint search: %d vantage(s) on a %dx%d grid (cell %.2f, sep %.2f): %s",
        len(waypoints), grid.nx, grid.ny, grid.cell, min_sep,
        [f"W{w.index} clr={w.clearance:.2f} view={w.view_distance:.1f}" for w in waypoints],
    )
    return waypoints


class CollisionWorld:
    """Splat-extent-aware clearance queries + optional motion clamping.

    Every solid gaussian counts as a sphere of radius 2 sigma (max scale,
    capped so room-sized fog splats can't block the whole scene) around its
    center. Clearance at a point is min_k(center_distance_k - radius_k) over
    the k nearest centers — one batched cKDTree query.

    collision selects the path clamp: "full" (original hard break), "low"
    (tiny keep-out), or "off" (free-cam, the default). Spawn search always
    uses clearance(); only move / move_toward go through clamp_motion.
    """

    K_NEIGHBORS = 16
    # Cap on a single splat's effective radius: large low-opacity "fog"
    # gaussians would otherwise mark whole rooms as blocked.
    MAX_SPLAT_RADIUS = 0.75

    def __init__(self, scene: GaussianScene, solid_opacity: float = 0.1,
                 clearance_radius: float = 0.25, collision: str = COLLISION_OFF):
        mask = scene.opacities >= solid_opacity
        self._tree = cKDTree(scene.means[mask].astype(np.float64))
        self._radii = np.minimum(
            2.0 * scene.scales[mask].max(axis=1).astype(np.float64),
            self.MAX_SPLAT_RADIUS,
        )
        self.clearance_radius = float(clearance_radius)
        mode = _normalize_collision(collision)
        if mode not in COLLISION_MODES:
            raise ValueError(
                f"navigation.collision must be one of {COLLISION_MODES}, got {collision!r}"
            )
        self.collision = mode
        logger.info(
            "Collision world: %d solid gaussians, clearance radius %.2f, collision=%s",
            int(mask.sum()), self.clearance_radius, self.collision,
        )

    def keep_out(self) -> float | None:
        """Path keep-out radius, or None when collision is off (free-cam)."""
        if self.collision == COLLISION_OFF:
            return None
        if self.collision == COLLISION_LOW:
            return LOW_CLEARANCE
        return self.clearance_radius

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
        """Largest travel along unit `direction` keeping the mode's keep-out.

        Returns (allowed_distance, was_clamped). One batched KD-tree query
        over path samples, so cost is O(samples * k * log N).

        collision=off is a no-op (free-cam). collision=full is the original
        hard break (clearance_radius). collision=low uses LOW_CLEARANCE.

        If the start pose is already inside the keep-out radius (e.g. after a
        previous dive toward the floor), motion that does not get *closer* to
        geometry is still allowed, so the agent can walk out horizontally.
        """
        distance = float(distance)
        if distance <= 0.0:
            return 0.0, False
        keep_out = self.keep_out()
        if keep_out is None:
            return distance, False
        start = np.asarray(start, dtype=np.float64)
        start_c = float(self.clearance(start))
        if start_c >= keep_out:
            required = keep_out
        elif start_c > 1e-3:
            required = start_c
        else:
            required = keep_out
        step = max(keep_out * 0.5, 1e-3)
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
    world plus the camera/depth the current observation was rendered from.

    waypoints / pose_history are used by jump_to_waypoint to teleport to a
    precomputed vantage or a past step's camera pose.
    """

    world: CollisionWorld | None = None
    camera: Camera | None = None
    depth: np.ndarray | None = None
    waypoints: list[Waypoint] | None = None
    pose_history: list[dict] | None = None


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
    waypoints: list[Waypoint] = field(default_factory=list)
    base_image: np.ndarray | None = None  # unmarked bird's-eye (path-map backdrop)
    camera: Camera | None = None          # camera used for the bird's-eye render

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

    def describe_waypoints(self) -> str:
        lines = []
        for w in self.waypoints:
            x, y, z = w.position
            lines.append(
                f"Waypoint {w.index} (W{w.index}): world coordinates "
                f"({x:.2f}, {y:.2f}, {z:.2f})"
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
    grid_kwargs = dict(
        ceiling_percentile=float(nav.ceiling_percentile),
        grid_cell=float(nav.grid_cell),
        solid_opacity=float(nav.solid_opacity),
        spawn_height_fraction=float(nav.spawn_height_fraction),
    )
    grid = _open_floor_grid(scene, cfg.camera.up_axis, world=world, **grid_kwargs)
    points = find_spawn_points(
        scene,
        up_axis=cfg.camera.up_axis,
        world=world,
        num_points=int(nav.num_spawn_points),
        centroid_pull=float(nav.centroid_pull),
        grid=grid,
        **grid_kwargs,
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
    base_image = np.asarray(image).copy()
    waypoints = find_waypoints(
        scene,
        up_axis=cfg.camera.up_axis,
        world=world,
        num_points=int(nav.get("num_waypoints", 8)),
        grid=grid,
        interior_image=base_image,
        interior_camera=camera,
        **grid_kwargs,
    )
    image = draw_spawn_markers(image, camera, np.stack([p.position for p in points]))
    return SpawnSelection(
        image=image, points=points, waypoints=waypoints,
        base_image=base_image, camera=camera,
    )
