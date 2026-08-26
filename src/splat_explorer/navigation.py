"""Scene-aware navigation: ceiling removal, spawn-point search, collision.

Spawn-point search
------------------
Interior gaussians (ceiling stripped by height percentile) are projected onto
the ground plane and binned into a 2D occupancy grid. A Euclidean distance
transform (scipy.ndimage.distance_transform_edt) then gives every free cell
its distance to the nearest occupied cell in O(cells) — an efficient way to
find spots that are "as far as possible from the closest splats" even for
multi-million-splat scenes (histogram + EDT, never pairwise distances). The
score subtracts a pull toward the scene's 2D center of mass, so winners sit
centrally (roughly equidistant from the surrounding geometry) instead of in
far corners, and greedy non-max suppression spaces the picks apart. A cell
only qualifies if solid floor splats exist beneath it, which rejects
open-space maxima outside the room shell.

Collision
---------
CollisionWorld holds a cKDTree over solid (opacity-filtered) splat centers.
Every camera move is clamped by sampling the path and stopping before the
clearance to the nearest center drops below clearance_radius, so the agent can
no longer end up inside walls or furniture.

move_toward
-----------
The VLM picks a pixel in its current RGB view plus an amount in [0, 1]. The
pixel is unprojected through the depth map rendered for that exact view (the
depth value is the first splat surface hit along that ray), giving a world
target; the camera travels amount x (distance to target), capped a safety
margin short of the surface and additionally collision-clamped along the path.
amount = 1 therefore means "right up to the surface", never inside it.
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

# Cap on occupancy grid cells; the cell size grows for huge scenes.
_MAX_GRID_CELLS = 1_500_000


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
    position: np.ndarray  # (3,) world
    clearance: float      # horizontal distance to the nearest occupied cell (scene units)


def find_spawn_points(
    scene: GaussianScene,
    up_axis: str,
    num_points: int = 5,
    ceiling_percentile: float = 25.0,
    grid_cell: float = 0.15,
    solid_opacity: float = 0.3,
    spawn_height_fraction: float = 0.5,
    centroid_pull: float = 0.35,
) -> list[SpawnPoint]:
    """Candidate start positions in open space with good vantage points."""
    up = up_vector(up_axis).astype(np.float64)
    e0, e1 = ground_basis(up)

    means = scene.means[scene.opacities >= solid_opacity].astype(np.float64)
    if len(means) < 100:
        logger.warning("Spawn search: too few solid gaussians (%d)", len(means))
        return []

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

    # Occupancy: everything at body height (above floor clutter noise, below
    # the ceiling cut). The floor band below tells us where the room actually
    # extends, so free space outside the walls never qualifies.
    body_lo = floor + 0.15 * interior
    body = (h >= body_lo) & (h <= cut) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    occ_hist, _, _ = np.histogram2d(x[body], y[body], bins=(x_edges, y_edges))
    occupied = occ_hist >= 2  # >=2 splats: robust to lone noise centers

    floor_band = (h >= floor - 0.05 * interior) & (h < body_lo)
    floor_hist, _, _ = np.histogram2d(x[floor_band], y[floor_band], bins=(x_edges, y_edges))
    has_floor = ndimage.binary_dilation(floor_hist >= 1, iterations=2)

    # Distance (scene units) from every free cell to the nearest occupied cell.
    clearance = ndimage.distance_transform_edt(~occupied) * cell

    cx_cells = 0.5 * (x_edges[:-1] + x_edges[1:])
    cy_cells = 0.5 * (y_edges[:-1] + y_edges[1:])
    CX, CY = np.meshgrid(cx_cells, cy_cells, indexing="ij")
    dist_to_com = np.hypot(CX - float(x.mean()), CY - float(y.mean()))

    score = clearance - centroid_pull * dist_to_com
    valid = (~occupied) & has_floor & (clearance >= max(2.0 * cell, 0.3))
    if not valid.any():
        logger.warning("Spawn search: no valid free cells found")
        return []
    score = np.where(valid, score, -np.inf)

    # Greedy top-k with non-max suppression so points spread across the room.
    separation = max(4.0 * cell, 0.15 * min(extent_x, extent_y))
    spawn_h = floor + spawn_height_fraction * interior
    points: list[SpawnPoint] = []
    for i in range(num_points):
        flat = int(np.argmax(score))
        gi, gj = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[gi, gj]):
            break
        px, py = float(cx_cells[gi]), float(cy_cells[gj])
        position = px * e0 + py * e1 + spawn_h * up
        points.append(SpawnPoint(index=i, position=position.astype(np.float64),
                                 clearance=float(clearance[gi, gj])))
        score[np.hypot(CX - px, CY - py) < separation] = -np.inf

    logger.info("Spawn search: %d candidate(s) on a %dx%d grid (cell %.2f): %s",
                len(points), nx, ny, cell,
                [f"#{p.index} clr={p.clearance:.2f}" for p in points])
    return points


class CollisionWorld:
    """cKDTree over solid splat centers: clearance queries + motion clamping."""

    def __init__(self, scene: GaussianScene, solid_opacity: float = 0.3,
                 clearance_radius: float = 0.25):
        mask = scene.opacities >= solid_opacity
        self._tree = cKDTree(scene.means[mask].astype(np.float64))
        self.clearance_radius = float(clearance_radius)
        logger.info("Collision world: %d solid gaussians, clearance radius %.2f",
                    int(mask.sum()), self.clearance_radius)

    def clearance(self, point: np.ndarray) -> float:
        return float(self._tree.query(np.asarray(point, dtype=np.float64))[0])

    def clamp_motion(
        self, start: np.ndarray, direction: np.ndarray, distance: float
    ) -> tuple[float, bool]:
        """Largest travel along unit `direction` keeping clearance_radius.

        Returns (allowed_distance, was_clamped). One batched KD-tree query
        over path samples, so cost is O(samples * log N).
        """
        distance = float(distance)
        if distance <= 0.0:
            return 0.0, False
        step = self.clearance_radius * 0.5
        n = int(np.ceil(distance / step))
        ts = np.linspace(0.0, distance, n + 1)[1:]
        pts = np.asarray(start, dtype=np.float64)[None, :] + np.asarray(direction)[None, :] * ts[:, None]
        dists, _ = self._tree.query(pts)
        bad = np.flatnonzero(dists < self.clearance_radius)
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
) -> dict | None:
    """Resolve a move_toward action into a new camera position.

    Returns None when there is no geometry anywhere near the picked pixel;
    otherwise a dict with new_position / target_distance / travelled / blocked.
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

    vec = target - position
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return None
    unit = vec / dist

    margin = world.clearance_radius if world is not None else 0.25
    travel = float(np.clip(amount, 0.0, 1.0)) * dist
    travel = min(travel, max(dist - margin, 0.0))
    blocked = False
    if world is not None and travel > 0.0:
        travel, blocked = world.clamp_motion(position, unit, travel)

    return {
        "new_position": position + unit * travel,
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
                f"~{p.clearance:.2f} units of open space around it"
            )
        return "\n".join(lines)


def prepare_spawn_selection(scene: GaussianScene, cfg) -> SpawnSelection | None:
    """Spawn-point search + annotated bird's-eye render, from the full config.

    Returns None when the search finds nothing usable (caller falls back to
    the legacy centroid spawn).
    """
    from .rendering.annotate import draw_spawn_markers
    from .rendering.birdseye import render_birdseye

    nav = cfg.navigation
    points = find_spawn_points(
        scene,
        up_axis=cfg.camera.up_axis,
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
