"""Discretize a 3D lidar point cloud into a 2.5D elevation grid + traversability cost.

A 2.5D grid is the standard map for legged locomotion: each XY cell stores a
single height surface plus statistics about local geometry. Unlike a flat 2D
occupancy grid it preserves steps, curbs, slopes and gaps -- exactly the features
a quadruped's footstep planner cares about. Unlike a full 3D voxel grid it stays
cheap to build and plan over.

Discretization steps:
    1. Bin every point into an (row, col) cell by XY position.
    2. Per cell, reduce the z-values to statistics: min, max, mean, count.
    3. Derive geometry layers:
         elevation  = representative ground height per cell (mean z)
         roughness  = z spread within the cell (max - min)
         slope      = magnitude of the elevation gradient to neighbours
    4. Classify each cell as FREE / LETHAL / UNKNOWN using QuadrupedParams,
       build a continuous traversal cost, then inflate lethal cells by the
       robot footprint radius.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .robot import QuadrupedParams


# Cell classification codes.
FREE = 0
LETHAL = 1      # untraversable: too tall a step, too steep, too rough, or inflated
UNKNOWN = 2     # no lidar returns landed here

# Sentinel cost meaning "do not enter".
LETHAL_COST = np.inf


@dataclass
class GridConfig:
    """Geometry of the discretization grid."""

    resolution: float = 0.10              # cell size in meters
    bounds: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    z_clip: Optional[Tuple[float, float]] = None  # drop points outside [zmin, zmax]
    unknown_cost: float = 50.0            # cost of crossing an un-observed cell
    cost_unknown_as_free: bool = False    # if True, unknown cells are freely traversable
    min_obstacle_cells: int = 3           # drop lethal blobs smaller than this (speckle)
    inflate_obstacles: bool = True        # set False to let nav2's InflationLayer inflate
    smooth_passes: int = 1                # elevation smoothing before slope/step (raise for noisy data)


@dataclass
class ElevationGrid:
    """Result of discretizing a point cloud.

    All 2D arrays are indexed [row, col] where row indexes +Y and col indexes +X.
    Use world_to_cell / cell_to_world to convert between meters and indices.
    """

    config: GridConfig
    origin: Tuple[float, float]           # world (x, y) of cell (0, 0) lower-left corner
    elevation: np.ndarray                 # float32, mean ground height (NaN where unknown)
    roughness: np.ndarray                 # float32, within-cell z std
    slope: np.ndarray                     # float32, rise/run gradient magnitude
    count: np.ndarray                     # int32, points per cell
    classes: np.ndarray                   # uint8, FREE / LETHAL / UNKNOWN
    cost: np.ndarray                      # float32, per-cell traversal cost (inf == lethal)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.elevation.shape

    @property
    def resolution(self) -> float:
        return self.config.resolution

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """World meters -> (row, col). Does not bounds-check."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """(row, col) -> world (x, y) at the cell center."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        h, w = self.shape
        return 0 <= row < h and 0 <= col < w


def _auto_bounds(xy: np.ndarray, pad: float) -> Tuple[float, float, float, float]:
    xmin, ymin = xy.min(axis=0) - pad
    xmax, ymax = xy.max(axis=0) + pad
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _cell_statistics(rows, cols, z, n_rows, n_cols):
    """Reduce scattered (row, col, z) points to per-cell statistics.

    Returns count, sum(z), sum(z^2), min(z), max(z) per cell. Vectorized with a
    flat cell index + np.add.at / np.minimum.at so it scales to millions of
    points without a Python loop over cells. The sum-of-squares lets us derive a
    robust within-cell std, which is far less sample-count-sensitive than max-min.
    """
    flat = rows * n_cols + cols
    n_cells = n_rows * n_cols

    count = np.zeros(n_cells, dtype=np.int64)
    np.add.at(count, flat, 1)

    zsum = np.zeros(n_cells, dtype=np.float64)
    np.add.at(zsum, flat, z)

    zsqsum = np.zeros(n_cells, dtype=np.float64)
    np.add.at(zsqsum, flat, z * z)

    zmin = np.full(n_cells, np.inf, dtype=np.float64)
    np.minimum.at(zmin, flat, z)

    zmax = np.full(n_cells, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, flat, z)

    return (
        count.reshape(n_rows, n_cols),
        zsum.reshape(n_rows, n_cols),
        zsqsum.reshape(n_rows, n_cols),
        zmin.reshape(n_rows, n_cols),
        zmax.reshape(n_rows, n_cols),
    )


def _smooth_nan(a: np.ndarray, passes: int = 1) -> np.ndarray:
    """NaN-aware 3x3 box smoothing: each cell becomes the mean of its finite
    neighbours. Used to denoise the elevation surface before computing slope and
    step, so per-cell sensor noise doesn't masquerade as terrain features.
    """
    out = a.astype(np.float32).copy()
    for _ in range(passes):
        finite = np.isfinite(out).astype(np.float32)
        vals = np.nan_to_num(out, nan=0.0)
        acc = np.zeros_like(vals)
        cnt = np.zeros_like(vals)
        h, w = out.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys = slice(max(0, dy), h + min(0, dy))
                xs = slice(max(0, dx), w + min(0, dx))
                sys = slice(max(0, -dy), h + min(0, -dy))
                sxs = slice(max(0, -dx), w + min(0, -dx))
                acc[ys, xs] += vals[sys, sxs]
                cnt[ys, xs] += finite[sys, sxs]
        with np.errstate(invalid="ignore"):
            smoothed = np.where(cnt > 0, acc / cnt, np.nan)
        # keep originally-unknown cells unknown
        smoothed[~np.isfinite(a)] = np.nan
        out = smoothed.astype(np.float32)
    return out


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift `a` by (dr, dc), filling exposed border with NaN."""
    out = np.full_like(a, np.nan)
    h, w = a.shape
    ys = slice(max(0, dr), h + min(0, dr))
    xs = slice(max(0, dc), w + min(0, dc))
    sys = slice(max(0, -dr), h + min(0, -dr))
    sxs = slice(max(0, -dc), w + min(0, -dc))
    out[ys, xs] = a[sys, sxs]
    return out


def _nan_partial(a: np.ndarray, dr: int, dc: int, resolution: float) -> np.ndarray:
    """Partial derivative of `a` along one axis using only finite neighbours.

    Central difference where both neighbours are known, one-sided where only one
    is, zero otherwise. Never invents values for unknown cells, so the slope of
    real ground next to a map hole or the grid border is not contaminated.
    """
    fwd = _shift(a, -dr, -dc)   # neighbour in +(dr,dc) direction
    bwd = _shift(a, dr, dc)     # neighbour in -(dr,dc) direction
    have_f = np.isfinite(fwd)
    have_b = np.isfinite(bwd)

    central = (fwd - bwd) / (2.0 * resolution)
    one_fwd = (fwd - a) / resolution
    one_bwd = (a - bwd) / resolution

    g = np.zeros_like(a)
    g = np.where(have_f & have_b, central, g)
    g = np.where(have_f & ~have_b, one_fwd, g)
    g = np.where(~have_f & have_b, one_bwd, g)
    return g


def _slope_magnitude(elevation: np.ndarray, resolution: float) -> np.ndarray:
    """Gradient magnitude of the elevation surface, as rise/run (NaN-aware)."""
    known = np.isfinite(elevation)
    if not known.any():
        return np.full_like(elevation, np.nan)
    gx = _nan_partial(elevation, 0, 1, resolution)
    gy = _nan_partial(elevation, 1, 0, resolution)
    slope = np.hypot(gx, gy).astype(np.float32)
    slope[~known] = np.nan
    return slope


def _disk_offsets(r: int):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dx * dx + dy * dy <= r * r]


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation of `mask` by a disk of the given radius (in cells).

    Pure-numpy (no scipy): OR-shift the mask over every offset within the disk.
    """
    if radius_cells <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    for dy, dx in _disk_offsets(radius_cells):
        ys = slice(max(0, dy), h + min(0, dy))
        xs = slice(max(0, dx), w + min(0, dx))
        sys = slice(max(0, -dy), h + min(0, -dy))
        sxs = slice(max(0, -dx), w + min(0, -dx))
        out[ys, xs] |= mask[sys, sxs]
    return out


def _erode(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary erosion: a cell survives only if every disk neighbour is set.

    Implemented as the complement of dilating the inverse mask, with out-of-grid
    treated as background so border cells erode away (conservative).
    """
    if radius_cells <= 0:
        return mask.copy()
    return ~_dilate(~mask, radius_cells)


def _remove_small_blobs(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected True-regions smaller than `min_size` cells.

    Unlike morphological opening this removes isolated sensor speckle without
    thinning genuine obstacles -- a 2-cell-wide wall is one large component and
    survives, whereas a lone false-positive cell is its own tiny component and is
    removed. 8-connectivity; iterative flood fill (no recursion / scipy).
    """
    if min_size <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    h, w = mask.shape
    visited = np.zeros_like(mask)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in nbrs:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            if len(comp) >= min_size:
                for r, c in comp:
                    out[r, c] = True
    return out


def discretize(
    points: np.ndarray,
    config: Optional[GridConfig] = None,
    robot: Optional[QuadrupedParams] = None,
) -> ElevationGrid:
    """Turn an (N,3+) lidar point cloud into a 2.5D elevation + cost grid.

    Args:
        points: array of shape (N, 3) or (N, >=3); columns 0,1,2 are x, y, z.
        config: grid geometry; defaults to 0.10 m cells and auto bounds.
        robot:  quadruped mobility limits; defaults to a Go2/Spot-class robot.

    Returns:
        ElevationGrid with elevation, roughness, slope, classes, and cost layers.
    """
    config = config or GridConfig()
    robot = robot or QuadrupedParams()

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3) with x,y,z in cols 0,1,2")

    xy = pts[:, :2]
    z = pts[:, 2]

    if config.z_clip is not None:
        zlo, zhi = config.z_clip
        keep = (z >= zlo) & (z <= zhi)
        xy, z = xy[keep], z[keep]
    if xy.shape[0] == 0:
        raise ValueError("no points left to discretize (check z_clip / input)")

    if config.bounds is not None:
        xmin, ymin, xmax, ymax = config.bounds
    else:
        # Tight bounds (no padding): a padded ring of unknown cells would let the
        # planner route around obstacles through never-sensed border space.
        xmin, ymin, xmax, ymax = _auto_bounds(xy, pad=0.0)

    res = config.resolution
    n_cols = max(1, int(np.ceil((xmax - xmin) / res)))
    n_rows = max(1, int(np.ceil((ymax - ymin) / res)))

    cols = np.floor((xy[:, 0] - xmin) / res).astype(np.int64)
    rows = np.floor((xy[:, 1] - ymin) / res).astype(np.int64)
    if config.bounds is None:
        # Points exactly on the max edge land one cell past the grid; clamp them
        # into the boundary cells rather than dropping them.
        cols = np.clip(cols, 0, n_cols - 1)
        rows = np.clip(rows, 0, n_rows - 1)
        inside = np.ones(rows.shape[0], dtype=bool)
    else:
        inside = (cols >= 0) & (cols < n_cols) & (rows >= 0) & (rows < n_rows)
    rows, cols, z = rows[inside], cols[inside], z[inside]

    count, zsum, zsqsum, zmin, zmax = _cell_statistics(rows, cols, z, n_rows, n_cols)

    observed = count > 0
    elevation = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    elevation[observed] = (zsum[observed] / count[observed]).astype(np.float32)

    # Robust within-cell roughness: std of z, not max-min. Std barely moves with a
    # single outlier return, whereas max-min grows with both noise and point count.
    roughness = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    mean_obs = zsum[observed] / count[observed]
    var_obs = np.maximum(0.0, zsqsum[observed] / count[observed] - mean_obs * mean_obs)
    roughness[observed] = np.sqrt(var_obs).astype(np.float32)

    # Denoise the surface before deriving slope/step so sensor jitter on flat
    # ground is not mistaken for terrain features. Raw elevation is kept for display.
    elevation_smooth = _smooth_nan(elevation, passes=config.smooth_passes)
    slope = _slope_magnitude(elevation_smooth, res)
    step = _neighbor_step(elevation_smooth)

    # --- Classification -------------------------------------------------------
    classes = np.full((n_rows, n_cols), UNKNOWN, dtype=np.uint8)
    classes[observed] = FREE

    too_rough = observed & (roughness > robot.max_roughness)
    too_steep = observed & np.isfinite(slope) & (slope > robot.max_slope_tan)
    # Step height vs 4-neighbours: a cell next to a much higher/lower one is an edge.
    too_tall = observed & np.isfinite(step) & (step > robot.max_step)

    lethal = too_rough | too_steep | too_tall

    # Remove isolated false-positive cells (sensor speckle) before inflation, so a
    # single noisy cell doesn't bloom into a robot-radius blob. Real obstacles form
    # larger connected components and survive.
    lethal = _remove_small_blobs(lethal, min_size=config.min_obstacle_cells)
    classes[lethal] = LETHAL

    # Inflate the cleaned obstacle mask by the robot footprint for clearance.
    # Disable when an external consumer (e.g. a nav2 InflationLayer) will inflate,
    # so the footprint clearance isn't applied twice.
    if config.inflate_obstacles:
        radius_cells = int(np.ceil(robot.radius / res))
        inflated = _dilate(lethal, radius_cells)
        classes[inflated & observed] = LETHAL

    # --- Continuous cost ------------------------------------------------------
    cost = np.ones((n_rows, n_cols), dtype=np.float32)

    # Smoothly penalize roughness and slope below the lethal thresholds so the
    # planner prefers flat, smooth ground when it has a choice.
    with np.errstate(invalid="ignore"):
        rough_term = np.nan_to_num(roughness / robot.max_roughness) * 3.0
        slope_term = np.nan_to_num(slope / robot.max_slope_tan) * 3.0
    cost += rough_term + slope_term

    cost[classes == LETHAL] = LETHAL_COST
    if config.cost_unknown_as_free:
        cost[classes == UNKNOWN] = 1.0
    else:
        cost[classes == UNKNOWN] = config.unknown_cost

    meta = {
        "n_points": int(pts.shape[0]),
        "n_points_used": int(rows.shape[0]),
        "observed_cells": int(observed.sum()),
        "lethal_cells": int((classes == LETHAL).sum()),
        "unknown_cells": int((classes == UNKNOWN).sum()),
        "robot": robot,
    }

    return ElevationGrid(
        config=config,
        origin=(xmin, ymin),
        elevation=elevation,
        roughness=roughness,
        slope=slope,
        count=count.astype(np.int32),
        classes=classes,
        cost=cost,
        meta=meta,
    )


def cost_to_occupancy(grid: "ElevationGrid", soft_cost_cap: Optional[float] = None) -> np.ndarray:
    """Encode a cost grid as ROS OccupancyGrid values (int8, row-major).

    Mapping (matches nav2 StaticLayer with trinary_costmap:false, track_unknown_space:true):
        unknown cell      -> -1   (NO_INFORMATION)
        lethal cell       -> 100  (LETHAL_OBSTACLE)
        free cell, cost c -> 1..99, linearly scaled from [1, soft_cost_cap]

    StaticLayer then rescales 0..99 back into the costmap's cost range, so the
    soft slope/roughness penalties survive into nav2's planning. ROS-free so it
    can be unit-tested without rclpy; the node just wraps the result in a message.

    Returns an (n_rows, n_cols) int8 array indexed [row, col] (row indexes +Y),
    which is exactly OccupancyGrid row-major order when flattened with .ravel().
    """
    cost = grid.cost
    occ = np.zeros(cost.shape, dtype=np.int8)

    unknown = grid.classes == UNKNOWN
    lethal = ~np.isfinite(cost)
    soft = ~unknown & ~lethal

    if soft.any():
        c = cost[soft]
        lo = 1.0
        hi = soft_cost_cap if soft_cost_cap is not None else float(np.percentile(c, 99))
        hi = max(hi, lo + 1e-6)
        scaled = (c - lo) / (hi - lo)
        occ[soft] = np.clip(np.round(scaled * 98.0) + 1, 1, 99).astype(np.int8)

    occ[lethal] = 100
    occ[unknown] = -1
    return occ


def _neighbor_step(elevation: np.ndarray) -> np.ndarray:
    """Max absolute height difference to the 4-connected neighbours of each cell.

    Captures curbs/steps/gap edges that single-cell roughness misses. Unknown
    neighbours are ignored.
    """
    h, w = elevation.shape
    step = np.zeros((h, w), dtype=np.float32)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in shifts:
        ys = slice(max(0, dy), h + min(0, dy))
        xs = slice(max(0, dx), w + min(0, dx))
        sys = slice(max(0, -dy), h + min(0, -dy))
        sxs = slice(max(0, -dx), w + min(0, -dx))
        diff = np.abs(elevation[ys, xs] - elevation[sys, sxs])
        np.nan_to_num(diff, copy=False, nan=0.0)
        step[ys, xs] = np.maximum(step[ys, xs], diff)
    step[~np.isfinite(elevation)] = np.nan
    return step
