"""Geometric path smoothing -- a simplified stand-in for the paper's TEB step.

The paper smooths the raw Dijkstra path with a Timed-Elastic-Band optimizer. Full
TEB jointly optimizes geometry *and* timing under kinodynamic constraints. Here we
implement the geometric core of an elastic band, which is what matters for a
2.5D footstep-level path: each interior waypoint is pulled toward the midpoint of
its neighbours (contraction -> shorter, smoother) and pushed away from obstacles
that are within an influence distance (clearance). Endpoints stay fixed.

This removes the staircasing of a grid path and keeps the robot off obstacles,
without claiming to be the full time-optimal TEB.
"""

from typing import List, Tuple

import numpy as np


def _chamfer_distance(obstacle: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance (in cells) to the nearest obstacle cell.

    Two-pass chamfer transform (3,4 weights / 3), pure numpy. Cells that are
    obstacles have distance 0; free cells get their distance to the closest one.
    """
    h, w = obstacle.shape
    BIG = float(h + w) * 2
    d = np.where(obstacle, 0.0, BIG).astype(np.float64)
    a, b = 1.0, np.sqrt(2.0)
    # forward pass
    for r in range(h):
        for c in range(w):
            v = d[r, c]
            if r > 0:
                v = min(v, d[r - 1, c] + a)
                if c > 0:
                    v = min(v, d[r - 1, c - 1] + b)
                if c < w - 1:
                    v = min(v, d[r - 1, c + 1] + b)
            if c > 0:
                v = min(v, d[r, c - 1] + a)
            d[r, c] = v
    # backward pass
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            v = d[r, c]
            if r < h - 1:
                v = min(v, d[r + 1, c] + a)
                if c > 0:
                    v = min(v, d[r + 1, c - 1] + b)
                if c < w - 1:
                    v = min(v, d[r + 1, c + 1] + b)
            if c < w - 1:
                v = min(v, d[r, c + 1] + a)
            d[r, c] = v
    return d


def smooth_path(path_cells: List[Tuple[int, int]], grid,
                influence: float = 0.5, w_smooth: float = 0.3,
                w_clear: float = 0.15, iterations: int = 80) -> List[Tuple[float, float]]:
    """Smooth a grid path into a world-frame (x, y) polyline.

    Args:
        path_cells: list of (row, col) from Dijkstra/A*.
        grid: an ElevationGrid (provides resolution, origin, classes/cost).
        influence: obstacle clearance influence distance, meters.
        w_smooth: contraction weight (toward neighbour midpoint).
        w_clear: obstacle-repulsion weight.
        iterations: gradient-descent steps.

    Returns the smoothed list of (x, y) world points.
    """
    if path_cells is None or len(path_cells) < 3:
        return [grid.cell_to_world(r, c) for r, c in (path_cells or [])]

    res = grid.resolution
    obstacle = ~np.isfinite(grid.cost)
    dist_cells = _chamfer_distance(obstacle)
    dist_m = dist_cells * res
    # gradient of the distance field points away from obstacles
    gy, gx = np.gradient(dist_m, res)

    pts = np.array([grid.cell_to_world(r, c) for r, c in path_cells], dtype=np.float64)
    h, w = grid.shape

    def sample(field, x, y):
        col = int(round((x - grid.origin[0]) / res - 0.5))
        row = int(round((y - grid.origin[1]) / res - 0.5))
        row = min(max(row, 0), h - 1)
        col = min(max(col, 0), w - 1)
        return field[row, col]

    for _ in range(iterations):
        new = pts.copy()
        for i in range(1, len(pts) - 1):
            x, y = pts[i]
            # contraction toward midpoint of neighbours
            mid = 0.5 * (pts[i - 1] + pts[i + 1])
            smooth = w_smooth * (mid - pts[i])
            # repulsion when closer than `influence` to an obstacle
            d = sample(dist_m, x, y)
            clear = np.zeros(2)
            if d < influence:
                grad = np.array([sample(gx, x, y), sample(gy, x, y)])
                norm = np.linalg.norm(grad)
                if norm > 1e-6:
                    strength = (influence - d) / influence
                    clear = w_clear * strength * grad / norm
            cand = pts[i] + smooth + clear
            # never push a waypoint into an obstacle cell
            if np.isfinite(grid.cost[_clamp_cell(grid, cand)]):
                new[i] = cand
        pts = new
    return [tuple(p) for p in pts]


def _clamp_cell(grid, xy):
    h, w = grid.shape
    res = grid.resolution
    col = int((xy[0] - grid.origin[0]) / res)
    row = int((xy[1] - grid.origin[1]) / res)
    return (min(max(row, 0), h - 1), min(max(col, 0), w - 1))
