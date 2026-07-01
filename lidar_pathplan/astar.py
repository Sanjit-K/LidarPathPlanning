"""A* path planning over a 2.5D traversability cost grid.

Operates on the `cost` layer produced by elevation_grid.discretize: each cell
holds a per-cell traversal cost, with np.inf marking lethal/inflated obstacles.
8-connected moves; the cost of stepping into a cell is the average of the two
cells' costs scaled by the move distance (1 or sqrt(2)). The heuristic is the
octile distance, which is admissible for 8-connected grids.
"""

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np


Cell = Tuple[int, int]

# 8-connected neighbour offsets with their base step distances.
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def _octile(a: Cell, b: Cell) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def astar(
    cost: np.ndarray,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool = True,
    heuristic_weight: float = 1.0,
) -> Optional[List[Cell]]:
    """Find a least-cost path from `start` to `goal` over a cost grid.

    Args:
        cost: 2D float array; cell value is traversal cost, np.inf == blocked.
        start, goal: (row, col) cells.
        allow_diagonal: enable 8-connectivity (False -> 4-connectivity).
        heuristic_weight: >1 makes search greedier/faster but sub-optimal.

    Returns:
        List of (row, col) cells from start to goal inclusive, or None if no path.
    """
    h, w = cost.shape

    def passable(c: Cell) -> bool:
        return 0 <= c[0] < h and 0 <= c[1] < w and np.isfinite(cost[c[0], c[1]])

    if not passable(start):
        raise ValueError("start cell is blocked or out of bounds")
    if not passable(goal):
        raise ValueError("goal cell is blocked or out of bounds")

    neighbors = _NEIGHBORS if allow_diagonal else _NEIGHBORS[:4]

    open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct(came_from, current)
        if current in closed:
            continue
        closed.add(current)

        cr, cc = current
        for dr, dc, dist in neighbors:
            nb = (cr + dr, cc + dc)
            if nb in closed or not passable(nb):
                continue
            # Prevent diagonal corner-cutting through two blocked orthogonals.
            if dr != 0 and dc != 0:
                if not (passable((cr + dr, cc)) and passable((cr, cc + dc))):
                    continue
            step_cost = dist * 0.5 * (cost[cr, cc] + cost[nb[0], nb[1]])
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(nb, math.inf):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + heuristic_weight * _octile(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    return None


def _reconstruct(came_from: dict, current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_length(path: List[Cell], resolution: float) -> float:
    """Euclidean length of a cell path in meters."""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        total += math.hypot(r1 - r0, c1 - c0) * resolution
    return total
