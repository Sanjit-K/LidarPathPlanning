"""Dijkstra global path planning over the traversability cost grid.

The paper uses Dijkstra on the grid map for the long-range global path. Dijkstra
is uniform-cost search: it is A* with a zero heuristic, so it expands strictly by
accumulated cost and is guaranteed shortest-cost. (The existing `astar` would give
the same path faster via its octile heuristic; this module exists to mirror the
paper's stated algorithm.)

8-connected; the cost of entering a cell is the mean of the two cells' costs times
the step distance (1 or sqrt(2)); np.inf cells are impassable.
"""

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def dijkstra(cost: np.ndarray, start: Cell, goal: Cell,
             allow_diagonal: bool = True) -> Optional[List[Cell]]:
    """Shortest-cost path from start to goal over `cost` (np.inf == blocked)."""
    h, w = cost.shape

    def passable(c: Cell) -> bool:
        return 0 <= c[0] < h and 0 <= c[1] < w and np.isfinite(cost[c[0], c[1]])

    if not passable(start):
        raise ValueError("start cell is blocked or out of bounds")
    if not passable(goal):
        raise ValueError("goal cell is blocked or out of bounds")

    neighbors = _NEIGHBORS if allow_diagonal else _NEIGHBORS[:4]
    dist = {start: 0.0}
    came_from = {}
    pq: List[Tuple[float, Cell]] = [(0.0, start)]
    visited = set()

    while pq:
        d, cur = heapq.heappop(pq)
        if cur == goal:
            return _reconstruct(came_from, cur)
        if cur in visited:
            continue
        visited.add(cur)
        cr, cc = cur
        for dr, dc, sd in neighbors:
            nb = (cr + dr, cc + dc)
            if nb in visited or not passable(nb):
                continue
            if dr != 0 and dc != 0:
                if not (passable((cr + dr, cc)) and passable((cr, cc + dc))):
                    continue
            nd = d + sd * 0.5 * (cost[cr, cc] + cost[nb[0], nb[1]])
            if nd < dist.get(nb, math.inf):
                dist[nb] = nd
                came_from[nb] = cur
                heapq.heappush(pq, (nd, nb))
    return None


def _reconstruct(came_from, cur) -> List[Cell]:
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path
