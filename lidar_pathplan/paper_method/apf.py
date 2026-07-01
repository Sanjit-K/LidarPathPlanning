"""Artificial Potential Field (APF) local obstacle avoidance -- Khatib formulation.

The paper's local planner: the goal exerts an attractive force, nearby obstacles
exert repulsive forces, and the robot moves along the resultant. This is the
classic Khatib model.

Attractive (eq. 14-15 in the paper):
    U_att = 1/2 * k_att * d(p, goal)^2
    F_att = k_att * (goal - p)          (conic far field: capped magnitude)

Repulsive (eq. 16):
    U_rep = 1/2 * k_rep * (1/rho - 1/rho0)^2   for rho <= rho0, else 0
    F_rep = k_rep * (1/rho - 1/rho0) * (1/rho^2) * grad(rho)
where rho is the distance to the nearest obstacle and rho0 its influence radius.

The goal may be **fixed** or a **moving target** (a callable of time): the latter
is the person-following extension -- attractive force simply tracks the person
each step while repulsion keeps the robot off terrain hazards.

Obstacle distances come from a chamfer distance field over the map's lethal cells,
so each force query is O(1) regardless of obstacle count.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .elastic_band import _chamfer_distance


@dataclass
class APFParams:
    k_att: float = 1.0          # attractive gain
    k_rep: float = 0.6          # repulsive gain
    rho0: float = 0.8           # obstacle influence radius (m)
    max_speed: float = 0.6      # m/s cap on the commanded velocity
    dt: float = 0.1             # integration step (s)
    goal_tol: float = 0.15      # reached-goal distance (m)
    att_cap_dist: float = 2.0   # beyond this, attractive force is constant (conic)
    rep_clip: float = 50.0      # clamp on repulsive magnitude (avoids blow-up near rho->0)


GoalLike = Union[Tuple[float, float], Callable[[float], Tuple[float, float]]]


class APFPlanner:
    """Khatib APF planner over a static obstacle field, fixed or moving goal."""

    def __init__(self, grid, params: Optional[APFParams] = None):
        self.grid = grid
        self.params = params or APFParams()
        self.res = grid.resolution
        self.origin = grid.origin
        obstacle = ~np.isfinite(grid.cost)
        self._dist = _chamfer_distance(obstacle) * self.res   # meters to nearest obstacle
        self._gy, self._gx = np.gradient(self._dist, self.res)
        self._h, self._w = grid.shape

    # ---------------------------------------------------------------- sampling
    def _sample(self, field, x, y):
        col = int(round((x - self.origin[0]) / self.res - 0.5))
        row = int(round((y - self.origin[1]) / self.res - 0.5))
        row = min(max(row, 0), self._h - 1)
        col = min(max(col, 0), self._w - 1)
        return field[row, col]

    def force(self, pos, goal) -> np.ndarray:
        """Resultant force (attractive + repulsive) at `pos` toward `goal`."""
        pos = np.asarray(pos, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        p = self.params

        # Attractive (conic far field so distant goals don't produce huge forces).
        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)
        if dist < 1e-9:
            f_att = np.zeros(2)
        elif dist <= p.att_cap_dist:
            f_att = p.k_att * to_goal
        else:
            f_att = p.k_att * p.att_cap_dist * to_goal / dist

        # Repulsive from the nearest obstacle (via the distance field gradient).
        rho = self._sample(self._dist, pos[0], pos[1])
        f_rep = np.zeros(2)
        if 0.0 < rho <= p.rho0:
            grad = np.array([self._sample(self._gx, pos[0], pos[1]),
                             self._sample(self._gy, pos[0], pos[1])])
            n = np.linalg.norm(grad)
            if n > 1e-6:
                mag = p.k_rep * (1.0 / rho - 1.0 / p.rho0) / (rho * rho)
                mag = min(mag, p.rep_clip)
                f_rep = mag * grad / n      # grad points away from the obstacle
        return f_att + f_rep

    # ---------------------------------------------------------------- rollout
    def run(self, start, goal: GoalLike, max_steps: int = 600,
            escape_kicks: bool = True) -> List[Tuple[float, float]]:
        """Roll out the trajectory from `start` until the goal is reached.

        `goal` is either a fixed (x, y) or a callable t -> (x, y) for a moving
        target (person-following). Returns the list of (x, y) poses.
        """
        p = self.params
        pos = np.asarray(start, dtype=np.float64)
        traj = [tuple(pos)]
        stuck = 0
        for k in range(max_steps):
            t = k * p.dt
            g = np.asarray(goal(t) if callable(goal) else goal, dtype=np.float64)
            if np.linalg.norm(g - pos) <= p.goal_tol:
                break
            f = self.force(pos, g)
            speed = np.linalg.norm(f)
            if speed > p.max_speed:
                f = f * (p.max_speed / speed)
            # local-minimum escape: nudge perpendicular if barely moving but not there
            if escape_kicks and speed < 0.05 and np.linalg.norm(g - pos) > p.goal_tol:
                stuck += 1
                if stuck > 3:
                    perp = np.array([-(g - pos)[1], (g - pos)[0]])
                    n = np.linalg.norm(perp)
                    if n > 1e-6:
                        f = f + 0.3 * p.max_speed * perp / n
            else:
                stuck = 0
            pos = pos + f * p.dt
            traj.append(tuple(pos))
        return traj

    def follow_path(self, path_world: List[Tuple[float, float]],
                    lookahead: float = 0.7, max_steps: int = 2000) -> List[Tuple[float, float]]:
        """Track a global (Dijkstra) path with APF local avoidance.

        This is the paper's two-level architecture: the global planner provides the
        route, and APF handles real-time deviation around obstacles. A carrot
        sub-goal slides along the path a `lookahead` ahead of the robot, so the
        attractive force never points across a wall (which is what traps a single
        far goal in a local minimum); repulsion still steers around hazards.
        """
        p = self.params
        path = [np.asarray(q, dtype=np.float64) for q in path_world]
        if len(path) < 2:
            return [tuple(q) for q in path]
        pos = path[0].copy()
        traj = [tuple(pos)]
        target_i = 0
        for _ in range(max_steps):
            while target_i < len(path) - 1 and np.linalg.norm(pos - path[target_i]) < lookahead:
                target_i += 1
            subgoal = path[target_i]
            if target_i == len(path) - 1 and np.linalg.norm(pos - subgoal) <= p.goal_tol:
                break
            f = self.force(pos, subgoal)
            speed = np.linalg.norm(f)
            if speed > p.max_speed:
                f = f * (p.max_speed / speed)
            pos = pos + f * p.dt
            traj.append(tuple(pos))
        return traj
