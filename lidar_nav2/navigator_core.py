"""ROS-free real-time navigation core for the quadruped.

This is the onboard planning loop, kept free of rclpy so it can be unit-tested
off-robot and reused outside ROS:

    every new scan:  build a robot-centric local traversability grid
    every replan :   A* from the robot's own cell to the goal (projected into
                     the local map if it lies beyond it)
    every tick   :   carrot-follow the current path -> (v_x, yaw_rate) command

Design points for real-time use on the dog:
  * The grid is a rolling square window centered on the robot, so cost stays
    bounded no matter how big the environment is. The START is always the
    robot's (lidar's) current cell.
  * A goal beyond the window is projected onto the window edge along the
    robot->goal line; the robot planning toward the projection walks the window
    toward the true goal (classic rolling-horizon planning).
  * Replanning every cycle makes obstacle avoidance implicit: anything the
    lidar sees enters the next grid and the next plan routes around it.
  * Safety: if the scan is stale or no path exists, the commanded velocity is
    zero (the dog stops rather than walking blind).

The thin rclpy wrapper in realtime_nav_node.py wires this to topics.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, astar


@dataclass
class NavConfig:
    map_size: float = 8.0          # side of the rolling local map (m)
    resolution: float = 0.10       # cell size (m)
    z_min: float = -1.0            # z-clip on incoming points (sensor/odom frame)
    z_max: float = 1.5
    goal_tolerance: float = 0.25   # reached when within this distance (m)
    lookahead: float = 0.6         # carrot distance along the path (m)
    max_speed: float = 0.5         # forward speed limit (m/s)
    max_yaw_rate: float = 1.2      # yaw rate limit (rad/s)
    k_yaw: float = 2.0             # heading-error gain
    rotate_threshold: float = 1.0  # |heading err| above this -> rotate in place (rad)
    scan_timeout: float = 1.0      # stop if newest scan is older than this (s)
    unknown_is_lethal: bool = True # never command the dog into unobserved space
    unknown_cost: float = 20.0     # soft cost of unknown cells if not lethal
    smooth_passes: int = 1
    min_range: float = 0.45        # drop returns this close to the robot (self-hits);
                                   # the Go2's L1 sees its own body at < ~0.4 m
    cloud_buffer: float = 2.0      # accumulate scans over this window (s); the L1
                                   # publishes sparse ~1.4k-pt sweeps, so a single
                                   # message under-observes the surroundings. 0 = off


class RealtimeNavigator:
    """Rolling-window mapper + replanner + path follower."""

    def __init__(self, config: Optional[NavConfig] = None,
                 robot: Optional[QuadrupedParams] = None):
        self.cfg = config or NavConfig()
        self.robot = robot or QuadrupedParams()
        self.goal: Optional[Tuple[float, float]] = None
        self.grid = None
        self.path_world: List[Tuple[float, float]] = []
        self._last_scan_time: float = -1e18
        self._scan_buffer: List[Tuple[float, np.ndarray]] = []   # (stamp, points)
        self.status = "waiting for goal"

    # ------------------------------------------------------------- inputs
    def set_goal(self, x: float, y: float):
        """Set the navigation target in the odom/world frame."""
        self.goal = (float(x), float(y))
        self.status = "goal set"

    def update_cloud(self, points_odom: np.ndarray, robot_xy: Tuple[float, float],
                     stamp: Optional[float] = None):
        """Ingest a new scan (already in the odom/world frame) and rebuild the
        robot-centric local grid. The robot's own position is the map center.

        Self-hits within `min_range` (XY) of the robot are dropped, and recent
        scans within `cloud_buffer` seconds are accumulated so sparse per-sweep
        lidars (Go2 L1: ~1.4k pts/sweep) still build a dense grid."""
        rx, ry = robot_xy
        stamp = time.monotonic() if stamp is None else stamp

        pts = np.asarray(points_odom, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] > 0 and self.cfg.min_range > 0.0:
            d2 = (pts[:, 0] - rx) ** 2 + (pts[:, 1] - ry) ** 2
            pts = pts[d2 > self.cfg.min_range ** 2]

        if self.cfg.cloud_buffer > 0.0:
            self._scan_buffer.append((stamp, pts))
            cutoff = stamp - self.cfg.cloud_buffer
            self._scan_buffer = [(t, p) for t, p in self._scan_buffer if t >= cutoff]
            merged = [p for _, p in self._scan_buffer if p.shape[0] > 0]
            if not merged:
                return self
            points_odom = np.concatenate(merged, axis=0)
        else:
            points_odom = pts

        half = self.cfg.map_size / 2.0
        cfg = GridConfig(
            resolution=self.cfg.resolution,
            bounds=(rx - half, ry - half, rx + half, ry + half),
            z_clip=(self.cfg.z_min, self.cfg.z_max),
            unknown_cost=self.cfg.unknown_cost,
            smooth_passes=self.cfg.smooth_passes,
            inflate_obstacles=True,       # we plan directly on this grid
        )
        try:
            grid = discretize(points_odom, cfg, self.robot)
            if self.cfg.unknown_is_lethal:
                # conservative onboard policy: unobserved space is untraversable.
                # The replan() fallback still makes progress by planning to the
                # frontier of observed space (rolling window reveals more).
                from lidar_pathplan.elevation_grid import UNKNOWN
                grid.cost[grid.classes == UNKNOWN] = np.inf
            self.grid = grid
            self._last_scan_time = stamp
        except ValueError:
            # empty / degenerate scan: keep the previous grid, timeout handles staleness
            pass

    # ------------------------------------------------------------- planning
    def _project_goal(self, robot_xy) -> Tuple[float, float]:
        """Goal clamped into the local window (toward it if it lies beyond)."""
        gx, gy = self.goal
        rx, ry = robot_xy
        half = self.cfg.map_size / 2.0 - 3.0 * self.cfg.resolution  # margin
        dx, dy = gx - rx, gy - ry
        scale = max(abs(dx) / half if half else 1.0, abs(dy) / half if half else 1.0)
        if scale > 1.0:
            dx, dy = dx / scale, dy / scale
        return rx + dx, ry + dy

    def _snap_free(self, cell, max_r=25):
        g = self.grid
        if g.in_bounds(*cell) and np.isfinite(g.cost[cell]):
            return cell
        h, w = g.shape
        for r in range(1, max_r):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr, cc = cell[0] + dr, cell[1] + dc
                    if 0 <= rr < h and 0 <= cc < w and np.isfinite(g.cost[rr, cc]):
                        return (rr, cc)
        return None

    def _reachable_from(self, start):
        """BFS over finite-cost cells; returns the boolean reachable mask."""
        g = self.grid
        h, w = g.shape
        free = np.isfinite(g.cost)
        seen = np.zeros((h, w), dtype=bool)
        seen[start] = True
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        return seen

    def replan(self, robot_xy) -> bool:
        """A* from the robot's current cell toward the (projected) goal.

        If the goal cell is unreachable in the current local map -- e.g. the
        nearest finite cell is an enclosed unknown pocket inside an obstacle --
        the plan targets the REACHABLE cell closest to the goal instead, so the
        robot keeps making progress and the rolling window reveals more map.
        Returns True if a path was produced."""
        if self.grid is None or self.goal is None:
            self.path_world = []
            return False
        g = self.grid
        start = self._snap_free(g.world_to_cell(*robot_xy))
        if start is None:
            self.path_world = []
            self.status = "no free space near robot"
            return False

        proj = self._project_goal(robot_xy)
        target = self._snap_free(g.world_to_cell(*proj))

        reachable = self._reachable_from(start)
        if target is None or not reachable[target]:
            # fall back: reachable cell nearest (euclidean) to the projected goal
            rows, cols = np.nonzero(reachable)
            if rows.size == 0:
                self.path_world = []
                self.status = "enclosed - no reachable space"
                return False
            xs = g.origin[0] + (cols + 0.5) * g.resolution
            ys = g.origin[1] + (rows + 0.5) * g.resolution
            k = int(np.argmin((xs - proj[0]) ** 2 + (ys - proj[1]) ** 2))
            target = (int(rows[k]), int(cols[k]))

        if target == start:
            self.path_world = [g.cell_to_world(*start)]
            self.status = "no progress possible from here"
            return False

        path = astar(g.cost, start, target)
        if path is None:                      # shouldn't happen: target is reachable
            self.path_world = []
            self.status = "no path"
            return False
        self.path_world = [g.cell_to_world(r, c) for r, c in path]
        self.status = "following path"
        return True

    # ------------------------------------------------------------- control
    def _carrot(self, robot_xy) -> Optional[Tuple[float, float]]:
        """Point on the path ~lookahead ahead of the robot."""
        if not self.path_world:
            return None
        pts = self.path_world
        # nearest path point, then advance by lookahead
        d = [math.hypot(px - robot_xy[0], py - robot_xy[1]) for px, py in pts]
        i = int(np.argmin(d))
        acc = 0.0
        while i < len(pts) - 1 and acc < self.cfg.lookahead:
            acc += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            i += 1
        return pts[i]

    def compute_cmd(self, pose, now: Optional[float] = None):
        """One control tick.

        Args:
            pose: (x, y, yaw) of the robot in the odom/world frame.
            now:  time source override (tests); defaults to time.monotonic().

        Returns:
            (v_x, yaw_rate, status) -- body-frame forward velocity and yaw rate.
        """
        x, y, yaw = pose
        now = time.monotonic() if now is None else now

        if self.goal is None:
            return 0.0, 0.0, "waiting for goal"
        if math.hypot(self.goal[0] - x, self.goal[1] - y) <= self.cfg.goal_tolerance:
            self.status = "goal reached"
            return 0.0, 0.0, self.status
        if now - self._last_scan_time > self.cfg.scan_timeout:
            return 0.0, 0.0, "scan timeout - stopped"
        if not self.path_world:
            return 0.0, 0.0, self.status or "no path"

        carrot = self._carrot((x, y))
        if carrot is None:
            return 0.0, 0.0, "no path"

        desired = math.atan2(carrot[1] - y, carrot[0] - x)
        err = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
        wz = max(-self.cfg.max_yaw_rate, min(self.cfg.max_yaw_rate, self.cfg.k_yaw * err))
        if abs(err) > self.cfg.rotate_threshold:
            return 0.0, wz, "rotating to heading"     # turn in place first
        vx = self.cfg.max_speed * max(0.0, math.cos(err))
        return vx, wz, "following path"
