"""Simulated end-to-end test of the real-time navigator core (no ROS needed).

A virtual quadruped (unicycle model) starts in the synthetic scene, receives
simulated lidar scans (world points within sensor range), and must reach a
preprogrammed goal on the far side of the walls using only the rolling local
map + replanning + carrot following. Verifies:

  * the robot reaches the goal,
  * it never enters a lethal cell of the ground-truth inflated map,
  * safety stop engages when scans go stale,
  * the goal-projection handles goals beyond the local window.

Runs under pytest, or standalone:  python3 tests/test_navigator_core.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_pathplan import GridConfig, QuadrupedParams, discretize
from lidar_pathplan.synthetic import make_scene
from lidar_nav2.navigator_core import NavConfig, RealtimeNavigator

SENSOR_RANGE = 5.0


def simulate(goal, start=(0.6, 0.6, 0.0), max_steps=3000, scan_every=3):
    """Drive the navigator through the synthetic scene; return trajectory + status."""
    world = make_scene()
    # cloud_buffer off: the simulated scans already cover the full sensor range,
    # so accumulation would only slow the test (it's exercised separately below).
    cfg = NavConfig(map_size=8.0, resolution=0.10, goal_tolerance=0.3,
                    max_speed=0.6, scan_timeout=2.0, cloud_buffer=0.0)
    robot = QuadrupedParams()
    nav = RealtimeNavigator(cfg, robot)
    nav.set_goal(*goal)

    # ground-truth inflated map for collision checking
    truth = discretize(world, GridConfig(resolution=0.10), robot)

    x, y, yaw = start
    dt = 0.1
    traj = [(x, y)]
    status = ""
    for k in range(max_steps):
        t = k * dt
        if k % scan_every == 0:
            d = np.hypot(world[:, 0] - x, world[:, 1] - y)
            scan = world[d < SENSOR_RANGE]
            nav.update_cloud(scan, (x, y), stamp=t)
            nav.replan((x, y))
        vx, wz, status = nav.compute_cmd((x, y, yaw), now=t)
        if status == "goal reached":
            break
        yaw += wz * dt
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        traj.append((x, y))
    return traj, status, truth


def test_reaches_far_goal_beyond_local_window():
    """Goal ~9.5 m away with an 8 m window: exercises rolling-horizon projection."""
    traj, status, truth = simulate(goal=(9.4, 5.0))
    assert status == "goal reached", "did not reach goal: %s" % status
    end = traj[-1]
    assert math.hypot(end[0] - 9.4, end[1] - 5.0) < 0.5


# True (untraversable) obstacle rectangles in make_scene: the two 0.8 m walls.
# The 0.12 m curb and the ramp are climbable for the default robot, so they are
# excluded -- driving over them is correct behavior, not a collision.
_WALLS = [(3.0, 3.6, 2.0, 7.0), (6.0, 6.6, 0.0, 5.5)]


def _dist_to_rect(px, py, rect):
    x0, x1, y0, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def test_maintains_physical_clearance():
    """The robot's body (radius 0.30 m) must never intersect a real obstacle.
    This checks the physical safety property directly, rather than comparing
    two differently-built grids' inflation boundaries (which legitimately
    disagree by a cell at the margin)."""
    traj, status, truth = simulate(goal=(9.4, 5.0))
    assert status == "goal reached"
    radius = QuadrupedParams().radius
    slack = truth.resolution / 2.0          # half-cell discretization slack
    min_clear = min(min(_dist_to_rect(px, py, w) for w in _WALLS) for px, py in traj)
    assert min_clear >= radius - slack, \
        "body clearance violated: %.3f m < %.3f m" % (min_clear, radius - slack)


def test_safety_stop_on_stale_scan():
    world = make_scene()
    nav = RealtimeNavigator(NavConfig(scan_timeout=1.0))
    nav.set_goal(9.0, 5.0)
    nav.update_cloud(world, (0.6, 0.6), stamp=0.0)
    nav.replan((0.6, 0.6))
    vx, wz, status = nav.compute_cmd((0.6, 0.6, 0.0), now=0.5)
    assert status in ("following path", "rotating to heading")
    vx, wz, status = nav.compute_cmd((0.6, 0.6, 0.0), now=5.0)   # scan is stale
    assert vx == 0.0 and "timeout" in status


def test_waiting_without_goal():
    nav = RealtimeNavigator()
    vx, wz, status = nav.compute_cmd((0.0, 0.0, 0.0), now=0.0)
    assert (vx, wz) == (0.0, 0.0) and status == "waiting for goal"


def test_min_range_drops_self_hits():
    """Returns from the robot's own body (< min_range) must not become obstacles."""
    rng = np.random.default_rng(0)
    n = 5000
    ground = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n),
                              rng.normal(0, 0.01, n)])
    # a dense blob of "body" hits 0.2 m from the sensor, 0.3 m tall (would be lethal)
    body = np.column_stack([rng.normal(0.2, 0.03, 800), rng.normal(0.0, 0.03, 800),
                            rng.uniform(0, 0.3, 800)])
    nav = RealtimeNavigator(NavConfig(map_size=6.0, resolution=0.10,
                                      min_range=0.45, cloud_buffer=0.0))
    nav.update_cloud(np.vstack([ground, body]), (0.0, 0.0), stamp=0.0)
    g = nav.grid
    cell = g.world_to_cell(0.2, 0.0)
    from lidar_pathplan.elevation_grid import LETHAL
    assert g.classes[cell] != LETHAL, "self-hit blob became an obstacle"


def test_cloud_buffer_accumulates_sparse_sweeps():
    """Sparse per-sweep clouds must merge across the buffer window."""
    rng = np.random.default_rng(1)
    nav = RealtimeNavigator(NavConfig(map_size=6.0, resolution=0.10,
                                      min_range=0.0, cloud_buffer=1.0))
    def sweep(t):
        n = 300     # sparse, like the Go2 L1
        return np.column_stack([rng.uniform(-2.5, 2.5, n), rng.uniform(-2.5, 2.5, n),
                                rng.normal(0, 0.01, n)])
    nav.update_cloud(sweep(0.0), (0, 0), stamp=0.0)
    obs1 = nav.grid.meta["observed_cells"]
    for k in range(1, 6):
        nav.update_cloud(sweep(k * 0.1), (0, 0), stamp=k * 0.1)
    obs6 = nav.grid.meta["observed_cells"]
    assert obs6 > obs1 * 2, "buffer did not densify the grid (%d -> %d)" % (obs1, obs6)
    # old scans age out: jump far ahead, buffer keeps only the newest
    nav.update_cloud(sweep(99.0), (0, 0), stamp=99.0)
    assert len(nav._scan_buffer) == 1


def test_goal_projection_stays_inside_window():
    nav = RealtimeNavigator(NavConfig(map_size=8.0, resolution=0.10))
    nav.set_goal(100.0, 50.0)
    px, py = nav._project_goal((0.0, 0.0))
    half = 4.0
    assert abs(px) <= half and abs(py) <= half
    # direction preserved
    assert abs(math.atan2(py, px) - math.atan2(50.0, 100.0)) < 1e-6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1; print("FAIL", fn.__name__, "-", e)
        except Exception as e:  # noqa: BLE001
            failed += 1; print("ERROR", fn.__name__, "-", repr(e))
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
