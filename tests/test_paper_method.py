"""Tests for the paper-method stack (Kalman map, Dijkstra, elastic band, APF).

Runs under pytest, or standalone:  python3 tests/test_paper_method.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_pathplan import QuadrupedParams
from lidar_pathplan.synthetic import make_scene
from lidar_pathplan.paper_method import (
    KalmanElevationMap, dijkstra, smooth_path, APFPlanner, APFParams)
from lidar_pathplan.paper_method.kalman_map import KalmanMapConfig
from lidar_pathplan.paper_method.elastic_band import _chamfer_distance


def _flat(n=15000, size=5.0, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, size, n); y = rng.uniform(0, size, n)
    z = rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


def test_kalman_single_cell_fusion_equation():
    """Two equal-variance measurements of one cell fuse to their mean, half var."""
    km = KalmanElevationMap(KalmanMapConfig(
        resolution=1.0, bounds=(0, 0, 2, 2), sensor_var=0.04, range_coeff=0.0))
    # one point in cell (0,0): z = 1.0, then a second scan z = 2.0
    km.update(np.array([[0.5, 0.5, 1.0]]), sensor_origin=(0.5, 0.5, 0.0))
    rc = km.world_to_cell(0.5, 0.5)
    assert abs(km.height[rc] - 1.0) < 1e-9
    assert abs(km.variance[rc] - 0.04) < 1e-9
    km.update(np.array([[0.5, 0.5, 2.0]]), sensor_origin=(0.5, 0.5, 0.0))
    assert abs(km.height[rc] - 1.5) < 1e-9        # mean of 1 and 2
    assert abs(km.variance[rc] - 0.02) < 1e-9     # half the variance


def test_kalman_variance_decreases_with_more_scans():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    rng = np.random.default_rng(0)
    prev = None
    for _ in range(5):
        s = base.copy(); s[:, 2] += rng.normal(0, 0.05, base.shape[0])
        km.update(s, sensor_origin=(5, 5, 1.5))
        obs = np.isfinite(km.height)
        mv = float(np.nanmean(km.variance[obs]))
        if prev is not None:
            assert mv <= prev + 1e-12             # monotonically non-increasing
        prev = mv


def test_kalman_predict_inflates_variance():
    km = KalmanElevationMap(KalmanMapConfig(1.0, (0, 0, 2, 2), sensor_var=0.04, range_coeff=0.0))
    km.update(np.array([[0.5, 0.5, 1.0]]), sensor_origin=(0.5, 0.5, 0.0))
    rc = km.world_to_cell(0.5, 0.5)
    v0 = km.variance[rc]
    km.predict(0.01)
    assert abs(km.variance[rc] - (v0 + 0.01)) < 1e-9


def test_kalman_map_to_grid_plannable():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    rng = np.random.default_rng(0)
    for _ in range(4):
        s = base.copy(); s[:, 2] += rng.normal(0, 0.04, base.shape[0])
        km.update(s, (5, 5, 1.5))
    g = km.to_grid(QuadrupedParams())
    assert g.meta["lethal_cells"] > 0            # the walls
    assert g.meta["observed_cells"] > 1000


def test_dijkstra_finds_path_and_rejects_blocked():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    km.update(base, (5, 5, 1.5))
    g = km.to_grid(QuadrupedParams())
    s = g.world_to_cell(0.5, 0.5); go = g.world_to_cell(9.5, 5.0)
    path = dijkstra(g.cost, s, go)
    assert path is not None and path[0] == s and path[-1] == go

    blocked = g.cost.copy(); blocked[s] = np.inf
    raised = False
    try:
        dijkstra(blocked, s, go)
    except ValueError:
        raised = True
    assert raised


def test_dijkstra_matches_astar_cost():
    """Dijkstra and A* (admissible heuristic) must return equal-cost paths."""
    from lidar_pathplan import astar
    cloud = make_scene(seed=2)
    from lidar_pathplan import discretize, GridConfig
    g = discretize(cloud, GridConfig(resolution=0.1))
    s = g.world_to_cell(0.5, 0.5); go = g.world_to_cell(9.5, 5.0)

    def cost_of(path):
        import math
        t = 0.0
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            d = math.hypot(r1 - r0, c1 - c0)
            t += d * 0.5 * (g.cost[r0, c0] + g.cost[r1, c1])
        return t

    pd = dijkstra(g.cost, s, go)
    pa = astar(g.cost, s, go)
    assert pd is not None and pa is not None
    assert abs(cost_of(pd) - cost_of(pa)) < 1e-6


def test_chamfer_distance_basic():
    obs = np.zeros((5, 5), dtype=bool)
    obs[2, 2] = True
    d = _chamfer_distance(obs)
    assert d[2, 2] == 0.0
    assert abs(d[2, 0] - 2.0) < 1e-9              # two cells away horizontally
    assert d[0, 0] > 0


def test_smooth_path_keeps_endpoints_and_clearance():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    km.update(base, (5, 5, 1.5))
    g = km.to_grid(QuadrupedParams())
    s = g.world_to_cell(0.5, 0.5); go = g.world_to_cell(9.5, 5.0)
    raw = dijkstra(g.cost, s, go)
    sm = smooth_path(raw, g)
    assert len(sm) == len(raw)
    # endpoints preserved
    assert np.allclose(sm[0], g.cell_to_world(*raw[0]))
    assert np.allclose(sm[-1], g.cell_to_world(*raw[-1]))
    # every smoothed waypoint lands in free space
    for x, y in sm:
        assert np.isfinite(g.cost[g.world_to_cell(x, y)])


def test_apf_force_directions():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    km.update(base, (5, 5, 1.5))
    g = km.to_grid(QuadrupedParams())
    apf = APFPlanner(g, APFParams())
    # In open space far from obstacles, force points toward the goal.
    pos = np.array([1.0, 1.0]); goal = np.array([2.0, 1.0])
    f = apf.force(pos, goal)
    assert f[0] > 0 and abs(f[1]) < abs(f[0])     # mostly +x toward goal


def test_apf_follow_path_reaches_goal():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    km.update(base, (5, 5, 1.5))
    g = km.to_grid(QuadrupedParams())
    s = g.world_to_cell(0.5, 0.5); go = g.world_to_cell(9.5, 5.0)
    sm = smooth_path(dijkstra(g.cost, s, go), g)
    apf = APFPlanner(g, APFParams())
    traj = apf.follow_path(sm)
    end = np.array(traj[-1]); goal = np.array(sm[-1])
    assert np.linalg.norm(end - goal) <= 0.3      # reached the path end


def test_apf_tracks_moving_goal():
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(0.1, (0, 0, 10, 10), z_clip=(-1, 2)))
    km.update(base, (5, 5, 1.5))
    g = km.to_grid(QuadrupedParams())
    apf = APFPlanner(g, APFParams(max_speed=0.9))
    # person walks slowly along open ground near y=8 (above the walls)
    def person(t):
        return (min(1.0 + 0.4 * t, 9.0), 8.2)
    traj = apf.run((0.6, 8.2), person, max_steps=400)
    gap = np.linalg.norm(np.array(traj[-1]) - np.array(person(len(traj) * apf.params.dt)))
    assert gap < 1.0                              # stays close to the person


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
