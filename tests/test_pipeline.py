"""Tests for the lidar discretization + planning pipeline.

Runs under pytest, or standalone:  python3 tests/test_pipeline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_pathplan import (GridConfig, QuadrupedParams, discretize, astar,
                            cost_to_occupancy)
from lidar_pathplan.elevation_grid import FREE, LETHAL, UNKNOWN, LETHAL_COST
from lidar_pathplan.astar import path_length
from lidar_pathplan.synthetic import make_scene
from lidar_pathplan.io_utils import load_point_cloud


def _flat_ground(n=20000, size=5.0, noise=0.005, seed=1, exclude=None):
    """Flat ground point cloud. `exclude(x, y) -> bool` masks out a footprint
    (so we don't place ground returns under a solid object the lidar can't see
    through)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)
    if exclude is not None:
        keep = ~exclude(x, y)
        x, y = x[keep], y[keep]
    z = rng.normal(0, noise, x.shape[0])
    return np.column_stack([x, y, z])


def test_flat_ground_is_traversable():
    """Flat ground should produce no lethal cells and a uniform low cost."""
    grid = discretize(_flat_ground(), GridConfig(resolution=0.1))
    assert grid.meta["lethal_cells"] == 0
    free_cost = grid.cost[grid.classes == FREE]
    assert free_cost.size > 0
    assert free_cost.max() < 2.0  # near-flat -> near-base cost on observed ground


def test_wall_becomes_lethal_and_blocks_path():
    """A tall wall across the map must be lethal and split free space."""
    wall_x = lambda x, y: (x >= 2.4) & (x <= 2.6)
    ground = _flat_ground(size=5.0, exclude=wall_x)
    # vertical wall at x in [2.4, 2.6], full y, height 1.0 m
    rng = np.random.default_rng(2)
    n = 5000
    wx = rng.uniform(2.4, 2.6, n)
    wy = rng.uniform(0, 5, n)
    wz = rng.uniform(0, 1.0, n)
    cloud = np.vstack([ground, np.column_stack([wx, wy, wz])])

    grid = discretize(cloud, GridConfig(resolution=0.1))
    assert grid.meta["lethal_cells"] > 0

    # A straight crossing of the wall must be impossible...
    left = grid.world_to_cell(0.5, 2.5)
    right = grid.world_to_cell(4.5, 2.5)
    assert astar(grid.cost, left, right) is None

    # ...but a map with a doorway in the wall should connect. Ground fills the gap.
    doorway_wall = lambda x, y: (x >= 2.4) & (x <= 2.6) & ~((y > 2.0) & (y < 3.0))
    ground2 = _flat_ground(size=5.0, exclude=doorway_wall)
    keep = ~((wy > 2.0) & (wy < 3.0))
    cloud2 = np.vstack([ground2, np.column_stack([wx[keep], wy[keep], wz[keep]])])
    grid2 = discretize(cloud2, GridConfig(resolution=0.1))
    path = astar(grid2.cost, grid2.world_to_cell(0.5, 2.5),
                 grid2.world_to_cell(4.5, 2.5))
    assert path is not None
    assert path_length(path, grid2.resolution) >= 4.0


def test_low_step_is_climbable_high_step_is_not():
    """A 0.10 m curb is traversable for default params; a 0.40 m one is not."""
    def scene_with_step(height):
        # raised platform occupies x in [2.5, 5.0]; no ground returns under it
        platform = lambda x, y: x >= 2.5
        ground = _flat_ground(size=5.0, exclude=platform)
        rng = np.random.default_rng(3)
        n = 8000
        sx = rng.uniform(2.5, 5.0, n)
        sy = rng.uniform(0, 5.0, n)
        sz = np.full(n, height) + rng.normal(0, 0.005, n)
        # include the riser face so the edge is sampled
        fx = rng.uniform(2.45, 2.55, n // 2)
        fy = rng.uniform(0, 5.0, n // 2)
        fz = rng.uniform(0, height, n // 2)
        return np.vstack([ground,
                          np.column_stack([sx, sy, sz]),
                          np.column_stack([fx, fy, fz])])

    robot = QuadrupedParams(max_step=0.18)
    low = discretize(scene_with_step(0.10), GridConfig(resolution=0.1), robot)
    high = discretize(scene_with_step(0.40), GridConfig(resolution=0.1), robot)

    p_low = astar(low.cost, low.world_to_cell(1.0, 2.5), low.world_to_cell(4.5, 2.5))
    p_high = astar(high.cost, high.world_to_cell(1.0, 2.5), high.world_to_cell(4.5, 2.5))
    assert p_low is not None          # can climb the 0.10 m curb
    assert p_high is None             # 0.40 m step blocks (no way around)


def test_coordinate_roundtrip():
    grid = discretize(_flat_ground(), GridConfig(resolution=0.1))
    for (x, y) in [(0.55, 1.05), (2.0, 3.3), (4.9, 0.2)]:
        rc = grid.world_to_cell(x, y)
        wx, wy = grid.cell_to_world(*rc)
        # recovered center must be within half a cell of the input
        assert abs(wx - x) <= grid.resolution
        assert abs(wy - y) <= grid.resolution


def test_astar_rejects_blocked_endpoints():
    grid = discretize(_flat_ground(), GridConfig(resolution=0.1))
    cost = grid.cost.copy()
    cost[5, 5] = LETHAL_COST
    raised = False
    try:
        astar(cost, (5, 5), (10, 10))
    except ValueError:
        raised = True
    assert raised


def test_full_synthetic_demo_finds_path():
    cloud = make_scene()
    grid = discretize(cloud, GridConfig(resolution=0.1))
    assert set(np.unique(grid.classes)).issubset({FREE, LETHAL, UNKNOWN})
    path = astar(grid.cost, grid.world_to_cell(0.5, 0.5),
                 grid.world_to_cell(9.5, 5.0))
    assert path is not None
    assert path[0] == grid.world_to_cell(0.5, 0.5)


def test_occupancy_encoding_for_nav2():
    """cost_to_occupancy maps lethal->100, unknown->-1, free->1..99."""
    cloud = make_scene()
    grid = discretize(cloud, GridConfig(resolution=0.1))
    occ = cost_to_occupancy(grid)

    assert occ.shape == grid.shape
    assert occ.dtype == np.int8
    assert occ.min() >= -1 and occ.max() <= 100
    # class-consistent encoding
    assert np.all(occ[grid.classes == LETHAL] == 100)
    assert np.all(occ[grid.classes == UNKNOWN] == -1)
    free = occ[grid.classes == FREE]
    assert free.size > 0
    assert free.min() >= 1 and free.max() <= 99
    # row-major flatten matches OccupancyGrid data ordering length
    assert len(occ.ravel(order="C")) == grid.shape[0] * grid.shape[1]


def test_inflate_flag_changes_lethal_count():
    """Disabling inflation (for nav2's InflationLayer) yields fewer lethal cells."""
    cloud = make_scene()
    inflated = discretize(cloud, GridConfig(resolution=0.1, inflate_obstacles=True))
    raw = discretize(cloud, GridConfig(resolution=0.1, inflate_obstacles=False))
    assert raw.meta["lethal_cells"] < inflated.meta["lethal_cells"]


def test_ply_loader_ascii_and_binary(tmp_path_factory=None):
    """The .ply loader reads x,y,z from both ascii and binary_little_endian,
    skipping extra per-vertex properties (rgb, normals)."""
    import struct, tempfile
    pts = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [-1.5, 0.0, 4.25]])

    d = tempfile.mkdtemp()
    # --- ascii, with extra rgb columns the loader must skip ---
    ap = os.path.join(d, "a.ply")
    with open(ap, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 3\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z) in pts:
            f.write("%g %g %g 10 20 30\n" % (x, y, z))
    a = load_point_cloud(ap)
    assert np.allclose(a[:, :3], pts, atol=1e-5)

    # --- binary little endian, xyz only ---
    bp = os.path.join(d, "b.ply")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(bp, "wb") as f:
        f.write(header.encode("ascii"))
        for (x, y, z) in pts:
            f.write(struct.pack("<fff", x, y, z))
    b = load_point_cloud(bp)
    assert np.allclose(b[:, :3], pts, atol=1e-5)


def test_smooth_passes_reduces_noisy_lethal():
    """More elevation smoothing reduces spurious lethal cells on noisy ground."""
    rng = np.random.default_rng(7)
    n = 40000
    x = rng.uniform(0, 5, n); y = rng.uniform(0, 5, n)
    z = rng.normal(0, 0.06, n)   # noisy-but-flat (RGBD-like) ground
    cloud = np.column_stack([x, y, z])
    few = discretize(cloud, GridConfig(resolution=0.1, smooth_passes=1))
    many = discretize(cloud, GridConfig(resolution=0.1, smooth_passes=5))
    assert many.meta["lethal_cells"] <= few.meta["lethal_cells"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, "-", e)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("ERROR", fn.__name__, "-", repr(e))
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
