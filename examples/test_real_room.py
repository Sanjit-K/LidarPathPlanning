#!/usr/bin/env python3
"""Test the pipeline on a real indoor RGBD-fused scan (Open3D room fragment).

This is a genuinely different dataset from KITTI: an indoor scene (floor, walls,
furniture) reconstructed from RGBD, ~196k points. It exercises two things the
automotive scan doesn't:

  * Arbitrary up-axis. RGBD clouds are not gravity-aligned the way a Velodyne is.
    We auto-detect the vertical axis (the one whose height histogram has the
    sharpest peak -- the flat floor) and remap so the floor is at height 0 with
    the other two axes forming the horizontal ground plane.
  * Higher, less structured noise than lidar, which stresses the roughness /
    slope thresholds.

The use case is an *indoor quadruped*: floor = traversable, furniture/walls =
obstacles to route around.

Usage:
    python3 examples/test_real_room.py
    python3 examples/test_real_room.py --ply path/to/scan.ply --res 0.05
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, astar
from lidar_pathplan.astar import path_length
from lidar_pathplan.elevation_grid import UNKNOWN
from lidar_pathplan.io_utils import load_point_cloud

SAMPLE_URL = ("https://github.com/isl-org/open3d_downloads/releases/download/"
              "20220201-data/fragment.ply")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch_sample() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, "room_fragment.ply")
    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        print("using cached", dest)
        return dest
    print("downloading", SAMPLE_URL)
    urllib.request.urlretrieve(SAMPLE_URL, dest)
    print("saved", dest)
    return dest


def detect_up_axis(pts: np.ndarray) -> int:
    """Return the index (0/1/2) of the vertical axis.

    The floor is a large flat plane, so the vertical axis is the one whose
    coordinate histogram concentrates the most points in a single bin.
    """
    best, best_frac = 2, -1.0
    for i in range(3):
        h, _ = np.histogram(pts[:, i], bins=50)
        frac = h.max() / len(pts)
        if frac > best_frac:
            best, best_frac = i, frac
    return best


def reorient(pts: np.ndarray, up: int):
    """Remap so columns are (ground_a, ground_b, height) with floor near height 0
    and height increasing away from the floor."""
    horiz = [i for i in range(3) if i != up]
    height = pts[:, up]
    # Floor is the densest height; put it at 0 and make "up" positive.
    h, edges = np.histogram(height, bins=50)
    floor = 0.5 * (edges[h.argmax()] + edges[h.argmax() + 1])
    sign = 1.0 if (height.max() - floor) >= (floor - height.min()) else -1.0
    z = sign * (height - floor)
    return np.column_stack([pts[:, horiz[0]], pts[:, horiz[1]], z]), floor


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply", help="PLY scan; omit to auto-download the room sample")
    # Defaults are tuned for noisy single-view RGBD reconstruction, NOT clean
    # lidar: the floor here has ~0.15-0.2 m of depth scatter per cell at grazing
    # angle, so the roughness/slope tolerances and smoothing are looser than the
    # lidar defaults in QuadrupedParams/GridConfig. See the printout / README.
    p.add_argument("--res", type=float, default=0.08, help="grid resolution (m)")
    p.add_argument("--radius", type=float, default=0.15, help="robot footprint radius (m)")
    p.add_argument("--max-step", type=float, default=0.15, help="robot max step (m)")
    p.add_argument("--max-roughness", type=float, default=0.22, help="max within-cell std (m)")
    p.add_argument("--max-slope", type=float, default=45.0, help="robot max slope (deg)")
    p.add_argument("--smooth-passes", type=int, default=4, help="elevation smoothing passes")
    p.add_argument("--out", default="out_room.png")
    args = p.parse_args(argv)

    path_ply = args.ply or fetch_sample()
    raw = load_point_cloud(path_ply)
    up = detect_up_axis(raw)
    cloud, floor = reorient(raw, up)
    print("loaded %d pts; up-axis=%s, floor at %.2f" % (raw.shape[0], "xyz"[up], floor))
    print("  ground x[%.2f,%.2f] y[%.2f,%.2f] height[%.2f,%.2f]" % (
        cloud[:, 0].min(), cloud[:, 0].max(),
        cloud[:, 1].min(), cloud[:, 1].max(),
        cloud[:, 2].min(), cloud[:, 2].max()))

    robot = QuadrupedParams(radius=args.radius, max_step=args.max_step,
                            max_roughness=args.max_roughness, max_slope_deg=args.max_slope)
    # Keep points from floor up to ~1.5 m; drop ceiling so it isn't a giant obstacle
    # roof (we care about what's on the floor plane).
    config = GridConfig(resolution=args.res, z_clip=(-0.10, 1.5),
                        smooth_passes=args.smooth_passes)
    grid = discretize(cloud, config=config, robot=robot)
    print("grid %dx%d @ %.2fm  observed=%d lethal=%d unknown=%d" % (
        grid.shape[0], grid.shape[1], grid.resolution,
        grid.meta["observed_cells"], grid.meta["lethal_cells"],
        grid.meta["unknown_cells"]))

    free_area = grid.meta["observed_cells"] - grid.meta["lethal_cells"]
    print("  traversable floor: %d cells (%.2f m^2)" % (
        free_area, free_area * grid.resolution ** 2))

    start, goal = _pick_endpoints(grid)
    path = astar(grid.cost, start, goal)
    if path is None:
        print("NO PATH between", start, goal)
    else:
        print("path: %d cells, %.2f m" % (len(path), path_length(path, grid.resolution)))

    _plot(grid, cloud, path, start, goal, args.out)
    print("wrote", args.out)
    return 0


def _pick_endpoints(grid):
    """Pick two free cells far apart (extreme corners of the free region)."""
    free = np.isfinite(grid.cost)
    rc = np.argwhere(free)
    if rc.shape[0] < 2:
        raise ValueError("not enough free space to plan")
    # farthest-apart pair along the dominant spread direction
    key = rc[:, 1] + rc[:, 0]
    start = tuple(rc[key.argmin()])
    goal = tuple(rc[key.argmax()])
    return start, goal


def _plot(grid, cloud, path, start, goal, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    extent = [grid.origin[0], grid.origin[0] + grid.shape[1] * grid.resolution,
              grid.origin[1], grid.origin[1] + grid.shape[0] * grid.resolution]

    ax = axes[0]
    sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], s=0.5, cmap="viridis")
    ax.set_title("Room scan (color = height above floor)")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="height (m)")

    ax = axes[1]
    im = ax.imshow(grid.elevation, origin="lower", extent=extent, cmap="terrain")
    ax.set_title("2.5D elevation grid")
    fig.colorbar(im, ax=ax, fraction=0.046, label="height (m)")

    ax = axes[2]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # classes: FREE=0 -> green, LETHAL=1 -> dark red, UNKNOWN=2 -> light grey
    cmap = ListedColormap(["#cfe8c4", "#7d2828", "#e8e8e8"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(grid.classes, origin="lower", extent=extent, cmap=cmap, norm=norm)
    if path:
        xs = [grid.cell_to_world(r, c)[0] for r, c in path]
        ys = [grid.cell_to_world(r, c)[1] for r, c in path]
        ax.plot(xs, ys, "-", color="dodgerblue", lw=2, label="path")
    ax.plot(*grid.cell_to_world(*start), "o", color="lime", ms=9, label="start")
    ax.plot(*grid.cell_to_world(*goal), "*", color="red", ms=14, label="goal")
    ax.set_title("Classification + A* path\n(green=free, red=obstacle, grey=unknown)")
    ax.legend(loc="upper right")

    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    sys.exit(main())
