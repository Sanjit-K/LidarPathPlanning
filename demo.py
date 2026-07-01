#!/usr/bin/env python3
"""End-to-end demo: lidar point cloud -> 2.5D discretization -> A* path.

Run with no arguments to use a built-in synthetic scene:

    python3 demo.py

Or point it at your own cloud (.npy/.bin/.pcd/.xyz):

    python3 demo.py --cloud scan.npy --res 0.1 --start 0.5,0.5 --goal 9.5,9.5

Writes a figure to out.png (and shows it if a display is available).
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe default; demo saves a PNG. Comment out for interactive.
import matplotlib.pyplot as plt

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, astar
from lidar_pathplan.astar import path_length
from lidar_pathplan.elevation_grid import LETHAL, UNKNOWN
from lidar_pathplan.synthetic import make_scene
from lidar_pathplan.io_utils import load_point_cloud


def parse_xy(s):
    x, y = s.split(",")
    return float(x), float(y)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cloud", help="point cloud file; omit to use synthetic scene")
    p.add_argument("--res", type=float, default=0.10, help="grid resolution (m)")
    p.add_argument("--start", type=parse_xy, default=(0.5, 0.5), help="x,y meters")
    p.add_argument("--goal", type=parse_xy, default=(9.5, 5.0), help="x,y meters")
    p.add_argument("--max-step", type=float, default=0.18, help="robot max step (m)")
    p.add_argument("--max-slope", type=float, default=30.0, help="robot max slope (deg)")
    p.add_argument("--out", default="out.png", help="output figure path")
    args = p.parse_args(argv)

    if args.cloud:
        cloud = load_point_cloud(args.cloud)
        print("loaded %d points from %s" % (cloud.shape[0], args.cloud))
    else:
        cloud = make_scene()
        print("generated synthetic scene: %d points" % cloud.shape[0])

    robot = QuadrupedParams(max_step=args.max_step, max_slope_deg=args.max_slope)
    config = GridConfig(resolution=args.res)
    grid = discretize(cloud, config=config, robot=robot)

    print("grid: %d x %d cells @ %.2f m" % (grid.shape[0], grid.shape[1], grid.resolution))
    print("  observed=%d lethal=%d unknown=%d" % (
        grid.meta["observed_cells"], grid.meta["lethal_cells"], grid.meta["unknown_cells"]))

    start_cell = grid.world_to_cell(*args.start)
    goal_cell = grid.world_to_cell(*args.goal)
    start_cell = _nudge_to_free(grid, start_cell)
    goal_cell = _nudge_to_free(grid, goal_cell)

    path = astar(grid.cost, start_cell, goal_cell)
    if path is None:
        print("NO PATH FOUND between %s and %s" % (start_cell, goal_cell))
    else:
        print("path: %d cells, %.2f m" % (len(path), path_length(path, grid.resolution)))

    _plot(grid, cloud, path, start_cell, goal_cell, args.out)
    print("wrote %s" % args.out)
    return 0


def _nudge_to_free(grid, cell, max_r=20):
    """If a requested cell is blocked, snap to the nearest finite-cost cell."""
    if np.isfinite(grid.cost[cell]):
        return cell
    h, w = grid.shape
    for r in range(1, max_r):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = cell[0] + dr, cell[1] + dc
                if 0 <= rr < h and 0 <= cc < w and np.isfinite(grid.cost[rr, cc]):
                    return (rr, cc)
    raise ValueError("no free cell near %s" % (cell,))


def _plot(grid, cloud, path, start_cell, goal_cell, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    extent = [grid.origin[0], grid.origin[0] + grid.shape[1] * grid.resolution,
              grid.origin[1], grid.origin[1] + grid.shape[0] * grid.resolution]

    # 1. raw point cloud colored by height
    ax = axes[0]
    sc = ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], s=1, cmap="viridis")
    ax.set_title("Lidar point cloud (color = height)")
    fig.colorbar(sc, ax=ax, label="z (m)", fraction=0.046)

    # 2. elevation grid
    ax = axes[1]
    im = ax.imshow(grid.elevation, origin="lower", extent=extent, cmap="terrain")
    ax.set_title("2.5D elevation grid")
    fig.colorbar(im, ax=ax, label="height (m)", fraction=0.046)

    # 3. traversability cost + path
    ax = axes[2]
    disp = grid.cost.copy()
    finite_max = np.nanmax(disp[np.isfinite(disp)]) if np.isfinite(disp).any() else 1.0
    disp[~np.isfinite(disp)] = finite_max * 1.5  # show lethal as the top of the scale
    ax.imshow(disp, origin="lower", extent=extent, cmap="magma_r")
    ax.imshow(np.where(grid.classes == UNKNOWN, 1.0, np.nan),
              origin="lower", extent=extent, cmap="Greys", alpha=0.25, vmin=0, vmax=1)
    if path:
        xs = [grid.cell_to_world(r, c)[0] for r, c in path]
        ys = [grid.cell_to_world(r, c)[1] for r, c in path]
        ax.plot(xs, ys, "-", color="cyan", linewidth=2, label="path")
    sx, sy = grid.cell_to_world(*start_cell)
    gx, gy = grid.cell_to_world(*goal_cell)
    ax.plot(sx, sy, "o", color="lime", markersize=10, label="start")
    ax.plot(gx, gy, "*", color="red", markersize=15, label="goal")
    ax.set_title("Traversability cost + A* path")
    ax.legend(loc="upper left")

    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    sys.exit(main())
