#!/usr/bin/env python3
"""Interactive path-planning demo: click to set start and goal.

Opens a window showing the traversability map. Click once to drop the START,
click again to drop the GOAL --- the A* path is planned and drawn instantly.
Click a third time (or press 'r') to reset and pick a new pair. Clicks on an
obstacle snap to the nearest free cell.

    python3 examples/interactive_plan.py                 # synthetic scene
    python3 examples/interactive_plan.py --cloud scan.npy # your own point cloud
    python3 examples/interactive_plan.py --cloud apt_sub.npy --rgbd

Requires an interactive matplotlib backend (default on a desktop). It is NOT
forced to Agg so a window can open.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, astar
from lidar_pathplan.astar import path_length
from lidar_pathplan.io_utils import load_point_cloud
from lidar_pathplan.synthetic import make_scene


def build_grid(args):
    """Return an ElevationGrid from the synthetic scene or a given cloud."""
    if args.cloud:
        cloud = load_point_cloud(args.cloud)
        if args.rgbd:
            # indoor RGB-D: auto-orient to gravity and use looser thresholds
            from test_real_room import detect_up_axis, reorient
            up = detect_up_axis(cloud)
            cloud, _ = reorient(cloud, up)
            robot = QuadrupedParams(radius=0.15, max_step=0.15,
                                    max_roughness=0.22, max_slope_deg=45)
            cfg = GridConfig(resolution=args.res, z_clip=(-0.10, 1.8), smooth_passes=4)
        else:
            robot = QuadrupedParams()
            cfg = GridConfig(resolution=args.res)
        print("loaded %d points from %s" % (cloud.shape[0], args.cloud))
    else:
        cloud = make_scene()
        robot = QuadrupedParams()
        cfg = GridConfig(resolution=args.res)
        print("synthetic scene: %d points" % cloud.shape[0])
    grid = discretize(cloud, cfg, robot)
    print("grid %dx%d @ %.2fm  free=%d lethal=%d unknown=%d" % (
        grid.shape[0], grid.shape[1], grid.resolution,
        grid.meta["observed_cells"] - grid.meta["lethal_cells"],
        grid.meta["lethal_cells"], grid.meta["unknown_cells"]))
    return grid


def snap_to_free(grid, cell, max_r=40):
    """Nearest free (finite-cost) cell to `cell`, or None if none within max_r."""
    if grid.in_bounds(*cell) and np.isfinite(grid.cost[cell]):
        return cell
    h, w = grid.shape
    for r in range(1, max_r):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = cell[0] + dr, cell[1] + dc
                if 0 <= rr < h and 0 <= cc < w and np.isfinite(grid.cost[rr, cc]):
                    return (rr, cc)
    return None


class InteractivePlanner:
    def __init__(self, grid):
        self.grid = grid
        self.start = None
        self.goal = None
        self.artists = []
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self._draw_map()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._set_title("Click to set START")

    def _draw_map(self):
        g = self.grid
        ext = [g.origin[0], g.origin[0] + g.shape[1] * g.resolution,
               g.origin[1], g.origin[1] + g.shape[0] * g.resolution]
        cmap = ListedColormap(["#cfe8c4", "#7d2828", "#e8e8e8"])
        self.ax.imshow(g.classes, origin="lower", extent=ext, cmap=cmap,
                       norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N))
        self.ax.set_xlabel("x (m)"); self.ax.set_ylabel("y (m)")
        self.ax.set_aspect("equal")
        self._extent = ext

    def _set_title(self, msg):
        self.ax.set_title("%s   (green=free, red=obstacle, grey=unknown; 'r'=reset)" % msg)
        self.fig.canvas.draw_idle()

    def _clear_artists(self):
        for a in self.artists:
            a.remove()
        self.artists = []

    def on_key(self, event):
        if event.key in ("r", "escape"):
            self.reset()

    def reset(self):
        self.start = self.goal = None
        self._clear_artists()
        self._set_title("Click to set START")

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        cell = snap_to_free(self.grid, self.grid.world_to_cell(event.xdata, event.ydata))
        if cell is None:
            self._set_title("No free cell there --- try again")
            return
        if self.start is not None and self.goal is not None:
            self.reset()
        if self.start is None:
            self.start = cell
            x, y = self.grid.cell_to_world(*cell)
            self.artists += self.ax.plot(x, y, "o", color="lime", ms=11,
                                         markeredgecolor="black", zorder=5)
            self._set_title("Click to set GOAL")
        else:
            self.goal = cell
            x, y = self.grid.cell_to_world(*cell)
            self.artists += self.ax.plot(x, y, "*", color="red", ms=16,
                                         markeredgecolor="black", zorder=5)
            self.plan()

    def plan(self):
        path = astar(self.grid.cost, self.start, self.goal)
        if path is None:
            self._set_title("No path found (start/goal in separate regions)")
            return
        xs = [self.grid.cell_to_world(r, c)[0] for r, c in path]
        ys = [self.grid.cell_to_world(r, c)[1] for r, c in path]
        self.artists += self.ax.plot(xs, ys, "-", color="dodgerblue", lw=2.5, zorder=4)
        self._set_title("Path: %.2f m   ---   click to plan another" %
                        path_length(path, self.grid.resolution))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cloud", help="point cloud file (.npy/.bin/.ply/.pcd); omit for synthetic")
    p.add_argument("--rgbd", action="store_true", help="treat cloud as indoor RGB-D (auto-orient)")
    p.add_argument("--res", type=float, default=0.10, help="grid resolution (m)")
    args = p.parse_args()
    grid = build_grid(args)
    InteractivePlanner(grid)
    print("\nClick START, then GOAL. Press 'r' to reset. Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
