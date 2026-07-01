#!/usr/bin/env python3
"""Run the discretization + planning pipeline on the Open3D apartment scan.

The full PLY is ~30M double-precision points (806 MB). We read a bounded leading
chunk sequentially (fast, memory-safe), reorient to a gravity-aligned frame, build
the 2.5D traversability map with RGB-D-appropriate parameters, and plan a path
between the two farthest cells of the largest connected free region (so a path
always exists). Saves out_apartment.png.
"""
import os
import sys
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, astar
from lidar_pathplan.astar import path_length
from lidar_pathplan.elevation_grid import FREE, LETHAL, UNKNOWN
from test_real_room import detect_up_axis, reorient

SRC = "test-data/aligned_low_apartment/apt.ply"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000


def read_chunk(path, n):
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.read(64)
        f.seek(hdr.find(b"end_header\n") + len(b"end_header\n"))
        block = np.fromfile(f, dtype=dt, count=n)
    xyz = np.stack([block["x"], block["y"], block["z"]], axis=-1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def largest_component_endpoints(grid):
    free = np.isfinite(grid.cost)
    seen = np.zeros_like(free)
    h, w = free.shape
    best = []
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r0 in range(h):
        for c0 in range(w):
            if not free[r0, c0] or seen[r0, c0]:
                continue
            comp = []
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in nbr:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            if len(comp) > len(best):
                best = comp
    arr = np.array(best)
    key = arr[:, 0] + arr[:, 1]
    return tuple(arr[key.argmin()]), tuple(arr[key.argmax()]), len(best)


def main():
    print("reading %d points from %s ..." % (N, SRC), flush=True)
    raw = read_chunk(SRC, N)
    up = detect_up_axis(raw)
    cloud, floor = reorient(raw, up)
    print("loaded %d pts; up=%s floor=%.2f; ground x[%.1f,%.1f] y[%.1f,%.1f] h[%.2f,%.2f]" % (
        raw.shape[0], "xyz"[up], floor,
        cloud[:, 0].min(), cloud[:, 0].max(), cloud[:, 1].min(), cloud[:, 1].max(),
        cloud[:, 2].min(), cloud[:, 2].max()), flush=True)

    robot = QuadrupedParams(radius=0.15, max_step=0.15, max_roughness=0.22, max_slope_deg=45)
    grid = discretize(cloud, GridConfig(resolution=0.05, z_clip=(-0.10, 1.8),
                                        smooth_passes=4), robot)
    print("grid %dx%d @ %.2fm  observed=%d lethal=%d unknown=%d" % (
        grid.shape[0], grid.shape[1], grid.resolution, grid.meta["observed_cells"],
        grid.meta["lethal_cells"], grid.meta["unknown_cells"]), flush=True)

    start, goal, comp_sz = largest_component_endpoints(grid)
    path = astar(grid.cost, start, goal)
    msg = "no path" if path is None else "%d cells, %.2f m" % (len(path), path_length(path, grid.resolution))
    print("largest free region: %d cells; path: %s" % (comp_sz, msg), flush=True)

    plot(grid, cloud, path, start, goal)
    print("wrote out_apartment.png", flush=True)


def plot(grid, cloud, path, start, goal):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ext = [grid.origin[0], grid.origin[0] + grid.shape[1] * grid.resolution,
           grid.origin[1], grid.origin[1] + grid.shape[0] * grid.resolution]
    s = cloud[::5]
    ax[0].scatter(s[:, 0], s[:, 1], c=s[:, 2], s=0.3, cmap="viridis")
    ax[0].set_title("apartment scan (color = height)")
    im = ax[1].imshow(grid.elevation, origin="lower", extent=ext, cmap="terrain")
    ax[1].set_title("2.5D elevation grid"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    cmap = ListedColormap(["#cfe8c4", "#7d2828", "#e8e8e8"])
    ax[2].imshow(grid.classes, origin="lower", extent=ext, cmap=cmap,
                 norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N))
    if path:
        xs = [grid.cell_to_world(r, c)[0] for r, c in path]
        ys = [grid.cell_to_world(r, c)[1] for r, c in path]
        ax[2].plot(xs, ys, "-", color="dodgerblue", lw=2, label="path")
    ax[2].plot(*grid.cell_to_world(*start), "o", color="lime", ms=8, label="start")
    ax[2].plot(*grid.cell_to_world(*goal), "*", color="red", ms=13, label="goal")
    ax[2].set_title("classification + path\n(green=free, red=obstacle, grey=unknown)")
    ax[2].legend(loc="upper right")
    for a in ax:
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig("out_apartment.png", dpi=120)


if __name__ == "__main__":
    main()
