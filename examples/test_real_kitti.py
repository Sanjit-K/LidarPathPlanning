#!/usr/bin/env python3
"""Test the discretization + planning pipeline on a real KITTI Velodyne scan.

KITTI scans are real outdoor lidar (Velodyne HDL-64, ~120k points, x,y,z,intensity
as float32) -- a good stress test for the pipeline versus the synthetic scene.
They differ from the synthetic data in ways this harness handles explicitly:

  * The sensor sits ~1.73 m above the road, so the ground plane is at z ~ -1.73,
    not 0. We don't need to re-zero it -- the elevation grid is relative -- but we
    z-clip to drop the sky / far-tall returns and the under-car noise.
  * Coverage is 360 deg out to ~80 m and gets sparse with range, so far cells end
    up UNKNOWN. That's expected, not a bug.
  * There is a no-return shadow directly under the vehicle, so (0,0) is not a valid
    start; we pick a start a few meters ahead on the road and snap to free space.

Usage:
    python3 examples/test_real_kitti.py                # auto-downloads a sample
    python3 examples/test_real_kitti.py --bin scan.bin # use your own KITTI .bin
    python3 examples/test_real_kitti.py --res 0.25 --map-size 60
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

# Public no-auth mirrors of a single KITTI Velodyne scan. Tried in order.
SAMPLE_URLS = [
    "https://raw.githubusercontent.com/azureology/kitti-velo2cam/master/data/000000.bin",
    "https://raw.githubusercontent.com/kuixu/kitti_object_vis/master/data/object/training/velodyne/000000.bin",
    "https://raw.githubusercontent.com/windowsub0406/KITTI_Tutorial/master/velodyne_points/data/0000000000.bin",
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch_sample() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, "kitti_000000.bin")
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print("using cached", dest)
        return dest
    for url in SAMPLE_URLS:
        try:
            print("downloading", url)
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 10000:
                print("saved", dest, "(%d bytes)" % os.path.getsize(dest))
                return dest
        except Exception as e:  # noqa: BLE001
            print("  failed:", e)
    raise RuntimeError("could not download a KITTI sample; pass --bin manually")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", help="KITTI .bin scan; omit to auto-download a sample")
    p.add_argument("--res", type=float, default=0.25, help="grid resolution (m)")
    p.add_argument("--map-size", type=float, default=60.0, help="square map side (m)")
    p.add_argument("--z-min", type=float, default=-3.0, help="z clip low (sensor frame)")
    p.add_argument("--z-max", type=float, default=1.0, help="z clip high (sensor frame)")
    p.add_argument("--out", default="out_kitti.png")
    args = p.parse_args(argv)

    path_bin = args.bin or fetch_sample()
    cloud = load_point_cloud(path_bin)
    print("loaded %d points; x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]" % (
        cloud.shape[0],
        cloud[:, 0].min(), cloud[:, 0].max(),
        cloud[:, 1].min(), cloud[:, 1].max(),
        cloud[:, 2].min(), cloud[:, 2].max()))

    # Center a square map on the sensor so we plan over the immediate surroundings.
    half = args.map_size / 2.0
    config = GridConfig(
        resolution=args.res,
        bounds=(-half, -half, half, half),
        z_clip=(args.z_min, args.z_max),
    )
    robot = QuadrupedParams()
    grid = discretize(cloud, config=config, robot=robot)
    print("grid %dx%d @ %.2fm  observed=%d lethal=%d unknown=%d" % (
        grid.shape[0], grid.shape[1], grid.resolution,
        grid.meta["observed_cells"], grid.meta["lethal_cells"],
        grid.meta["unknown_cells"]))

    # Plan along the road ahead (+x), starting a few meters from the sensor shadow.
    start = _nudge_to_free(grid, grid.world_to_cell(4.0, 0.0))
    goal = _nudge_to_free(grid, grid.world_to_cell(min(half - 2.0, 22.0), 0.0))
    path = astar(grid.cost, start, goal)
    if path is None:
        print("NO PATH (try larger --map-size or different goal)")
    else:
        print("path: %d cells, %.1f m" % (len(path), path_length(path, grid.resolution)))

    _plot(grid, cloud, path, start, goal, args.out)
    print("wrote", args.out)
    return 0


def _nudge_to_free(grid, cell, max_r=40):
    if grid.in_bounds(*cell) and np.isfinite(grid.cost[cell]):
        return cell
    h, w = grid.shape
    for r in range(1, max_r):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = cell[0] + dr, cell[1] + dc
                if 0 <= rr < h and 0 <= cc < w and np.isfinite(grid.cost[rr, cc]):
                    return (rr, cc)
    raise ValueError("no free cell near %s" % (cell,))


def _plot(grid, cloud, path, start, goal, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    extent = [grid.origin[0], grid.origin[0] + grid.shape[1] * grid.resolution,
              grid.origin[1], grid.origin[1] + grid.shape[0] * grid.resolution]

    ax = axes[0]
    m = (np.abs(cloud[:, 0]) < extent[1]) & (np.abs(cloud[:, 1]) < extent[3])
    sc = ax.scatter(cloud[m, 0], cloud[m, 1], c=cloud[m, 2], s=0.5, cmap="viridis")
    ax.set_title("KITTI scan (color = height)")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="z (m)")

    ax = axes[1]
    im = ax.imshow(grid.elevation, origin="lower", extent=extent, cmap="terrain")
    ax.set_title("2.5D elevation grid")
    fig.colorbar(im, ax=ax, fraction=0.046, label="height (m)")

    ax = axes[2]
    disp = grid.cost.copy()
    fmax = np.nanmax(disp[np.isfinite(disp)]) if np.isfinite(disp).any() else 1.0
    disp[~np.isfinite(disp)] = fmax * 1.5
    ax.imshow(disp, origin="lower", extent=extent, cmap="magma_r")
    ax.imshow(np.where(grid.classes == UNKNOWN, 1.0, np.nan),
              origin="lower", extent=extent, cmap="Greys", alpha=0.3, vmin=0, vmax=1)
    if path:
        xs = [grid.cell_to_world(r, c)[0] for r, c in path]
        ys = [grid.cell_to_world(r, c)[1] for r, c in path]
        ax.plot(xs, ys, "-", color="cyan", lw=2, label="path")
    ax.plot(*grid.cell_to_world(*start), "o", color="lime", ms=9, label="start")
    ax.plot(*grid.cell_to_world(*goal), "*", color="red", ms=14, label="goal")
    ax.set_title("Cost grid + A* path (grey = unknown)")
    ax.legend(loc="upper right")

    for ax in axes:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    sys.exit(main())
