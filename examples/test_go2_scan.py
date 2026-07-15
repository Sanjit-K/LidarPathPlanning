#!/usr/bin/env python3
"""Run the pipeline on a real Unitree Go2 L1 scan captured from the robot.

The Go2's /utlidar/cloud is in the tilted sensor frame (`utlidar_lidar`) and
includes returns from the robot's own body. This harness:
  1. drops self-hits (range < min_range) and far outliers,
  2. levels the cloud by fitting the dominant ground plane (SVD) and rotating
     its normal to +z, with the ground re-zeroed,
  3. discretizes + plans between the two farthest free cells.

    python3 examples/test_go2_scan.py [scan.npy]           # writes out_go2.png
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

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, astar
from lidar_pathplan.astar import path_length

SRC = sys.argv[1] if len(sys.argv) > 1 else "test-data/go2/go2_scan.npy"
MIN_RANGE, MAX_RANGE = 0.5, 12.0


def _rot_to_z(n):
    """Rodrigues rotation matrix taking unit vector n -> +z."""
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3) if n[2] > 0 else np.diag([1.0, -1.0, -1.0])
    axis /= s
    ang = np.arccos(np.clip(n[2], -1, 1))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def level_cloud(pts, sensor_height_range=(0.15, 0.9)):
    """Level the cloud using the FLOOR plane, correctly oriented.

    The Go2's L1 is mounted upside-down and pitched, so the raw sensor frame
    cannot be trusted for 'up'. We find candidate planes with RANSAC and use two
    physical priors to identify the floor and its true up direction:
      * the sensor (origin) stands 0.15--0.9 m above the floor (standing dog),
        which also rejects the ceiling (sensor would be ~2 m from it);
      * the floor normal points from the plane toward the sensor.
    Finally the scene is yaw-aligned so the densest near-field obstacles (what
    the dog faces) lie toward +x.

    Returns (leveled points, tilt_deg, sensor_height).
    """
    rng = np.random.default_rng(0)
    n_pts = pts.shape[0]
    candidates = []          # (inliers, n, c)
    for _ in range(300):
        p0, p1, p2 = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = np.abs((pts - p0) @ n)
        inl = int((d < 0.05).sum())
        candidates.append((inl, n, p0))
    candidates.sort(key=lambda t: -t[0])

    chosen = None
    for inl, n, p0 in candidates[:30]:
        # refine plane on its inliers
        inliers = pts[np.abs((pts - p0) @ n) < 0.05]
        c = inliers.mean(axis=0)
        _, _, vt = np.linalg.svd(inliers - c, full_matrices=False)
        n = vt[2]
        # orient the normal from the plane toward the sensor (origin)
        if np.dot(-c, n) < 0:
            n = -n
        h = float(np.dot(-c, n))           # sensor height above this plane
        if sensor_height_range[0] <= h <= sensor_height_range[1]:
            chosen = (n, c, h, inl)
            break
    if chosen is None:                      # fallback: largest plane, oriented to sensor
        inl, n, p0 = candidates[0]
        inliers = pts[np.abs((pts - p0) @ n) < 0.05]
        c = inliers.mean(axis=0)
        _, _, vt = np.linalg.svd(inliers - c, full_matrices=False)
        n = vt[2]
        if np.dot(-c, n) < 0:
            n = -n
        chosen = (n, c, float(np.dot(-c, n)), inl)

    n, c, h_sensor, _ = chosen
    tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), -1, 1))))
    R = _rot_to_z(n)
    out = (pts - c) @ R.T                  # floor at z ~ 0, +z is truly up

    # yaw-align: put the near-field above-floor structure (table/box the dog
    # faces) toward +x for a readable map. Pure rotation about z.
    near = out[(np.hypot(out[:, 0], out[:, 1]) < 4.0) &
               (out[:, 2] > 0.08) & (out[:, 2] < 1.2)]
    R_total = R
    if near.shape[0] > 50:
        fx, fy = near[:, 0].mean(), near[:, 1].mean()
        yaw = np.arctan2(fy, fx)
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        out = out @ Rz.T
        R_total = Rz @ R
    # R_total maps raw sensor-frame vectors into the leveled, forward=+x frame:
    # this IS the lidar->base mount rotation (translation excluded).
    return out, tilt, h_sensor, R_total


def main():
    raw = np.load(SRC).astype(np.float64)
    r = np.linalg.norm(raw, axis=1)
    pts = raw[(r > MIN_RANGE) & (r < MAX_RANGE)]
    print("loaded %d pts; %d after range filter [%.1f, %.1f] m" % (
        raw.shape[0], pts.shape[0], MIN_RANGE, MAX_RANGE))

    cloud, tilt, h_sensor, _R = level_cloud(pts)
    print("estimated mount tilt vs ground: %.1f deg; sensor height %.2f m" % (tilt, h_sensor))
    print("leveled extent x[%.1f,%.1f] y[%.1f,%.1f] z[%.2f,%.2f]" % (
        cloud[:, 0].min(), cloud[:, 0].max(), cloud[:, 1].min(), cloud[:, 1].max(),
        cloud[:, 2].min(), cloud[:, 2].max()))

    robot = QuadrupedParams(radius=0.25, max_step=0.15,
                            max_roughness=0.10, max_slope_deg=35)
    grid = discretize(cloud, GridConfig(resolution=0.10, z_clip=(-0.3, 1.2),
                                        smooth_passes=2), robot)
    free = grid.meta["observed_cells"] - grid.meta["lethal_cells"]
    print("grid %dx%d @ %.2fm  observed=%d free=%d lethal=%d unknown=%d" % (
        grid.shape[0], grid.shape[1], grid.resolution, grid.meta["observed_cells"],
        free, grid.meta["lethal_cells"], grid.meta["unknown_cells"]))

    start, goal = far_endpoints(grid)
    path = astar(grid.cost, start, goal) if start else None
    if path:
        print("path: %d cells, %.2f m" % (len(path), path_length(path, grid.resolution)))
    else:
        print("no path in largest free region")

    plot(grid, cloud, path, start, goal, tilt)
    print("wrote out_go2.png")


def far_endpoints(grid):
    free = np.isfinite(grid.cost)
    seen = np.zeros_like(free)
    best = []
    h, w = free.shape
    for r0 in range(h):
        for c0 in range(w):
            if not free[r0, c0] or seen[r0, c0]:
                continue
            comp, q = [], deque([(r0, c0)])
            seen[r0, c0] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            if len(comp) > len(best):
                best = comp
    if len(best) < 2:
        return None, None
    arr = np.array(best)
    key = arr[:, 0] + arr[:, 1]
    return tuple(arr[key.argmin()]), tuple(arr[key.argmax()])


def plot(grid, cloud, path, start, goal, tilt):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    ext = [grid.origin[0], grid.origin[0] + grid.shape[1] * grid.resolution,
           grid.origin[1], grid.origin[1] + grid.shape[0] * grid.resolution]
    sc = ax[0].scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], s=1.2,
                       cmap="viridis", vmin=-0.3, vmax=1.5)
    ax[0].set_title("Go2 L1 scan, leveled (tilt %.1f deg)" % tilt)
    fig.colorbar(sc, ax=ax[0], fraction=0.046, label="height (m)")
    im = ax[1].imshow(grid.elevation, origin="lower", extent=ext, cmap="terrain")
    ax[1].set_title("2.5D elevation grid")
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="height (m)")
    cmap = ListedColormap(["#cfe8c4", "#7d2828", "#e8e8e8"])
    ax[2].imshow(grid.classes, origin="lower", extent=ext, cmap=cmap,
                 norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N))
    if path:
        xs = [grid.cell_to_world(r, c)[0] for r, c in path]
        ys = [grid.cell_to_world(r, c)[1] for r, c in path]
        ax[2].plot(xs, ys, "-", color="dodgerblue", lw=2, label="path")
        ax[2].plot(*grid.cell_to_world(*start), "o", color="lime", ms=9, label="start")
        ax[2].plot(*grid.cell_to_world(*goal), "*", color="red", ms=14, label="goal")
        ax[2].legend(loc="upper right")
    ax[2].set_title("classification + path")
    for a in ax:
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("out_go2.png", dpi=120)


if __name__ == "__main__":
    main()
