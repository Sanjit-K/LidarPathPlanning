#!/usr/bin/env python3
"""Demo of the paper's classical navigation stack (Wang et al.).

    Kalman terrain map  ->  Dijkstra global path  ->  elastic-band (TEB) smooth
                                                   ->  APF local obstacle avoidance

Two modes:
    --mode stack    full fixed-goal stack: fuse scans, plan globally with Dijkstra,
                    smooth, then follow with APF local avoidance (default).
    --mode follow   person-following: the goal is a *moving* target and APF tracks
                    it directly with local obstacle avoidance (the modern extension
                    where the global Dijkstra planner matters less).

    python3 examples/paper_navigation_demo.py
    python3 examples/paper_navigation_demo.py --mode follow

Writes a figure to out_paper.png.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_pathplan import QuadrupedParams
from lidar_pathplan.synthetic import make_scene
from lidar_pathplan.elevation_grid import LETHAL, UNKNOWN
from lidar_pathplan.paper_method import (
    KalmanElevationMap, dijkstra, smooth_path, APFPlanner, APFParams)
from lidar_pathplan.paper_method.kalman_map import KalmanMapConfig


def build_kalman_map(n_scans=6, noise=0.05, seed=0):
    """Fuse several noisy LiDAR scans of the same scene; return (map, var_trace)."""
    base = make_scene(seed=1)
    km = KalmanElevationMap(KalmanMapConfig(
        resolution=0.10, bounds=(0, 0, 10, 10), z_clip=(-1.0, 2.0)))
    rng = np.random.default_rng(seed)
    var_trace = []
    for _ in range(n_scans):
        scan = base.copy()
        scan[:, 2] += rng.normal(0, noise, base.shape[0])   # per-scan sensor noise
        # simulate partial coverage from a moving sensor: keep a random 70%
        keep = rng.random(scan.shape[0]) < 0.7
        km.update(scan[keep], sensor_origin=(5.0, 5.0, 1.5))
        obs = np.isfinite(km.height)
        var_trace.append(float(np.nanmean(km.variance[obs])))
    return km, var_trace


def nudge_free(grid, cell, max_r=30):
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["stack", "follow"], default="stack")
    p.add_argument("--scans", type=int, default=6)
    p.add_argument("--noise", type=float, default=0.05, help="per-scan sensor noise std (m)")
    p.add_argument("--out", default="out_paper.png")
    args = p.parse_args(argv)

    km, var_trace = build_kalman_map(args.scans, args.noise)
    print("Kalman fusion over %d scans; mean cell variance: %s" % (
        args.scans, " -> ".join("%.4f" % v for v in var_trace)))
    grid = km.to_grid(QuadrupedParams())
    print("fused grid %dx%d  observed=%d lethal=%d unknown=%d  mean_var=%.4f" % (
        grid.shape[0], grid.shape[1], grid.meta["observed_cells"],
        grid.meta["lethal_cells"], grid.meta["unknown_cells"], grid.meta["mean_variance"]))

    if args.mode == "stack":
        _run_stack(grid, var_trace, args.out)
    else:
        _run_follow(grid, var_trace, args.out)
    print("wrote", args.out)
    return 0


def _run_stack(grid, var_trace, out):
    start = nudge_free(grid, grid.world_to_cell(0.5, 0.5))
    goal = nudge_free(grid, grid.world_to_cell(9.5, 5.0))
    raw = dijkstra(grid.cost, start, goal)
    if raw is None:
        print("Dijkstra found NO global path"); smoothed, traj = [], []
    else:
        smoothed = smooth_path(raw, grid)
        apf = APFPlanner(grid, APFParams())
        traj = apf.follow_path(smoothed)
        print("Dijkstra: %d cells; smoothed: %d pts; APF: %d steps" % (
            len(raw), len(smoothed), len(traj)))
    _plot(grid, var_trace, raw, smoothed, traj, None, out,
          title="Paper stack: Kalman map -> Dijkstra -> TEB smooth -> APF")


def _run_follow(grid, var_trace, out):
    # The person walks a guaranteed-traversable route (a Dijkstra path through the
    # free map) at a steady pace; the robot APF-tracks the person as a moving goal,
    # avoiding terrain hazards on the way. This is the person-following regime where
    # the robot has no fixed destination -- only the (moving) person.
    pstart = nudge_free(grid, grid.world_to_cell(0.8, 0.8))
    pgoal = nudge_free(grid, grid.world_to_cell(9.2, 8.5))
    route = dijkstra(grid.cost, pstart, pgoal)
    route_xy = np.array([grid.cell_to_world(r, c) for r, c in route])
    seglen = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seglen)])
    speed = 0.5  # person walking speed, m/s

    def person(t):
        s = min(speed * t, arc[-1])
        return (float(np.interp(s, arc, route_xy[:, 0])),
                float(np.interp(s, arc, route_xy[:, 1])))

    apf = APFPlanner(grid, APFParams(max_speed=0.9))   # robot a bit faster, can keep up
    # robot starts ~1 m behind the person
    start = grid.cell_to_world(*nudge_free(grid, grid.world_to_cell(0.6, 0.6)))
    traj = apf.run(start, person, max_steps=600)
    person_traj = [person(k * apf.params.dt) for k in range(len(traj))]
    gap = np.linalg.norm(np.array(traj[-1]) - np.array(person_traj[-1]))
    gaps = [np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(traj, person_traj)]
    print("person-following: %d steps; mean gap %.2f m, final gap %.2f m" % (
        len(traj), float(np.mean(gaps)), gap))
    _plot(grid, var_trace, None, None, traj, person_traj, out,
          title="Person-following: APF tracks a moving target with local avoidance")


def _class_panel(ax, grid, extent):
    cmap = ListedColormap(["#cfe8c4", "#7d2828", "#e8e8e8"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(grid.classes, origin="lower", extent=extent, cmap=cmap, norm=norm)


def _plot(grid, var_trace, raw, smoothed, traj, person_traj, out, title):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4))
    extent = [grid.origin[0], grid.origin[0] + grid.shape[1] * grid.resolution,
              grid.origin[1], grid.origin[1] + grid.shape[0] * grid.resolution]

    # 1. Kalman variance reduction curve (the headline contribution)
    ax = axes[0]
    ax.plot(range(1, len(var_trace) + 1), var_trace, "o-", color="purple")
    ax.set_title("Kalman fusion: mean cell variance")
    ax.set_xlabel("scan #"); ax.set_ylabel("variance (m$^2$)"); ax.grid(alpha=0.3)

    # 2. fused elevation (with confidence = 1/std as alpha would over-complicate; show std)
    ax = axes[1]
    std = grid.roughness  # we stored per-cell std here
    im = ax.imshow(std, origin="lower", extent=extent, cmap="inferno_r", vmax=np.nanpercentile(std, 95))
    ax.set_title("Per-cell height std (confidence)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="std (m)")
    ax.set_aspect("equal")

    # 3. classification + global Dijkstra path + smoothed path
    ax = axes[2]
    _class_panel(ax, grid, extent)
    if raw:
        xs = [grid.cell_to_world(r, c)[0] for r, c in raw]
        ys = [grid.cell_to_world(r, c)[1] for r, c in raw]
        ax.plot(xs, ys, "--", color="orange", lw=1.5, label="Dijkstra (raw)")
    if smoothed:
        ax.plot([q[0] for q in smoothed], [q[1] for q in smoothed],
                "-", color="navy", lw=2, label="TEB-smoothed")
    ax.set_title("Global plan (green=free,red=obstacle)")
    if raw or smoothed:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    # 4. classification + APF trajectory (+ person)
    ax = axes[3]
    _class_panel(ax, grid, extent)
    if traj:
        ax.plot([q[0] for q in traj], [q[1] for q in traj],
                "-", color="dodgerblue", lw=2, label="APF robot")
        ax.plot(traj[0][0], traj[0][1], "o", color="lime", ms=9, label="start")
    if person_traj:
        ax.plot([q[0] for q in person_traj], [q[1] for q in person_traj],
                "-", color="red", lw=1.5, label="person")
        ax.plot(person_traj[-1][0], person_traj[-1][1], "*", color="red", ms=15)
    elif traj:
        ax.plot(traj[-1][0], traj[-1][1], "*", color="red", ms=15, label="goal")
    ax.set_title("Local avoidance (APF)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")

    for ax in axes[1:]:
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
