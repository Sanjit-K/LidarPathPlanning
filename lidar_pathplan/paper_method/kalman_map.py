"""Probabilistic 2.5D terrain map with per-cell Kalman height fusion.

This is the paper's "Terrain Mapping with LiDAR" block: point clouds are placed
into a fixed grid in the world frame, and each cell maintains a height *estimate*
and its *variance*. Successive LiDAR scans are fused with a 1-D Kalman filter, so
repeated observations drive the variance down and average out sensor noise --
exactly the noise-reduction the paper attributes to its Kalman-filtered map.

Per-cell state: height h and variance P.

Measurement model: a lidar return at range r has variance
    R = sensor_var + range_coeff * r**2
(lidar noise grows with range). Multiple returns landing in one cell during one
scan are combined in information form before the Kalman update.

Kalman update of cell (h, P) with a measurement (z, R):
    unobserved cell (P = inf):  h <- z,                P <- R
    otherwise:                  K = P / (P + R)
                                h <- h + K (z - h)
                                P <- (1 - K) P

predict(q) inflates every variance by process noise q (call it when the world may
have changed or pose drift accumulated) so the map can adapt to new measurements.

The fused map exposes a traversability cost grid (height-difference / slope based,
following the paper's grid-height-difference obstacle test) for the Dijkstra and
APF planners. It reuses the geometry helpers from elevation_grid.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..elevation_grid import (
    ElevationGrid, GridConfig, FREE, LETHAL, UNKNOWN, LETHAL_COST,
    _neighbor_step, _slope_magnitude, _smooth_nan, _remove_small_blobs, _dilate,
)
from ..robot import QuadrupedParams


@dataclass
class KalmanMapConfig:
    resolution: float = 0.10
    bounds: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0)  # xmin,ymin,xmax,ymax
    sensor_var: float = 0.0009            # base measurement variance (m^2), ~0.03 m std
    range_coeff: float = 0.0004           # variance added per m^2 of range
    z_clip: Optional[Tuple[float, float]] = None
    init_variance: float = 1e6            # variance of an unobserved cell (~inf)
    max_variance_traversable: float = 0.05  # cells noisier than this are UNKNOWN in the cost grid


class KalmanElevationMap:
    """Fixed-frame probabilistic elevation map fused over multiple scans."""

    def __init__(self, config: Optional[KalmanMapConfig] = None):
        self.config = config or KalmanMapConfig()
        xmin, ymin, xmax, ymax = self.config.bounds
        res = self.config.resolution
        self.origin = (xmin, ymin)
        self.n_cols = max(1, int(np.ceil((xmax - xmin) / res)))
        self.n_rows = max(1, int(np.ceil((ymax - ymin) / res)))
        shape = (self.n_rows, self.n_cols)
        self.height = np.full(shape, np.nan, dtype=np.float64)
        self.variance = np.full(shape, self.config.init_variance, dtype=np.float64)
        self.hits = np.zeros(shape, dtype=np.int32)
        self.n_scans = 0

    @property
    def shape(self):
        return (self.n_rows, self.n_cols)

    @property
    def resolution(self):
        return self.config.resolution

    def world_to_cell(self, x, y):
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row, col):
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def predict(self, process_var: float):
        """Inflate all variances by process noise so the filter stays adaptive."""
        self.variance += process_var

    def update(self, points: np.ndarray, sensor_origin=(0.0, 0.0, 0.0)):
        """Fuse one LiDAR scan (already in the map/world frame) into the map.

        Args:
            points: (N,3+) array; columns 0,1,2 are x,y,z in the world frame.
            sensor_origin: (x,y,z) of the sensor, used for the range-based noise.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be (N,>=3)")
        xy = pts[:, :2]
        z = pts[:, 2]

        if self.config.z_clip is not None:
            lo, hi = self.config.z_clip
            keep = (z >= lo) & (z <= hi)
            xy, z, pts = xy[keep], z[keep], pts[keep]
        if z.size == 0:
            return self

        res = self.config.resolution
        cols = np.floor((xy[:, 0] - self.origin[0]) / res).astype(np.int64)
        rows = np.floor((xy[:, 1] - self.origin[1]) / res).astype(np.int64)
        inside = (cols >= 0) & (cols < self.n_cols) & (rows >= 0) & (rows < self.n_rows)
        rows, cols, z, pts = rows[inside], cols[inside], z[inside], pts[inside]
        if z.size == 0:
            return self

        # Per-point measurement variance from range to the sensor.
        d = pts[:, :3] - np.asarray(sensor_origin, dtype=np.float64)
        r2 = np.einsum("ij,ij->i", d, d)
        meas_var = self.config.sensor_var + self.config.range_coeff * r2
        info = 1.0 / meas_var                       # measurement information weight

        # Combine all returns in a cell *this scan* in information form:
        #   z_meas = (sum z*info) / (sum info),  R_meas = 1 / (sum info)
        flat = rows * self.n_cols + cols
        n = self.n_rows * self.n_cols
        sum_info = np.zeros(n)
        sum_zinfo = np.zeros(n)
        cnt = np.zeros(n, dtype=np.int64)
        np.add.at(sum_info, flat, info)
        np.add.at(sum_zinfo, flat, z * info)
        np.add.at(cnt, flat, 1)

        observed = sum_info > 0
        z_meas = np.zeros(n)
        z_meas[observed] = sum_zinfo[observed] / sum_info[observed]
        R_meas = np.full(n, np.inf)
        R_meas[observed] = 1.0 / sum_info[observed]

        z_meas = z_meas.reshape(self.shape)
        R_meas = R_meas.reshape(self.shape)
        cnt = cnt.reshape(self.shape)
        obs2d = observed.reshape(self.shape)

        # Snapshot which cells already had an estimate BEFORE we mutate anything,
        # so a first observation is only initialized, not also fused against itself.
        had = np.isfinite(self.height)

        fresh = obs2d & ~had                         # first-ever observation
        self.height[fresh] = z_meas[fresh]
        self.variance[fresh] = R_meas[fresh]

        upd = obs2d & had                            # fuse with existing estimate
        P = self.variance[upd]
        R = R_meas[upd]
        K = P / (P + R)
        self.height[upd] = self.height[upd] + K * (z_meas[upd] - self.height[upd])
        self.variance[upd] = (1.0 - K) * P

        self.hits += cnt.astype(np.int32)
        self.n_scans += 1
        return self

    # ------------------------------------------------------------------ cost grid
    def to_grid(self, robot: Optional[QuadrupedParams] = None,
                inflate: bool = True, smooth_passes: int = 1) -> ElevationGrid:
        """Build a traversability cost grid from the fused map.

        Obstacle test follows the paper's grid-height-difference idea (a cell whose
        height differs from its neighbours by more than the robot can step is an
        obstacle), augmented with slope and a variance gate: cells still too noisy
        (high Kalman variance) are marked UNKNOWN rather than trusted.
        """
        robot = robot or QuadrupedParams()
        res = self.resolution
        elevation = self.height.astype(np.float32)
        observed = np.isfinite(elevation) & (self.hits > 0)

        elev_s = _smooth_nan(elevation, passes=smooth_passes)
        slope = _slope_magnitude(elev_s, res)
        step = _neighbor_step(elev_s)               # paper's |z_i - z_d| height difference

        classes = np.full(self.shape, UNKNOWN, dtype=np.uint8)
        classes[observed] = FREE

        too_tall = observed & np.isfinite(step) & (step > robot.max_step)
        too_steep = observed & np.isfinite(slope) & (slope > robot.max_slope_tan)
        lethal = too_tall | too_steep
        lethal = _remove_small_blobs(lethal, min_size=3)
        classes[lethal] = LETHAL

        if inflate:
            radius_cells = int(np.ceil(robot.radius / res))
            classes[_dilate(lethal, radius_cells) & observed] = LETHAL

        # Variance gate: confident-but-untrusted cells fall back to UNKNOWN.
        noisy = observed & (self.variance > self.config.max_variance_traversable)
        classes[noisy & (classes == FREE)] = UNKNOWN

        cost = np.ones(self.shape, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            cost += np.nan_to_num(slope / robot.max_slope_tan) * 3.0
        # Prefer lower-variance (better-known) ground, all else equal.
        cost += np.nan_to_num(np.sqrt(np.clip(self.variance, 0, None)) * 2.0).astype(np.float32)
        cost[classes == LETHAL] = LETHAL_COST
        cost[classes == UNKNOWN] = 50.0

        gconf = GridConfig(resolution=res, bounds=tuple(self.config.bounds))
        return ElevationGrid(
            config=gconf, origin=self.origin, elevation=elevation,
            roughness=np.sqrt(np.clip(self.variance, 0, None)).astype(np.float32),
            slope=slope, count=self.hits.copy(), classes=classes, cost=cost,
            meta={"n_scans": self.n_scans,
                  "observed_cells": int(observed.sum()),
                  "lethal_cells": int((classes == LETHAL).sum()),
                  "unknown_cells": int((classes == UNKNOWN).sum()),
                  "mean_variance": float(np.nanmean(self.variance[observed])) if observed.any() else float("nan"),
                  "robot": robot},
        )
