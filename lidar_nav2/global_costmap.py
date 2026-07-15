"""Sparse persistent fusion of rolling 2.5D costmaps.

The onboard navigator deliberately plans on a small robot-centred grid.  This
module keeps a second, world-indexed layer for visualization and export: known
cells from every local grid are fused into a sparse map, while unknown cells
never erase terrain observed earlier in the run.
"""

import os
import threading
import time
from typing import Dict, Tuple

import numpy as np

from lidar_pathplan.elevation_grid import UNKNOWN


class SparseGlobalCostmap:
    """Accumulate known cells from rolling ``ElevationGrid`` instances.

    Cell indices are anchored to world coordinate zero at a fixed resolution,
    so local grids may slide by arbitrary sub-cell offsets without moving data
    that was already fused.  The newest class wins; elevation is a capped
    count-weighted mean to reduce sweep noise without making old data immutable.
    """

    def __init__(self, resolution: float = 0.10):
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        self.resolution = float(resolution)
        self._cells: Dict[Tuple[int, int], Tuple[float, int, int]] = {}
        self._lock = threading.Lock()
        self.revision = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cells)

    def update(self, grid, sensor_xy=None, existing_update_radius=None) -> int:
        """Fuse known cells from ``grid`` and return the number changed.

        New cells are always admitted.  When ``existing_update_radius`` is set,
        an already-mapped cell is refreshed only when its centre is within that
        many metres of ``sensor_xy``.  This lets distant returns discover new
        space without allowing their higher uncertainty to rewrite established
        terrain repeatedly.
        """
        if abs(float(grid.resolution) - self.resolution) > 1e-6:
            raise ValueError("grid resolution does not match global map")
        if existing_update_radius is not None:
            if sensor_xy is None:
                raise ValueError("sensor_xy is required when update radius is set")
            if existing_update_radius < 0.0:
                raise ValueError("existing_update_radius must be non-negative")

        known = (grid.classes != UNKNOWN) & np.isfinite(grid.elevation)
        rows, cols = np.nonzero(known)
        if rows.size == 0:
            return 0

        wx = float(grid.origin[0]) + (cols.astype(np.float64) + 0.5) * self.resolution
        wy = float(grid.origin[1]) + (rows.astype(np.float64) + 0.5) * self.resolution
        ix = np.floor(wx / self.resolution).astype(np.int32)
        iy = np.floor(wy / self.resolution).astype(np.int32)
        elevations = grid.elevation[rows, cols].astype(np.float64)
        classes = grid.classes[rows, cols].astype(np.uint8)
        counts = np.maximum(1, grid.count[rows, cols]).astype(np.int32)

        changed = 0
        with self._lock:
            for x, y, world_x, world_y, z, cls, count in zip(
                    ix, iy, wx, wy, elevations, classes, counts):
                key = (int(x), int(y))
                old = self._cells.get(key)
                if old is not None and existing_update_radius is not None:
                    dx = float(world_x) - float(sensor_xy[0])
                    dy = float(world_y) - float(sensor_xy[1])
                    if dx * dx + dy * dy > float(existing_update_radius) ** 2:
                        continue
                new_weight = min(int(count), 20)
                if old is None:
                    fused_z, total_weight = float(z), new_weight
                else:
                    old_z, _, old_weight = old
                    old_weight = min(int(old_weight), 80)
                    total_weight = min(old_weight + new_weight, 100)
                    fused_z = (old_z * old_weight + float(z) * new_weight) / \
                              (old_weight + new_weight)
                # Classification is deliberately current rather than averaged:
                # an obstacle that moves away must eventually become traversable.
                self._cells[key] = (fused_z, int(cls), total_weight)
                changed += 1
            if changed:
                self.revision += 1
        return changed

    def snapshot(self):
        """Return deterministic arrays suitable for network transport or NPZ."""
        with self._lock:
            items = sorted(self._cells.items(), key=lambda item: (item[0][1], item[0][0]))
            revision = self.revision
        n = len(items)
        cells = np.empty((n, 2), dtype=np.int32)
        elevation = np.empty(n, dtype=np.float32)
        classes = np.empty(n, dtype=np.uint8)
        observations = np.empty(n, dtype=np.uint16)
        for i, ((x, y), (z, cls, weight)) in enumerate(items):
            cells[i] = (x, y)
            elevation[i] = z
            classes[i] = cls
            observations[i] = min(weight, np.iinfo(np.uint16).max)
        return {
            "resolution": self.resolution,
            "revision": revision,
            "cells": cells,
            "elevation": elevation,
            "classes": classes,
            "observations": observations,
        }

    def save(self, path: str) -> int:
        """Atomically save the current sparse map to a compressed NPZ file."""
        snap = self.snapshot()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            format_version=np.asarray([1], dtype=np.int16),
            resolution=np.asarray([snap["resolution"]], dtype=np.float32),
            revision=np.asarray([snap["revision"]], dtype=np.int64),
            saved_at=np.asarray([time.time()], dtype=np.float64),
            cells=snap["cells"],
            elevation=snap["elevation"],
            classes=snap["classes"],
            observations=snap["observations"],
        )
        os.replace(tmp, path)
        return int(snap["cells"].shape[0])

    @classmethod
    def load(cls, path: str):
        """Load a map written by :meth:`save`.

        Callers must ensure the current world/odometry frame matches the saved
        session; this module cannot infer whether a robot reboot reset odometry.
        """
        with np.load(path) as data:
            resolution = float(data["resolution"][0])
            cells = np.asarray(data["cells"], dtype=np.int32)
            elevation = np.asarray(data["elevation"], dtype=np.float32)
            classes = np.asarray(data["classes"], dtype=np.uint8)
            observations = np.asarray(data["observations"], dtype=np.uint16)
            revision = int(data["revision"][0])
        if cells.ndim != 2 or cells.shape[1] != 2:
            raise ValueError("invalid saved global map cells")
        if not (len(cells) == len(elevation) == len(classes) == len(observations)):
            raise ValueError("saved global map arrays have inconsistent lengths")
        result = cls(resolution)
        result._cells = {
            (int(cell[0]), int(cell[1])): (float(z), int(cell_class), int(weight))
            for cell, z, cell_class, weight in zip(
                cells, elevation, classes, observations)
        }
        result.revision = revision
        return result
