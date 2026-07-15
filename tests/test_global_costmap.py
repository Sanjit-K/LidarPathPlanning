"""Tests for persistent fusion of rolling local costmaps."""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_nav2.global_costmap import SparseGlobalCostmap
from lidar_pathplan.elevation_grid import FREE, LETHAL, UNKNOWN


def grid(origin, elevation, classes, count=None, resolution=0.1):
    elevation = np.asarray(elevation, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.uint8)
    if count is None:
        count = np.ones(elevation.shape, dtype=np.int32)
    return SimpleNamespace(
        origin=origin, resolution=resolution, elevation=elevation,
        classes=classes, count=np.asarray(count, dtype=np.int32))


def test_accumulates_cells_outside_latest_window():
    world = SparseGlobalCostmap(0.1)
    world.update(grid((0.0, 0.0), [[0.0, 0.1]], [[FREE, FREE]]))
    world.update(grid((2.0, 0.0), [[0.2, 0.3]], [[FREE, LETHAL]]))
    snap = world.snapshot()
    assert len(world) == 4
    assert set(map(tuple, snap["cells"])) == {(0, 0), (1, 0), (20, 0), (21, 0)}


def test_unknown_never_erases_observed_cell():
    world = SparseGlobalCostmap(0.1)
    world.update(grid((0.0, 0.0), [[0.4]], [[LETHAL]]))
    assert world.update(grid((0.0, 0.0), [[np.nan]], [[UNKNOWN]])) == 0
    snap = world.snapshot()
    assert snap["classes"].tolist() == [LETHAL]
    assert np.allclose(snap["elevation"], [0.4])


def test_latest_class_wins_and_elevation_is_fused():
    world = SparseGlobalCostmap(0.1)
    world.update(grid((0.0, 0.0), [[0.0]], [[LETHAL]], [[2]]))
    world.update(grid((0.0, 0.0), [[0.2]], [[FREE]], [[2]]))
    snap = world.snapshot()
    assert snap["classes"].tolist() == [FREE]
    assert np.allclose(snap["elevation"], [0.1])


def test_only_near_existing_cells_are_refreshed():
    world = SparseGlobalCostmap(0.1)
    # Establish one near and one far cell.
    world.update(grid((0.0, 0.0), [[0.0, 0.0]], [[LETHAL, LETHAL]]))
    world.update(grid((4.0, 0.0), [[0.0]], [[LETHAL]]))

    # A wide local grid observes both again, plus a brand-new far cell.  With a
    # 1 m refresh radius, only the near existing cell may change; the new far
    # cell is still admitted for exploration.
    elevation = np.full((1, 42), np.nan, dtype=np.float32)
    classes = np.full((1, 42), UNKNOWN, dtype=np.uint8)
    count = np.zeros((1, 42), dtype=np.int32)
    for col in (0, 1, 40, 41):
        elevation[0, col] = 0.2
        classes[0, col] = FREE
        count[0, col] = 2
    changed = world.update(
        grid((0.0, 0.0), elevation, classes, count),
        sensor_xy=(0.0, 0.0), existing_update_radius=1.0)
    assert changed == 3  # two near existing cells + one new far cell

    snap = world.snapshot()
    by_cell = {tuple(cell): int(cls) for cell, cls in zip(snap["cells"], snap["classes"])}
    assert by_cell[(0, 0)] == FREE
    assert by_cell[(1, 0)] == FREE
    assert by_cell[(40, 0)] == LETHAL  # far existing classification is preserved
    assert by_cell[(41, 0)] == FREE    # far but new is admitted


def test_atomic_npz_save():
    world = SparseGlobalCostmap(0.1)
    world.update(grid((0.0, 0.0), [[0.25]], [[FREE]], [[3]]))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "map.npz")
        assert world.save(path) == 1
        data = np.load(path)
        assert data["cells"].tolist() == [[0, 0]]
        assert np.allclose(data["elevation"], [0.25])
        assert not os.path.exists(path + ".tmp.npz")

        restored = SparseGlobalCostmap.load(path).snapshot()
        assert restored["cells"].tolist() == [[0, 0]]
        assert restored["classes"].tolist() == [FREE]
        assert np.allclose(restored["elevation"], [0.25])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception as exc:
            failed += 1; print("FAIL", fn.__name__, "-", repr(exc))
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
