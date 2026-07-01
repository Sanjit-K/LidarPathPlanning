"""Generate a synthetic lidar point cloud of a small outdoor scene.

Used by the demo so the pipeline runs with no external data. The scene has:
    - a gently sloped ground plane with noise (rough terrain)
    - a raised platform / curb (a climbable-or-not step depending on height)
    - two box obstacles (walls) the planner must route around
    - a ramp the robot *can* take

Returns an (N, 3) array of x, y, z in meters.
"""

import numpy as np


def make_scene(seed: int = 0, density: float = 400.0) -> np.ndarray:
    """Build the demo point cloud.

    Args:
        seed: RNG seed for reproducibility.
        density: approximate points per square meter on the ground.
    """
    rng = np.random.default_rng(seed)
    pts = []

    # --- Ground plane: 10m x 10m, gentle slope in +x, small roughness ---------
    area = 10.0 * 10.0
    n_ground = int(area * density)
    gx = rng.uniform(0, 10, n_ground)
    gy = rng.uniform(0, 10, n_ground)
    gz = 0.04 * gx + rng.normal(0, 0.02, n_ground)  # ~2.3 deg slope + noise
    pts.append(np.column_stack([gx, gy, gz]))

    # --- Two box obstacles (tall, lethal) -------------------------------------
    pts.append(_box(rng, x0=3.0, x1=3.6, y0=2.0, y1=7.0, z_top=0.8, density=density * 4))
    pts.append(_box(rng, x0=6.0, x1=6.6, y0=0.0, y1=5.5, z_top=0.8, density=density * 4))

    # --- A curb / step ~0.12 m high (climbable for default params) ------------
    pts.append(_box(rng, x0=1.0, x1=9.0, y0=8.6, y1=9.0, z_top=0.12, density=density * 2))

    # --- A ramp the robot can ascend (slope within limit) ---------------------
    # Sits on top of the sloped ground so it reads as a rise, not a pit. The added
    # rise of ~0.10 m/m (~5.7 deg) plus the 2.3 deg base stays within max_slope.
    rx = rng.uniform(7.5, 9.5, int(2.0 * density))
    ry = rng.uniform(6.0, 8.0, int(2.0 * density))
    ground_z = 0.04 * rx
    rz = ground_z + 0.10 * (rx - 7.5) + rng.normal(0, 0.01, rx.shape[0])
    pts.append(np.column_stack([rx, ry, rz]))

    cloud = np.vstack(pts)
    rng.shuffle(cloud)
    return cloud


def _box(rng, x0, x1, y0, y1, z_top, density):
    """Points on the top face + vertical sides of an axis-aligned box."""
    top_n = int((x1 - x0) * (y1 - y0) * density)
    tx = rng.uniform(x0, x1, top_n)
    ty = rng.uniform(y0, y1, top_n)
    tz = np.full(top_n, z_top) + rng.normal(0, 0.01, top_n)
    top = np.column_stack([tx, ty, tz])

    side_n = int(2 * ((x1 - x0) + (y1 - y0)) * z_top * density)
    sz = rng.uniform(0, z_top, side_n)
    # distribute along the 4 walls
    sx, sy = [], []
    for _ in range(side_n):
        wall = rng.integers(0, 4)
        if wall == 0:
            sx.append(x0); sy.append(rng.uniform(y0, y1))
        elif wall == 1:
            sx.append(x1); sy.append(rng.uniform(y0, y1))
        elif wall == 2:
            sx.append(rng.uniform(x0, x1)); sy.append(y0)
        else:
            sx.append(rng.uniform(x0, x1)); sy.append(y1)
    sides = np.column_stack([np.array(sx), np.array(sy), sz])
    return np.vstack([top, sides])
