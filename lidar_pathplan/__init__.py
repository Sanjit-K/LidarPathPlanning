"""Lidar discretization and path planning for a quadruped robot.

Pipeline:
    point cloud (Nx3) -> 2.5D elevation grid -> traversability cost grid -> A* path

Modules:
    elevation_grid : discretize a 3D point cloud into a 2.5D elevation + cost grid
    astar          : grid path planner over the traversability cost grid
    robot          : quadruped traversability parameters
    io_utils       : load point clouds from .npy / .bin / .pcd / .xyz
"""

from .robot import QuadrupedParams
from .elevation_grid import ElevationGrid, GridConfig, discretize, cost_to_occupancy
from .astar import astar

__all__ = [
    "QuadrupedParams",
    "ElevationGrid",
    "GridConfig",
    "discretize",
    "cost_to_occupancy",
    "astar",
]
