"""Classical quadruped navigation stack from Wang et al., "Research on path
planning of quadruped robot based on globally mapping localization" (IEEE).

This implements the paper's pipeline as an additive layer on top of the existing
2.5D discretization, reusing its geometry helpers where useful:

    Kalman terrain map  ->  Dijkstra global path  ->  elastic-band (TEB) smooth
                                                   ->  APF local obstacle avoidance

Modules:
    kalman_map   : KalmanElevationMap -- probabilistic, incrementally-fused 2.5D map
    dijkstra     : Dijkstra shortest-path over the traversability cost grid
    elastic_band : geometric path smoother (simplified stand-in for TEB)
    apf          : Khatib artificial-potential-field local planner (moving goal ok)

The paper's SLAM/ICP localization block is assumed (poses are provided to the map
update); the modern person-following extension is supported by APF accepting a
moving goal.
"""

from .kalman_map import KalmanElevationMap
from .dijkstra import dijkstra
from .elastic_band import smooth_path
from .apf import APFPlanner, APFParams

__all__ = [
    "KalmanElevationMap",
    "dijkstra",
    "smooth_path",
    "APFPlanner",
    "APFParams",
]
