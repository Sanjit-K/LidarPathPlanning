"""Quadruped traversability parameters.

These define what terrain the robot can negotiate, and turn raw geometry
(step height, slope) into the thresholds the cost grid uses to mark cells
lethal or expensive.
"""

from dataclasses import dataclass
import math


@dataclass
class QuadrupedParams:
    """Mobility limits for a medium quadruped (defaults ~ Unitree Go2 / Spot class).

    Lengths in meters, angles in degrees, used by elevation_grid to classify cells.
    """

    # Footprint radius used to inflate obstacles so the body/legs clear them.
    radius: float = 0.30

    # Largest vertical step the robot can climb in a single cell-to-cell move.
    max_step: float = 0.18

    # Largest terrain slope the robot can stand on / traverse.
    max_slope_deg: float = 30.0

    # Local roughness (within-cell height std) that still counts as ground.
    # Above this the cell is treated as an obstacle (clutter, vegetation, edge).
    max_roughness: float = 0.05

    @property
    def max_slope_rad(self) -> float:
        return math.radians(self.max_slope_deg)

    @property
    def max_slope_tan(self) -> float:
        """Slope limit as rise/run, convenient for comparing against gradients."""
        return math.tan(self.max_slope_rad)
