#!/usr/bin/env python3
"""ROS2 (Humble) node: lidar PointCloud2 -> nav2-consumable OccupancyGrid.

Subscribes to a sensor_msgs/PointCloud2, discretizes it into a 2.5D
traversability grid with the project's `discretize()` pipeline, and republishes
it as a nav_msgs/OccupancyGrid. A nav2 StaticLayer pointed at this topic folds
the grid into the costmap (see config/nav2_costmap_params.yaml).

The grid is published WITHOUT robot-footprint inflation (inflate_obstacles=False)
so nav2's own InflationLayer can inflate against the configured footprint without
double-counting.

Frames: the cloud is assumed already expressed in `target_frame` (e.g. `odom`).
If your driver publishes in a sensor frame, transform it upstream (a tf2 lookup
+ do_transform_cloud) before this node, or run a point_cloud2 transformer.

Parameters (all have defaults):
    input_topic        (str)   incoming PointCloud2            [/points]
    output_topic       (str)   outgoing OccupancyGrid          [/lidar_costmap]
    target_frame       (str)   frame_id stamped on the grid    [odom]
    resolution         (float) cell size, m                    [0.10]
    map_size           (float) square grid side, m (0=auto)    [0.0]
    z_min, z_max       (float) height clip on input points     [-1.0, 2.0]
    max_step           (float) robot max climbable step, m     [0.18]
    max_slope_deg      (float) robot max slope, deg            [30.0]
    max_roughness      (float) robot max within-cell std, m    [0.05]
    unknown_cost       (float) soft cost of unobserved cells   [50.0]
    publish_period     (float) min seconds between grids       [0.2]
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from lidar_pathplan import GridConfig, QuadrupedParams, discretize, cost_to_occupancy


class LidarCostmapNode(Node):
    def __init__(self):
        super().__init__("lidar_costmap_node")

        p = self.declare_parameters("", [
            ("input_topic", "/points"),
            ("output_topic", "/lidar_costmap"),
            ("target_frame", "odom"),
            ("resolution", 0.10),
            ("map_size", 0.0),
            ("z_min", -1.0),
            ("z_max", 2.0),
            ("max_step", 0.18),
            ("max_slope_deg", 30.0),
            ("max_roughness", 0.05),
            ("unknown_cost", 50.0),
            ("publish_period", 0.2),
        ])
        self._g = {param.name: param.value for param in p}

        self._robot = QuadrupedParams(
            max_step=self._g["max_step"],
            max_slope_deg=self._g["max_slope_deg"],
            max_roughness=self._g["max_roughness"],
        )
        self._last_pub = self.get_clock().now()

        # Latched (transient-local) publisher so a costmap StaticLayer that
        # subscribes late still receives the most recent grid.
        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(OccupancyGrid, self._g["output_topic"], latched)
        self._sub = self.create_subscription(
            PointCloud2, self._g["input_topic"], self._on_cloud, qos_profile=10)

        self.get_logger().info(
            "lidar_costmap: %s -> %s @ %.2f m, frame=%s" % (
                self._g["input_topic"], self._g["output_topic"],
                self._g["resolution"], self._g["target_frame"]))

    def _on_cloud(self, msg: PointCloud2):
        # Rate-limit: discretizing every scan is wasteful for a costmap source.
        now = self.get_clock().now()
        if (now - self._last_pub).nanoseconds < self._g["publish_period"] * 1e9:
            return

        pts = self._read_xyz(msg)
        if pts.shape[0] < 100:
            self.get_logger().warn("too few points (%d), skipping" % pts.shape[0])
            return

        bounds = None
        if self._g["map_size"] > 0.0:
            half = self._g["map_size"] / 2.0
            cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
            bounds = (cx - half, cy - half, cx + half, cy + half)

        config = GridConfig(
            resolution=self._g["resolution"],
            bounds=bounds,
            z_clip=(self._g["z_min"], self._g["z_max"]),
            unknown_cost=self._g["unknown_cost"],
            inflate_obstacles=False,   # let nav2's InflationLayer do it
        )

        try:
            grid = discretize(pts, config=config, robot=self._robot)
        except ValueError as e:
            self.get_logger().warn("discretize failed: %s" % e)
            return

        self._pub.publish(self._to_msg(grid, msg.header.stamp))
        self._last_pub = now

    def _read_xyz(self, msg: PointCloud2) -> np.ndarray:
        """Extract an (N,3) float64 array of finite x,y,z from a PointCloud2."""
        structured = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)
        if structured.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        xyz = np.stack([structured["x"], structured["y"], structured["z"]], axis=-1)
        xyz = np.asarray(xyz, dtype=np.float64)
        return xyz[np.isfinite(xyz).all(axis=1)]

    def _to_msg(self, grid, stamp) -> OccupancyGrid:
        occ = cost_to_occupancy(grid)            # (rows, cols) int8, row-major in +Y
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self._g["target_frame"]
        msg.info.resolution = float(grid.resolution)
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]
        msg.info.origin.position.x = float(grid.origin[0])
        msg.info.origin.position.y = float(grid.origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0      # grid axes aligned with the frame
        msg.data = occ.ravel(order="C").tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = LidarCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
