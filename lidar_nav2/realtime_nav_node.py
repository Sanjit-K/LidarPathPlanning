#!/usr/bin/env python3
"""ROS2 real-time navigator for the quadruped (Foxy+; no distro-specific APIs).

Onboard loop: the START is always the dog's own position (from odometry); the
GOAL is a world/odom coordinate -- preprogrammed via parameters or sent at
runtime on a topic (e.g. RViz "2D Goal Pose"). Each new lidar scan rebuilds a
robot-centric traversability grid and the path is replanned continuously, so
the dog routes around obstacles as it discovers them.

    subscribes  /points     sensor_msgs/PointCloud2   (in `odom_frame`)
                /odom       nav_msgs/Odometry         (robot pose in `odom_frame`)
                /goal_pose  geometry_msgs/PoseStamped (optional runtime goal)
    publishes   /cmd_vel        geometry_msgs/Twist   (v_x forward, yaw rate)
                /planned_path   nav_msgs/Path         (for RViz)
                /local_costmap  nav_msgs/OccupancyGrid (for RViz)

Preprogrammed goals: set `waypoints:=[x1, y1, x2, y2, ...]` -- the node walks
them in order, advancing when each is reached.

Example (Unitree Go2 style topics):
    ros2 run lidar_nav2 realtime_nav --ros-args \
        -p input_topic:=/utlidar/cloud_deskewed -p odom_topic:=/utlidar/robot_odom \
        -p waypoints:="[5.0, 0.0, 5.0, 5.0]" -p max_speed:=0.4

Frames: the cloud must be in `odom_frame` (matching the odometry). If your
driver outputs the cloud in the sensor frame, transform it upstream with tf2.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import PointCloud2

from lidar_pathplan import QuadrupedParams, cost_to_occupancy
from .navigator_core import NavConfig, RealtimeNavigator
from .pc2 import cloud_to_xyz    # Foxy-compatible (no sensor_msgs_py)


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RealtimeNavNode(Node):
    def __init__(self):
        super().__init__("realtime_nav")
        p = self.declare_parameters("", [
            ("input_topic", "/points"),
            ("odom_topic", "/odom"),
            ("goal_topic", "/goal_pose"),
            ("cmd_vel_topic", "/cmd_vel"),
            ("odom_frame", "odom"),
            ("cloud_in_odom_frame", True), # False: cloud is in base frame -> rotate by odom pose
            # Sensor->base mount rotation, ZYX euler degrees [roll, pitch, yaw].
            # Applied to the raw cloud BEFORE the odom transform, so it is only
            # meaningful with cloud_in_odom_frame:=false. Default is the Go2 L1
            # calibrated from real scans (mounted upside-down, pitched ~14 deg).
            # [0.0, 0.0, 0.0] disables it.
            ("mount_rpy", [-177.93, 13.77, 176.36]),
            ("waypoints", [0.0]),          # flat [x1,y1,x2,y2,...]; [0.0] = none
            ("map_size", 8.0),
            ("resolution", 0.10),
            ("z_min", -1.0), ("z_max", 1.5),
            ("goal_tolerance", 0.25),
            ("max_speed", 0.5), ("max_yaw_rate", 1.2),
            ("lookahead", 0.6),
            ("replan_period", 0.3),        # s between replans
            ("control_period", 0.1),       # s between cmd_vel ticks
            ("min_range", 0.45),           # drop self-hits closer than this (m)
            ("cloud_buffer", 2.0),         # accumulate scans over this window (s)
            ("max_step", 0.18), ("max_slope_deg", 30.0),
            ("max_roughness", 0.05), ("robot_radius", 0.30),
        ])
        g = {x.name: x.value for x in p}
        self._g = g

        cfg = NavConfig(map_size=g["map_size"], resolution=g["resolution"],
                        z_min=g["z_min"], z_max=g["z_max"],
                        goal_tolerance=g["goal_tolerance"],
                        lookahead=g["lookahead"], max_speed=g["max_speed"],
                        max_yaw_rate=g["max_yaw_rate"],
                        min_range=g["min_range"], cloud_buffer=g["cloud_buffer"])
        robot = QuadrupedParams(radius=g["robot_radius"], max_step=g["max_step"],
                                max_slope_deg=g["max_slope_deg"],
                                max_roughness=g["max_roughness"])
        self.nav = RealtimeNavigator(cfg, robot)

        # preprogrammed waypoint queue
        wp = list(g["waypoints"])
        self.waypoints = [(wp[i], wp[i + 1]) for i in range(0, len(wp) - 1, 2)] \
            if len(wp) >= 2 else []
        if self.waypoints:
            self.nav.set_goal(*self.waypoints[0])
            self.get_logger().info("waypoints: %s" % self.waypoints)

        self.pose = None          # (x, y, yaw)
        self._quat = None
        self._pos3 = (0.0, 0.0, 0.0)

        # precompute the sensor->base mount rotation (identity if disabled)
        rpy = [math.radians(v) for v in g["mount_rpy"]]
        self._R_mount = self._rpy_to_R(*rpy) if any(abs(v) > 1e-9 for v in rpy) else None
        if self._R_mount is not None and not g["cloud_in_odom_frame"]:
            self.get_logger().info("mount_rpy: %s deg" % (list(g["mount_rpy"]),))

        self.create_subscription(PointCloud2, g["input_topic"], self.on_cloud, 5)
        self.create_subscription(Odometry, g["odom_topic"], self.on_odom, 20)
        self.create_subscription(PoseStamped, g["goal_topic"], self.on_goal, 5)
        self.pub_cmd = self.create_publisher(Twist, g["cmd_vel_topic"], 10)
        self.pub_path = self.create_publisher(Path, "planned_path", 5)
        latched = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(OccupancyGrid, "local_costmap", latched)

        self.create_timer(g["replan_period"], self.tick_replan)
        self.create_timer(g["control_period"], self.tick_control)
        self._last_status = ""
        self.get_logger().info("realtime_nav up: cloud=%s odom=%s goal=%s" % (
            g["input_topic"], g["odom_topic"], g["goal_topic"]))

    # ------------------------------------------------------------ callbacks
    def on_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.pose = (pos.x, pos.y, yaw_from_quat(msg.pose.pose.orientation))
        self._quat = msg.pose.pose.orientation
        self._pos3 = (pos.x, pos.y, pos.z)

    def on_goal(self, msg: PoseStamped):
        self.waypoints = []                     # runtime goal overrides the queue
        self.nav.set_goal(msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info("new goal: (%.2f, %.2f)" % (
            msg.pose.position.x, msg.pose.position.y))

    def on_cloud(self, msg: PointCloud2):
        if self.pose is None:
            return
        xyz = cloud_to_xyz(msg)
        if xyz.shape[0] == 0:
            return
        if not self._g["cloud_in_odom_frame"]:
            if self._R_mount is not None:      # sensor frame -> body frame first
                xyz = xyz @ self._R_mount.T
            xyz = self._body_to_odom(xyz)
        self.nav.update_cloud(xyz, (self.pose[0], self.pose[1]))

    @staticmethod
    def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX euler (radians) -> rotation matrix R = Rz(yaw) Ry(pitch) Rx(roll)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _body_to_odom(self, xyz: np.ndarray) -> np.ndarray:
        """Rotate+translate base-frame points into odom using the latest pose."""
        q = self._quat
        # quaternion -> rotation matrix
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return xyz @ R.T + np.asarray(self._pos3)

    # ------------------------------------------------------------ timers
    def tick_replan(self):
        if self.pose is None or self.nav.grid is None:
            return
        self.nav.replan((self.pose[0], self.pose[1]))
        self.publish_path()
        self.publish_map()

    def tick_control(self):
        if self.pose is None:
            return
        vx, wz, status = self.nav.compute_cmd(self.pose)
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.angular.z = float(wz)
        self.pub_cmd.publish(cmd)
        if status != self._last_status:
            self.get_logger().info("status: %s" % status)
            self._last_status = status
        # advance the preprogrammed queue
        if status == "goal reached" and self.waypoints:
            self.waypoints.pop(0)
            if self.waypoints:
                self.nav.set_goal(*self.waypoints[0])
                self.get_logger().info("next waypoint: %s" % (self.waypoints[0],))

    # ------------------------------------------------------------ outputs
    def publish_path(self):
        msg = Path()
        msg.header.frame_id = self._g["odom_frame"]
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.nav.path_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def publish_map(self):
        g = self.nav.grid
        if g is None:
            return
        occ = cost_to_occupancy(g)
        msg = OccupancyGrid()
        msg.header.frame_id = self._g["odom_frame"]
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(g.resolution)
        msg.info.width = g.shape[1]
        msg.info.height = g.shape[0]
        msg.info.origin.position.x = float(g.origin[0])
        msg.info.origin.position.y = float(g.origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = occ.ravel(order="C").tolist()
        self.pub_map.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
