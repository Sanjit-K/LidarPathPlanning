#!/usr/bin/env python3
"""Twist -> Unitree Go2 Sport-API bridge.

The Go2 does not accept raw /cmd_vel: locomotion goes through the Sport API as
unitree_api/msg/Request messages on /api/sport/request (JSON parameters). This
node bridges the navigator's geometry_msgs/Twist to Sport `Move` requests, with
a watchdog that issues `StopMove` if commands stop arriving.

    subscribes  /cmd_vel  geometry_msgs/Twist
    publishes   /api/sport/request  unitree_api/msg/Request

Requires the `unitree_api` message package from unitree_ros2
(https://github.com/unitreerobotics/unitree_ros2) built in the same workspace.

Parameters:
    cmd_vel_topic   (str)   [/cmd_vel]
    request_topic   (str)   [/api/sport/request]
    watchdog_period (float) seconds without a Twist before StopMove [0.5]
    max_vx, max_vyaw(float) hard clamps on what is forwarded [0.6, 1.2]
    enable          (bool)  if False, log instead of publishing (dry run) [True]
"""

import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    from unitree_api.msg import Request
    HAVE_UNITREE = True
except ImportError:      # allows the package to build/run elsewhere for dry runs
    HAVE_UNITREE = False

# Sport-mode API ids (unitree_sdk2 sport client)
SPORT_API_ID_MOVE = 1008
SPORT_API_ID_STOPMOVE = 1003


class Go2TwistBridge(Node):
    def __init__(self):
        super().__init__("go2_twist_bridge")
        p = self.declare_parameters("", [
            ("cmd_vel_topic", "/cmd_vel"),
            ("request_topic", "/api/sport/request"),
            ("watchdog_period", 0.5),
            ("max_vx", 0.6),
            ("max_vyaw", 1.2),
            ("enable", True),
        ])
        self._g = {x.name: x.value for x in p}

        if not HAVE_UNITREE and self._g["enable"]:
            raise RuntimeError(
                "unitree_api messages not found. Build unitree_ros2 in this "
                "workspace, or run with -p enable:=false for a dry run.")

        self.create_subscription(Twist, self._g["cmd_vel_topic"], self.on_twist, 10)
        if HAVE_UNITREE:
            self.pub = self.create_publisher(Request, self._g["request_topic"], 10)
        self._last_cmd_time = self.get_clock().now()
        self._stopped = True
        self.create_timer(self._g["watchdog_period"] / 2.0, self.watchdog)
        self.get_logger().info("go2_twist_bridge up (enable=%s)" % self._g["enable"])

    def _request(self, api_id: int, params: dict):
        if not self._g["enable"] or not HAVE_UNITREE:
            self.get_logger().info("[dry-run] api %d %s" % (api_id, params))
            return
        msg = Request()
        msg.header.identity.api_id = api_id
        msg.parameter = json.dumps(params)
        self.pub.publish(msg)

    def on_twist(self, msg: Twist):
        vx = max(-self._g["max_vx"], min(self._g["max_vx"], msg.linear.x))
        vy = max(-0.4, min(0.4, msg.linear.y))
        wz = max(-self._g["max_vyaw"], min(self._g["max_vyaw"], msg.angular.z))
        self._last_cmd_time = self.get_clock().now()
        if abs(vx) < 1e-3 and abs(vy) < 1e-3 and abs(wz) < 1e-3:
            if not self._stopped:
                self._request(SPORT_API_ID_STOPMOVE, {})
                self._stopped = True
            return
        self._stopped = False
        self._request(SPORT_API_ID_MOVE, {"x": vx, "y": vy, "z": wz})

    def watchdog(self):
        dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        if dt > self._g["watchdog_period"] and not self._stopped:
            self.get_logger().warn("cmd_vel silent %.2fs -> StopMove" % dt)
            self._request(SPORT_API_ID_STOPMOVE, {})
            self._stopped = True


def main(args=None):
    rclpy.init(args=args)
    node = Go2TwistBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
