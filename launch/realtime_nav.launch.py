"""Launch the onboard real-time navigator.

    ros2 launch lidar_nav2 realtime_nav.launch.py \
        input_topic:=/utlidar/cloud_deskewed odom_topic:=/utlidar/robot_odom \
        waypoints:="[5.0, 0.0, 5.0, 5.0]" max_speed:=0.4

Send a runtime goal any time (overrides the waypoint queue):
    ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
        "{header: {frame_id: odom}, pose: {position: {x: 5.0, y: 2.0}}}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        ("input_topic", "/points"),
        ("odom_topic", "/odom"),
        ("goal_topic", "/goal_pose"),
        ("cmd_vel_topic", "/cmd_vel"),
        ("odom_frame", "odom"),
        ("waypoints", "[0.0]"),
        ("map_size", "8.0"),
        ("resolution", "0.10"),
        ("max_speed", "0.5"),
        ("goal_tolerance", "0.25"),
        ("replan_period", "0.3"),
    ]
    declared = [DeclareLaunchArgument(n, default_value=v) for n, v in args]
    node = Node(
        package="lidar_nav2",
        executable="realtime_nav",
        name="realtime_nav",
        output="screen",
        parameters=[{n: LaunchConfiguration(n) for n, _ in args}],
    )
    return LaunchDescription(declared + [node])
