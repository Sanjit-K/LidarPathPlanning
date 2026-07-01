"""Launch the lidar->OccupancyGrid costmap source node.

    ros2 launch lidar_nav2 lidar_costmap.launch.py \
        input_topic:=/velodyne_points target_frame:=odom resolution:=0.1

This launches only the map-source node. Wire the published grid into your nav2
costmaps using config/nav2_costmap_params.yaml (a StaticLayer pointed at
output_topic), then bring up nav2 as usual.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        ("input_topic", "/points"),
        ("output_topic", "/lidar_costmap"),
        ("target_frame", "odom"),
        ("resolution", "0.10"),
        ("map_size", "0.0"),
        ("max_step", "0.18"),
        ("max_slope_deg", "30.0"),
        ("max_roughness", "0.05"),
        ("unknown_cost", "50.0"),
        ("publish_period", "0.2"),
    ]
    declared = [DeclareLaunchArgument(name, default_value=default) for name, default in args]

    params = {name: _typed(name, LaunchConfiguration(name)) for name, _ in args}

    node = Node(
        package="lidar_nav2",
        executable="costmap_node",
        name="lidar_costmap_node",
        output="screen",
        parameters=[params],
    )
    return LaunchDescription(declared + [node])


def _typed(name, value):
    # LaunchConfiguration resolves to strings; ROS param typing coerces numeric
    # strings fine for these declared float params, so pass through directly.
    return value
