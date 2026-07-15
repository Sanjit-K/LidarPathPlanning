#!/usr/bin/env bash
# Push this repo to the robot / lab computer and build it there.
#
#   ./deploy/deploy.sh <user@host> [workspace]     # default workspace: ~/go2_ws
#
# Example:
#   ./deploy/deploy.sh unitree@192.168.123.18
set -euo pipefail

DEST="${1:?usage: deploy.sh <user@host> [workspace]}"
WS="${2:-~/go2_ws}"
ROS_DISTRO="${ROS_DISTRO:-foxy}"     # override: ROS_DISTRO=humble ./deploy.sh ...
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> rsync source to $DEST:$WS/src/lidar_nav2_repo"
rsync -av --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'test-data' --exclude 'test-scans' --exclude 'examples/data' \
  --exclude 'out*.png' --exclude '*.html' --exclude 'report' \
  "$REPO_DIR/" "$DEST:$WS/src/lidar_nav2_repo/"

echo "==> colcon build on $DEST (ROS_DISTRO=$ROS_DISTRO)"
ssh "$DEST" "source /opt/ros/$ROS_DISTRO/setup.bash && cd $WS && \
  colcon build --symlink-install --packages-select lidar_nav2 && \
  source install/setup.bash && \
  ros2 pkg executables lidar_nav2"

echo "==> done. On the robot:"
echo "    source $WS/install/setup.bash"
echo "    ros2 run lidar_nav2 realtime_nav --ros-args -p input_topic:=/utlidar/cloud_deskewed \\"
echo "        -p odom_topic:=/utlidar/robot_odom -p waypoints:=\"[2.0, 0.0]\" -p max_speed:=0.2"
