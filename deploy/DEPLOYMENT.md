# Go2 Deployment Runbook

Step-by-step guide to run the real-time navigator on a Unitree Go2. Written to be
followed at the lab with no prior setup on the robot side.

---

## 0. Architecture: where does the code run?

Two options — pick based on which Go2 you have:

| Option | Hardware | When |
|---|---|---|
| **A. External computer** (recommended first) | Ubuntu 22.04 laptop/NUC/Jetson connected to the Go2 via **Ethernet** | Works with any Go2 (Air/Pro/EDU). The Go2 publishes lidar + odometry over DDS; your computer runs the navigator and sends motion requests back. |
| **B. Onboard (EDU only)** | The Go2 EDU's internal Jetson (ssh access) | Only EDU units expose onboard compute. Same steps, executed on the Jetson. |

Start with Option A — it is easier to debug (RViz locally) and requires no changes
on the robot.

## 1. Prepare the computer (once)

Ubuntu 20.04 + **ROS 2 Foxy** (the distro on the robot / used by unitree_ros2;
the code is distro-portable — Python 3.8+, no `sensor_msgs_py` dependency — so
Humble on 22.04 also works if that's what the lab machine has):

```bash
# ROS 2 Foxy (skip if installed) — https://docs.ros.org/en/foxy/Installation.html
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe
sudo apt install ros-foxy-desktop ros-foxy-rmw-cyclonedds-cpp python3-colcon-common-extensions -y
```

> Note: Foxy is end-of-life upstream — fine for the lab, but pin what works and
> don't expect apt updates.

Workspace with this repo + Unitree's message/driver package:

```bash
mkdir -p ~/go2_ws/src && cd ~/go2_ws/src
git clone <YOUR-REPO-URL> lidar_nav2_repo        # this project
git clone https://github.com/unitreerobotics/unitree_ros2.git
cd ~/go2_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 2. Connect to the Go2

1. Ethernet cable from the computer to the Go2's RJ45 port.
2. Set your wired interface to a static IP on the robot's subnet, e.g.
   `192.168.123.99/24` (the Go2 is `192.168.123.161`).
3. Unitree uses **CycloneDDS** on a specific interface. Export (adjust `eth0`):

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
  <NetworkInterface name="eth0" priority="default" multicast="default"/>
</Interfaces></General></Domain></CycloneDDS>'
```

4. Verify you see the robot's topics:

```bash
ros2 topic list            # expect /utlidar/robot_odom, /api/sport/request, ...
ros2 topic echo /utlidar/robot_odom --once
```

5. **Enable the lidar cloud publisher** — it is OFF by default (verified on our
   unit; `/utlidar/cloud` does not appear until switched on):

```bash
ros2 topic pub --once /utlidar/switch std_msgs/msg/String "data: 'ON'"
ros2 topic hz /utlidar/cloud       # now ~10 Hz, ~1.4k pts/sweep
```

> On our robot the onboard env is already set up: `ssh unitree@<ip>` then
> `source /opt/ros/foxy/setup.bash; source ~/cyclonedds_ws/install/setup.bash;`
> `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml`
> (the robot's `~/.bashrc` does this interactively). Note: this firmware has
> **no `/utlidar/cloud_deskewed`** — use `/utlidar/cloud` with
> `cloud_in_odom_frame:=false` so the navigator transforms by the odom pose.

## 3. Verify frames (important, 2 minutes)

The navigator needs the cloud and the odometry in the **same frame**.

```bash
ros2 topic echo /utlidar/cloud_deskewed --once | head -5    # note frame_id
ros2 topic echo /utlidar/robot_odom --once | head -8        # note frame_id
```

- Cloud frame == odom frame → launch with `cloud_in_odom_frame:=true` (default).
- Cloud is in the **base/sensor** frame → launch with `cloud_in_odom_frame:=false`
  (the node rotates points by the odom pose itself; no tf setup needed).

## 4. Dry run — robot standing, motion DISABLED

Everything runs except actual motion (`enable:=false` on the bridge logs
requests instead of sending them):

```bash
# terminal 1: the navigator (goal 2 m ahead as a first test)
ros2 run lidar_nav2 realtime_nav --ros-args \
  -p input_topic:=/utlidar/cloud_deskewed -p odom_topic:=/utlidar/robot_odom \
  -p waypoints:="[2.0, 0.0]" -p max_speed:=0.2 -p cloud_in_odom_frame:=true

# terminal 2: bridge in DRY-RUN
ros2 run lidar_nav2 go2_twist_bridge --ros-args -p enable:=false

# terminal 3: watch it think
ros2 topic echo /cmd_vel
rviz2   # add /planned_path (Path) and /local_costmap (OccupancyGrid), frame 'odom'
```

Checklist before going live:
- [ ] `/local_costmap` in RViz shows sensible free/obstacle structure around the dog
- [ ] `/planned_path` points from the dog toward the goal and avoids obstacles
- [ ] `/cmd_vel` shows small forward velocities (≤ max_speed), zero when you
      cover the lidar or kill the navigator (safety stop)
- [ ] bridge dry-run log shows `Move {x: ..., z: ...}` requests

## 5. Live — first motion

Safety: open area, ≥2 m clearance around the robot, one person on the Unitree
remote ready to take over (manual sticks override), `max_speed` at 0.2.

```bash
# make sure the dog is standing (use the remote / app), then:
ros2 run lidar_nav2 go2_twist_bridge --ros-args -p enable:=true -p max_vx:=0.25
```

The dog should walk ~2 m and stop ("goal reached"). Then:
- Longer waypoint routes: `-p waypoints:="[3.0, 0.0, 3.0, 3.0, 0.0, 3.0]"`.
- Runtime goals from RViz: set RViz's 2D Goal Pose topic to `/goal_pose`.
- Put a box in its path mid-walk — the next replan (~0.3 s) routes around it.

## 6. Tuning on the robot

| Symptom | Change |
|---|---|
| Hesitant / stops often | raise `scan_timeout`, lower `replan_period` |
| Cuts corners too close | raise `robot_radius` (inflation) |
| Won't cross a doorway | lower `robot_radius` or `resolution` (finer grid) |
| Jerky heading | lower `k_yaw` / `max_yaw_rate` in NavConfig |
| Flags flat ground as obstacle | raise `max_roughness` (lidar noise), see README tuning table |
| Too timid about unseen areas | it's by design (`unknown_is_lethal`); widen `map_size` so more is observed |

## 7. Troubleshooting

- **No topics visible** → wrong interface in `CYCLONEDDS_URI`, or IP not on
  192.168.123.x. `ping 192.168.123.161` first.
- **`unitree_api` import error** → unitree_ros2 not built/sourced in this shell.
- **Dog ignores Move requests** → it must be in sport (walking) mode and standing;
  use the app/remote to stand it up first.
- **Navigator says `scan timeout - stopped`** → cloud topic name wrong or not
  arriving; check `ros2 topic hz`.
- **Path goes through walls in RViz** → frame mismatch; redo step 3.
```
