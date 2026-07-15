#!/usr/bin/env python3
"""Live discretization + click-to-go navigator: runs ON the Go2.

Serves a browser page showing the 2.5D traversability grid as elevation planes
in real time. CLICK anywhere on the map to set the navigation goal: the onboard
navigator A*-replans continuously on the live grid, and the dog tracks a
geometric lookahead point on the latest path, publishing commands through the
lab's existing go2_controller AvoidanceClient to
/api/obstacles_avoid/request.

Pipeline (all onboard): /utlidar/cloud -> mount rotation -> odom pose ->
rolling 2 s cloud buffer -> lidar_pathplan.discretize (robot-centered 8 m
window) -> A* -> geometric path follower -> obstacles_avoid API.

Safety: velocities clamped, stale-scan timeout stops the dog, STOP button in
the UI releases the API override so the manual remote regains control.
NOTE: while tracking, the firmware treats API commands as the remote
(is_remote_commands_from_api=true) -- use the UI STOP to hand control back.

Needs lidar_pathplan/ and navigator_core.py importable (deployed to /tmp),
and the go2_controller package at /home/unitree/micah_ws/src/go2_controller.

On the robot (env sourced):   python3 live_grid.py [port]
On your computer:             open http://<robot-ip>:8767/

Foxy / Python 3.8, stdlib + numpy + rclpy (+ unitree_api for motion).
"""
import base64
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/unitree/micah_ws/src/go2_controller")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from navigator_core import NavConfig, RealtimeNavigator   # ROS-free core
from global_costmap import SparseGlobalCostmap

try:
    from unitree_api.msg import Request
    from go2_controller.trajectory_follow import AvoidanceClient
    HAVE_CTRL = True
except ImportError:                      # viewer still works, motion disabled
    HAVE_CTRL = False

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8767
MIN_RANGE, MAX_RANGE = 0.5, 12.0
MOUNT_RPY_DEG = [-177.93, 13.77, 176.36]   # calibrated Go2 L1 sensor->base
GRID_PERIOD = 0.33                         # s between grid broadcasts
GLOBAL_PERIOD = 1.0                        # s between full sparse-map broadcasts
SAVE_PERIOD = 5.0                          # s between atomic map saves
DEFAULT_FUSION_RADIUS = 2.5                 # refresh existing cells only this near (m)
MIN_FUSION_RADIUS, MAX_FUSION_RADIUS = 0.5, 6.0
GLOBAL_MAP_PATH = os.environ.get(
    "LIDAR_GLOBAL_MAP", "/home/unitree/lidar_maps/global_costmap_latest.npz")
MAX_V, MAX_W = 0.4, 1.0                    # hard clamps on published cmds
GOAL_TOL = 0.3                             # goal reached within this (m)

_DT = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def cloud_to_xyz(msg):
    names, formats, offs = [], [], []
    for f in msg.fields:
        names.append(f.name)
        formats.append(("<" if not msg.is_bigendian else ">") + _DT[f.datatype])
        offs.append(f.offset)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offs,
                      "itemsize": msg.point_step})
    n = msg.width * msg.height
    if msg.row_step == msg.point_step * msg.width:
        arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    else:
        rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.row_step)
        arr = rows[:, :msg.point_step * msg.width].reshape(-1).view(dtype)
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


R_MOUNT = rpy_to_R(*[math.radians(v) for v in MOUNT_RPY_DEG])

STATE = {
    "seq": -1, "frame": b"",           # latest raw sweep (sensor frame)
    "gseq": -1, "grid": b"",           # latest discretized grid + plan
    "mseq": -1, "map": b"",             # persistent sparse global costmap
    "lock": threading.Lock(),
    "fev": threading.Event(), "gev": threading.Event(), "mev": threading.Event(),
}
NODE = None                            # set in main; used by HTTP handlers


class LiveGrid(Node):
    def __init__(self):
        super().__init__("live_grid")
        self.pose = None               # [x y z qx qy qz qw]
        self.yaw = 0.0
        self.status = "no goal - click the map"
        self._stopped = True
        self._override = False         # API override (1004) currently held
        # min_range 0 here: the sensor-range filter below already drops self-hits
        self.nav = RealtimeNavigator(NavConfig(
            map_size=8.0, resolution=0.10, min_range=0.0, cloud_buffer=2.0,
            scan_timeout=1.0))
        if os.environ.get("LIDAR_RESUME_MAP") == "1" and os.path.exists(GLOBAL_MAP_PATH):
            self.global_map = SparseGlobalCostmap.load(GLOBAL_MAP_PATH)
            self.get_logger().info(
                "resumed global costmap: %d cells from %s" %
                (len(self.global_map), GLOBAL_MAP_PATH))
        else:
            self.global_map = SparseGlobalCostmap(self.nav.cfg.resolution)
        self.fusion_radius = DEFAULT_FUSION_RADIUS
        self.create_subscription(PointCloud2, "/utlidar/cloud", self.on_cloud, 5)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, 20)
        if HAVE_CTRL:
            self.avoid = AvoidanceClient()
            self.avoid_pub = self.create_publisher(Request, "/api/obstacles_avoid/request", 10)
            self.get_logger().info("go2_controller loaded - holonomic motion ENABLED")
        else:
            self.get_logger().warn("go2_controller/unitree_api not found - motion DISABLED")
        pub = self.create_publisher(String, "/utlidar/switch", 1)
        m = String(); m.data = "ON"
        for _ in range(3):
            pub.publish(m)
        self.create_timer(0.3, self.tick_replan)
        self.create_timer(0.1, self.tick_control)

    # --------------------------------------------------------------- inputs
    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def on_cloud(self, m):
        if self.pose is None:
            return
        xyz = cloud_to_xyz(m).astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > MIN_RANGE) & (r < MAX_RANGE)]
        if xyz.shape[0] == 0:
            return
        pose = list(self.pose)
        Ro = quat_to_R(*pose[3:])
        world = (xyz @ R_MOUNT.T) @ Ro.T + np.asarray(pose[:3])
        self.nav.update_cloud(world, (pose[0], pose[1]))
        # publish the raw sweep for the point overlay
        q16 = np.clip(np.round(xyz * 100.0), -32000, 32000).astype("<i2")
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with STATE["lock"]:
            STATE["seq"] += 1
            STATE["frame"] = json.dumps({
                "seq": STATE["seq"], "t": round(t, 3),
                "pose": [round(v, 4) for v in pose],
                "b64": base64.b64encode(q16.tobytes()).decode("ascii"),
            }).encode()
        STATE["fev"].set()
        STATE["fev"].clear()

    # --------------------------------------------------------- plan + drive
    def tick_replan(self):
        if self.pose is None or self.nav.grid is None or self.nav.goal is None:
            return
        # Replanning replaces only the path.  The geometric follower below
        # selects its lookahead from the latest path on every control tick, so
        # it does not restart a time-based trajectory every 0.3 seconds.
        self.nav.replan((self.pose[0], self.pose[1]))

    def _avoid_req(self, api_id, param):
        msg = Request()
        msg.header.identity.api_id = api_id
        msg.parameter = param
        self.avoid_pub.publish(msg)

    def _move(self, vx, vy, vyaw):
        self._avoid_req(*self.avoid.move_func(
            max(-MAX_V, min(MAX_V, vx)),
            max(-MAX_V, min(MAX_V, vy)),
            max(-MAX_W, min(MAX_W, vyaw))))

    def _acquire_override(self):
        if not self._override:
            self._avoid_req(*self.avoid.set_obstacle_avoidance(True))
            self._avoid_req(*self.avoid.use_api_command(True))
            self._override = True
            self.get_logger().info("obstacles_avoid API override ON")

    def _release_override(self):
        if self._override:
            self._move(0.0, 0.0, 0.0)
            self._avoid_req(*self.avoid.use_api_command(False))
            self._override = False
            self.get_logger().info("obstacles_avoid API override released")

    def tick_control(self):
        if self.pose is None:
            return
        if self.nav.goal is None:
            self.status = "no goal - click the map"
            if HAVE_CTRL:
                self._release_override()
            return
        if not HAVE_CTRL:
            self.status = "motion disabled: go2_controller not found"
            return
        gx, gy = self.nav.goal
        if math.hypot(gx - self.pose[0], gy - self.pose[1]) <= GOAL_TOL:
            self.status = "goal reached"
            self.nav.goal = None
            self.nav.path_world = []
            self._release_override()
            return
        now = time.monotonic()
        if now - self.nav._last_scan_time > self.nav.cfg.scan_timeout:
            self.status = "scan timeout - stopped"
            if self._override:
                self._move(0.0, 0.0, 0.0)
            return
        if not self.nav.path_world:
            self.status = self.nav.status or "planning..."
            if self._override:
                self._move(0.0, 0.0, 0.0)
            return
        self._acquire_override()
        vx, wz, status = self.nav.compute_cmd(
            (self.pose[0], self.pose[1], self.yaw), now=now)
        self._move(vx, 0.0, wz)
        self.status = status

    def stop(self):
        """UI STOP: drop the goal, zero velocity, hand control back to remote."""
        self.nav.goal = None
        self.nav.path_world = []
        if HAVE_CTRL:
            self._release_override()
        self.status = "stopped by user"

    def set_fusion_radius(self, metres):
        """Set the live radius used to refresh already-mapped global cells."""
        self.fusion_radius = max(
            MIN_FUSION_RADIUS, min(MAX_FUSION_RADIUS, float(metres)))
        self.get_logger().info(
            "global existing-cell refresh radius: %.1f m" % self.fusion_radius)
        return self.fusion_radius


def grid_broadcaster(node):
    """Broadcast the local plan and fuse/save the persistent global map."""
    last_global = 0.0
    last_save = 0.0
    while True:
        time.sleep(GRID_PERIOD)
        g = node.nav.grid
        if g is None or node.pose is None:
            continue
        node.global_map.update(
            g, sensor_xy=(node.pose[0], node.pose[1]),
            existing_update_radius=node.fusion_radius)
        elev = np.where(np.isfinite(g.elevation),
                        np.clip(np.round(g.elevation * 100.0), -32000, 32000),
                        -32768).astype("<i2")
        cls = g.classes.astype(np.uint8)
        goal = node.nav.goal
        path = [[round(px, 2), round(py, 2)] for px, py in node.nav.path_world][:400]
        with STATE["lock"]:
            STATE["gseq"] += 1
            STATE["grid"] = json.dumps({
                "seq": STATE["gseq"],
                "origin": [round(float(g.origin[0]), 3), round(float(g.origin[1]), 3)],
                "res": g.resolution, "h": g.shape[0], "w": g.shape[1],
                "elev": base64.b64encode(elev.tobytes()).decode("ascii"),
                "cls": base64.b64encode(cls.tobytes()).decode("ascii"),
                "pose": [round(v, 4) for v in node.pose],
                "goal": [round(goal[0], 2), round(goal[1], 2)] if goal else None,
                "path": path,
                "status": node.status,
            }).encode()
        STATE["gev"].set()
        STATE["gev"].clear()

        now = time.monotonic()
        if now - last_global >= GLOBAL_PERIOD:
            snap = node.global_map.snapshot()
            cells = snap["cells"].astype("<i4", copy=False)
            map_elev = np.clip(np.round(snap["elevation"] * 100.0),
                               -32000, 32000).astype("<i2")
            map_cls = snap["classes"].astype(np.uint8, copy=False)
            with STATE["lock"]:
                STATE["mseq"] += 1
                STATE["map"] = json.dumps({
                    "seq": STATE["mseq"], "revision": snap["revision"],
                    "res": snap["resolution"], "count": int(cells.shape[0]),
                    "cells": base64.b64encode(cells.tobytes()).decode("ascii"),
                    "elev": base64.b64encode(map_elev.tobytes()).decode("ascii"),
                    "cls": base64.b64encode(map_cls.tobytes()).decode("ascii"),
                    "pose": [round(v, 4) for v in node.pose],
                    "fusion_radius": round(node.fusion_radius, 1),
                    "saved_to": GLOBAL_MAP_PATH,
                }).encode()
            STATE["mev"].set()
            STATE["mev"].clear()
            last_global = now

        if now - last_save >= SAVE_PERIOD:
            try:
                n_saved = node.global_map.save(GLOBAL_MAP_PATH)
                node.get_logger().info(
                    "saved global costmap: %d cells -> %s" % (n_saved, GLOBAL_MAP_PATH))
            except Exception as exc:
                node.get_logger().warn("global costmap save failed: %s" % exc)
            last_save = now


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if u.path in ("/frame", "/grid", "/global"):
            if u.path == "/frame":
                key, ev, body_key = "seq", "fev", "frame"
            elif u.path == "/grid":
                key, ev, body_key = "gseq", "gev", "grid"
            else:
                key, ev, body_key = "mseq", "mev", "map"
            since = int(parse_qs(u.query).get("since", ["-1"])[0])
            if STATE[key] <= since:
                STATE[ev].wait(timeout=1.0)
            with STATE["lock"]:
                if STATE[key] <= since:
                    self._send(204)
                else:
                    self._send(200, STATE[body_key])
            return
        if u.path == "/goal":
            qs = parse_qs(u.query)
            try:
                x, y = float(qs["x"][0]), float(qs["y"][0])
            except (KeyError, ValueError):
                self._send(400, b'{"ok":false}')
                return
            NODE.nav.set_goal(x, y)
            NODE.get_logger().info("web goal: (%.2f, %.2f)" % (x, y))
            self._send(200, b'{"ok":true}')
            return
        if u.path == "/stop":
            NODE.stop()
            NODE.get_logger().info("web STOP")
            self._send(200, b'{"ok":true}')
            return
        if u.path == "/fusion_radius":
            qs = parse_qs(u.query)
            try:
                metres = float(qs["meters"][0])
                actual = NODE.set_fusion_radius(metres)
            except (KeyError, ValueError):
                self._send(400, b'{"ok":false}')
                return
            self._send(200, json.dumps({
                "ok": True, "meters": actual}).encode())
            return
        self._send(404, b"{}")


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Go2 live nav</title>
<style>
 html,body{margin:0;height:100%;background:#0e1116;color:#d7dae0;font-family:system-ui,sans-serif;overflow:hidden}
 #hud{position:fixed;top:12px;left:12px;background:#171b22;border:1px solid #2a2f38;border-radius:8px;
   padding:10px 12px;font-size:13px;line-height:1.7;z-index:2;max-width:380px}
 #bar{position:fixed;left:50%;transform:translateX(-50%);bottom:14px;background:#171b22;
   border:1px solid #2a2f38;border-radius:10px;padding:9px 14px;z-index:2;display:flex;
   gap:12px;align-items:center;font-size:13px;white-space:nowrap}
 button{background:#2b3442;color:#e6e9ee;border:0;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}
 button:hover{background:#38445a}
 #stop{background:#8e2f2f;font-weight:600} #stop:hover{background:#b03a3a}
 canvas{display:block} label{user-select:none} .dim{color:#8b93a0}
 #dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#e05555;margin-right:6px}
 #dot.on{background:#4fc06a}
 #status{color:#7fd0ff}
 .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
</style></head><body>
<div id="hud">
 <b><span id="dot"></span>Unitree Go2 &mdash; LIVE nav &middot; click map to send</b><br>
 <span id="status">connecting&hellip;</span><br>
 <span id="stats" class="dim"></span><br>
 <span class="sw" style="background:#4a9e57"></span>traversable
 <span class="sw" style="background:#c24444"></span>lethal
 <span class="sw" style="background:#4f8ef7"></span>A* path
 <span class="sw" style="background:#41e0c8;border-radius:50%"></span>goal
</div>
<div id="bar">
 <button id="stop">&#9632; STOP</button>
 <label><input id="follow" type="checkbox" checked> follow robot</label>
 <label><input id="showpts" type="checkbox" checked> sweep points</label>
 <label><input id="showgrid" type="checkbox" checked> grid planes</label>
 <label>refresh radius <input id="fusion" type="range" min="0.5" max="6" step="0.1" value="2.5">
  <span id="fusionval">2.5 m</span></label>
</div>
<canvas id="cv"></canvas>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const MOUNT_RPY=__MOUNT_RPY__, deg=Math.PI/180;
function rpyToR(r,p,y){
 const cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);
 return [cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr, sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr, -sp,cp*sr,cp*cr];}
function quatToR(x,y,z,w){
 return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w), 2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
         2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];}
function mul3(a,b){const o=new Array(9);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)o[i*3+j]=a[i*3]*b[j]+a[i*3+1]*b[3+j]+a[i*3+2]*b[6+j];
 return o;}
const Rm=(()=>{const[r,p,y]=MOUNT_RPY.map(v=>v*deg);return rpyToR(r,p,y);})();
function decode(b64){const s=atob(b64),u=new Uint8Array(s.length);
 for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}

let origin=null;   // world offset so the scene stays near (0,0)

const canvas=document.getElementById('cv');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.05,500);
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
function resize(){renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();}

// ---- persistent global costmap planes ----
const MAXC=250000;
const gpos=new Float32Array(MAXC*4*3), gcol=new Float32Array(MAXC*4*3);
const gidx=new Uint32Array(MAXC*6);
for(let c=0;c<MAXC;c++){const v=c*4,i=c*6;
 gidx[i]=v;gidx[i+1]=v+2;gidx[i+2]=v+1;gidx[i+3]=v;gidx[i+4]=v+3;gidx[i+5]=v+2;}
const gg=new THREE.BufferGeometry();
gg.setAttribute('position',new THREE.BufferAttribute(gpos,3));
gg.setAttribute('color',new THREE.BufferAttribute(gcol,3));
gg.setIndex(new THREE.BufferAttribute(gidx,1));
gg.setDrawRange(0,0);
const gridMesh=new THREE.Mesh(gg,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}));
scene.add(gridMesh);

function rampFree(t){t=Math.min(1,Math.max(0,t));
 if(t<0.5){const u=t*2;return[0.16+0.24*u,0.45+0.30*u,0.22+0.16*u];}
 const u=(t-0.5)*2;return[0.40+0.42*u,0.75+0.05*u,0.38+0.14*u];}

function applyGlobal(j){
 const cells=new Int32Array(decode(j.cells).buffer);
 const elev=new Int16Array(decode(j.elev).buffer), cls=decode(j.cls);
 const res=j.res;
 if(!origin)origin=[j.pose[0],j.pose[1]];
 const inset=res*0.06;
 const n=Math.min(j.count,MAXC);
 for(let k=0;k<n;k++){
  const cl=cls[k];
  let z=elev[k]*0.01;
  let col;
  if(cl===1)col=[0.76,0.27,0.27];
  else col=rampFree((z+0.15)/0.9);
  const x0=cells[k*2]*res-origin[0]+inset, x1=x0+res-2*inset;
  const y0=cells[k*2+1]*res-origin[1]+inset, y1=y0+res-2*inset;
  const v=k*4*3;
  gpos[v]=x0;gpos[v+1]=z;gpos[v+2]=y0;
  gpos[v+3]=x1;gpos[v+4]=z;gpos[v+5]=y0;
  gpos[v+6]=x1;gpos[v+7]=z;gpos[v+8]=y1;
  gpos[v+9]=x0;gpos[v+10]=z;gpos[v+11]=y1;
  for(let q=0;q<4;q++){gcol[v+q*3]=col[0];gcol[v+q*3+1]=col[1];gcol[v+q*3+2]=col[2];}
 }
 gg.setDrawRange(0,n*6);
 gg.attributes.position.needsUpdate=true;
 gg.attributes.color.needsUpdate=true;
 fusion.value=j.fusion_radius;
 fusionVal.textContent=Number(j.fusion_radius).toFixed(1)+' m';
 return n;
}

// ---- A* path + goal marker ----
const MAXPL=512;
const plpos=new Float32Array(MAXPL*3);
const plg=new THREE.BufferGeometry();
plg.setAttribute('position',new THREE.BufferAttribute(plpos,3));
plg.setDrawRange(0,0);
scene.add(new THREE.Line(plg,new THREE.LineBasicMaterial({color:0x4f8ef7,linewidth:2})));

const goalMarker=new THREE.Group();
const ring=new THREE.Mesh(new THREE.TorusGeometry(0.20,0.035,10,28),
 new THREE.MeshBasicMaterial({color:0x41e0c8}));
ring.rotation.x=Math.PI/2; ring.position.y=0.03;
const pole=new THREE.Mesh(new THREE.CylinderGeometry(0.015,0.015,0.6,8),
 new THREE.MeshBasicMaterial({color:0x41e0c8}));
pole.position.y=0.3;
goalMarker.add(ring); goalMarker.add(pole);
goalMarker.visible=false;
scene.add(goalMarker);

function applyPlan(j){
 if(j.goal&&origin){goalMarker.visible=true;
  goalMarker.position.set(j.goal[0]-origin[0],0,j.goal[1]-origin[1]);}
 else goalMarker.visible=false;
 const p=j.path||[], n=Math.min(p.length,MAXPL);
 for(let i=0;i<n;i++){plpos[i*3]=p[i][0]-origin[0];plpos[i*3+1]=0.07;plpos[i*3+2]=p[i][1]-origin[1];}
 plg.setDrawRange(0,n);
 plg.attributes.position.needsUpdate=true;
 document.getElementById('status').textContent=j.status||'';
}

// ---- current sweep points ----
const MAXP=8192;
const ppos=new Float32Array(MAXP*3);
const pg=new THREE.BufferGeometry();
pg.setAttribute('position',new THREE.BufferAttribute(ppos,3));
pg.setDrawRange(0,0);
const sweep=new THREE.Points(pg,new THREE.PointsMaterial({size:0.055,color:0xffc24d,sizeAttenuation:true}));
scene.add(sweep);

function applyFrame(j){
 if(!origin)origin=[j.pose[0],j.pose[1]];
 const S=new Int16Array(decode(j.b64).buffer), n=Math.min(S.length/3,MAXP);
 const q=j.pose, Ro=quatToR(q[3],q[4],q[5],q[6]), R=mul3(Ro,Rm);
 const tx=q[0]-origin[0], ty=q[1]-origin[1], tz=q[2];
 for(let i=0;i<n;i++){
  const x=S[i*3]*0.01,y=S[i*3+1]*0.01,z=S[i*3+2]*0.01;
  ppos[i*3]  =R[0]*x+R[1]*y+R[2]*z+tx;
  ppos[i*3+1]=R[6]*x+R[7]*y+R[8]*z+tz;
  ppos[i*3+2]=R[3]*x+R[4]*y+R[5]*z+ty;
 }
 pg.setDrawRange(0,n);
 pg.attributes.position.needsUpdate=true;
 robot.position.set(tx,q[2]+0.05,ty);
 if(trailN<MAXT){trail[trailN*3]=tx;trail[trailN*3+1]=q[2]+0.02;trail[trailN*3+2]=ty;trailN++;
  tg.setDrawRange(0,trailN);tg.attributes.position.needsUpdate=true;}
}

const MAXT=20000;
const trail=new Float32Array(MAXT*3); let trailN=0;
const tg=new THREE.BufferGeometry();
tg.setAttribute('position',new THREE.BufferAttribute(trail,3));
tg.setDrawRange(0,0);
scene.add(new THREE.Line(tg,new THREE.LineBasicMaterial({color:0xff5252})));
const robot=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,16),
 new THREE.MeshBasicMaterial({color:0xff5252}));
scene.add(robot);
scene.add(new THREE.GridHelper(30,30,0x39404c,0x232830));
scene.add(new THREE.AmbientLight(0xffffff,1));

// ---- controls / camera / click-to-goal ----
const followBox=document.getElementById('follow');
const fusion=document.getElementById('fusion'),fusionVal=document.getElementById('fusionval');
document.getElementById('showpts').onchange=e=>sweep.visible=e.target.checked;
document.getElementById('showgrid').onchange=e=>gridMesh.visible=e.target.checked;
document.getElementById('stop').onclick=()=>{fetch('stop');};
fusion.oninput=()=>{fusionVal.textContent=Number(fusion.value).toFixed(1)+' m';};
fusion.onchange=()=>{fetch('fusion_radius?meters='+encodeURIComponent(fusion.value));};

let theta=-0.9,phi=1.0,radius=7,drag=false,lx=0,ly=0,downX=0,downY=0;
function cam(){
 const t=followBox.checked?robot.position:new THREE.Vector3(0,0.3,0);
 camera.position.set(t.x+radius*Math.sin(phi)*Math.cos(theta),
  t.y+radius*Math.cos(phi), t.z+radius*Math.sin(phi)*Math.sin(theta));
 camera.lookAt(t.x,t.y+0.2,t.z);}
canvas.addEventListener('pointerdown',e=>{drag=true;lx=downX=e.clientX;ly=downY=e.clientY;});
addEventListener('pointerup',e=>{
 if(drag&&Math.hypot(e.clientX-downX,e.clientY-downY)<6)clickGoal(e);
 drag=false;});
addEventListener('pointermove',e=>{if(!drag)return;
 theta-=(e.clientX-lx)*0.006;phi-=(e.clientY-ly)*0.006;
 phi=Math.max(0.12,Math.min(1.52,phi));lx=e.clientX;ly=e.clientY;});
canvas.addEventListener('wheel',e=>{e.preventDefault();
 radius=Math.min(40,Math.max(2,radius*(1+(e.deltaY>0?0.08:-0.08))));},{passive:false});

function clickGoal(e){
 if(!origin)return;
 const ndc=new THREE.Vector2(e.clientX/innerWidth*2-1,-(e.clientY/innerHeight)*2+1);
 const rc=new THREE.Raycaster(); rc.setFromCamera(ndc,camera);
 const pt=new THREE.Vector3();
 if(!rc.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),pt))return;
 goalMarker.visible=true;
 goalMarker.position.set(pt.x,0,pt.z);
 const gx=(pt.x+origin[0]).toFixed(2), gy=(pt.z+origin[1]).toFixed(2);
 document.getElementById('status').textContent='goal sent ('+gx+', '+gy+')…';
 fetch('goal?x='+gx+'&y='+gy);
}

// ---- streams ----
const dot=document.getElementById('dot'),stats=document.getElementById('stats');
let fSince=-1,gSince=-1,mSince=-1,nGrids=0,mapCells=0;
async function pollFrames(){
 for(;;){try{
   const r=await fetch('frame?since='+fSince);
   if(r.status===200){const j=await r.json();fSince=j.seq;dot.classList.add('on');applyFrame(j);}
  }catch(e){dot.classList.remove('on');await new Promise(res=>setTimeout(res,700));}}
}
async function pollGrid(){
 for(;;){try{
   const r=await fetch('grid?since='+gSince);
   if(r.status===200){const j=await r.json();gSince=j.seq;
    applyPlan(j);nGrids++;
    stats.textContent=mapCells+' saved cells · local grid #'+nGrids+
     (j.goal?' · goal ('+j.goal[0]+', '+j.goal[1]+')':'');}
  }catch(e){await new Promise(res=>setTimeout(res,700));}}
}
async function pollGlobal(){
 for(;;){try{
   const r=await fetch('global?since='+mSince);
   if(r.status===200){const j=await r.json();mSince=j.seq;
    mapCells=applyGlobal(j);}
  }catch(e){await new Promise(res=>setTimeout(res,700));}}
}
function loop(){cam();renderer.render(scene,camera);requestAnimationFrame(loop);}
resize();addEventListener('resize',resize);loop();pollFrames();pollGrid();pollGlobal();
</script></body></html>
""".replace("__MOUNT_RPY__", json.dumps(MOUNT_RPY_DEG))


def main():
    global NODE
    rclpy.init()
    NODE = LiveGrid()
    threading.Thread(target=grid_broadcaster, args=(NODE,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("live nav viewer on http://0.0.0.0:%d/ (ctrl-c to stop)" % PORT, flush=True)
    try:
        rclpy.spin(NODE)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if HAVE_CTRL:
                NODE._release_override()
        except Exception:
            pass
        try:
            NODE.global_map.save(GLOBAL_MAP_PATH)
        except Exception as exc:
            NODE.get_logger().warn("final global costmap save failed: %s" % exc)
        srv.shutdown()
        NODE.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
