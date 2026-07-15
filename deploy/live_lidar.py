#!/usr/bin/env python3
"""Live lidar streamer: runs ON the Go2, serves a browser viewer to the LAN.

Subscribes /utlidar/cloud + /utlidar/robot_odom, quantizes each sweep to int16
centimeters (sensor frame), and serves it over HTTP long-poll together with the
odometry pose. The viewer page (served at /) applies the calibrated mount
rotation + per-frame pose in the browser and accumulates a rolling world map.

On the robot (env sourced):   python3 live_lidar.py [port]
On your computer:             open http://<robot-ip>:8766/

Foxy / Python 3.8, stdlib + rclpy only.
"""
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
MIN_RANGE, MAX_RANGE = 0.5, 12.0
MOUNT_RPY_DEG = [-177.93, 13.77, 176.36]   # calibrated Go2 L1 sensor->base

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


STATE = {"seq": -1, "payload": b"", "lock": threading.Lock(), "event": threading.Event()}


class Grab(Node):
    def __init__(self):
        super().__init__("live_lidar")
        self.pose = None
        self.create_subscription(PointCloud2, "/utlidar/cloud", self.on_cloud, 5)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, 20)
        # make sure the lidar publisher is switched on
        pub = self.create_publisher(String, "/utlidar/switch", 1)
        m = String(); m.data = "ON"
        for _ in range(3):
            pub.publish(m)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    def on_cloud(self, m):
        if self.pose is None:
            return
        xyz = cloud_to_xyz(m)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > MIN_RANGE) & (r < MAX_RANGE)]
        q = np.clip(np.round(xyz * 100.0), -32000, 32000).astype("<i2")
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with STATE["lock"]:
            STATE["seq"] += 1
            STATE["payload"] = json.dumps({
                "seq": STATE["seq"], "t": round(t, 3),
                "pose": [round(v, 4) for v in self.pose],
                "b64": base64.b64encode(q.tobytes()).decode("ascii"),
            }).encode()
        STATE["event"].set()
        STATE["event"].clear()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # quiet
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
        elif u.path == "/frame":
            since = int(parse_qs(u.query).get("since", ["-1"])[0])
            if STATE["seq"] <= since:
                STATE["event"].wait(timeout=1.0)      # long-poll
            with STATE["lock"]:
                if STATE["seq"] <= since:
                    self._send(204)
                else:
                    self._send(200, STATE["payload"])
        else:
            self._send(404, b"{}")


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Go2 live lidar</title>
<style>
 html,body{margin:0;height:100%;background:#0e1116;color:#d7dae0;font-family:system-ui,sans-serif;overflow:hidden}
 #hud{position:fixed;top:12px;left:12px;background:#171b22;border:1px solid #2a2f38;border-radius:8px;
   padding:10px 12px;font-size:13px;line-height:1.7;z-index:2;max-width:340px}
 #bar{position:fixed;left:50%;transform:translateX(-50%);bottom:14px;background:#171b22;
   border:1px solid #2a2f38;border-radius:10px;padding:9px 14px;z-index:2;display:flex;
   gap:12px;align-items:center;font-size:13px;white-space:nowrap}
 button{background:#2b3442;color:#e6e9ee;border:0;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}
 button:hover{background:#38445a}
 input[type=range]{accent-color:#4f8ef7}
 canvas{display:block} label{user-select:none} .dim{color:#8b93a0}
 #dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#e05555;margin-right:6px;vertical-align:0}
 #dot.on{background:#4fc06a}
</style></head><body>
<div id="hud">
 <b><span id="dot"></span>Unitree Go2 &mdash; LIVE lidar</b><br>
 <span id="stats" class="dim">connecting&hellip;</span><br>
 <span id="posetxt" class="dim"></span>
</div>
<div id="bar">
 <button id="pause">&#10074;&#10074; pause</button>
 <button id="clear">clear map</button>
 <label><input id="follow" type="checkbox" checked> follow robot</label>
 <label class="dim">yaw trim <input id="trim" type="range" min="-30" max="30" value="0" step="0.5">
  <span id="trimv">0&deg;</span></label>
</div>
<canvas id="cv"></canvas>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const MOUNT_RPY=__MOUNT_RPY__;
const MAXPTS=300000, MAXTRAIL=20000, deg=Math.PI/180;

function rpyToR(r,p,y){
 const cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);
 return [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
         sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
         -sp,   cp*sr,          cp*cr];
}
function quatToR(x,y,z,w){
 return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
         2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
         2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];
}
function mul3(a,b){const o=new Array(9);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)
  o[i*3+j]=a[i*3]*b[j]+a[i*3+1]*b[3+j]+a[i*3+2]*b[6+j];
 return o;}
function ramp(t){t=Math.min(1,Math.max(0,t));
 if(t<0.5){const u=t*2;return[0.21+0.05*u,0.34+0.41*u,0.76-0.39*u];}
 const u=(t-0.5)*2;return[0.26+0.70*u,0.75+0.13*u,0.37-0.06*u];}

let trimDeg=0, Rm=null;
function updRm(){const[r,p,y]=MOUNT_RPY.map(v=>v*deg);Rm=rpyToR(r,p,y+trimDeg*deg);}
updRm();

// rolling store of raw frames (sensor-frame int16 + pose) for rebuilds
const frames=[]; let totalPts=0, origin=null;

const pos=new Float32Array(MAXPTS*3), col=new Float32Array(MAXPTS*3);
const curPos=new Float32Array(8192*3);
const trail=new Float32Array(MAXTRAIL*3); let trailN=0;
let used=0;

const canvas=document.getElementById('cv');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.05,500);
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
function resize(){renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();}

const g=new THREE.BufferGeometry();
g.setAttribute('position',new THREE.BufferAttribute(pos,3));
g.setAttribute('color',new THREE.BufferAttribute(col,3));
g.setDrawRange(0,0);
scene.add(new THREE.Points(g,new THREE.PointsMaterial({size:0.05,vertexColors:true,sizeAttenuation:true})));

const g2=new THREE.BufferGeometry();
g2.setAttribute('position',new THREE.BufferAttribute(curPos,3));
g2.setDrawRange(0,0);
scene.add(new THREE.Points(g2,new THREE.PointsMaterial({size:0.09,color:0xffc24d,sizeAttenuation:true})));

const tg=new THREE.BufferGeometry();
tg.setAttribute('position',new THREE.BufferAttribute(trail,3));
tg.setDrawRange(0,0);
scene.add(new THREE.Line(tg,new THREE.LineBasicMaterial({color:0xff5252})));

const robot=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,16),
 new THREE.MeshBasicMaterial({color:0xff5252}));
scene.add(robot);
scene.add(new THREE.GridHelper(30,30,0x39404c,0x232830));
scene.add(new THREE.AmbientLight(0xffffff,1));

function xform(f,outArr,outOff){       // frame -> world -> three (y-up), returns count
 const S=f.pts, n=S.length/3, q=f.pose;
 const Ro=quatToR(q[3],q[4],q[5],q[6]), R=mul3(Ro,Rm);
 const tx=q[0]-origin[0], ty=q[1]-origin[1], tz=q[2];
 for(let i=0;i<n;i++){
  const x=S[i*3]*0.01,y=S[i*3+1]*0.01,z=S[i*3+2]*0.01;
  const wx=R[0]*x+R[1]*y+R[2]*z+tx;
  const wy=R[3]*x+R[4]*y+R[5]*z+ty;
  const wz=R[6]*x+R[7]*y+R[8]*z+tz;
  const o=(outOff+i)*3;
  outArr[o]=wx; outArr[o+1]=wz; outArr[o+2]=wy;
  if(outArr===pos){const c=ramp((wz+0.2)/1.6);col[o]=c[0];col[o+1]=c[1];col[o+2]=c[2];}
 }
 return n;
}
function rebuildAll(){
 used=0;
 for(const f of frames){used+=xform(f,pos,used);}
 g.setDrawRange(0,used);
 g.attributes.position.needsUpdate=true; g.attributes.color.needsUpdate=true;
}
function addFrame(f){
 if(!origin)origin=[f.pose[0],f.pose[1]];
 frames.push(f); totalPts+=f.pts.length/3;
 let dropped=false;
 while(totalPts>MAXPTS && frames.length>1){totalPts-=frames.shift().pts.length/3;dropped=true;}
 if(dropped){rebuildAll();}
 else{used+=xform(f,pos,used); g.setDrawRange(0,used);
  g.attributes.position.needsUpdate=true; g.attributes.color.needsUpdate=true;}
 // current-sweep highlight
 const n=Math.min(f.pts.length/3,8192);
 xform({pts:f.pts.subarray(0,n*3),pose:f.pose},curPos,0);
 g2.setDrawRange(0,n); g2.attributes.position.needsUpdate=true;
 // trail + robot marker
 const tx=f.pose[0]-origin[0], ty=f.pose[1]-origin[1];
 if(trailN<MAXTRAIL){trail[trailN*3]=tx;trail[trailN*3+1]=f.pose[2]+0.02;trail[trailN*3+2]=ty;trailN++;
  tg.setDrawRange(0,trailN); tg.attributes.position.needsUpdate=true;}
 robot.position.set(tx,f.pose[2]+0.05,ty);
 document.getElementById('posetxt').textContent=
  'odom ('+f.pose[0].toFixed(2)+', '+f.pose[1].toFixed(2)+') m';
}

// ---------- controls ----------
let paused=false;
const pauseBtn=document.getElementById('pause');
pauseBtn.onclick=()=>{paused=!paused;
 pauseBtn.innerHTML=paused?'&#9654; resume':'&#10074;&#10074; pause';};
document.getElementById('clear').onclick=()=>{frames.length=0;totalPts=0;used=0;trailN=0;
 g.setDrawRange(0,0);g2.setDrawRange(0,0);tg.setDrawRange(0,0);};
const trim=document.getElementById('trim'),trimv=document.getElementById('trimv');
trim.oninput=()=>{trimDeg=parseFloat(trim.value);trimv.textContent=trim.value+'°';updRm();rebuildAll();};
const followBox=document.getElementById('follow');

// ---------- camera ----------
let theta=-0.9,phi=1.0,radius=8,drag=false,lx=0,ly=0;
function cam(){
 const t=followBox.checked?robot.position:new THREE.Vector3(0,0.3,0);
 camera.position.set(t.x+radius*Math.sin(phi)*Math.cos(theta),
  t.y+radius*Math.cos(phi), t.z+radius*Math.sin(phi)*Math.sin(theta));
 camera.lookAt(t.x,t.y+0.3,t.z);
}
canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{if(!drag)return;
 theta-=(e.clientX-lx)*0.006;phi-=(e.clientY-ly)*0.006;
 phi=Math.max(0.12,Math.min(1.52,phi));lx=e.clientX;ly=e.clientY;});
canvas.addEventListener('wheel',e=>{e.preventDefault();
 radius=Math.min(40,Math.max(2,radius*(1+(e.deltaY>0?0.08:-0.08))));},{passive:false});

// ---------- stream ----------
const dot=document.getElementById('dot'),stats=document.getElementById('stats');
let since=-1,nFrames=0,rateT=performance.now(),rateN=0,fps=0;
function decode(b64){const s=atob(b64),u=new Uint8Array(s.length);
 for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Int16Array(u.buffer);}
async function poll(){
 for(;;){
  try{
   const r=await fetch('frame?since='+since);
   if(r.status===200){
    const j=await r.json(); since=j.seq;
    dot.classList.add('on');
    if(!paused){addFrame({pts:decode(j.b64),pose:j.pose}); nFrames++; rateN++;}
   }
   const now=performance.now();
   if(now-rateT>2000){fps=rateN*1000/(now-rateT);rateN=0;rateT=now;}
   stats.textContent=nFrames+' sweeps · '+used.toLocaleString()+' pts in map · '+fps.toFixed(1)+' Hz';
  }catch(e){
   dot.classList.remove('on');
   stats.textContent='disconnected - retrying…';
   await new Promise(res=>setTimeout(res,700));
  }
 }
}
function loop(){cam();renderer.render(scene,camera);requestAnimationFrame(loop);}
resize();addEventListener('resize',resize);loop();poll();
</script></body></html>
""".replace("__MOUNT_RPY__", json.dumps(MOUNT_RPY_DEG))


def main():
    rclpy.init()
    node = Grab()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("live lidar viewer on http://0.0.0.0:%d/ (ctrl-c to stop)" % PORT, flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
