#!/usr/bin/env python3
"""Render a recorded Go2 'lidar video' (grab_video.py npz) as an interactive
world-frame 3D animation in a standalone HTML file.

Each frame's points are stored in the SENSOR frame; the browser applies the
calibrated lidar->base mount rotation and the per-frame odometry pose, so the
scan accumulates into a consistent world map as the dog walks. Includes
play/pause, scrubber, speed, accumulation toggle and a live mount-yaw trim.

    python3 examples/export_lidar_video.py [video.npz] [out.html]
"""
import base64
import json
import os
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "test-data/go2/lidar_video.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "lidar_video.html"
MIN_RANGE, MAX_RANGE = 0.5, 12.0

# calibrated Go2 L1 mount (sensor -> leveled body), from test_go2_scan.level_cloud
MOUNT_RPY_DEG = [-177.93, 13.77, 176.36]

d = np.load(SRC)
pts, off, stamps, poses = d["pts"], d["offsets"], d["stamps"], d["poses"]

# per-frame range filter in the sensor frame (drop self-hits + far noise)
keep_pts, new_off = [], [0]
for f in range(len(stamps)):
    p = pts[off[f]:off[f + 1]]
    r = np.linalg.norm(p, axis=1)
    p = p[(r > MIN_RANGE) & (r < MAX_RANGE)]
    keep_pts.append(p)
    new_off.append(new_off[-1] + p.shape[0])
pts = np.vstack(keep_pts)
off = np.array(new_off, dtype=np.int64)
print("frames %d, %d pts after range filter [%.1f, %.1f] m" % (
    len(stamps), pts.shape[0], MIN_RANGE, MAX_RANGE))

# quantize sensor-frame points to int16 centimeters -> base64
q = np.clip(np.round(pts * 100.0), -32000, 32000).astype("<i2")
b64 = base64.b64encode(q.tobytes()).decode("ascii")

t0 = float(stamps[0])
data = {
    "b64": b64,
    "off": [int(v) for v in off],
    "t": [round(float(s - t0), 3) for s in stamps],
    "poses": [[round(float(v), 4) for v in p] for p in poses],
    "mount_rpy": MOUNT_RPY_DEG,
}

HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Go2 lidar video</title>
<style>
 html,body{margin:0;height:100%;background:#0e1116;color:#d7dae0;font-family:system-ui,sans-serif;overflow:hidden}
 #hud{position:fixed;top:12px;left:12px;background:#171b22;border:1px solid #2a2f38;border-radius:8px;
   padding:10px 12px;font-size:13px;line-height:1.7;z-index:2;max-width:330px}
 #bar{position:fixed;left:50%;transform:translateX(-50%);bottom:14px;background:#171b22;
   border:1px solid #2a2f38;border-radius:10px;padding:9px 14px;z-index:2;display:flex;
   gap:10px;align-items:center;font-size:13px;white-space:nowrap}
 button{background:#2b3442;color:#e6e9ee;border:0;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}
 button:hover{background:#38445a}
 input[type=range]{accent-color:#4f8ef7}
 #seek{width:min(34vw,340px)} #trim{width:110px}
 canvas{display:block}
 label{user-select:none}
 .dim{color:#8b93a0}
</style></head><body>
<div id="hud">
 <b>Unitree Go2 &mdash; 10 s lidar video</b><br>
 <span id="stats" class="dim"></span><br>
 <span class="dim">sensor&rarr;base mount + odometry applied per frame</span>
</div>
<div id="bar">
 <button id="play">&#10074;&#10074;</button>
 <input id="seek" type="range" min="0" max="1000" value="0">
 <span id="clock" class="dim" style="min-width:86px">0.0 s &middot; f0</span>
 <select id="speed"><option value="0.5">0.5&times;</option><option value="1" selected>1&times;</option><option value="2">2&times;</option></select>
 <label><input id="acc" type="checkbox" checked> accumulate</label>
 <label class="dim">yaw trim <input id="trim" type="range" min="-30" max="30" value="0" step="0.5">
  <span id="trimv">0&deg;</span></label>
</div>
<canvas id="cv"></canvas>
<script id="data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const raw=atob(D.b64), nb=raw.length, u8=new Uint8Array(nb);
for(let i=0;i<nb;i++)u8[i]=raw.charCodeAt(i);
const S=new Int16Array(u8.buffer);            // sensor-frame cm
const OFF=D.off, T=D.t, P=D.poses, F=T.length, N=S.length/3;

function rpyToR(r,p,y){
 const cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);
 return [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
         sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
         -sp,   cp*sr,          cp*cr];
}
function quatToR(x,y,z,w){
 return [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
         2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
         2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)];
}
function mul3(a,b){ // 3x3 row-major product a*b
 const o=new Array(9);
 for(let i=0;i<3;i++)for(let j=0;j<3;j++)
  o[i*3+j]=a[i*3]*b[j]+a[i*3+1]*b[3+j]+a[i*3+2]*b[6+j];
 return o;
}

// world-frame positions (three.js y-up: world x->x, z->y, y->z)
const pos=new Float32Array(N*3), col=new Float32Array(N*3);
const trail=new Float32Array(F*3);
const deg=Math.PI/180;
function ramp(t){t=Math.min(1,Math.max(0,t));
 if(t<0.5){const u=t*2;return[0.21+0.05*u,0.34+0.41*u,0.76-0.39*u];}
 const u=(t-0.5)*2;return[0.26+0.70*u,0.75+0.13*u,0.37-0.06*u];}

let cx=0, cz=0;   // scene center (world x,y)
function rebuild(trimDeg){
 const [r0,p0,y0]=D.mount_rpy.map(v=>v*deg);
 const Rm=rpyToR(r0,p0,y0+trimDeg*deg);
 // scene center = mean robot xy
 let sx=0,sy=0;
 for(let f=0;f<F;f++){sx+=P[f][0];sy+=P[f][1];}
 cx=sx/F; cz=sy/F;
 for(let f=0;f<F;f++){
  const q=P[f], Ro=quatToR(q[3],q[4],q[5],q[6]), R=mul3(Ro,Rm);
  const tx=q[0]-cx, ty=q[1]-cz, tz=q[2];
  trail[f*3]=tx; trail[f*3+1]=tz+0.02; trail[f*3+2]=ty;
  for(let i=OFF[f];i<OFF[f+1];i++){
   const x=S[i*3]*0.01, y=S[i*3+1]*0.01, z=S[i*3+2]*0.01;
   const wx=R[0]*x+R[1]*y+R[2]*z+tx;
   const wy=R[3]*x+R[4]*y+R[5]*z+ty;
   const wz=R[6]*x+R[7]*y+R[8]*z+tz;
   pos[i*3]=wx; pos[i*3+1]=wz; pos[i*3+2]=wy;
   const c=ramp((wz+0.2)/1.6);
   col[i*3]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2];
  }
 }
}
rebuild(0);

const canvas=document.getElementById('cv');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.05,500);
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
function resize(){renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();}

const g=new THREE.BufferGeometry();
g.setAttribute('position',new THREE.BufferAttribute(pos,3));
g.setAttribute('color',new THREE.BufferAttribute(col,3));
const cloud=new THREE.Points(g,new THREE.PointsMaterial({size:0.05,vertexColors:true,sizeAttenuation:true}));
scene.add(cloud);

// current-frame highlight (re-uses the same buffers via drawRange on a clone)
const g2=new THREE.BufferGeometry();
g2.setAttribute('position',new THREE.BufferAttribute(pos,3));
const cur=new THREE.Points(g2,new THREE.PointsMaterial({size:0.09,color:0xffc24d,sizeAttenuation:true}));
scene.add(cur);

const tg=new THREE.BufferGeometry();
tg.setAttribute('position',new THREE.BufferAttribute(trail,3));
const trailLine=new THREE.Line(tg,new THREE.LineBasicMaterial({color:0xff5252}));
scene.add(trailLine);

const robot=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,16),
 new THREE.MeshBasicMaterial({color:0xff5252}));
scene.add(robot);
scene.add(new THREE.GridHelper(24,24,0x39404c,0x232830));
scene.add(new THREE.AmbientLight(0xffffff,1));

let theta=-0.9,phi=1.0,radius=8,drag=false,lx=0,ly=0;
function cam(){camera.position.set(radius*Math.sin(phi)*Math.cos(theta),radius*Math.cos(phi),
 radius*Math.sin(phi)*Math.sin(theta));camera.lookAt(0,0.3,0);}
canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{if(!drag)return;
 theta-=(e.clientX-lx)*0.006;phi-=(e.clientY-ly)*0.006;
 phi=Math.max(0.12,Math.min(1.52,phi));lx=e.clientX;ly=e.clientY;});
canvas.addEventListener('wheel',e=>{e.preventDefault();
 radius=Math.min(40,Math.max(2,radius*(1+(e.deltaY>0?0.08:-0.08))));},{passive:false});

// ---------- playback ----------
const dur=T[F-1];
let playing=true,tPlay=0,last=performance.now(),fi=0;
const seek=document.getElementById('seek'),playBtn=document.getElementById('play');
const clock=document.getElementById('clock'),speedSel=document.getElementById('speed');
const accBox=document.getElementById('acc');
function frameAt(t){let lo=0,hi=F-1;while(lo<hi){const m=(lo+hi+1)>>1;if(T[m]<=t)lo=m;else hi=m-1;}return lo;}
function apply(){
 fi=frameAt(tPlay);
 if(accBox.checked){cloud.visible=true;g.setDrawRange(0,OFF[fi+1]);}
 else{cloud.visible=false;}
 g2.setDrawRange(OFF[fi],OFF[fi+1]-OFF[fi]);
 tg.setDrawRange(0,fi+1);
 robot.position.set(trail[fi*3],trail[fi*3+1]+0.05,trail[fi*3+2]);
 seek.value=Math.round(1000*tPlay/dur);
 clock.textContent=tPlay.toFixed(1)+' s · f'+fi;
}
playBtn.onclick=()=>{playing=!playing;playBtn.innerHTML=playing?'&#10074;&#10074;':'&#9654;';last=performance.now();};
seek.oninput=()=>{tPlay=dur*seek.value/1000;playing=false;playBtn.innerHTML='&#9654;';apply();};
accBox.onchange=apply;
const trim=document.getElementById('trim'),trimv=document.getElementById('trimv');
trim.oninput=()=>{trimv.textContent=trim.value+'°';rebuild(parseFloat(trim.value));
 g.attributes.position.needsUpdate=true;g.attributes.color.needsUpdate=true;
 g2.attributes.position.needsUpdate=true;tg.attributes.position.needsUpdate=true;apply();};

document.getElementById('stats').textContent=
 F+' frames · '+N.toLocaleString()+' points · '+dur.toFixed(1)+' s';

function loop(){
 const now=performance.now();
 if(playing){tPlay+=(now-last)/1000*parseFloat(speedSel.value);
  if(tPlay>dur)tPlay=0; apply();}
 last=now;
 cam();renderer.render(scene,camera);requestAnimationFrame(loop);
}
resize();addEventListener('resize',resize);apply();loop();
</script></body></html>
"""

html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
with open(OUT, "w") as f:
    f.write(html)
print("wrote %s (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
