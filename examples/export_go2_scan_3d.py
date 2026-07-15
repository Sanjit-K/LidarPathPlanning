#!/usr/bin/env python3
"""Export the captured Go2 lidar scan as an interactive 3D HTML point-cloud viewer.

    python3 examples/export_go2_scan_3d.py [scan.npy] [out.html]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_go2_scan import level_cloud, MIN_RANGE, MAX_RANGE

SRC = sys.argv[1] if len(sys.argv) > 1 else "test-data/go2/go2_scan.npy"
OUT = sys.argv[2] if len(sys.argv) > 2 else "go2_scan_3d.html"

raw = np.load(SRC).astype(np.float64)
r = np.linalg.norm(raw, axis=1)
pts = raw[(r > MIN_RANGE) & (r < MAX_RANGE)]
cloud, tilt, h_sensor, _R = level_cloud(pts)
print("points: %d (of %d raw), mount tilt %.1f deg, sensor height %.2f m" % (
    cloud.shape[0], raw.shape[0], tilt, h_sensor))

data = {"pts": [[round(float(v), 3) for v in p] for p in cloud],
        "tilt": round(tilt, 1)}

HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Go2 lidar scan (3D)</title>
<style>
 html,body{margin:0;height:100%;background:#101318;color:#d7dae0;font-family:system-ui,sans-serif;overflow:hidden}
 #hud{position:fixed;top:12px;left:12px;background:#1a1e25;border:1px solid #2a2f38;border-radius:8px;
   padding:10px 12px;font-size:13px;line-height:1.8;z-index:2}
 #hint{position:fixed;bottom:12px;right:14px;font-size:13px;color:#7f8590;z-index:2}
 canvas{display:block}
 .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
</style></head><body>
<div id="hud">
 <b>Unitree Go2 &mdash; live L1 lidar capture</b><br>
 __N__ points &middot; 20 sweeps merged &middot; leveled (tilt __TILT__&deg;)<br>
 <span class="sw" style="background:#3557c2"></span>low &rarr;
 <span class="sw" style="background:#43c05f"></span>mid &rarr;
 <span class="sw" style="background:#f5e04e"></span>high&nbsp;
 <span class="sw" style="background:#ff5252;border-radius:50%"></span>robot (sensor origin)
</div>
<div id="hint">drag to rotate &middot; scroll to zoom</div>
<canvas id="cv"></canvas>
<script id="data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const P=D.pts, N=P.length;
const canvas=document.getElementById('cv');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.05, 500);
const renderer=new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
function resize(){renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();}

const pos=new Float32Array(N*3), col=new Float32Array(N*3);
let zmin=1e9, zmax=-1e9;
for(const p of P){ if(p[2]<zmin)zmin=p[2]; if(p[2]>zmax)zmax=p[2]; }
const zlo=Math.max(zmin,-0.4), zhi=Math.min(zmax,2.2);
function ramp(t){ // blue -> green -> yellow
 t=Math.min(1,Math.max(0,t));
 if(t<0.5){const u=t*2; return [0.21+(0.26-0.21)*u, 0.34+(0.75-0.34)*u, 0.76+(0.37-0.76)*u];}
 const u=(t-0.5)*2; return [0.26+(0.96-0.26)*u, 0.75+(0.88-0.75)*u, 0.37+(0.31-0.37)*u];
}
for(let i=0;i<N;i++){
 pos[i*3]=P[i][0]; pos[i*3+1]=P[i][2]; pos[i*3+2]=P[i][1];   // y-up for display
 const c=ramp((P[i][2]-zlo)/(zhi-zlo));
 col[i*3]=c[0]; col[i*3+1]=c[1]; col[i*3+2]=c[2];
}
const g=new THREE.BufferGeometry();
g.setAttribute('position', new THREE.BufferAttribute(pos,3));
g.setAttribute('color', new THREE.BufferAttribute(col,3));
scene.add(new THREE.Points(g, new THREE.PointsMaterial({size:0.07, vertexColors:true, sizeAttenuation:true})));

// robot marker at origin + ground grid
const robot=new THREE.Mesh(new THREE.SphereGeometry(0.12,16,16),
  new THREE.MeshBasicMaterial({color:0xff5252}));
robot.position.set(0,0.1,0); scene.add(robot);
scene.add(new THREE.GridHelper(24,24,0x39404c,0x232830));
scene.add(new THREE.AmbientLight(0xffffff,1));

let theta=-1.0, phi=1.05, radius=9, drag=false, lx=0, ly=0, auto=true;
function cam(){camera.position.set(radius*Math.sin(phi)*Math.cos(theta), radius*Math.cos(phi),
 radius*Math.sin(phi)*Math.sin(theta)); camera.lookAt(0,0.4,0);}
canvas.addEventListener('pointerdown',e=>{drag=true;auto=false;lx=e.clientX;ly=e.clientY;});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{if(!drag)return;
 theta-=(e.clientX-lx)*0.006; phi-=(e.clientY-ly)*0.006;
 phi=Math.max(0.15,Math.min(1.5,phi)); lx=e.clientX; ly=e.clientY;});
canvas.addEventListener('wheel',e=>{e.preventDefault();
 radius=Math.min(40,Math.max(2,radius*(1+(e.deltaY>0?0.08:-0.08))));},{passive:false});
function loop(){if(auto)theta+=0.002;cam();renderer.render(scene,camera);requestAnimationFrame(loop);}
resize(); addEventListener('resize',resize); loop();
</script></body></html>
"""

html = (HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__N__", "{:,}".format(cloud.shape[0]))
        .replace("__TILT__", str(data["tilt"])))
with open(OUT, "w") as f:
    f.write(html)
print("wrote %s (%.0f KB)" % (OUT, os.path.getsize(OUT) / 1024))
