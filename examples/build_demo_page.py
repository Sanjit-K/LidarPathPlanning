#!/usr/bin/env python3
"""Build a single self-contained project demo page (index.html).

Academic-report styling (white background, Times New Roman). Explains the pipeline
and embeds an interactive path planner with a scene selector (synthetic + real
scans baked in). Offline: no server, no CDN.

    python3 examples/build_demo_page.py --out index.html
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lidar_pathplan import GridConfig, QuadrupedParams, discretize
from lidar_pathplan.io_utils import load_point_cloud
from lidar_pathplan.synthetic import make_scene


def synthetic():
    return discretize(make_scene(), GridConfig(0.10), QuadrupedParams())


def kitti(path, half=22.5, res=0.25):
    cloud = load_point_cloud(path)
    cfg = GridConfig(resolution=res, bounds=(-half, -half, half, half), z_clip=(-3.0, 1.0))
    return discretize(cloud, cfg, QuadrupedParams())


def rgbd(path, res=0.05):
    from test_real_room import detect_up_axis, reorient
    cloud = load_point_cloud(path)
    up = detect_up_axis(cloud)
    cloud, _ = reorient(cloud, up)
    robot = QuadrupedParams(radius=0.15, max_step=0.15, max_roughness=0.22, max_slope_deg=45)
    return discretize(cloud, GridConfig(res, z_clip=(-0.10, 1.8), smooth_passes=4), robot)


def pack(grid):
    cost = [(-1.0 if not np.isfinite(v) else round(float(v), 2)) for v in grid.cost.ravel()]
    free = grid.meta["observed_cells"] - grid.meta["lethal_cells"]
    return {
        "W": int(grid.shape[1]), "H": int(grid.shape[0]), "res": float(grid.resolution),
        "cost": cost, "cls": [int(v) for v in grid.classes.ravel()],
        "free": int(free), "lethal": int(grid.meta["lethal_cells"]),
        "unknown": int(grid.meta["unknown_cells"]),
    }


def collect():
    scenes = {}
    print("building synthetic ..."); scenes["Synthetic terrain"] = pack(synthetic())
    for name, p in [("KITTI street — 000001", "test-scans/000001.bin"),
                    ("KITTI street — 000002", "test-scans/000002.bin")]:
        if os.path.exists(p):
            print("building", name, "..."); scenes[name] = pack(kitti(p))
    room = "examples/data/room_fragment.ply"
    if os.path.exists(room):
        print("building room ..."); scenes["Indoor room (RGB-D)"] = pack(rgbd(room))
    apt = "test-data/aligned_low_apartment/apt_sub.npy"
    if os.path.exists(apt):
        print("building apartment ..."); scenes["Apartment (RGB-D)"] = pack(rgbd(apt))
    return scenes


PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiDAR-Based Path Planning for a Quadruped Robot</title>
<style>
 html{font-size:17px}
 body{margin:0;background:#ffffff;color:#111;
   font-family:"Times New Roman",Times,serif;line-height:1.55}
 .wrap{max-width:760px;margin:0 auto;padding:48px 28px 72px}
 .titleblock{text-align:center;border-bottom:1.5px solid #111;padding-bottom:20px;margin-bottom:26px}
 h1{font-size:27px;font-weight:700;margin:0 0 6px;line-height:1.25}
 .subtitle{font-size:18px;font-style:italic;color:#333;margin-bottom:14px}
 .authors{font-size:16px;margin:2px 0}
 .affil{font-size:14px;color:#444}
 .date{font-size:14px;color:#444;margin-top:6px}
 h2{font-size:19px;font-weight:700;margin:30px 0 8px;border-bottom:0.5px solid #bbb;padding-bottom:3px}
 h3{font-size:16px;font-weight:700;margin:18px 0 4px}
 p{margin:0 0 12px;text-align:justify}
 .abstract{font-size:15.5px;margin:0 22px 8px;text-align:justify}
 .abstract b{font-weight:700}
 ol,ul{margin:0 0 12px;padding-left:26px}
 li{margin:5px 0;text-align:justify}
 .fig{border:1px solid #ccc;background:#fbfbfb;border-radius:4px;padding:16px;margin:14px 0}
 .figcap{font-size:14px;color:#333;margin-top:12px;text-align:justify}
 .figcap b{font-weight:700}
 .controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px;
   font-family:"Times New Roman",Times,serif}
 label.lbl{font-size:15px}
 select,button{font-family:inherit;font-size:14px;background:#fff;color:#111;
   border:1px solid #999;border-radius:4px;padding:5px 10px;cursor:pointer}
 select:hover,button:hover{border-color:#555}
 #status{font-style:italic;color:#222;margin-left:auto;font-size:15px}
 #meta{font-size:13px;color:#555;display:block;margin-bottom:8px}
 canvas{display:block;max-width:100%;border:1px solid #bbb;background:#fff;
   cursor:crosshair;image-rendering:pixelated}
 .legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13.5px;color:#333;margin-top:12px}
 .legend i{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:1px;border:0.5px solid #999}
 code{font-family:"Courier New",monospace;font-size:14px;background:#f2f2f2;padding:1px 4px;border-radius:3px}
 footer{font-size:13.5px;color:#555;border-top:0.5px solid #bbb;margin-top:36px;padding-top:14px;text-align:justify}
</style></head><body>
<div class="wrap">

<div class="titleblock">
 <h1>LiDAR-Based Terrain Mapping and Path Planning<br>for a Quadruped Robot</h1>
 <div class="subtitle">A classical mapping&ndash;planning stack with an interactive demonstration</div>
 <div class="authors">Sanjit K.</div>
 <div class="affil">Southern Methodist University</div>
 <div class="date">2026</div>
</div>

<p class="abstract"><b>Abstract&mdash;</b>This report describes a navigation pipeline that converts a
raw three-dimensional LiDAR point cloud into a traversability map and a collision-free path for a
legged robot operating on uneven terrain. The environment is discretized into a 2.5D elevation grid;
per-cell slope, step height, and roughness are compared against the robot's mobility limits to obtain a
cost map, over which A* and Dijkstra search compute a global route. Repeated scans are fused with a
Kalman filter, and an artificial potential field provides reactive local avoidance. The method is
implemented from first principles in Python and validated on synthetic terrain, automotive LiDAR, and
indoor RGB-D data. An interactive planner (Section 3) lets the reader select start and goal locations
on each map and observe the resulting path.</p>

<h2>1. Overview</h2>
<p>Legged robots can traverse terrain&mdash;steps, slopes, and gaps&mdash;that defeats wheeled
platforms, but doing so autonomously requires a terrain representation that preserves these geometric
features, together with planners that produce both a long-range route and reactive avoidance of nearby
hazards. This work reproduces and extends a classical stack for this task: LiDAR mapping, localization,
global planning, and local obstacle avoidance.</p>

<h2>2. Method</h2>
<p>The pipeline proceeds in four stages. Each cell of a top-down grid stores a terrain height, which is
converted to a traversal cost that the planner minimizes.</p>
<ol>
 <li><b>LiDAR point cloud.</b> Millions of three-dimensional points from a LiDAR or RGB-D sensor,
 expressed in a fixed world frame.</li>
 <li><b>2.5D elevation grid.</b> Points are binned into planar cells; each cell retains a representative
 height and a robust roughness estimate. Successive scans are fused with a per-cell Kalman filter to
 suppress sensor noise. Unlike a flat two-dimensional occupancy grid, this representation preserves the
 vertical structure&mdash;curbs, steps, and slopes&mdash;that determines footing.</li>
 <li><b>Traversability cost map.</b> The within-cell height standard deviation (roughness), the local
 slope, and the step height relative to neighbouring cells are compared against the robot's limits to
 classify every cell as <i>free</i>, <i>obstacle</i>, or <i>unknown</i>, with a continuous cost assigned
 to the remainder. Isolated false positives are removed, and obstacles are inflated by the robot
 footprint for clearance.</li>
 <li><b>Path planning.</b> Dijkstra's algorithm, or equivalently A* with an admissible heuristic,
 computes the least-cost global route. An artificial potential field then performs real-time local
 avoidance and can track a moving target for person-following.</li>
</ol>

<h2>3. Interactive Demonstration</h2>
<p>The figure below embeds the discretized cost map for several datasets. Select a scan from the menu,
then <b>click once on the map to place the start and again to place the goal</b>; the A* path is computed
in the browser and drawn immediately, with its length reported. A click on an obstacle snaps to the
nearest traversable cell. The planner is the same algorithm used in the Python implementation.</p>

<div class="fig">
 <div class="controls">
  <label class="lbl">Dataset:</label>
  <select id="scene"></select>
  <button id="reset">Reset</button>
  <span id="status">Click to set start</span>
 </div>
 <span id="meta"></span>
 <canvas id="cv"></canvas>
 <div class="legend">
  <span><i style="background:#a8d08d"></i>traversable</span>
  <span><i style="background:#c0504d"></i>obstacle</span>
  <span><i style="background:#dcdcdc"></i>unknown / unobserved</span>
  <span><i style="background:#548235"></i>start</span>
  <span><i style="background:#c00000"></i>goal</span>
  <span><i style="background:#1f4e79"></i>planned path</span>
 </div>
 <div class="figcap"><b>Figure 1.</b> Interactive traversability map and planner. Green cells are
 traversable, red cells are obstacles, and grey cells were not observed by the sensor. The blue curve is
 the least-cost A* path between the user-selected start (green) and goal (red).</div>
</div>

<h2>4. Experimental Findings</h2>
<p>Validation across synthetic, automotive, and indoor data produced three transferable observations.</p>
<ol>
 <li><b>Traversability thresholds are sensor-specific.</b> The same implementation handles clean
 automotive LiDAR (KITTI) at default settings, whereas a single-view indoor RGB-D floor exhibits
 approximately 15&ndash;20&nbsp;cm of depth noise per cell and requires looser tolerances and stronger
 smoothing.</li>
 <li><b>Multi-scan fusion requires accurate localization.</b> Fusing two scans captured from different
 robot poses inflated the obstacle count through ghosting, whereas fusing a scan with itself did not.
 This demonstrates empirically why the stack requires an ICP/SLAM localization stage.</li>
 <li><b>The global planner is essential.</b> Potential-field avoidance alone becomes trapped in local
 minima at large obstacles; pairing it with a global path resolves the entrapment.</li>
</ol>

<h2>5. Implementation</h2>
<p>The mapping, planning, smoothing, and avoidance components are implemented in Python using only the
NumPy numerical library, with no dependence on external robotics frameworks for the core algorithms, and
are covered by an automated test suite. A ROS 2 / Nav2 integration path publishes the map as a costmap
layer for deployment. For this demonstration page, the A* planner is reimplemented in client-side
JavaScript so that it runs entirely in the browser.</p>

<footer>Interactive planner runs client-side; all terrain maps are precomputed by the Python pipeline.
This page is self-contained and requires no network connection.</footer>

</div>

<script id="scenes" type="application/json">__DATA__</script>
<script>
const SC=JSON.parse(document.getElementById('scenes').textContent);
const names=Object.keys(SC);
const sel=document.getElementById('scene');
names.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;sel.appendChild(o);});
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
const CLR={0:'#a8d08d',1:'#c0504d',2:'#dcdcdc'};
let D,cell,start,goal;

function setStatus(m){document.getElementById('status').textContent=m;}
function load(name){
 D=SC[name]; start=goal=null;
 cell=Math.max(3,Math.floor(700/Math.max(D.W,D.H)));
 cv.width=D.W*cell; cv.height=D.H*cell;
 document.getElementById('meta').textContent=
   D.W+' × '+D.H+' cells at '+D.res+' m resolution — '+D.free+' traversable, '+D.lethal+' obstacle, '+D.unknown+' unknown';
 base(); setStatus('Click to set start');
}
function base(){for(let r=0;r<D.H;r++)for(let c=0;c<D.W;c++){
 cx.fillStyle=CLR[D.cls[r*D.W+c]];cx.fillRect(c*cell,(D.H-1-r)*cell,cell,cell);}}
function px(r,c){return [c*cell+cell/2,(D.H-1-r)*cell+cell/2];}
function passable(r,c){return r>=0&&r<D.H&&c>=0&&c<D.W&&D.cost[r*D.W+c]>=0;}
function snap(r,c){if(passable(r,c))return[r,c];
 for(let k=1;k<60;k++)for(let dr=-k;dr<=k;dr++)for(let dc=-k;dc<=k;dc++)
   if(passable(r+dr,c+dc))return[r+dr,c+dc];return null;}
function dot(r,c,col,rad){const[x,y]=px(r,c);cx.beginPath();cx.arc(x,y,rad,0,7);cx.fillStyle=col;cx.fill();
 cx.lineWidth=1.5;cx.strokeStyle='#333';cx.stroke();}

function hpush(h,it){h.push(it);let i=h.length-1;while(i>0){let p=(i-1)>>1;if(h[p][0]<=h[i][0])break;[h[p],h[i]]=[h[i],h[p]];i=p;}}
function hpop(h){const t=h[0],l=h.pop();if(h.length){h[0]=l;let i=0,n=h.length;for(;;){let a=2*i+1,b=2*i+2,s=i;
 if(a<n&&h[a][0]<h[s][0])s=a;if(b<n&&h[b][0]<h[s][0])s=b;if(s===i)break;[h[s],h[i]]=[h[i],h[s]];i=s;}}return t;}
const NB=[[-1,0,1],[1,0,1],[0,-1,1],[0,1,1],[-1,-1,Math.SQRT2],[-1,1,Math.SQRT2],[1,-1,Math.SQRT2],[1,1,Math.SQRT2]];
function astar(s,g){const W=D.W,co=D.cost,si=s[0]*W+s[1],gi=g[0]*W+g[1],N=W*D.H;
 const gs=new Float64Array(N).fill(Infinity),cm=new Int32Array(N).fill(-1),cl=new Uint8Array(N);gs[si]=0;
 const oc=(a,b)=>{const dr=Math.abs((a/W|0)-(b/W|0)),dc=Math.abs(a%W-b%W);return (dr+dc)+(Math.SQRT2-2)*Math.min(dr,dc);};
 const h=[];hpush(h,[0,si]);
 while(h.length){const cur=hpop(h)[1];
  if(cur===gi){const p=[];let x=cur;while(x!==-1){p.push([x/W|0,x%W]);x=cm[x];}return p.reverse();}
  if(cl[cur])continue;cl[cur]=1;const cr=cur/W|0,cc=cur%W;
  for(const[dr,dc,dd]of NB){const nr=cr+dr,nc=cc+dc;if(!passable(nr,nc))continue;
   if(dr&&dc&&!(passable(cr+dr,cc)&&passable(cr,cc+dc)))continue;
   const ni=nr*W+nc,t=gs[cur]+dd*0.5*(co[cur]+co[ni]);
   if(t<gs[ni]){gs[ni]=t;cm[ni]=cur;hpush(h,[t+oc(ni,gi),ni]);}}}
 return null;}

cv.addEventListener('click',e=>{const rect=cv.getBoundingClientRect();
 const sx=cv.width/rect.width, sy=cv.height/rect.height;
 const c=Math.floor((e.clientX-rect.left)*sx/cell);
 const r=(D.H-1)-Math.floor((e.clientY-rect.top)*sy/cell);
 const s=snap(r,c); if(!s){setStatus('No traversable cell there — try again');return;}
 if(start&&goal){start=goal=null;base();}
 if(!start){start=s;dot(start[0],start[1],'#548235',cell*0.55+3);setStatus('Click to set goal');}
 else{goal=s;dot(goal[0],goal[1],'#c00000',cell*0.6+3);
  const p=astar(start,goal);
  if(!p){setStatus('No path — start and goal are in separate regions');return;}
  cx.beginPath();p.forEach((q,i)=>{const[x,y]=px(q[0],q[1]);i?cx.lineTo(x,y):cx.moveTo(x,y);});
  cx.strokeStyle='#1f4e79';cx.lineWidth=Math.max(2,cell*0.4);cx.lineJoin='round';cx.stroke();
  dot(start[0],start[1],'#548235',cell*0.55+3);dot(goal[0],goal[1],'#c00000',cell*0.6+3);
  let L=0;for(let i=1;i<p.length;i++)L+=Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1])*D.res;
  setStatus('Path length: '+L.toFixed(2)+' m — click to plan another');}
});
sel.addEventListener('change',()=>load(sel.value));
document.getElementById('reset').addEventListener('click',()=>{start=goal=null;base();setStatus('Click to set start');});
load(names[0]);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="index.html")
    args = ap.parse_args()
    scenes = collect()
    html = PAGE.replace("__DATA__", json.dumps(scenes, separators=(",", ":")))
    with open(args.out, "w") as f:
        f.write(html)
    kb = os.path.getsize(args.out) / 1024
    print("wrote %s (%.0f KB) with %d scenes: %s" % (
        args.out, kb, len(scenes), ", ".join(scenes)))


if __name__ == "__main__":
    main()
