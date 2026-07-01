#!/usr/bin/env python3
"""Write a self-contained interactive path-planning HTML demo.

Click the map to set START, click again for GOAL --- A* runs in the browser and
draws the path. Works offline (no server, no CDN). Regenerate for any cloud.

    python3 examples/export_plan_html.py                       # synthetic scene
    python3 examples/export_plan_html.py --cloud apt_sub.npy --rgbd --out apt_plan.html
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


def build_grid(args):
    if args.cloud:
        cloud = load_point_cloud(args.cloud)
        if args.rgbd:
            from test_real_room import detect_up_axis, reorient
            up = detect_up_axis(cloud)
            cloud, _ = reorient(cloud, up)
            robot = QuadrupedParams(radius=0.15, max_step=0.15,
                                    max_roughness=0.22, max_slope_deg=45)
            cfg = GridConfig(resolution=args.res, z_clip=(-0.10, 1.8), smooth_passes=4)
        elif args.kitti:
            # automotive LiDAR: crop a square window around the sensor and z-clip,
            # else auto-bounds spans ~160 m and the grid explodes.
            half = args.map_size / 2.0
            robot = QuadrupedParams()
            cfg = GridConfig(resolution=args.res, bounds=(-half, -half, half, half),
                             z_clip=(args.zmin, args.zmax))
        else:
            robot = QuadrupedParams()
            cfg = GridConfig(resolution=args.res)
    else:
        cloud = make_scene()
        robot = QuadrupedParams()
        cfg = GridConfig(resolution=args.res)
    return discretize(cloud, cfg, robot)


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Path planning demo</title>
<style>
 body{margin:0;background:#15171c;color:#cfd2d8;font-family:system-ui,sans-serif;
   display:flex;flex-direction:column;align-items:center;padding:16px;gap:10px}
 #bar{display:flex;gap:14px;align-items:center;font-size:14px}
 #status{font-weight:600;color:#eaecef}
 button{background:#262a31;color:#dfe2e8;border:1px solid #3a3f47;border-radius:7px;
   padding:6px 12px;font-size:13px;cursor:pointer}
 button:hover{background:#31363e}
 canvas{border:1px solid #2c2f36;border-radius:6px;cursor:crosshair;image-rendering:pixelated}
 .legend{display:flex;gap:14px;font-size:12px;color:#9aa0aa}
 .legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:1px}
</style></head><body>
<div id="bar"><span id="status">Click to set START</span>
 <button id="reset">Reset</button></div>
<div class="legend">
 <span><i style="background:#8cc772"></i>traversable</span>
 <span><i style="background:#b82929"></i>obstacle</span>
 <span><i style="background:#cccccc"></i>unknown</span>
 <span><i style="background:#39d353"></i>start</span>
 <span><i style="background:#e2483f"></i>goal</span>
 <span><i style="background:#3aa0ff"></i>path</span>
</div>
<canvas id="cv"></canvas>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const W=D.W,H=D.H,cost=D.cost,cls=D.cls;
const cell=Math.max(3,Math.floor(760/Math.max(W,H)));
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
cv.width=W*cell; cv.height=H*cell;
const CLR={0:'#8cc772',1:'#b82929',2:'#cccccc'};
function base(){
 for(let r=0;r<H;r++)for(let c=0;c<W;c++){
   cx.fillStyle=CLR[cls[r*W+c]];
   cx.fillRect(c*cell,(H-1-r)*cell,cell,cell);
 }
}
function px(r,c){return [c*cell+cell/2,(H-1-r)*cell+cell/2];}
function passable(r,c){return r>=0&&r<H&&c>=0&&c<W&&cost[r*W+c]>=0;}

function snap(r,c){
 if(passable(r,c))return [r,c];
 for(let rad=1;rad<50;rad++)for(let dr=-rad;dr<=rad;dr++)for(let dc=-rad;dc<=rad;dc++){
   if(passable(r+dr,c+dc))return [r+dr,c+dc];
 }
 return null;
}

function heapPush(h,item){h.push(item);let i=h.length-1;
 while(i>0){let p=(i-1)>>1;if(h[p][0]<=h[i][0])break;[h[p],h[i]]=[h[i],h[p]];i=p;}}
function heapPop(h){const top=h[0],last=h.pop();
 if(h.length){h[0]=last;let i=0;const n=h.length;
   for(;;){let l=2*i+1,r=2*i+2,s=i;
     if(l<n&&h[l][0]<h[s][0])s=l; if(r<n&&h[r][0]<h[s][0])s=r;
     if(s===i)break;[h[s],h[i]]=[h[i],h[s]];i=s;}}
 return top;}

const NB=[[-1,0,1],[1,0,1],[0,-1,1],[0,1,1],
  [-1,-1,Math.SQRT2],[-1,1,Math.SQRT2],[1,-1,Math.SQRT2],[1,1,Math.SQRT2]];
function octile(a,b){const dr=Math.abs((a/W|0)-(b/W|0)),dc=Math.abs(a%W-b%W);
 return (dr+dc)+(Math.SQRT2-2)*Math.min(dr,dc);}

function astar(s,g){
 const si=s[0]*W+s[1],gi=g[0]*W+g[1],N=W*H;
 const gs=new Float64Array(N).fill(Infinity),came=new Int32Array(N).fill(-1);
 const closed=new Uint8Array(N); gs[si]=0;
 const heap=[]; heapPush(heap,[0,si]);
 while(heap.length){
   const cur=heapPop(heap)[1];
   if(cur===gi){const p=[];let x=cur;while(x!==-1){p.push([x/W|0,x%W]);x=came[x];}return p.reverse();}
   if(closed[cur])continue; closed[cur]=1;
   const cr=cur/W|0,cc=cur%W;
   for(const [dr,dc,dd] of NB){
     const nr=cr+dr,ncc=cc+dc; if(!passable(nr,ncc))continue;
     if(dr&&dc&&!(passable(cr+dr,cc)&&passable(cr,cc+dc)))continue;
     const ni=nr*W+ncc, step=dd*0.5*(cost[cur]+cost[ni]);
     const t=gs[cur]+step;
     if(t<gs[ni]){gs[ni]=t;came[ni]=cur;heapPush(heap,[t+octile(ni,gi),ni]);}
   }
 }
 return null;
}

let start=null,goal=null;
function setStatus(m){document.getElementById('status').textContent=m;}
function dot(r,c,color,rad){const [x,y]=px(r,c);cx.beginPath();cx.arc(x,y,rad,0,7);cx.fillStyle=color;cx.fill();
 cx.lineWidth=1.5;cx.strokeStyle='#111';cx.stroke();}
function reset(){start=goal=null;base();setStatus('Click to set START');}

cv.addEventListener('click',e=>{
 const rect=cv.getBoundingClientRect();
 const c=Math.floor((e.clientX-rect.left)/cell);
 const r=(H-1)-Math.floor((e.clientY-rect.top)/cell);
 const snapped=snap(r,c);
 if(!snapped){setStatus('No free cell there --- try again');return;}
 if(start&&goal)reset();
 if(!start){start=snapped;dot(start[0],start[1],'#39d353',cell*0.55+3);setStatus('Click to set GOAL');}
 else{goal=snapped;dot(goal[0],goal[1],'#e2483f',cell*0.6+3);
   const path=astar(start,goal);
   if(!path){setStatus('No path (start/goal in separate regions)');return;}
   cx.beginPath();for(let i=0;i<path.length;i++){const [x,y]=px(path[i][0],path[i][1]);i?cx.lineTo(x,y):cx.moveTo(x,y);}
   cx.strokeStyle='#3aa0ff';cx.lineWidth=Math.max(2,cell*0.35);cx.lineJoin='round';cx.stroke();
   dot(start[0],start[1],'#39d353',cell*0.55+3);dot(goal[0],goal[1],'#e2483f',cell*0.6+3);
   let len=0;for(let i=1;i<path.length;i++){const dr=path[i][0]-path[i-1][0],dc=path[i][1]-path[i-1][1];len+=Math.hypot(dr,dc)*D.res;}
   setStatus('Path: '+len.toFixed(2)+' m  ---  click to plan another');
 }
});
document.getElementById('reset').addEventListener('click',reset);
base();
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cloud")
    p.add_argument("--rgbd", action="store_true", help="indoor RGB-D (auto-orient)")
    p.add_argument("--kitti", action="store_true", help="automotive LiDAR (crop + z-clip)")
    p.add_argument("--map-size", type=float, default=50.0, help="KITTI crop side (m)")
    p.add_argument("--zmin", type=float, default=-3.0)
    p.add_argument("--zmax", type=float, default=1.0)
    p.add_argument("--res", type=float, default=0.10)
    p.add_argument("--out", default="plan_demo.html")
    args = p.parse_args()

    grid = build_grid(args)
    cost = grid.cost
    flat_cost = [(-1.0 if not np.isfinite(v) else round(float(v), 3)) for v in cost.ravel()]
    data = {"W": int(grid.shape[1]), "H": int(grid.shape[0]),
            "res": float(grid.resolution),
            "cost": flat_cost, "cls": [int(v) for v in grid.classes.ravel()]}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(args.out, "w") as f:
        f.write(html)
    print("grid %dx%d @ %.2fm  free=%d lethal=%d unknown=%d" % (
        grid.shape[0], grid.shape[1], grid.resolution,
        grid.meta["observed_cells"] - grid.meta["lethal_cells"],
        grid.meta["lethal_cells"], grid.meta["unknown_cells"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
