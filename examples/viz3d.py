#!/usr/bin/env python3
"""3D visualization of the 2.5D terrain map with the planned path draped on it.

Renders the elevation grid as a 3D surface coloured by traversability
(green = free, red = obstacle), with the A*/Dijkstra path floating just above the
terrain so you can see *where* it goes and *what* it avoids. Unknown cells are left
as holes in the surface.

    python3 examples/viz3d.py                       # synthetic scene
    python3 examples/viz3d.py --bin test-scans/000001.bin --range 25
    python3 examples/viz3d.py --room                 # Open3D indoor room scan

Writes out_3d.png (two viewing angles).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401 (registers 3d)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_pathplan import QuadrupedParams, GridConfig, discretize, astar
from lidar_pathplan.elevation_grid import FREE, LETHAL, UNKNOWN
from lidar_pathplan.synthetic import make_scene
from lidar_pathplan.io_utils import load_point_cloud


# ---------------------------------------------------------------- terrain source
def synthetic_grid():
    cloud = make_scene()
    grid = discretize(cloud, GridConfig(resolution=0.10), QuadrupedParams())
    start = _nudge(grid, grid.world_to_cell(0.5, 0.5))
    goal = _nudge(grid, grid.world_to_cell(9.5, 5.0))
    return grid, astar(grid.cost, start, goal), 0.5


def kitti_grid(path_bin, half):
    cloud = load_point_cloud(path_bin)
    config = GridConfig(resolution=0.25, bounds=(-half, -half, half, half),
                        z_clip=(-3.0, 1.0))
    grid = discretize(cloud, config=config, robot=QuadrupedParams())
    start = _nudge(grid, grid.world_to_cell(4.0, 0.0))
    goal = _nudge(grid, grid.world_to_cell(min(half - 2.0, 22.0), 0.0))
    return grid, astar(grid.cost, start, goal), 0.6


def room_grid():
    # Reuse the room harness loader + reorientation.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_real_room import fetch_sample, detect_up_axis, reorient, _pick_endpoints
    raw = load_point_cloud(fetch_sample())
    up = detect_up_axis(raw)
    cloud, _ = reorient(raw, up)
    robot = QuadrupedParams(radius=0.15, max_step=0.15, max_roughness=0.22, max_slope_deg=45)
    grid = discretize(cloud, GridConfig(0.08, z_clip=(-0.10, 1.5), smooth_passes=4), robot)
    s, g = _pick_endpoints(grid)
    return grid, astar(grid.cost, s, g), 0.15


def _nudge(grid, cell, max_r=40):
    if grid.in_bounds(*cell) and np.isfinite(grid.cost[cell]):
        return cell
    h, w = grid.shape
    for r in range(1, max_r):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                rr, cc = cell[0] + dr, cell[1] + dc
                if 0 <= rr < h and 0 <= cc < w and np.isfinite(grid.cost[rr, cc]):
                    return (rr, cc)
    raise ValueError("no free cell near %s" % (cell,))


# ---------------------------------------------------------------- 3d rendering
def _class_facecolors(grid):
    """RGBA array colouring each cell by traversability class."""
    rgba = np.zeros(grid.shape + (4,), dtype=np.float32)
    rgba[grid.classes == FREE] = (0.55, 0.78, 0.45, 1.0)     # green
    rgba[grid.classes == LETHAL] = (0.72, 0.16, 0.16, 1.0)   # red
    rgba[grid.classes == UNKNOWN] = (0.85, 0.85, 0.85, 0.25) # faint grey
    return rgba


def _draped_path(grid, path, z_offset):
    """World-frame (x, y, z) of the path, floated z_offset above the terrain."""
    xs, ys, zs = [], [], []
    elev = grid.elevation
    fallback = np.nanmin(elev) if np.isfinite(elev).any() else 0.0
    for r, c in path:
        x, y = grid.cell_to_world(r, c)
        z = elev[r, c]
        if not np.isfinite(z):
            z = fallback
        xs.append(x); ys.append(y); zs.append(z + z_offset)
    return np.array(xs), np.array(ys), np.array(zs)


def render(grid, path, z_offset, out, stride=1, z_exag=0.5, title=""):
    res = grid.resolution
    h, w = grid.shape
    xs = grid.origin[0] + (np.arange(w) + 0.5) * res
    ys = grid.origin[1] + (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    Z = grid.elevation.astype(float).copy()

    # Downsample huge grids so plot_surface stays responsive.
    if stride > 1:
        X, Y, Z = X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride]
        face = _class_facecolors(grid)[::stride, ::stride]
    else:
        face = _class_facecolors(grid)

    fig = plt.figure(figsize=(18, 8))
    for i, (elev_ang, azim) in enumerate([(72, -88), (22, -55)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=face, rcount=Z.shape[0], ccount=Z.shape[1],
                        linewidth=0, antialiased=False, shade=False)
        if path:
            px, py, pz = _draped_path(grid, path, z_offset)
            # black underlay + bright cyan on top so the path reads against terrain
            ax.plot(px, py, pz, color="black", linewidth=6, solid_capstyle="round")
            ax.plot(px, py, pz, color="cyan", linewidth=3.5, label="path", solid_capstyle="round")
            ax.scatter(px[0], py[0], pz[0], color="lime", s=130, edgecolor="black",
                       depthshade=False, label="start")
            ax.scatter(px[-1], py[-1], pz[-1], color="red", s=200, marker="*",
                       edgecolor="black", depthshade=False, label="goal")
        # vertical exaggeration so steps/walls are legible
        xr = xs[-1] - xs[0]; yr = ys[-1] - ys[0]
        ax.set_box_aspect((xr, yr, z_exag * max(xr, yr)))
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("height (m)")
        ax.view_init(elev=elev_ang, azim=azim)
        if i == 0 and path:
            ax.legend(loc="upper left")
        ax.set_title("view %d" % (i + 1), fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#15171c;color:#cfd2d8;font-family:system-ui,sans-serif;overflow:hidden}
  #legend{position:fixed;top:12px;left:12px;background:#1e2127;border:1px solid #2c2f36;
          border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.9;z-index:2}
  #legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:7px;vertical-align:1px}
  #hint{position:fixed;bottom:12px;right:14px;font-size:13px;color:#80858f;z-index:2}
  #title{position:fixed;top:12px;right:14px;font-size:13px;color:#a0a4ad;z-index:2}
  canvas{display:block}
</style></head><body>
<div id="legend">
  <div><i style="background:#8cc772"></i>traversable</div>
  <div><i style="background:#b82929"></i>obstacle</div>
  <div><i style="background:#cccccc"></i>unknown</div>
  <div><i style="background:#27c4d4"></i>planned path</div>
</div>
<div id="title">__TITLE__</div>
<div id="hint">drag to rotate &middot; scroll to zoom</div>
<canvas id="cv"></canvas>
<script id="data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
  var D=JSON.parse(document.getElementById('data').textContent);
  var canvas=document.getElementById('cv');
  var W=D.W,H=D.H,res=D.res,exag=D.exag;
  var cx=D.ox+W*res/2, cy=D.oy+H*res/2;
  var scene=new THREE.Scene();
  var camera=new THREE.PerspectiveCamera(48, innerWidth/innerHeight, 0.1, 5000);
  var renderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  function resize(){renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();}
  var pos=new Float32Array(W*H*3),col=new Float32Array(W*H*3);
  var FREE=[0.55,0.78,0.45],OBS=[0.72,0.16,0.16],UNK=[0.80,0.80,0.80];
  for(var r=0;r<H;r++)for(var c=0;c<W;c++){var i=r*W+c;
    pos[i*3]=D.ox+(c+0.5)*res-cx; pos[i*3+1]=D.z[i]*exag; pos[i*3+2]=D.oy+(r+0.5)*res-cy;
    var k=D.cls[i],cc=k===1?OBS:(k===2?UNK:FREE); col[i*3]=cc[0];col[i*3+1]=cc[1];col[i*3+2]=cc[2];}
  var idx=[];for(var r2=0;r2<H-1;r2++)for(var c2=0;c2<W-1;c2++){
    var a=r2*W+c2,b=a+1,d=a+W,e=d+1; if(D.z[a]>-90&&D.z[b]>-90&&D.z[d]>-90&&D.z[e]>-90){idx.push(a,d,b,b,d,e);}}
  var g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));
  g.setAttribute('color',new THREE.BufferAttribute(col,3));
  g.setIndex(idx); g.computeVertexNormals();
  scene.add(new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.95,side:THREE.DoubleSide})));
  if(D.path.length>1){
    var pts=D.path.map(function(p){return new THREE.Vector3(p[0]-cx,p[2]*exag+0.18*exag/4,p[1]-cy);});
    var span=Math.max(W,H)*res;
    var tube=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),Math.max(60,pts.length*2),span*0.006+0.05,10,false);
    scene.add(new THREE.Mesh(tube,new THREE.MeshStandardMaterial({color:0x27c4d4,emissive:0x0c6b75,roughness:0.4})));
    [[pts[0],0x39d353],[pts[pts.length-1],0xe2483f]].forEach(function(m){
      var s=new THREE.Mesh(new THREE.SphereGeometry(span*0.018+0.1,16,16),
        new THREE.MeshStandardMaterial({color:m[1],emissive:m[1],emissiveIntensity:0.35}));
      s.position.copy(m[0]); scene.add(s);});
  }
  scene.add(new THREE.HemisphereLight(0xffffff,0x303038,0.9));
  var dir=new THREE.DirectionalLight(0xffffff,0.8); dir.position.set(6,12,4); scene.add(dir);
  var theta=-0.9,phi=0.95,radius=Math.max(W,H)*res*1.5,drag=false,lx=0,ly=0,auto=true;
  function cam(){camera.position.set(radius*Math.sin(phi)*Math.cos(theta),radius*Math.cos(phi),radius*Math.sin(phi)*Math.sin(theta));camera.lookAt(0,0.6,0);}
  canvas.addEventListener('pointerdown',function(e){drag=true;auto=false;lx=e.clientX;ly=e.clientY;});
  addEventListener('pointerup',function(){drag=false;});
  addEventListener('pointermove',function(e){if(!drag)return;theta-=(e.clientX-lx)*0.006;phi-=(e.clientY-ly)*0.006;phi=Math.max(0.18,Math.min(1.45,phi));lx=e.clientX;ly=e.clientY;});
  canvas.addEventListener('wheel',function(e){e.preventDefault();radius*=(1+(e.deltaY>0?0.08:-0.08));radius=Math.min(Math.max(radius,span_min()),5000);},{passive:false});
  function span_min(){return Math.max(W,H)*res*0.25;}
  function loop(){if(auto)theta+=0.0022;cam();renderer.render(scene,camera);requestAnimationFrame(loop);}
  resize(); addEventListener('resize',resize); loop();
})();
</script></body></html>
"""


def export_html(grid, path, out, z_exag, title):
    """Write a self-contained interactive 3D HTML page openable in any browser."""
    import json
    elev = np.nan_to_num(grid.elevation, nan=-99.0)
    H, W = grid.shape
    data = {
        "W": int(W), "H": int(H), "res": float(grid.resolution),
        "ox": float(grid.origin[0]), "oy": float(grid.origin[1]),
        "exag": float(z_exag),
        "z": [round(float(v), 3) for v in elev.ravel()],
        "cls": [int(v) for v in grid.classes.ravel()],
        "path": ([] if path is None else
                 [[round(grid.cell_to_world(r, c)[0], 3),
                   round(grid.cell_to_world(r, c)[1], 3),
                   round(float(elev[r, c] if elev[r, c] > -90 else 0.0), 3)] for r, c in path]),
    }
    html = (_HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__TITLE__", title))
    with open(out, "w") as f:
        f.write(html)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", help="KITTI .bin scan to render in 3D")
    p.add_argument("--room", action="store_true", help="render the indoor room scan")
    p.add_argument("--range", type=float, default=25.0, help="half-size for KITTI crop (m)")
    p.add_argument("--html", action="store_true",
                   help="write an interactive standalone HTML file instead of a PNG")
    p.add_argument("--out", default=None, help="output path (defaults to out_3d.png / .html)")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = "out_3d.html" if args.html else "out_3d.png"

    if args.bin:
        grid, path, zoff = kitti_grid(args.bin, args.range)
        stride, z_exag, title = 1, 0.25, "KITTI scan: terrain + path (%s)" % os.path.basename(args.bin)
    elif args.room:
        grid, path, zoff = room_grid()
        stride, z_exag, title = 1, 0.5, "Indoor room scan: terrain + path"
    else:
        grid, path, zoff = synthetic_grid()
        stride, z_exag, title = 1, 0.45, "Synthetic terrain: 2.5D map + planned path"

    if path is None:
        print("WARNING: no path found; rendering terrain only")
    if args.html:
        # HTML uses a 3-4x vertical exaggeration factor directly (not the matplotlib
        # box-aspect ratio), so translate the small box ratios into a real z scale.
        html_exag = 4.0 if zoff < 0.3 else 3.0
        export_html(grid, path, args.out, html_exag, title)
    else:
        render(grid, path, zoff, args.out, stride=stride, z_exag=z_exag, title=title)
    print("wrote", args.out,
          "| grid %dx%d, path %s cells" % (grid.shape[0], grid.shape[1],
                                           "0" if path is None else len(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
