#!/usr/bin/env python3
"""Does a forward-only path past a box exist from poses along a route? Full map, long search.

Geometry only: known walls and boxes, no sensor limits, no tracking error. usage: o3_feasible.py <sim csv>
"""
import csv, math, re, sys
import numpy as np, yaml
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, 'src')
from hybrid_astar_local import HybridAStar, footprint_clearance
src = open('tools/exp/run_monitor.py').read(); ns = {'math': math}
exec(re.search(r'^OBSTACLES = .*$', src, re.M).group(0), ns)
OB = ns['OBSTACLES']
cfg = yaml.safe_load(open('src/Raceline-Optimization/maps/track_map3.yaml'))
img = np.array(Image.open('src/Raceline-Optimization/maps/track_map3.pgm')); H, W = img.shape
res, (ox, oy) = cfg['resolution'], cfg['origin'][:2]
r, c = np.nonzero(img < 100)
walls = np.c_[ox + (c + .5) * res, oy + (H - r - .5) * res]
cx, cy, hx, hy = OB['o3']
gx, gy = np.meshgrid(np.arange(cx - hx, cx + hx + .001, .02), np.arange(cy - hy, cy + hy + .001, .02))
box = np.c_[gx.ravel(), gy.ravel()]
box = box[(np.abs(box[:, 0] - cx) > hx - .021) | (np.abs(box[:, 1] - cy) > hy - .021)]
ALL = np.vstack([walls, box])
xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
d = np.diff(xy, axis=0); th = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); th = np.r_[th, th[-1]]
s_o3 = np.argmin(np.hypot(xy[:, 0] + 8.3, xy[:, 1] - 0.42))
print('route nearest south of o3 idx', s_o3, xy[s_o3], 'route yaw', round(float(th[s_o3]), 2))
goal_ids = np.arange(s_o3 + 8, s_o3 + 16) % len(xy)
print('goal', xy[goal_ids[0]], xy[goal_ids[-1]])
ha = HybridAStar(0.24, 0.42)   # same search settings as the planner
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
rows = list(csv.DictReader(open(sys.argv[1])))
seen = -9
for row in rows:
    pr = float(row['prog'])
    if pr < 24.5 or pr - seen < 0.5 or pr > 28.0:
        continue
    seen = pr
    x, y, yaw = float(row['x']), float(row['y']), float(row['yaw'])
    q, s = math.cos(yaw), math.sin(yaw)
    def base(a):
        dx, dy = a[:, 0] - x, a[:, 1] - y
        return np.c_[q * dx + s * dy, -s * dx + q * dy]
    P = base(ALL); P = P[(np.abs(P[:, 0]) < 5) & (np.abs(P[:, 1]) < 5)]
    out = []
    for rad in (0.20, 0.19, 0.18, 0.17):
        m = max(0, rad - 0.155)
        path = ha.plan(P, base(xy[goal_ids]), th[goal_ids] - yaw, 0.165 + m, 0.155 + m, 0.0, budget=BUDGET)
        st = ha.stats
        tag = f"{rad:.2f}:{st['result'][:5]}/{st['expanded']}/{st['ms']:.0f}ms"
        if path is not None:
            bare = footprint_clearance(path, P, 0.165, 0.155).min()
            tag += f"/len{len(path)*ha.step/ha.samples:.1f}/clr{bare:.3f}"
        out.append(tag)
    print(f"prog {pr:5.2f} ({x:6.2f},{y:5.2f},{yaw:5.2f})  " + ' '.join(out), flush=True)
