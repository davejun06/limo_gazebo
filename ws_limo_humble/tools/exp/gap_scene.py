#!/usr/bin/env python3
"""One scene: real corridor at a box, what the scan sees, the raw and refined path through it."""
import math, sys
import numpy as np, yaml
from PIL import Image
sys.dont_write_bytecode = True
sys.path.insert(0, 'src')
from hybrid_astar_local import REFINE_FAIL, HybridAStar, center_and_smooth, footprint_clearance
cfg = yaml.safe_load(open('src/Raceline-Optimization/maps/track_map3.yaml'))
img = np.array(Image.open('src/Raceline-Optimization/maps/track_map3.pgm')); H, W = img.shape
res, (ox, oy) = cfg['resolution'], cfg['origin'][:2]; occ = img < 100
LX, FOV = 0.124, 2.0943951; ANG = np.linspace(-FOV, FOV, 720); STEPS = np.arange(0.2, 5.2, 0.01)
BOXES = [(5.0, -0.087847578, .15, .4), (-3.0, 3.054242265, .15, .4), (-8.0, 1.145160234, .15, .5)]
def lidar(x, y, yaw):
    lx, ly = x + LX * math.cos(yaw), y + LX * math.sin(yaw)
    wx = lx + np.outer(STEPS, np.cos(ANG + yaw)); wy = ly + np.outer(STEPS, np.sin(ANG + yaw))
    c = np.clip(((wx - ox) / res).astype(int), 0, W - 1); r = np.clip((H - 1 - (wy - oy) / res).astype(int), 0, H - 1)
    hit = occ[r, c]
    for cx, cy, hx, hy in BOXES:
        hit = hit | ((np.abs(wx - cx) <= hx) & (np.abs(wy - cy) <= hy))
    f = hit.argmax(axis=0); ok = hit.any(axis=0) & (f > 0); d = STEPS[f[ok]]
    return np.c_[d * np.cos(ANG[ok]) + LX, d * np.sin(ANG[ok])]
print('wall above o2 from the map, lowest occupied edge y by x:')
for x in (-2.6, -2.8, -3.0, -3.2, -3.4):
    col = int((x - ox) / res); rows = np.nonzero(occ[:, col])[0]; ys = oy + (H - rows) * res - res
    up = ys[ys > 3.46]; print(f'  x={x}: wall y={up.min():.3f}, gap to box top 3.454 = {up.min() - 3.454:.3f}, centre {(up.min() + 3.454) / 2:.3f}')
x, y, yaw = -1.6, 3.85, 3.4
c, s = math.cos(yaw), math.sin(yaw)
pts = lidar(x, y, yaw)
def world(b): return np.c_[x + c * b[:, 0] - s * b[:, 1], y + s * b[:, 0] + c * b[:, 1]]
near = world(pts); m = (near[:, 0] < -2.7) & (near[:, 0] > -3.3) & (near[:, 1] > 3.4)
print('scan points beside o2: below', np.round(near[m & (near[:, 1] < 3.6)][:, 1].max(), 3) if (m & (near[:, 1] < 3.6)).any() else None, 'above', np.round(near[m & (near[:, 1] > 3.6)][:, 1].min(), 3) if (m & (near[:, 1] > 3.6)).any() else None)
goal_w = np.array([[-4.5, 3.95], [-4.3, 3.85], [-4.1, 3.76]]); dx, dy = goal_w[:, 0] - x, goal_w[:, 1] - y
goal = np.c_[c * dx + s * dy, -s * dx + c * dy]; gy = np.full(3, 2.8 - yaw)
ha = HybridAStar(0.24, math.atan(0.9 * math.tan(0.42)))
ha = HybridAStar(0.24, math.atan(0.9 * math.tan(0.42) / 0.24 * 0.24))
path = ha.plan(pts, goal, gy, 0.21, 0.20, 0.0, 3.0); print('search', ha.stats)
def at(xy, q): return round(float(xy[np.argmin(np.abs(xy[:, 0] - q)), 1]), 3)
if path is not None:
    wr = world(path); print('raw path y at x -2.6..-3.4:', [at(wr, q) for q in (-2.6, -2.8, -3.0, -3.2, -3.4)])
    for iters in (10, 16, 30):
        ref = center_and_smooth(path, pts, 0.21, 0.20, 0.9 * math.tan(0.42) / 0.24, iters=iters)
        print(f'iters {iters}:', None if ref is None else [at(world(ref), q) for q in (-2.6, -2.8, -3.0, -3.2, -3.4)], REFINE_FAIL)
