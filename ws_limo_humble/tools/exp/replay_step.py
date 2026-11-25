#!/usr/bin/env python3
"""Open-loop replay of a logged run through FrenetPlanner.step() with a simulated LiDAR.

usage: replay_step.py <planner.py> [--no-memory] [--since SECONDS_BEFORE_END]
Ray-casts the world boxes and the map walls from the logged Gazebo poses at 8 Hz. The planner sees
logged map poses and logged Pure Pursuit commands. Its output does not move the car, so this
shows what it would choose at those poses, not what the car would have done. Map walls are the
SLAM map's, not Gazebo's.
"""
import argparse
import csv
import importlib.util
import math

import numpy as np
import yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('planner')
ap.add_argument('--run', default='logs/exp/exp-006b/run.csv')
ap.add_argument('--no-memory', action='store_true')
ap.add_argument('--since', type=float, default=14.0)
ap.add_argument('--every', type=float, default=0.4)
ap.add_argument('--set', action='append', default=[], help='planner param, e.g. r_min=0.175')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('fp', a.planner)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)
LX, RMIN, RMAX, FOV, N = 0.124, 0.2, 8.0, 2.0943951, 720
ANG = fp.np.linspace(-FOV, FOV, N)
BOXES = [(5.0, -0.0878, .15, .4), (-3.0, 3.0542, .15, .4), (-8.0, 1.1452, .15, .5)]
cfg = yaml.safe_load(open('src/Raceline-Optimization/maps/track_map3.yaml'))
img = np.array(Image.open('src/Raceline-Optimization/maps/track_map3.pgm'))
H, W = img.shape
res, (ox, oy) = cfg['resolution'], cfg['origin'][:2]
occ = img < 100
STEPS = np.arange(RMIN, 5.2, 0.01)


def lidar(x, y, yaw):
    lx, ly = x + LX * math.cos(yaw), y + LX * math.sin(yaw)
    wx = lx + np.outer(STEPS, np.cos(ANG + yaw))
    wy = ly + np.outer(STEPS, np.sin(ANG + yaw))
    col = np.clip(((wx - ox) / res).astype(int), 0, W - 1)
    row = np.clip((H - 1 - (wy - oy) / res).astype(int), 0, H - 1)
    hit = occ[row, col]
    for cx, cy, hx, hy in BOXES:
        hit |= (np.abs(wx - cx) <= hx) & (np.abs(wy - cy) <= hy)
    first = hit.argmax(axis=0)
    ok = hit.any(axis=0) & (first > 0)     # a beam already inside at range_min returns nothing
    r = STEPS[first[ok]]
    return np.c_[r * np.cos(ANG[ok]) + LX, r * np.sin(ANG[ok])]


rows = list(csv.DictReader(open(a.run)))
col = {k: np.array([float(r[k]) for r in rows]) for k in
       ('t', 'x', 'y', 'yaw', 'map_x', 'map_y', 'map_yaw', 'pp_v', 'pp_steer', 'cmd_v', 'cmd_steer')}
xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
p = fp.default_params()
for item in a.set:
    k, v = item.split('=')
    old = getattr(p, k)
    setattr(p, k, tuple(float(x) for x in v.split(',')) if isinstance(old, tuple) else type(old)(float(v)))
P = fp.FrenetPlanner(xy, p)
mem = fp.UnseenMemory(p.memory_hold, p.memory_keep_r, p.memory_cell, p.memory_rmin_margin)
t_end = col['t'][-1]
last_scan, pts, last_print = -1e9, np.zeros((0, 2)), -1e9
print('  t_end  mode    n_ok  rad    min_rad v_out steer logged_v logged_st mem fb')
rads = []
no_cand = zero_v = ticks = escape = creep = 0
t0 = t_end - a.since
for i in range(len(rows)):
    t = col['t'][i]
    if t < t0 - 6.0:
        continue
    gt = (col['x'][i], col['y'][i], col['yaw'][i])
    if t - last_scan >= 0.125:
        pts = lidar(*gt)
        mem.add(t, pts, gt)
        last_scan = t
    unseen = None if a.no_memory else mem.unseen(t, gt, (LX, 0.0), RMIN, FOV)
    if t < t0:
        continue
    v, steer, info = P.step((col['map_x'][i], col['map_y'][i], col['map_yaw'][i]), pts,
                            col['pp_v'][i], col['pp_steer'][i], t, unseen)
    n = len(info['cands']) if info.get('cands') is not None else 0
    # last_rad is stale in FOLLOW and resets on no-candidate ticks: count real passes only.
    if info['mode'] == 'AVOID' and n > 0:
        rads.append(P.last_rad)
    no_cand += n == 0 and info['mode'] != 'FOLLOW'
    zero_v += v == 0.0
    ticks += 1
    if info['mode'] == 'CRAWL':
        if 'escape_clearance' in info:
            escape += 1
        else:
            creep += 1
    if t - last_print >= a.every:
        last_print = t
        print(f"{t - t_end:6.1f}  {info['mode']:6s} {n:4d}  {P.last_rad:.3f}  {P.min_rad:.3f}  "
              f"{v:.2f}  {steer:+.2f}  {col['cmd_v'][i]:.2f}     {col['cmd_steer'][i]:+.2f}     "
              f"{0 if unseen is None else len(unseen):3d} {info.get('fallback', 0)}")
        P.min_rad = p.r_robot
rads = np.array(rads)
print('ticks', ticks, 'no-candidate', no_cand, 'zero-speed', zero_v, 'escape arc', escape, 'creep', creep)
print('AVOID ticks with a path', len(rads), 'share by radius', {r: round(float((rads == r).mean()), 3) for r in sorted(set(rads.tolist()))})
