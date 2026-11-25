#!/usr/bin/env python3
"""Replay exp-006b poses with a synthetic LiDAR: does the box corner stay in the collision points?

usage: test_unseen_memory.py <planner.py>. Boxes only, no walls, recorded poses: this tests what
the point set contains, not how the car would have driven with it.
"""
import csv
import importlib.util
import math
import sys

import numpy as np

spec = importlib.util.spec_from_file_location('fp', sys.argv[1])
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)
BOX = (-8.15, -7.85, 0.645, 1.645)
LX, RMIN, RMAX, FOV, N = 0.124, 0.2, 8.0, 2.0943951, 720
ANG = np.linspace(-FOV, FOV, N)


def scan(pose):
    """Ray-cast one box from the laser; returns base-frame points the sensor reports."""
    x, y, yaw = pose
    ox, oy = x + LX * math.cos(yaw), y + LX * math.sin(yaw)
    dx, dy = np.cos(ANG + yaw), np.sin(ANG + yaw)
    best = np.full(N, np.inf)
    for fixed, lo, hi, vertical in ((BOX[0], BOX[2], BOX[3], True), (BOX[1], BOX[2], BOX[3], True),
                                    (BOX[2], BOX[0], BOX[1], False), (BOX[3], BOX[0], BOX[1], False)):
        with np.errstate(divide='ignore', invalid='ignore'):
            if vertical:
                r = (fixed - ox) / dx
                ok = (r > 0) & (oy + r * dy >= lo) & (oy + r * dy <= hi)
            else:
                r = (fixed - oy) / dy
                ok = (r > 0) & (ox + r * dx >= lo) & (ox + r * dx <= hi)
        best = np.where(ok & (r < best), r, best)
    keep = (best > RMIN) & (best < RMAX)
    return np.c_[best[keep] * np.cos(ANG[keep]) + LX, best[keep] * np.sin(ANG[keep])]


def gap(pts):
    if not len(pts):
        return float('inf')
    return float(np.hypot(np.maximum(np.abs(pts[:, 0]) - .165, 0),
                          np.maximum(np.abs(pts[:, 1]) - .155, 0)).min())


rows = list(csv.DictReader(open('logs/exp/exp-006b/run.csv')))
col = {k: np.array([float(r[k]) for r in rows]) for k in ('t', 'x', 'y', 'yaw')}
T0 = 3878.7
idx = np.where((col['t'] > T0 - 12) & (col['t'] < T0 + 0.3))[0]
mem = fp.UnseenMemory(15.0, 1.0, 0.02, 0.03)
last = -1e9
fails = 0
print(' t-T0  true_gap  scan_only_gap  merged_gap  unseen_pts')
for n, i in enumerate(idx):
    t, pose = col['t'][i], (col['x'][i], col['y'][i], col['yaw'][i])
    pts = scan(pose)
    if t - last >= 0.125:
        mem.add(t, pts, pose)
        last = t
    extra = mem.unseen(t, pose, (LX, 0.0), RMIN, FOV)
    merged = np.vstack([pts, extra]) if len(extra) else pts
    c, s = math.cos(pose[2]), math.sin(pose[2])
    corner = (BOX[0] - pose[0], BOX[2] - pose[1])
    u, v = c * corner[0] + s * corner[1], -s * corner[0] + c * corner[1]
    true = math.hypot(max(abs(u) - .165, 0), max(abs(v) - .155, 0))
    if t - T0 > -1.4 and n % 3 == 0:
        print(f'{t - T0:5.1f}  {true:.3f}     {gap(pts):.3f}          {gap(merged):.3f}       {len(extra)}')
    if true < 0.1 and gap(merged) > true + 0.04:
        fails += 1
print('PASS' if fails == 0 else f'FAIL {fails}: merged points miss the corner')

# Visible area must not use memory: a point dead ahead at 0.5 m, then gone, is not returned.
m = fp.UnseenMemory(15.0, 1.0, 0.02, 0.03)
m.add(0.0, np.array([[0.5, 0.0]]), (0.0, 0.0, 0.0))
print('visible point not returned:', len(m.unseen(0.1, (0.0, 0.0, 0.0), (LX, 0.0), RMIN, FOV)) == 0)
# Same point once the car has moved so it is beside it, inside range_min: returned, then expires.
pose = (0.5, 0.15, 0.0)
print('blind point returned:', len(m.unseen(1.0, pose, (LX, 0.0), RMIN, FOV)) == 1)
print('blind point expires:', len(m.unseen(16.0, pose, (LX, 0.0), RMIN, FOV)) == 0)
# Behind the field of view.
print('rear point returned:', len(m.unseen(1.0, (0.9, 0.0, 0.0), (LX, 0.0), RMIN, FOV)) == 1)
