#!/usr/bin/env python3
"""Every path center_and_smooth accepts must meet the lattice's adjacent-point curvature limit."""
import math, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, 'src')
from hybrid_astar_local import REFINE_FAIL, HybridAStar, center_and_smooth
LIM = 0.9 * math.tan(0.42) / 0.24
ha = HybridAStar(0.24, math.atan(LIM * 0.24))
rng = np.random.default_rng(3)
ok = worst = raw_worst = 0
n = 300
for _ in range(n):
    goal = np.array([[3.0 + rng.uniform(0, 1), rng.uniform(-1.5, 1.5)]])
    box = rng.uniform([0.8, -1.0], [2.5, 1.0])
    pts = box + rng.uniform(-0.2, 0.2, (40, 2))
    path = ha.plan(pts, goal, np.array([0.0]), 0.21, 0.20, 0.0, 1.0)
    if path is None:
        continue
    d = np.diff(path[:, :2], axis=0); h = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); s = np.hypot(*d.T)
    raw_worst = max(raw_worst, float((np.abs(np.diff(h)) / (.5 * (s[:-1] + s[1:]))).max()))
    out = center_and_smooth(path, pts, 0.21, 0.20, LIM)
    if out is None:
        continue
    ok += 1
    d = np.diff(out[:, :2], axis=0); h = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); s = np.hypot(*d.T)
    worst = max(worst, float((np.abs(np.diff(h)) / (.5 * (s[:-1] + s[1:]))).max()))
print(f'limit {LIM:.3f}; accepted {ok}/{n}; worst refined {worst:.3f}; worst raw {raw_worst:.3f}; rejected {REFINE_FAIL}')
sys.exit(0 if worst <= LIM + 1e-9 else 1)
