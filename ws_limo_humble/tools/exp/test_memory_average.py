#!/usr/bin/env python3
"""Replay the exp-006b contact poses with noise: is averaged memory still safe at the blind corner?

Gates fixed before seeing results. Fail if the remembered points overstate the footprint's gap to
the box by more than 0.02 m, or if a blind box corner has no remembered point within 0.03 m.
Recorded poses only: this tests the point set, not how the car would have driven with it.
"""
import csv, importlib.util, math, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('fp', 'src/frenet_planner_optimized.py')
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
sys.path.insert(0, 'src')
from hybrid_astar_local import footprint_clearance
BOX = (-8.15, -7.85, 0.645, 1.645)
LX, RMIN, FOV = 0.124, 0.2, 2.0943951
ANG = np.linspace(-FOV, FOV, 720); STEPS = np.arange(RMIN, 5.2, 0.005)
CORNERS = np.array([[BOX[i], BOX[j]] for i in (0, 1) for j in (2, 3)])
HL, HW = 0.165, 0.155

def scan(x, y, yaw, noise):
    lx, ly = x + LX * math.cos(yaw), y + LX * math.sin(yaw)
    wx = lx + np.outer(STEPS, np.cos(ANG + yaw)); wy = ly + np.outer(STEPS, np.sin(ANG + yaw))
    hit = (wx >= BOX[0]) & (wx <= BOX[1]) & (wy >= BOX[2]) & (wy <= BOX[3])
    f = hit.argmax(axis=0); ok = hit.any(axis=0) & (f > 0)
    d = STEPS[f[ok]] + np.random.normal(0.0, noise, int(ok.sum()))
    return np.c_[d * np.cos(ANG[ok]) + LX, d * np.sin(ANG[ok])]

def true_gap(x, y, yaw):
    gx, gy = np.meshgrid(np.arange(BOX[0], BOX[1] + 1e-6, 0.005), np.arange(BOX[2], BOX[3] + 1e-6, 0.005))
    edge = (np.abs(gx - BOX[0]) < 1e-3) | (np.abs(gx - BOX[1]) < 1e-3) | (np.abs(gy - BOX[2]) < 1e-3) | (np.abs(gy - BOX[3]) < 1e-3)
    p = np.c_[gx[edge], gy[edge]]
    c, s = math.cos(yaw), math.sin(yaw); dx, dy = p[:, 0] - x, p[:, 1] - y
    return np.hypot(np.maximum(np.abs(c * dx + s * dy) - HL, 0), np.maximum(np.abs(-s * dx + c * dy) - HW, 0))

def run(average, cell, seed):
    np.random.seed(seed)
    R = list(csv.DictReader(open('logs/exp/exp-006b/run.csv')))
    t_end = 3878.7
    m = fp.UnseenMemory(15.0, 1.0, cell, 0.03, average)
    last = -9.0; worst_opt = -9.0; worst_corner = 0.0; n_corner = 0; pts = np.zeros((0, 2))
    for r in R:
        t = float(r['t'])
        if t < t_end - 30 or t > t_end:
            continue
        pose = (float(r['x']), float(r['y']), float(r['yaw']))
        if t - last >= 0.125:
            pts = scan(*pose, 0.01); m.add(t, pts, pose); last = t
        else:
            continue
        un = m.unseen(t, pose, (LX, 0.0), RMIN, FOV)
        seen = np.vstack([pts, un]) if len(un) else pts
        if not len(seen):
            continue
        if true_gap(*pose).min() > 0.6:
            continue
        est = float(footprint_clearance(np.array([pose[:2] + (pose[2],)]) * 0 + 0, seen, HL, HW)[0])
        worst_opt = max(worst_opt, est - float(true_gap(*pose).min()))
        c, s = math.cos(pose[2]), math.sin(pose[2]); dx, dy = CORNERS[:, 0] - pose[0], CORNERS[:, 1] - pose[1]
        cb = np.c_[c * dx + s * dy, -s * dx + c * dy]
        rx, ry = cb[:, 0] - LX, cb[:, 1]
        blind = ((np.hypot(rx, ry) < RMIN) | (np.abs(np.arctan2(ry, rx)) > FOV)) & (np.hypot(cb[:, 0], cb[:, 1]) < 1.0)
        for q in cb[blind]:
            if len(un):
                worst_corner = max(worst_corner, float(np.hypot(un[:, 0] - q[0], un[:, 1] - q[1]).min()))
                n_corner += 1
    return worst_opt, worst_corner, n_corner

def synth(average, cell, seed):
    """Straight passes beside the box on both sides at low speed: many blind-corner checks."""
    np.random.seed(seed)
    worst_opt = -9.0; worst_corner = 0.0; n = 0
    for y0 in (0.42, 0.47, 1.82, 1.87):
        m = fp.UnseenMemory(15.0, 1.0, cell, 0.03, average)
        x = -9.2; t = 0.0
        while x < -6.9:
            pose = (x, y0, 0.0)
            pts = scan(*pose, 0.01); m.add(t, pts, pose)
            un = m.unseen(t, pose, (LX, 0.0), RMIN, FOV)
            seen = np.vstack([pts, un]) if len(un) else pts
            if len(seen):
                worst_opt = max(worst_opt, float(footprint_clearance(np.zeros((1, 3)), seen, HL, HW)[0]) - float(true_gap(*pose).min()))
            cb = CORNERS - np.array([x, y0])
            rx, ry = cb[:, 0] - LX, cb[:, 1]
            blind = ((np.hypot(rx, ry) < RMIN) | (np.abs(np.arctan2(ry, rx)) > FOV)) & (np.hypot(cb[:, 0], cb[:, 1]) < 1.0)
            for q in cb[blind]:
                if len(un):
                    worst_corner = max(worst_corner, float(np.hypot(un[:, 0] - q[0], un[:, 1] - q[1]).min())); n += 1
            x += 0.02; t += 0.125
    return worst_opt, worst_corner, n

for name, avg, cell in (('newest 0.02 (current)', False, 0.02), ('mean 0.04', True, 0.04)):
    res = [synth(avg, cell, s) for s in (1, 2)]
    print(f'synthetic passes, {name}: optimism max {max(r[0] for r in res):.3f}, blind corner max {max(r[1] for r in res):.3f} over {min(r[2] for r in res)} checks')

ok = True
for name, avg, cell in (('newest per 0.02 cell (current)', False, 0.02), ('mean per 0.04 cell', True, 0.04), ('mean per 0.02 cell', True, 0.02)):
    res = [run(avg, cell, s) for s in (1, 2, 3)]
    o, c, n = max(r[0] for r in res), max(r[1] for r in res), min(r[2] for r in res)
    v = o <= 0.02 and c <= 0.03
    print(f'{name}: optimism max {o:.3f} m, blind corner distance max {c:.3f} m over {n} checks -> {"PASS" if v else "FAIL"}')
