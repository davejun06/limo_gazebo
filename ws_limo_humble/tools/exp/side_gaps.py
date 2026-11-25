#!/usr/bin/env python3
"""Both side gaps of the body beside each obstacle along a logged run, from the world's own boxes. Reads only.
usage: side_gaps.py <exp> ...   Per 0.1 m of travel along x (o1, o2) or y (o3): box gap, wall gap, centring."""
import csv, json, math, sys
import numpy as np
import xml.etree.ElementTree as ET
root = ET.parse('src/limo_ros2/limo_car/worlds/roboracer_obstacles_1/ifac_roboracer_obstacles_1.world').getroot()
B = []
for model in root.iter('model'):
    if 'obstacle' in model.get('name', '').lower():
        continue
    mp = [float(x) for x in (model.findtext('pose') or '0 0 0 0 0 0').split()]
    for link in model.iter('link'):
        lp = [float(x) for x in (link.findtext('pose') or '0 0 0 0 0 0').split()]
        for col in link.iter('collision'):
            size = col.find('geometry/box/size')
            if size is None:
                continue
            cp = [float(x) for x in (col.findtext('pose') or '0 0 0 0 0 0').split()]
            sx, sy, _ = [float(x) for x in size.text.split()]
            rot = lambda a, p: (p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a))
            lx, ly = rot(mp[5], lp[:2]); cx, cy = rot(mp[5] + lp[5], cp[:2])
            yaw = mp[5] + lp[5] + cp[5]; c, s_ = math.cos(yaw), math.sin(yaw)
            q = np.arange(-sx / 2, sx / 2 + 1e-6, 0.01); w_ = np.arange(-sy / 2, sy / 2 + 1e-6, 0.01)
            P = np.vstack([np.c_[q, np.full(len(q), -sy / 2)], np.c_[q, np.full(len(q), sy / 2)], np.c_[np.full(len(w_), -sx / 2), w_], np.c_[np.full(len(w_), sx / 2), w_]])
            B.append(np.c_[mp[0] + lx + cx + c * P[:, 0] - s_ * P[:, 1], mp[1] + ly + cy + s_ * P[:, 0] + c * P[:, 1]])
W = np.vstack(B)
OB = {'o1': (5.0, -0.0878, .15, .4), 'o2': (-3.0, 3.0542, .15, .4), 'o3': (-8.0, 1.1452, .15, .5)}
u = np.linspace(-.165, .165, 9); v = np.linspace(-.155, .155, 9)
BX = np.r_[u, u, np.full(9, -.165), np.full(9, .165)]; BY = np.r_[np.full(9, -.155), np.full(9, .155), v, v]
def body(x, y, w):
    c, s = math.cos(w), math.sin(w)
    return x + c * BX - s * BY, y + s * BX + c * BY
def gap_box(X, Y, b):
    return float(np.hypot(np.maximum(np.abs(X - b[0]) - b[2], 0), np.maximum(np.abs(Y - b[1]) - b[3], 0)).min())
def gap_wall(x, y, yw):
    m = (np.abs(W[:, 0] - x) < 1.0) & (np.abs(W[:, 1] - y) < 1.0)
    if not m.any():
        return 9.0
    c, s_ = math.cos(yw), math.sin(yw); dx, dy = W[m, 0] - x, W[m, 1] - y
    return float(np.hypot(np.maximum(np.abs(dx * c + dy * s_) - .165, 0), np.maximum(np.abs(-dx * s_ + dy * c) - .155, 0)).min())
print('wall points', len(W))
for e in sys.argv[1:]:
    R = list(csv.DictReader(open(f'logs/exp/{e}/run.csv')))
    for n, b in OB.items():
        rows = [(float(r['x']), float(r['y']), float(r['yaw']), float(r['t']), float(r['v'])) for r in R if r['x'] not in ('', 'None')]
        A = np.array(rows); d = np.hypot(np.maximum(np.abs(A[:, 0] - b[0]) - b[2], 0), np.maximum(np.abs(A[:, 1] - b[1]) - b[3], 0))
        beside = np.where(d < 0.55)[0]
        if not len(beside):
            continue
        # one pass only: the first contiguous stretch
        cut = np.where(np.diff(beside) > 40)[0]; beside = beside[:cut[0] + 1] if len(cut) else beside
        out = []; last = None
        for i in beside[::5]:
            X, Y = body(*A[i, :3]); gb = gap_box(X, Y, b); gw = gap_wall(*A[i, :3])
            out.append((A[i, 0], A[i, 1], gb, gw, A[i, 4]))
        O = np.array(out); slack = O[:, 2] + O[:, 3]; mb = O[:, 2].argmin()
        print(f"{e} {n}: rows {len(beside)} {A[beside[-1],3]-A[beside[0],3]:.1f} s, lowest box gap {O[mb,2]:.3f} at ({O[mb,0]:.2f},{O[mb,1]:.2f}) with wall {O[mb,3]:.3f} v {O[mb,4]:.2f}; wall lowest {O[:,3].min():.3f}; slack min {slack.min():.3f} at box gap<0.12: mean off-centre {np.mean((O[:,3]-O[:,2])[O[:,2]<.12])/2:.3f}")
