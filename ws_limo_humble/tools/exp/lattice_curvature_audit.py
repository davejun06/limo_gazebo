#!/usr/bin/env python3
"""Offline audit: how many accepted candidates exceed the curvature limit on their own arc length?

usage: lattice_curvature_audit.py <planner.py> ...   Synthetic box points, stationary poses on the
route, every 0.5 m plus 3/2/1.5/1 m before each box. Says nothing about tracking or real scans.
"""
import importlib.util
import sys

import numpy as np

XY = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
OBS = [(5, -.0878, .15, .4), (-3, 3.054, .15, .4), (-8, 1.145, .15, .5)]


def box_points():
    out = []
    for cx, cy, hx, hy in OBS:
        for t in np.arange(-hx, hx + 1e-6, .025):
            out += [(cx + t, cy - hy), (cx + t, cy + hy)]
        for t in np.arange(-hy, hy + 1e-6, .025):
            out += [(cx - hx, cy + t), (cx + hx, cy + t)]
    return np.array(out)


def true_kmax(b):
    seg = np.hypot(*np.diff(b, axis=0).T)
    h = np.unwrap(np.arctan2(*np.diff(b, axis=0).T[::-1]))
    return float(np.abs(np.diff(h) / np.maximum(.5 * (seg[:-1] + seg[1:]), 1e-6)).max())


def audit(path):
    spec = importlib.util.spec_from_file_location('fp', path)
    fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fp)
    P = fp.FrenetPlanner(XY)
    limit = .9 * fp.MAX_CURV
    B = box_points()
    cases = [(i, False) for i in range(0, P.n, 10)]
    for cx, cy, _, _ in OBS:
        j = int(np.argmin(np.hypot(*(P.ref - (cx, cy)).T)))
        cases += [((j - int(back / P.ds)) % P.n, back) for back in (3., 2., 1.5, 1.)]
    total = over = chosen_over = none_empty = 0
    worst = 0.
    box_counts = []
    for i, back in cases:
        pose = (P.ref[i, 0], P.ref[i, 1], P.theta[i])
        if back:
            pts = fp.to_base(B, pose)
            r = np.hypot(*pts.T)
            pts = pts[(r < 5) & (r > .12)]
        else:
            pts = np.zeros((0, 2))
        best, cands, _ = P.lattice(i, 0., 0., pose, pts, free=False, block_s=(back - .3) if back else 1.0)
        if back:
            box_counts.append((len(cands), round(P.last_rad, 3)))
        elif not cands:
            none_empty += 1
        ks = [true_kmax(c) for c in cands]
        total += len(ks)
        over += sum(k > limit for k in ks)
        worst = max([worst] + ks)
        if best is not None and true_kmax(best) > limit:
            chosen_over += 1
    print(f'{path.split("/")[-1]}: limit {limit:.3f}; accepted {total}, over limit {over}, '
          f'chosen over limit {chosen_over} of {len(cases)} cases, worst {worst:.2f} 1/m; '
          f'empty-scene cases with no candidate {none_empty}/{P.n // 10 + 1}; '
          f'box cases o1/o2/o3 at 3,2,1.5,1 m (count, radius): {box_counts}')


for p in sys.argv[1:]:
    audit(p)
