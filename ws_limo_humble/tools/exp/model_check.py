#!/usr/bin/env python3
"""Checks the planner's arc model against logged motion. Reads only.
From each tick beside a box (true gap under 0.08), predicts the body 0.1 m of travel ahead from the commanded steering,
by turning about base_link (planner's arc) and about the rear axle (lr behind base_link), and compares with the logged pose.
usage: model_check.py <lr> <exp> ..."""
import csv, math, runpy, sys
import numpy as np
lr = float(sys.argv[1]); E = sys.argv[2:]; sys.argv = ['x']; G = runpy.run_path('tools/exp/side_gaps.py')
OB, body, gap_box = G['OB'], G['body'], G['gap_box']; L = 0.24
def step(x, y, w, st, d, lr_):
    k = math.tan(st) / L; n = 10
    for _ in range(n):
        b = math.atan(lr_ * k); x += d / n * math.cos(w + b); y += d / n * math.sin(w + b); w += d / n * k * math.cos(b)
    return x, y, w
for e in E:
    R = list(csv.DictReader(open(f'logs/exp/{e}/run.csv')))
    A = np.array([[float(r[k]) if r[k] not in ('', 'None') else np.nan for k in ('t', 'x', 'y', 'yaw', 'v', 'cmd_steer')] for r in R])
    _, u = np.unique(A[:, 0], return_index=True); A = A[u]
    d = np.r_[0, np.cumsum(np.hypot(np.diff(A[:, 1]), np.diff(A[:, 2])))]
    res = {0.0: [0, 0, []], lr: [0, 0, []]}; ns = 0
    only = __import__('os').environ.get('BOX')
    for i in range(len(A)):
        if A[i, 4] < 0.03 or abs(A[i, 5]) < 0.05: continue
        for n_, b in OB.items():
            if only and n_ != only: continue
            X, Y = body(*A[i, 1:4]); g0 = gap_box(X, Y, b)
            if g0 >= 0.08: continue
            j = np.searchsorted(d, d[i] + 0.1)
            if j >= len(A) or np.isnan(A[j, 1]): continue
            # hold only ticks where the steering stayed about the same over the step
            if np.nanmax(np.abs(A[i:j + 1, 5] - A[i, 5])) > 0.12: continue
            ns += 1
            Xa, Ya = body(*A[j, 1:4]); ga = gap_box(Xa, Ya, b)
            for l_ in (0.0, lr):
                xp, yp, wp = step(*A[i, 1:4], A[i, 5], d[j] - d[i], l_)
                Xp, Yp = body(xp, yp, wp); gp = gap_box(Xp, Yp, b)
                res[l_][2].append(math.hypot(xp - A[j, 1], yp - A[j, 2]))
                if abs(ga - g0) > 0.001:
                    res[l_][1] += 1; res[l_][0] += (gp - g0) * (ga - g0) > 0
    print(e, 'steady ticks beside a box', ns, {('base_link' if k == 0.0 else f'rear lr {k}'): dict(sign=f'{v[0]}/{v[1]}', pose_err_p50=round(float(np.median(v[2])), 4) if v[2] else None, p90=round(float(np.percentile(v[2], 90)), 4) if v[2] else None) for k, v in res.items()})
