#!/usr/bin/env python3
"""How fast can the body close on a box it runs beside? For narrow_drift_time. Reads only.
Samples with the body beside a box, speed at least 0.1, command above the floor. D = largest fall of the gap within tau,
up to 0.6 s; T = D / v. Includes planned sideways motion, so it is on the high side.
usage: drift_measure.py <exp> ..."""
import csv, runpy, sys
import numpy as np
E = sys.argv[1:]; sys.argv = ['x']; G = runpy.run_path('tools/exp/side_gaps.py'); OB = G['OB']
for e in E:
    R = list(csv.DictReader(open(f'logs/exp/{e}/run.csv')))
    f = lambda k: np.array([float(r[k]) if r[k] not in ('', 'None') else np.nan for r in R])
    t, x, y, w, v, c = f('t'), f('x'), f('y'), f('yaw'), f('v'), f('cmd_v'); _, u = np.unique(t, return_index=True); t, x, y, w, v, c = [a[u] for a in (t, x, y, w, v, c)]
    for n, b in OB.items():
        g = np.array([G['gap_box'](*G['body'](x[i], y[i], w[i]), b) for i in range(len(t))])
        along = np.cos(w) * (b[0] - x) + np.sin(w) * (b[1] - y)
        near = g < 0.15
        Ts = []; Ds = []; used = 0
        for i in np.where(near & (v >= 0.1) & (c > 0.065))[0]:
            j = np.searchsorted(t, t[i] + 0.6); D = float(g[i] - g[i:j + 1].min())
            used += 1; Ds.append(D); Ts.append(D / v[i])
        if used: print(e, n, 'samples', used, 'D p95/max', np.round(np.percentile(Ds, 95), 3), round(max(Ds), 3), 'T p95/max', np.round(np.percentile(Ts, 95), 3), round(max(Ts), 3), 'speed mean', round(float(np.mean(v[near & (v >= .1) & (c > .065)])), 3), 'gap min there', round(float(g[near & (v >= .1) & (c > .065)].min()), 3))
        else: print(e, n, 'no samples')
