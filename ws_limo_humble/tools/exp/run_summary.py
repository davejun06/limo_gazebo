#!/usr/bin/env python3
"""Summary of one kept run. usage: run_summary.py <exp>"""
import csv, json, runpy, sys
import numpy as np
e = sys.argv[1]; J = json.load(open(f'logs/exp/{e}/run.json'))
print({k: J.get(k) for k in ('laps', 'stop_reason', 'contacts', 'contact_stream_ok', 'min_obstacle_gap', 'rtf', 'stuck_lines', 'distance', 'loc_err_mean', 'loc_err_p95', 'loc_err_max', 'yaw_err_p95', 'max_cmd_v', 'max_odom_v', 'max_route_dev')})
R = list(csv.DictReader(open(f'logs/exp/{e}/run.csv')))
f = lambda k: np.array([float(r[k]) if r[k] not in ('', 'None') else np.nan for r in R])
t, v, c, x, y, yw, q, st, pr = f('t'), f('v'), f('cmd_v'), f('x'), f('y'), f('yaw'), f('loc_err'), f('cmd_steer'), f('progress')
_, u = np.unique(t, return_index=True); t, v, c, x, y, yw, q, st, pr = [a[u] for a in (t, v, c, x, y, yw, q, st, pr)]
m = t[np.argmax(np.abs(v) > .03)]; nz = np.where(c > .001)[0][-1]; print('motion', m, 'last cmd', t[nz], 'from motion', round(t[nz] - m, 1), 'end', round(x[-1], 3), round(y[-1], 3), 's', pr[-1])
sys.argv = ['x']; G = runpy.run_path('tools/exp/side_gaps.py'); OB = G['OB']
for n, b in OB.items():
    d = np.array([G['gap_box'](*G['body'](a, bb, cc), b) for a, bb, cc in zip(x[::3], y[::3], yw[::3])]); i = d.argmin()
    D = np.array([G['gap_box'](*G['body'](a, bb, cc), b) for a, bb, cc in zip(x, y, yw)]); j = int(D.argmin())   # every row, for the speeds
    near = D < 1.0; app = near & (np.arange(len(D)) < j); dep = near & (np.arange(len(D)) > j); mx = lambda m: round(float(np.nanmax(v[m])), 3) if m.any() else None
    print(n, 'lowest', round(d.min(), 3), 'at', round(x[::3][i], 2), round(y[::3][i], 2), 'wall there', round(G['gap_wall'](x[::3][i], y[::3][i], yw[::3][i]), 3), 'within 1 m: max approaching', mx(app), 'leaving', mx(dep), 'mean', round(float(np.nanmean(v[near])), 2) if near.any() else None, 'within 0.12:', round(float(np.nanmean(v[D < .12])), 3) if (D < .12).any() else None)
dt = np.r_[np.diff(t), 0]; mv = (t >= m) & (t <= t[nz]); print('time under 0.08', round(dt[mv & (c < .08)].sum(), 1), 'mean v', round(float(np.nansum(np.hypot(np.diff(x[mv]), np.diff(y[mv])))) / max(t[nz] - m, .1), 3), 'v x |tan steer| / L max', round(float(np.nanmax(v * v * np.abs(np.tan(st)) / .24)), 3))
for T in np.linspace(t[nz] - 7, t[-1], 8):
    i = np.argmin(abs(t - T)); gs = min(G['gap_box'](*G['body'](x[i], y[i], yw[i]), bb) for bb in OB.values()); print(f"t {t[i]:.1f} ({x[i]:.3f},{y[i]:.3f}) yaw {yw[i]:.3f} v {v[i]:.2f} st {st[i]:+.2f} box {gs:.3f} wall {G['gap_wall'](x[i], y[i], yw[i]):.3f} err {q[i]:.3f}")
