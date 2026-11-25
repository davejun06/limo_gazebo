#!/usr/bin/env python3
"""AMCL error per lap over route stretches, leaving out samples from the stop request on.

usage: lap_err.py <exp> [stop_t]. stop_t defaults to the monitor's STOP time mapped from run.json laps.
"""
import csv, json, sys
import numpy as np
e = sys.argv[1]
J = json.load(open(f'logs/exp/{e}/run.json'))
R = list(csv.DictReader(open(f'logs/exp/{e}/run.csv')))
f = lambda r, k: float(r[k]) if r[k] not in ('', 'None') else float('nan')
t = np.array([f(r, 't') for r in R]); _, i = np.unique(t, return_index=True); R = [R[j] for j in i]
t = np.array([f(r, 't') for r in R]); v = np.array([f(r, 'v') for r in R]); c = np.array([f(r, 'cmd_v') for r in R])
err = np.array([f(r, 'loc_err') for r in R]); s = np.array([f(r, 'progress') for r in R]); cx = np.array([f(r, 'cov_x') for r in R]); cy = np.array([f(r, 'cov_y') for r in R])
mv = t[np.argmax(np.abs(v) > .03)]
stop = float(sys.argv[2]) if len(sys.argv) > 2 else t[np.where(c > 0)[0][-1]]
ed = [mv] + [mv + sum(J['laps'][:k + 1]) for k in range(len(J['laps']))] + [stop]
out = []
for k in range(len(ed) - 1):
    m = (t >= ed[k]) & (t < ed[k + 1])
    a = (s >= 20) & (s <= 29.19) & m; b = (s > 29.19) & (s <= 44) & m
    g = lambda q: (round(float(err[q].max()), 3), round(float(np.percentile(err[q], 95)), 3)) if q.any() else None
    o = err[m] / np.sqrt(np.maximum(cx[m] + cy[m], 1e-9))
    print(f'{e} lap{k + 1}', 's20-29', g(a), 's29-44', g(b), 'all', g(m), 'cov_x median', round(float(np.nanmedian(cx[m])), 4), 'err/sigma over 3 s', round(float((o > 3).sum() * 0.1), 1))
    out.append((g(a), g(b)))
A = [x[0] for x in out if x[0]]; B = [x[1] for x in out if x[1]]
print(f'{e} n', len(out), 'means s20-29', np.round(np.mean(A, axis=0), 3).tolist() if A else None, 's29-44', np.round(np.mean(B, axis=0), 3).tolist() if B else None)
