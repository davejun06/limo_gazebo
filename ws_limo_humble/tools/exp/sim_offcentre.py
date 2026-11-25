#!/usr/bin/env python3
"""Mean off-centre and lowest wall gap beside each obstacle over simulator csv files. Reads only.
usage: sim_offcentre.py <csv> ...   Pass counted where the box gap is under 0.12 m, as in side_gaps.py."""
import csv, runpy, sys
import numpy as np
F = sys.argv[1:]; sys.argv = ['x']; G = runpy.run_path('tools/exp/side_gaps.py')
out = {n: [] for n in G['OB']}; wall = {n: 9.0 for n in G['OB']}
for f in F:
    R = list(csv.DictReader(open(f)))
    for r in R[::6]:
        x, y, w = float(r['x']), float(r['y']), float(r['yaw'])
        for n, b in G['OB'].items():
            if abs(x - b[0]) < .8 and abs(y - b[1]) < .9:
                X, Y = G['body'](x, y, w); gb = G['gap_box'](X, Y, b)
                if gb < 0.12:
                    gw = G['gap_wall'](x, y, w); out[n].append((gw - gb) / 2); wall[n] = min(wall[n], gw)
print({n: (round(float(np.mean(v)), 3) if v else None, len(v)) for n, v in out.items()}, 'all', round(float(np.mean(sum(out.values(), []))), 3), 'wall', {k: round(v, 3) for k, v in wall.items()})
