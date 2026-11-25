#!/usr/bin/env python3
"""Footprint gap to the world's own wall boxes along a logged run, instead of the SLAM map."""
import csv, math, sys
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
            yaw = mp[5] + lp[5] + cp[5]
            c, s = math.cos(yaw), math.sin(yaw)
            u = np.arange(-sx / 2, sx / 2 + 1e-6, 0.01); v = np.arange(-sy / 2, sy / 2 + 1e-6, 0.01)
            P = np.vstack([np.c_[u, np.full(len(u), -sy / 2)], np.c_[u, np.full(len(u), sy / 2)], np.c_[np.full(len(v), -sx / 2), v], np.c_[np.full(len(v), sx / 2), v]])
            B.append(np.c_[mp[0] + lx + cx + c * P[:, 0] - s * P[:, 1], mp[1] + ly + cy + s * P[:, 0] + c * P[:, 1]])
W = np.vstack(B)
for e in sys.argv[1:]:
    R = list(csv.DictReader(open(f'logs/exp/{e}/run.csv')))
    best = (9.0,); n05 = 0; n = 0; seen = set()
    for r in R:
        if r['t'] in seen:
            continue
        seen.add(r['t']); n += 1
        x, y, w = float(r['x']), float(r['y']), float(r['yaw'])
        m = (np.abs(W[:, 0] - x) < 0.8) & (np.abs(W[:, 1] - y) < 0.8)
        if not m.any():
            continue
        c, s = math.cos(w), math.sin(w); dx, dy = W[m, 0] - x, W[m, 1] - y
        g = float(np.hypot(np.maximum(np.abs(dx * c + dy * s) - .165, 0), np.maximum(np.abs(-dx * s + dy * c) - .155, 0)).min())
        n05 += g < .05
        if g < best[0]:
            best = (round(g, 3), round(x, 2), round(y, 2), r['t'], r['cmd_v'][:4])
    print(e, 'lowest world wall gap', best, 'samples under 0.05', n05, 'of', n)
