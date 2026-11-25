#!/usr/bin/env python3
"""A map against the world's wall boxes, both ways, with the adoption criteria fixed in advance.

usage: map_vs_world.py <map.yaml>. Walls: the world's collision boxes without the obstacle boxes,
sampled every 0.02 m. Visible walls: rays cast through those boxes from route points. Offline.
"""
import math, sys
import numpy as np, yaml
import xml.etree.ElementTree as ET
from PIL import Image
from scipy.spatial import cKDTree
ym = sys.argv[1] if len(sys.argv) > 1 else 'src/Raceline-Optimization/maps/track_map3.yaml'
root = ET.parse('src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world').getroot()
W = []
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
            lx, ly = rot(mp[5], lp[:2]); cx, cy = rot(mp[5] + lp[5], cp[:2]); yw = mp[5] + lp[5] + cp[5]
            c, s = math.cos(yw), math.sin(yw)
            u = np.arange(-sx / 2, sx / 2 + 1e-6, 0.02); v = np.arange(-sy / 2, sy / 2 + 1e-6, 0.02)
            P = np.vstack([np.c_[u, np.full(len(u), -sy / 2)], np.c_[u, np.full(len(u), sy / 2)], np.c_[np.full(len(v), -sx / 2), v], np.c_[np.full(len(v), sx / 2), v]])
            W.append(np.c_[mp[0] + lx + cx + c * P[:, 0] - s * P[:, 1], mp[1] + ly + cy + s * P[:, 0] + c * P[:, 1]])
W = np.vstack(W)
Y = yaml.safe_load(open(ym)); d = ym.rsplit('/', 1)[0] + '/'; I = np.array(Image.open(d + Y['image']).convert('L')); res = Y['resolution']; ox, oy = Y['origin'][:2]
if Y.get('negate', 0):
    I = 255 - I
H = I.shape[0]; r, c = np.nonzero(I < 100); M = np.c_[ox + (c + .5) * res, oy + (H - r - .5) * res]
print(ym, 'map', I.shape, 'res', res, 'origin', Y['origin'][:2], 'occupied', len(M))
tw = cKDTree(W); tm = cKDTree(M)
D = tw.query(M)[0]
med, p95, mx = float(np.median(D)), float(np.percentile(D, 95)), float(D.max())
print(f'map cell to wall: median {med:.3f} p95 {p95:.3f} max {mx:.3f} over 0.1: {int((D > 0.1).sum())}')
xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
gx = np.round(W / 0.02).astype(int); occ = set(map(tuple, gx)); occ |= set(map(tuple, gx + [1, 0])) | set(map(tuple, gx + [0, 1])) | set(map(tuple, gx - [1, 0])) | set(map(tuple, gx - [0, 1]))
hits = []
for x0, y0 in xy[::3]:
    for a in np.linspace(-math.pi, math.pi, 180, endpoint=False):
        for dd in np.arange(0.05, 6.0, 0.02):
            q = (int(round((x0 + dd * math.cos(a)) / 0.02)), int(round((y0 + dd * math.sin(a)) / 0.02)))
            if q in occ:
                hits.append((x0 + dd * math.cos(a), y0 + dd * math.sin(a)))
                break
hits = np.array(hits)
dh = tm.query(hits)[0]
cover = float((dh <= 0.10).mean())
print(f'visible wall hits {len(hits)} within 0.10 m of an occupied cell: {cover:.3f}')
best = None
for dx in np.arange(-0.25, 0.2501, 0.025):
    for dy in np.arange(-0.25, 0.2501, 0.025):
        m = float(np.median(tw.query(M + [dx, dy])[0]))
        if best is None or m < best[0]:
            best = (m, dx, dy)
shift = math.hypot(best[1], best[2])
print(f'best shift ({best[1]:.3f}, {best[2]:.3f}) size {shift:.3f}, median then {best[0]:.3f}, gain {med - best[0]:.3f}')
zones = {'o3 south pinch': (-8.38, 0.43, 0.7), 'o3': (-8.0, 1.15, 1.2), 'o2': (-3, 3.05, 1.2), 'o1': (5, -0.09, 1.2), 's 39': (3.2, -0.25, 1.0), 's 24': (-10.2, 3.5, 1.0)}
zone = {}
for name, (cx, cy, rad) in zones.items():
    m = np.hypot(M[:, 0] - cx, M[:, 1] - cy) < rad
    if not m.any():
        print(name, 'no cells'); zone[name] = 9.0; continue
    _, j = tw.query(M[m]); v = (W[j] - M[m]).mean(axis=0); zone[name] = float(np.hypot(*v))
    print(name, 'cells', int(m.sum()), 'median', round(float(np.median(D[m])), 3), 'mean offset world-map', np.round(v, 3).tolist(), 'size', round(zone[name], 3))
ok = dict(median=med <= .03, p95=p95 <= .07, max=mx <= .12, zones=all(zone[k] <= .05 for k in ('o3 south pinch', 'o2', 'o1')), cover=cover >= .95, shift=shift <= .05, gain=med - best[0] <= .005)
print('criteria', ok, 'ALL' if all(ok.values()) else 'NOT ALL')
