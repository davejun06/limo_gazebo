#!/usr/bin/env python3
"""Open loop on a bag: dwa_planner.py with dwa_keep_gap 0 against a value. Reads only.
Passes: from the first tick with true box gap under 0.06 to a stop or gap over 0.08. In them, measured about the rear axle
(lr 0.12, as the car moves): share of choices whose 0.4 m pose is closer than now, lowest box gap over the first 0.4 m,
virtual monitor fires (gap + min(rate, 0) x 0.7 under 0.02 at the chosen arc's 0.1 m pose), wall gap, zero-speed ticks.
usage (in the container): replay_keep.py <exp> <bag> <o1|o2|o3> <t_from> <t_to> <value> [k=v ...]"""
import csv, importlib.util, math, runpy, sys, time
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
exp, bag, box, ta, tz, val = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]); kv = [a.split('=') for a in sys.argv[7:]]
_a = sys.argv; sys.argv = ['x']; SG = runpy.run_path('tools/exp/side_gaps.py'); sys.argv = _a
OB = SG['OB'][box]; body, gap_box, gap_wall = SG['body'], SG['gap_box'], SG['gap_wall']
s_ = importlib.util.spec_from_file_location('dp', 'src/dwa_planner.py'); dp = importlib.util.module_from_spec(s_); s_.loader.exec_module(dp)
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr')); r.set_filter(rosbag2_py.StorageFilter(topics=['/scan'])); sc = []
while r.has_next():
    _, d, _ = r.read_next(); m = deserialize_message(d, LaserScan); sc.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m))
ts = np.array([q[0] for q in sc]); R = list(csv.DictReader(open(f'logs/exp/{exp}/run.csv')))
g = lambda q, k: float(q[k]) if q[k] not in ('', 'None') else float('nan')
OD = np.array([[g(q, 't'), g(q, 'x'), g(q, 'y'), g(q, 'yaw')] for q in R]); xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
def run(v_):
    p = dp.default_params(); p.dwa_keep_gap = v_
    for k, v in kv: setattr(p, k, float(v))
    P = dp.DWAPlanner(xy, p); M = dp.UnseenMemory(p.memory_hold, p.memory_keep_r, p.memory_cell, p.memory_rmin_margin)
    LS = (0.1236, 0.0, 0.0); si = -1; on = False; res = dict(n=0, closer=0, low=9.0, wall=9.0, fires=0, zero=0, signs=0, step=0.0, ms=0.0, none=None, passes=0, vmax=0.0, low_cut=9.0, low_all=9.0, low_off=9.0, t_cut=0, t_all=0, t_off=0); last = None
    for q in R:
        t = g(q, 't')
        if not (ts[0] <= t <= tz) or math.isnan(g(q, 'map_x')) or math.isnan(g(q, 'pp_v')): continue
        i = int(np.searchsorted(ts, t, side='right') - 1); gt = (g(q, 'x'), g(q, 'y'), g(q, 'yaw'))
        if i != si:
            si = i; m = sc[i][1]; pts = dp.scan_to_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, p.plan_range, LS); k = int(np.argmin(np.abs(OD[:, 0] - ts[i]))); M.add(t, pts, tuple(OD[k, 1:]))
        un = M.unseen(t, gt, LS[:2], m.range_min, m.angle_max); a = time.perf_counter(); all0 = P.dwa.stats['keep_all']; on0 = P.dwa.stats['keep_on']
        v, st, info = P.step((g(q, 'map_x'), g(q, 'map_y'), g(q, 'map_yaw')), pts, g(q, 'pp_v'), g(q, 'pp_steer'), t, un, gt); ms = (time.perf_counter() - a) * 1e3
        if t < ta: continue
        res['ms'] = max(res['ms'], ms); X, Y = body(*gt); g0 = gap_box(X, Y, OB)
        if not on and g0 < 0.06: on = True; res['passes'] += 1
        elif on and g0 > 0.08: on = False
        if not on: continue
        res['zero'] += v == 0.0
        if info.get('stuck') and res['none'] is None: res['none'] = t
        if v == 0.0 or info['best'] is None: continue
        res['n'] += 1; res['vmax'] = max(res['vmax'], v)
        if last is not None:
            res['signs'] += st * last < 0; res['step'] = max(res['step'], abs(st - last))
        last = st
        c_, s2 = math.cos(gt[2]), math.sin(gt[2]); kk = math.tan(st) / 0.24; lr = 0.12
        F = []
        for d in (0.1, 0.2, 0.3, 0.4):
            h = kk * d; ax = (math.sin(h) / kk if abs(kk) > 1e-9 else d); ay = ((1 - math.cos(h)) / kk if abs(kk) > 1e-9 else 0.0)
            bx, by = ax - lr + lr * math.cos(h), ay + lr * math.sin(h); F.append((gt[0] + c_ * bx - s2 * by, gt[1] + s2 * bx + c_ * by, gt[2] + h))
        G = [gap_box(*body(*f), OB) for f in F]; res['low'] = min(res['low'], min(G)); kind = ('off' if P.dwa.stats['keep_on'] == on0 else 'all' if P.dwa.stats['keep_all'] > all0 else 'cut'); res['low_' + kind] = min(res['low_' + kind], min(G)); res['t_' + kind] += 1; res['wall'] = min(res['wall'], min(gap_wall(*f) for f in F)); res['closer'] += G[3] < g0
        rate = (G[0] - g0) / (0.1 / max(v, 0.03)); res['fires'] += g0 + min(rate, 0.0) * 0.7 < 0.02
    return res, P.dwa.stats
for v_ in (0.0, val):
    a, st = run(v_); print(exp, box, 'keep', v_, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in a.items()}, 'share', round(a['closer'] / max(a['n'], 1), 3), st)
