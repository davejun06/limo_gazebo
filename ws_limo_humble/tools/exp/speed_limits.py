#!/usr/bin/env python3
"""Which rule sets the speed in AVOID? Open loop on a bag through dwa_planner.py. Reads only.
usage (in the container): speed_limits.py <exp> <bag> [k=v ...]"""
import csv, importlib.util, math, sys
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
exp, bag = sys.argv[1:3]; kv = [a.split('=') for a in sys.argv[3:]]
s_ = importlib.util.spec_from_file_location('dp', 'src/dwa_planner.py'); dp = importlib.util.module_from_spec(s_); s_.loader.exec_module(dp)
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr')); r.set_filter(rosbag2_py.StorageFilter(topics=['/scan'])); sc = []
while r.has_next():
    _, d, _ = r.read_next(); m = deserialize_message(d, LaserScan); sc.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m))
ts = np.array([q[0] for q in sc]); R = list(csv.DictReader(open(f'logs/exp/{exp}/run.csv')))
g = lambda q, k: float(q[k]) if q[k] not in ('', 'None') else float('nan')
OD = np.array([[g(q, 't'), g(q, 'x'), g(q, 'y'), g(q, 'yaw')] for q in R]); xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_line_pp.csv', delimiter=',')[:, :2]
p = dp.default_params()
for k, v in kv: setattr(p, k, float(v))
P = dp.DWAPlanner(xy, p); M = dp.UnseenMemory(p.memory_hold, p.memory_keep_r, p.memory_cell, p.memory_rmin_margin)
_ns = P.narrow_speed; rec = {}
def ns(b, pts):
    v = _ns(b, pts); rec['narrow'] = v; rec['poses'] = b; return v
P.narrow_speed = ns
LS = (0.1236, 0.0, 0.0); si = -1; last = None; T = {}; bins = {}; sh = {}
for q in R:
    t = g(q, 't')
    if not (ts[0] <= t <= ts[-1]) or math.isnan(g(q, 'map_x')) or math.isnan(g(q, 'pp_v')): continue
    i = int(np.searchsorted(ts, t, side='right') - 1); gt = (g(q, 'x'), g(q, 'y'), g(q, 'yaw'))
    if i != si:
        si = i; m = sc[i][1]; pts = dp.scan_to_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, p.plan_range, LS); k = int(np.argmin(np.abs(OD[:, 0] - ts[i]))); M.add(t, pts, tuple(OD[k, 1:]))
    un = M.unseen(t, gt, LS[:2], m.range_min, m.angle_max); rec.clear(); pv, ps = g(q, 'pp_v'), g(q, 'pp_steer')
    v, st, info = P.step((g(q, 'map_x'), g(q, 'map_y'), g(q, 'map_yaw')), pts, pv, ps, t, un, gt)
    dt = 0.0 if last is None else max(0.0, t - last); last = t
    if g(q, 'v') < 0.01 and g(q, 'cmd_v') < 0.01: continue
    if info['mode'] == 'FOLLOW':
        cap = info['cap']; who = 'follow:pp' if pv <= cap else 'follow:cap(wall/corner/steer)'
    else:
        blocked = info['blocked']; base = min(float(np.clip(pv, p.min_speed, p.free_speed if not blocked else p.avoid_max_speed)), max(info['cap'], p.wall_speed))
        k_ = abs(math.tan(st)) / dp.WHEELBASE; a = max(p.curve_min, math.sqrt(p.a_lat / max(k_, 1e-3))); ss = p.max_speed - (p.max_speed - p.curve_min) * min(1.0, abs(st) / p.steer_slow); sq = p.squeeze_speed if P.last_rad < 0.9 * p.r_robot else 9.0; n = rec.get('narrow', 9.0)
        c = {'avoid/free cap': base, 'narrow': n, 'a_lat': a, 'steer_slow': ss, 'squeeze': sq}; who = 'avoid:' + (min(c, key=c.get) if v > 0 else 'stop')
        if who == 'avoid:narrow': who = 'avoid:narrow floor' if abs(v - p.narrow_min_speed) < 1e-6 else 'avoid:narrow'
    T[who] = T.get(who, 0.0) + dt
    s = int(g(q, 'progress') // 4) * 4; bins.setdefault(s, {}); bins[s][who] = bins[s].get(who, 0.0) + dt
print(exp, kv, 'total', round(sum(T.values()), 1)); [print(f'  {k:32s} {v:6.1f} s') for k, v in sorted(T.items(), key=lambda x: -x[1])]
for s in sorted(bins):
    print(f' s {s:2d}-{s+4:2d}', {k.replace('avoid:', 'a:').replace('follow:', 'f:')[:16]: round(v, 1) for k, v in sorted(bins[s].items(), key=lambda x: -x[1])[:4]})
