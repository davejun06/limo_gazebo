#!/usr/bin/env python3
"""Tick by tick on a bag: what the closing gate saw and chose. Reads only.
usage (in the container): gate_ticks.py <exp> <bag> <t_from> <t_to> [k=v ...]"""
import csv, importlib.util, math, sys
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
exp, bag, ta, tz = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); kv = [a.split('=') for a in sys.argv[5:]]
s_ = importlib.util.spec_from_file_location('dp', 'src/dwa_planner.py'); dp = importlib.util.module_from_spec(s_); s_.loader.exec_module(dp)
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr')); r.set_filter(rosbag2_py.StorageFilter(topics=['/scan'])); sc = []
while r.has_next():
    _, d, _ = r.read_next(); m = deserialize_message(d, LaserScan); sc.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m))
ts = np.array([q[0] for q in sc]); R = list(csv.DictReader(open(f'logs/exp/{exp}/run.csv')))
g = lambda q, k: float(q[k]) if q[k] not in ('', 'None') else float('nan')
OD = np.array([[g(q, 't'), g(q, 'x'), g(q, 'y'), g(q, 'yaw')] for q in R]); xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_line_pp.csv', delimiter=',')[:, :2]
p = dp.default_params()
for k, v in kv: setattr(p, k, float(v))
P = dp.DWAPlanner(xy, p); M = dp.UnseenMemory(p.memory_hold, p.memory_keep_r, p.memory_cell, p.memory_rmin_margin); D = P.dwa
LS = (0.1236, 0.0, 0.0); si = -1; lastp = -1e9; print('steers', np.round(D.first_steers, 2).tolist())
for q in R:
    t = g(q, 't')
    if not (ts[0] <= t <= tz) or math.isnan(g(q, 'map_x')) or math.isnan(g(q, 'pp_v')): continue
    i = int(np.searchsorted(ts, t, side='right') - 1); gt = (g(q, 'x'), g(q, 'y'), g(q, 'yaw'))
    if i != si:
        si = i; m = sc[i][1]; pts = dp.scan_to_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, p.plan_range, LS); k = int(np.argmin(np.abs(OD[:, 0] - ts[i]))); M.add(t, pts, tuple(OD[k, 1:]))
    un = M.unseen(t, gt, LS[:2], m.range_min, m.angle_max); a0 = D.stats['keep_all']; on0 = D.stats['keep_on']
    v, st, info = P.step((g(q, 'map_x'), g(q, 'map_y'), g(q, 'map_yaw')), pts, g(q, 'pp_v'), g(q, 'pp_steer'), t, un, gt)
    if t < ta or t - lastp < 0.5: continue
    lastp = t
    fg = D.first_gap; ok = D.first_ok
    print(f"t {t:.1f} ({gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f}) v {v:.2f} st {st:+.2f} log {g(q,'cmd_steer'):+.2f} rad {P.last_rad:.3f} gate {'off' if D.stats['keep_on']==on0 else ('none-left' if D.stats['keep_all']>a0 else 'cut')} mem {len(un)}", 'first_gap', None if fg is None else np.round(fg, 3).tolist(), 'ok', None if ok is None else ok.astype(int).tolist())
