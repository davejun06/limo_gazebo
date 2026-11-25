#!/usr/bin/env python3
"""Old planner and dwa_planner.py on one bag, same inputs: are speed and steering equal on every tick? Reads only.
usage (in the container): compare_planners.py <exp> <bag dir> [k=v ...]"""
import csv, importlib.util, math, sys, time
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
exp, bag = sys.argv[1:3]; kv = [a.split('=') for a in sys.argv[3:]]
def load(n, f):
    s = importlib.util.spec_from_file_location(n, f); m = importlib.util.module_from_spec(s); sys.path.insert(0, 'src'); s.loader.exec_module(m); return m
old = load('old', 'src/frenet_planner_optimized.py'); new = load('new', 'src/dwa_planner.py')
po = old.default_params(); old.apply_profile(po, 'dwa_lap'); pn = new.default_params()
pn.max_speed = po.max_speed; pn.dwa_keep_gap = 0.0   # defaults moved on: compare like with like
for k, v in kv:
    setattr(po, k, float(v)); setattr(pn, k, float(v))
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
r.set_filter(rosbag2_py.StorageFilter(topics=['/scan'])); scans = []
while r.has_next():
    _, d, _ = r.read_next(); m = deserialize_message(d, LaserScan); scans.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m))
ts = np.array([s[0] for s in scans])
R = list(csv.DictReader(open(f'logs/exp/{exp}/run.csv')))
g = lambda q, k: float(q[k]) if q[k] not in ('', 'None') else float('nan')
xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
O = old.FrenetPlanner(xy, po); N = new.DWAPlanner(xy, pn)
mo = old.UnseenMemory(po.memory_hold, po.memory_keep_r, po.memory_cell, po.memory_rmin_margin); mn = new.UnseenMemory(pn.memory_hold, pn.memory_keep_r, pn.memory_cell, pn.memory_rmin_margin)
LASER = (0.1236, 0.0, 0.0); si = -1; bad = n = 0; to = []; tn = []; first = None; modes = {}; maxv = 0.0; stuck = 0
OD = np.array([[g(q, 't'), g(q, 'x'), g(q, 'y'), g(q, 'yaw')] for q in R])
for q in R:
    t = g(q, 't')
    if not (ts[0] <= t <= ts[-1]) or math.isnan(g(q, 'map_x')) or math.isnan(g(q, 'pp_v')): continue
    i = int(np.searchsorted(ts, t, side='right') - 1); gt = (g(q, 'x'), g(q, 'y'), g(q, 'yaw'))
    if i != si:
        si = i; m = scans[i][1]; pts = old.scan_to_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, po.plan_range, LASER)
        k = int(np.argmin(np.abs(OD[:, 0] - ts[i])))
        mo.add(t, pts, tuple(OD[k, 1:])); mn.add(t, pts, tuple(OD[k, 1:]))
    uo = mo.unseen(t, gt, LASER[:2], m.range_min, m.angle_max); un = mn.unseen(t, gt, LASER[:2], m.range_min, m.angle_max)
    pose = (g(q, 'map_x'), g(q, 'map_y'), g(q, 'map_yaw')); pv, ps = g(q, 'pp_v'), g(q, 'pp_steer')
    a = time.perf_counter(); vo, so, io = O.step(pose, pts, pv, ps, t, uo, gt); b = time.perf_counter(); vn, sn, inn = N.step(pose, pts, pv, ps, t, un, gt); c = time.perf_counter()
    to.append((b - a) * 1e3); tn.append((c - b) * 1e3); n += 1; maxv = max(maxv, vn); modes[inn['mode']] = modes.get(inn['mode'], 0) + 1; stuck += bool(inn.get('stuck'))
    same = abs(vo - vn) < 1e-9 and abs(so - sn) < 1e-9 and io['mode'] == inn['mode'] and bool(io.get('stuck')) == bool(inn.get('stuck')) and O.last_rad == N.last_rad
    if not same:
        bad += 1
        if first is None: first = (t, vo, vn, so, sn, io['mode'], inn['mode'], O.last_rad, N.last_rad)
to, tn = np.array(to), np.array(tn)
print(exp, 'ticks', n, 'different', bad, 'first', first, 'modes', modes, 'stuck', stuck, 'max v', round(maxv, 3))
print('  ms old mean/p95/max', to.mean().round(1), np.percentile(to, 95).round(1), to.max().round(1), '| new', tn.mean().round(1), np.percentile(tn, 95).round(1), tn.max().round(1), '| dwa calls', O.dwa2.stats, N.dwa.stats)
