#!/usr/bin/env python3
"""Open-loop: real /scan from a bag + logged poses and Pure Pursuit commands -> FrenetPlanner.step().
Shows what the planner picks at those poses, not what the car would do. Reads only.
Inputs: /scan and header stamps from the bag; odom pose (x, y, yaw), map pose, pp_v, pp_steer, t from
run.csv, the monitor's sim clock at 20 Hz; route from track_map3_pp.csv. Laser mount from the bag's TF.
usage (in the container): bag_planner_replay.py <exp> <bag dir> <t_from> <t_to> [k=v ...]
"""
import csv, importlib.util, math, os, sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
exp, bag, t_from, t_to = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
spec = importlib.util.spec_from_file_location('fp', 'src/frenet_planner_optimized.py')
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
p = fp.default_params(); fp.apply_profile(p, 'dwa_lap')
for kv in sys.argv[5:]:
    k, v = kv.split('='); setattr(p, k, type(getattr(p, k))(float(v)))
LASER = (0.1236, 0.0, 0.0)
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
r.set_filter(rosbag2_py.StorageFilter(topics=['/scan']))
scans = []
while r.has_next():
    _, d, _ = r.read_next(); m = deserialize_message(d, LaserScan)
    scans.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m))
print('scans', len(scans), 'from', round(scans[0][0], 2), 'to', round(scans[-1][0], 2))
R = list(csv.DictReader(open(f'logs/exp/{exp}/run.csv')))
g = lambda r_, k: float(r_[k]) if r_[k] not in ('', 'None') else float('nan')
xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
P = fp.FrenetPlanner(xy, p)
mem = fp.UnseenMemory(p.memory_hold, p.memory_keep_r, p.memory_cell, p.memory_rmin_margin, p.memory_average)
OB = {'o1': (5.0, -0.0878, .15, .4), 'o2': (-3.0, 3.05424, .15, .4), 'o3': (-8.0, 1.14516, .15, .5)}[os.environ.get('BOX', 'o2')]; _plan = P.dwa2.plan; sel = {'p': None}
def plan_(*k, **kw):
    r_ = _plan(*k, **kw); sel['p'] = None if r_ is None else r_[1]; return r_
P.dwa2.plan = plan_
import runpy
_sv = sys.argv; sys.argv = ['x']; SG = runpy.run_path('tools/exp/side_gaps.py'); sys.argv = _sv
gap_wall = SG['gap_wall']; wall_low = 9.0; low_ranges = []; arc_ok = arc_n = 0
def plan_(pts_, target, ty, steer0, hl, hw, route=None, path_gain=1.0):
    r_ = _plan(pts_, target, ty, steer0, hl, hw, route=route, path_gain=path_gain); sel['p'] = None if r_ is None else r_[1]
    if r_ is not None:
        sh = P.dwa2.shapes; A_ = pts_[(np.abs(pts_[:, 0]) < 2.0) & (np.abs(pts_[:, 1]) < 2.0)]
        c_ = fp.footprint_clearance(sh.reshape(-1, 3), A_, hl, hw).reshape(sh.shape[0], sh.shape[1]) if len(A_) else None
        if c_ is not None:
            h_ = c_ <= 0.0; fr = np.where(h_.any(axis=1), h_.argmax(axis=1), sh.shape[1]); ok_ = fr * P.dwa2.spacing >= P.dwa2.min_free
            low_ = np.where(np.arange(sh.shape[1])[None, :] < fr[:, None], c_, np.inf).min(axis=1)[ok_]
            sel['range'] = float(low_.max() - low_.min()); sel['n'] = int(ok_.sum())
    return r_
P.dwa2.plan = plan_
def box_gap(x, y, yaw):
    """True gap of the 0.33 x 0.31 body at a pose to the o2 box, sampled on the body outline."""
    u = np.linspace(-.165, .165, 12); v = np.linspace(-.155, .155, 12)
    bx = np.r_[u, u, np.full(12, -.165), np.full(12, .165)]; by = np.r_[np.full(12, -.155), np.full(12, .155), v, v]
    c, s = math.cos(yaw), math.sin(yaw); X = x + c * bx - s * by; Y = y + s * bx + c * by
    return float(np.hypot(np.maximum(np.abs(X - OB[0]) - OB[2], 0), np.maximum(np.abs(Y - OB[1]) - OB[3], 0)).min())
sel_low = 9.0; none_late = 0; sel_n = closer = 0; first_low = wall_near = 9.0; v_near = 0.0; zero_near = 0
S = scans[0][1]; si = -1; pts = None; last_print = -1e9; none0 = 0; first_none = None; rows = 0
ts = np.array([s[0] for s in scans]); OD = np.array([[g(q, 't'), g(q, 'x'), g(q, 'y'), g(q, 'yaw')] for q in R])
def layers(pts, un):
    out = []
    for name, A in (('scan', pts), ('scan+mem', np.vstack([pts, un]) if len(un) else pts)):
        A = A[np.hypot(A[:, 0], A[:, 1]) > p.body_r]
        for rad in (0.20, 0.17):
            hl, hw = P.rect(rad); sh = P.dwa2.shapes
            c = fp.footprint_clearance(sh.reshape(-1, 3), A, hl, hw).reshape(sh.shape[0], sh.shape[1])
            h = c <= 0.0; free = np.where(h.any(axis=1), h.argmax(axis=1), sh.shape[1])
            st = float(fp.footprint_clearance(np.zeros((1, 3)), A, hl, hw)[0])
            out.append(f"{name}@{rad:.2f} free {free.max() * P.dwa2.spacing:.2f} start {st:+.3f}")
    return ' | '.join(out)
for q in R:
    t = g(q, 't')
    if not (ts[0] <= t <= t_to) or math.isnan(g(q, 'map_x')) or math.isnan(g(q, 'pp_v')): continue
    i = int(np.searchsorted(ts, t, side='right') - 1); gt = (g(q, 'x'), g(q, 'y'), g(q, 'yaw'))
    if i != si:
        si = i; m = scans[i][1]
        pts = fp.scan_to_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max, p.plan_range, LASER)
        k = int(np.argmin(np.abs(OD[:, 0] - ts[i]))); mem.add(t, pts, tuple(OD[k, 1:]))
    un = mem.unseen(t, gt, LASER[:2], m.range_min, m.angle_max)
    if os.environ.get('NOMEM'): un = np.zeros((0, 2))
    v, st, info = P.step((g(q, 'map_x'), g(q, 'map_y'), g(q, 'map_yaw')), pts, g(q, 'pp_v'), g(q, 'pp_steer'), t, un, gt)
    n = P.dwa2.stats['none']; rows += 1
    if t >= t_from and sel['p'] is not None and v > 0:
        c_, s_ = math.cos(gt[2]), math.sin(gt[2]); Q = sel['p']
        Q = np.c_[Q[:, 0] - (0.12 - p.dwa_rear_lr) + (0.12 - p.dwa_rear_lr) * np.cos(Q[:, 2]), Q[:, 1] + (0.12 - p.dwa_rear_lr) * np.sin(Q[:, 2]), Q[:, 2]]
        W4 = lambda q_: (gt[0] + c_ * q_[0] - s_ * q_[1], gt[1] + s_ * q_[0] + c_ * q_[1], gt[2] + q_[2])
        now_ = box_gap(*gt)
        if now_ < 0.12:
            sel_n += 1; first = [box_gap(*W4(q_)) for q_ in Q[:4]]; first_low = min(first_low, min(first))
            if len(Q) >= 4 and first[3] < now_: closer += 1
            if now_ < 0.06: v_near = max(v_near, v)
            wall_near = min(wall_near, min(gap_wall(*W4(q_)) for q_ in Q[:4]))
        Q2 = Q[::2]; wall_low = min(wall_low, min(gap_wall(gt[0] + c_ * q_[0] - s_ * q_[1], gt[1] + s_ * q_[0] + c_ * q_[1], gt[2] + q_[2]) for q_ in Q2))
        g_now = box_gap(*gt)
        if g_now < 0.12 and len(Q) >= 4:
            q_ = Q[3]; arc_n += 1; arc_ok += box_gap(gt[0] + c_ * q_[0] - s_ * q_[1], gt[1] + s_ * q_[0] + c_ * q_[1], gt[2] + q_[2]) >= g_now
            low_ranges.append(sel.get('range', float('nan')))
        sel_low = min(sel_low, min(box_gap(gt[0] + c_ * q_[0] - s_ * q_[1], gt[1] + s_ * q_[0] + c_ * q_[1], gt[2] + q_[2]) for q_ in Q[::2]))
    if t >= t_from and v == 0.0: zero_near += 1
    if t > float(os.environ.get('LATE', 457.5)) and v == 0.0: none_late += 1
    if t >= t_from and t - globals().get('last_dump', -1e9) >= 1.0 and len(un):
        globals()['last_dump'] = t; c_, s_ = math.cos(gt[2]), math.sin(gt[2]); W = np.c_[gt[0] + c_ * un[:, 0] - s_ * un[:, 1], gt[1] + s_ * un[:, 0] + c_ * un[:, 1]]
        hl, hw = P.rect(0.17); ins = (np.abs(un[:, 0]) < hl + .02) & (np.abs(un[:, 1]) < hw + .02) & (np.hypot(un[:, 0], un[:, 1]) > p.body_r)
        dB = np.hypot(np.maximum(np.abs(W[:, 0] - OB[0]) - OB[2], 0), np.maximum(np.abs(W[:, 1] - OB[1]) - OB[3], 0))
        print(f"DUMP t {t:.1f} true ({gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f}) mem {len(un)} body-rect hits {int(ins.sum())} world {np.round(W[ins][:6], 3).tolist()} dist to o2 {np.round(dB[ins][:6], 3).tolist()} base {np.round(un[ins][:6], 3).tolist()} | mem within 0.03 of box {int((dB < .03).sum())} p90 {np.percentile(dB[dB < .3], 90) if (dB < .3).any() else -1:.3f} max {dB[dB < .3].max() if (dB < .3).any() else -1:.3f}")
    if n > none0:
        if t >= t_from and first_none is None: first_none = t
        if t >= t_from: print(f"NONE t {t:.2f} calls {P.dwa2.stats['calls']} none {n} true ({gt[0]:.3f},{gt[1]:.3f}) mem {len(un)}")
        none0 = n
    if t >= t_from and t - last_print >= 0.4:
        last_print = t
        print(f"t {t:.2f} {info['mode']} v {v:.2f} st {st:+.2f} log {g(q, 'cmd_v'):.2f} {g(q, 'cmd_steer'):+.2f} rad {P.last_rad:.3f} min {P.min_rad:.3f} mem {len(un)} pts {len(pts)} | {layers(pts, un)}")
        P.min_rad = p.r_robot
print('selected path lowest true box gap', round(sel_low, 3), 'zero-speed ticks late', none_late)
print('selected path lowest true wall gap', round(wall_low, 3), 'first-arc-end gap >= current', arc_ok, 'of', arc_n, 'low range p50/p90 at box gap<0.12', np.round(np.nanpercentile(low_ranges, [50, 90]), 3) if low_ranges else None)
print('EXP029 ticks', sel_n, 'closer at 0.4', closer, 'share', round(closer / max(sel_n, 1), 3), 'first 0.4 lowest box', round(first_low, 3), 'wall', round(wall_near, 3), 'max v under 0.06', v_near, 'zero-speed', zero_near)
print('rows', rows, 'first none', first_none, 'stats', P.dwa2.stats)
