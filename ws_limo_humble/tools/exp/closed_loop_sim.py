#!/usr/bin/env python3
"""Offline closed-loop drive of FrenetPlanner over the map and world boxes. Not Gazebo.

Kinematic bicycle with first-order speed and steering lag, simulated LiDAR, optional AMCL-like lag.
Walls are the SLAM map's (0.05 m cells). It compares planners; it is not evidence of driving.
usage: closed_loop_sim.py <planner.py> [--start-s S] [--time T] [--set k=v] [--lag M] [--no-stop-rule]
"""
import argparse, csv, importlib.util, json, math, re, sys, time
import numpy as np, yaml
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('planner'); ap.add_argument('--time', type=float, default=300.0)
ap.add_argument('--start-s', type=float, default=None)
ap.add_argument('--lag', type=float, default=0.15, help='along-track map pose lag [m]')
ap.add_argument('--yaw-bias', type=float, default=0.05)
ap.add_argument('--laps', type=int, default=1)
ap.add_argument('--map-dy', type=float, default=0.0, help='constant world-y bias of the map pose')
ap.add_argument('--stall-limit', type=float, default=12.0)
ap.add_argument('--noise', type=float, default=0.01, help='range noise stddev, as in sensor.xacro')
ap.add_argument('--profile', default='')
ap.add_argument('--pp-max', type=float, default=0.8)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--rear-lr', type=float, default=0.0, help='base_link ahead of the rear axle [m]; 0 is the old form')
ap.add_argument('--no-stop-rule', action='store_true')
ap.add_argument('--out', default='')
ap.add_argument('--debug', action='store_true')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('fp', a.planner); fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
src = open('tools/exp/run_monitor.py').read(); ns = {'math': math}
exec(re.search(r'^OBSTACLES = .*$', src, re.M).group(0), ns); exec(re.search(r'^FOOT_HALF = .*$', src, re.M).group(0), ns)
i0 = src.index('def box_clearance'); exec(src[i0:re.compile(r'^\S', re.M).search(src, i0 + 10).start()], ns)
gap_box, OB = ns['box_clearance'], ns['OBSTACLES']; FL, FW = ns['FOOT_HALF']
LX, RMIN, RMAX, FOV, N = 0.124, 0.2, 8.0, 2.0943951, 720
ANG = np.linspace(-FOV, FOV, N)
cfg = yaml.safe_load(open('src/Raceline-Optimization/maps/track_map3.yaml'))
img = np.array(Image.open('src/Raceline-Optimization/maps/track_map3.pgm')); H, W = img.shape
res, (ox, oy) = cfg['resolution'], cfg['origin'][:2]; occ = img < 100
rows, cols = np.nonzero(occ); WX = ox + (cols + .5) * res; WY = oy + (H - rows - .5) * res
STEPS = np.arange(RMIN, 5.2, 0.01)

def lidar(x, y, yaw):
    lx, ly = x + LX * math.cos(yaw), y + LX * math.sin(yaw)
    wx = lx + np.outer(STEPS, np.cos(ANG + yaw)); wy = ly + np.outer(STEPS, np.sin(ANG + yaw))
    c = np.clip(((wx - ox) / res).astype(int), 0, W - 1); r = np.clip((H - 1 - (wy - oy) / res).astype(int), 0, H - 1)
    hit = occ[r, c]
    for cx, cy, hx, hy in OB.values():
        hit = hit | ((np.abs(wx - cx) <= hx) & (np.abs(wy - cy) <= hy))
    f = hit.argmax(axis=0); ok = hit.any(axis=0) & (f > 0)
    d = STEPS[f[ok]] + np.random.normal(0.0, a.noise, int(ok.sum()))
    return np.c_[d * np.cos(ANG[ok]) + LX, d * np.sin(ANG[ok])]

def wall_gap(x, y, yaw):
    m = (np.abs(WX - x) < 1) & (np.abs(WY - y) < 1)
    if not m.any(): return 9.0
    c, s = math.cos(yaw), math.sin(yaw); dx, dy = WX[m] - x, WY[m] - y
    u, v = np.abs(dx * c + dy * s), np.abs(-dx * s + dy * c)
    return float(np.hypot(np.maximum(u - FL, 0), np.maximum(v - FW, 0)).min())

xy = np.loadtxt('src/Raceline-Optimization/outputs/track_map3_pp.csv', delimiter=',')[:, :2]
p = fp.default_params()
if a.profile: fp.apply_profile(p, a.profile)
for it in a.set:
    k, v = it.split('='); old = getattr(p, k)
    if isinstance(old, bool): val = v in ('1', 'true', 'True')
    elif isinstance(old, tuple): val = tuple(float(z) for z in v.split(','))
    else: val = type(old)(float(v))
    setattr(p, k, val)
P = fp.FrenetPlanner(xy, p)
mem = fp.UnseenMemory(p.memory_hold, p.memory_keep_r, p.memory_avg_cell if p.memory_average else p.memory_cell, p.memory_rmin_margin, p.memory_average)
ref, theta, ds, n = P.ref, P.theta, P.ds, P.n
if a.start_s is None:
    x, y, yaw = 7.4926918, 0.0260169, 0.6297
    i = int(np.argmin(np.hypot(ref[:, 0] - x, ref[:, 1] - y)))
else:
    i0 = int(np.argmin(np.hypot(ref[:, 0] - 7.4926918, ref[:, 1] - 0.0260169)))
    i = (i0 + int(a.start_s / ds)) % n; x, y = ref[i]; yaw = float(theta[i])
L = n * ds
dt, t, v, st, last_scan, pts = 0.05, 0.0, 0.0, 0.0, -1e9, np.zeros((0, 2))
cmd_q = [(0.0, 0.0)] * 2     # about 0.1 s command delay
idx = i; prog = 0.0; hist = {k: [] for k in OB}; mins = {k: 9.0 for k in OB}; min_w = 9.0; stuck = 0; linear = {}
worst_view_box = worst_view_wall = max_alat = max_dst = 0.0; first_move = None; worst_view = 0.0; blind_ticks = 0; stop = None; ms = []; zero_time = 0.0; hx = []; laps = []; last_lap = 0.0; plan_ticks = 0
while t < a.time:
    # map pose: truth with along-track lag and a yaw bias
    c, s = math.cos(yaw), math.sin(yaw)
    mp = (x - a.lag * c, y - a.lag * s + a.map_dy, yaw + a.yaw_bias)
    odom = (x, y, yaw)
    if t - last_scan >= 0.125:
        pts = lidar(x, y, yaw); mem.add(t, pts, odom); last_scan = t
    un = mem.unseen(t, odom, (LX, 0.0), RMIN, FOV)
    # stand-in for the Pure Pursuit node
    j = (idx + int(max(0.35, 0.5 * v) / ds)) % n
    tb = fp.to_base(ref[[j]], mp)[0]
    pp_steer = math.atan2(2 * 0.24 * math.sin(math.atan2(tb[1], tb[0])), max(0.2, float(np.hypot(*tb))))
    t1 = time.perf_counter()
    cv, cs, info = P.step(mp, pts, min(a.pp_max, v + 0.3), pp_steer, t, un, odom)
    ms.append((time.perf_counter() - t1) * 1e3)
    stuck += bool(info.get('stuck')); plan_ticks += info.get('planner') == 'HA'
    cmd_q.append((cv, cs)); dv, ds_cmd = cmd_q.pop(0)
    v += (dv - v) * (dt / 0.15); st += (float(np.clip(ds_cmd, -.42, .42)) - st) * (dt / 0.12)
    beta_ = math.atan(a.rear_lr * math.tan(st) / 0.24)
    x += v * math.cos(yaw + beta_) * dt; y += v * math.sin(yaw + beta_) * dt; yaw += v * math.cos(beta_) * math.tan(st) / 0.24 * dt; t += dt
    w = np.hypot(ref[:, 0] - x, ref[:, 1] - y); ids = (idx + np.arange(-20, 60)) % n; new = int(ids[np.argmin(w[ids])])
    prog += ((new - idx + n // 2) % n - n // 2) * ds; idx = new
    if prog - last_lap >= L: laps.append(round(t, 1)); last_lap = prog
    wg = wall_gap(x, y, yaw); min_w = min(min_w, wg)
    for k, b in OB.items():
        g = gap_box(x, y, yaw, b); mins[k] = min(mins[k], g); h = hist[k]; h.append((t, g)); del h[:-6]
        rate = (g - h[0][1]) / max(t - h[0][0], 1e-3) if len(h) >= 2 else 0.0
        if g + min(rate, 0.0) * 0.7 < 0.02 and k not in linear:
            linear[k] = dict(t=round(t, 1), s=round(prog, 2), gap=round(g, 3), v=round(v, 2))
            if not a.no_stop_rule: stop = f'near-contact rule {k}'
        if g <= 0.0: stop = f'box contact {k}'
    if wg <= 0.0: stop = 'wall contact (map cells)'
    # what the planner could see of the nearest box: does its view explain the speed it chose?
    near_box = min(OB, key=lambda k: gap_box(x, y, yaw, OB[k]))
    allp = np.vstack([pts, un]) if un is not None and len(un) else pts
    seen_clr = float(fp.footprint_clearance(np.zeros((1, 3)), allp, FL, FW).min()) if len(allp) else 9.0
    true_clr = min(gap_box(x, y, yaw, OB[near_box]), wg)
    worst_view = max(worst_view, (true_clr < 0.1) * (seen_clr - true_clr))
    box_true = gap_box(x, y, yaw, OB[near_box])
    if box_true < 0.1 and box_true <= wg:
        worst_view_box = max(worst_view_box, seen_clr - box_true)
    if wg < 0.1 and wg < box_true:
        worst_view_wall = max(worst_view_wall, seen_clr - wg)
    max_alat = max(max_alat, v * v * abs(math.tan(st)) / 0.24)
    if hx:
        max_dst = max(max_dst, abs(st - hx[-1][5]))
    if true_clr < 0.1 and seen_clr - true_clr > 0.05:
        blind_ticks += 1
    if first_move is None and v > 0.03:
        first_move = round(t, 2)
    hp = getattr(P, 'ha_path', None)
    t_err, path_y = '', ''
    if hp is not None and len(hp):
        dd = np.hypot(hp[:, 0] - x, hp[:, 1] - y); t_err = round(float(dd.min()), 3)
        path_y = round(float(hp[int(np.argmin(np.abs(hp[:, 0] + 3.0))), 1]), 3)
    hx.append((t, x, y, yaw, v, st, prog, info['mode'], info.get('planner', ''), P.last_rad, round(seen_clr, 3), round(true_clr, 3), 0 if un is None else len(un), t_err, path_y))
    if a.debug and info.get('planner') == 'HA' and abs(t / 1.5 - round(t / 1.5)) < 0.02 and (hasattr(P, 'ha_path') and P.ha_path is not None):
        hp = P.ha_path
        near = hp[np.argsort(np.hypot(hp[:, 0] - x, hp[:, 1] - y))[0]:]
        print(f"t={t:.1f} car=({x:.2f},{y:.3f},{yaw:.2f}) rad={P.ha_rad:.2f} path_end=({hp[-1,0]:.2f},{hp[-1,1]:.2f}) "
              + ' '.join(f"x{q}:{hp[np.argmin(abs(hp[:,0]-q)),1]:.3f}" for q in (-2.4, -2.85, -3.0, -3.15, -3.5)), flush=True)
    if stop or len(laps) >= a.laps: break
    if v < 0.01 and cv == 0.0:
        zero_time += dt
        if zero_time > a.stall_limit:
            stop = 'stall 12 s'
            if a.debug and p.use_dwa:
                allp = np.vstack([pts, un]) if un is not None and len(un) else pts
                rr = np.hypot(allp[:, 0], allp[:, 1]); allp = allp[rr > p.body_r]
                print('DSTALL pose', round(x, 3), round(y, 3), round(yaw, 3), 'pts', len(pts), 'mem', 0 if un is None else len(un), 'v_arg floor', p.dwa_min_speed)
                near_ = allp[np.argsort(np.hypot(allp[:, 0], allp[:, 1]))[:10]]
                print(' nearest', np.round(near_, 3).tolist())
                for rad in (0.20, 0.17):
                    hl_, hw_ = P.rect(rad); o_ = []
                    for s_, k_ in zip(P.dwa.steers, P.dwa.curvs):
                        o_.append(round(float(fp.footprint_clearance(fp.arc(float(k_), 0.35, 7), allp, hl_, hw_).min()), 3))
                    print(' rad', rad, 'clearance per steer', o_)
                for rad in (0.20, 0.17):
                    hl_, hw_ = P.rect(rad); sh = P.dwa2.shapes
                    c_ = fp.footprint_clearance(sh.reshape(-1, 3), allp, hl_, hw_).reshape(sh.shape[0], sh.shape[1])
                    hit_ = c_ <= 0.0; free_ = np.where(hit_.any(axis=1), hit_.argmax(axis=1), sh.shape[1])
                    print(' two-stage rad', rad, 'best free m', round(float(free_.max()) * P.dwa2.spacing, 2), 'first steer of it', round(float(P.dwa2.firsts[free_.argmax()]), 2), 'shapes clear for 0.3 m', int((free_ >= 3).sum()), 'of', len(free_))
                print(' start clear r_min', round(float(fp.footprint_clearance(np.zeros((1, 3)), allp, *P.rect(0.17))[0]), 3), 'target', np.round(fp.to_base(ref[[(idx + 30) % n]], mp)[0], 2).tolist(), 'mode', info['mode'], 'cv', cv)
            if a.debug and not p.use_dwa:
                allp = np.vstack([pts, un]) if un is not None and len(un) else pts
                allp = allp[np.hypot(allp[:, 0], allp[:, 1]) > p.body_r]
                print('STALL pose', round(x, 3), round(y, 3), round(yaw, 3), 'steer', round(st, 3), 'pts', len(pts), 'mem', 0 if un is None else len(un), 'blocks', len(P.ha_blocks), 'try', P.ha_try)
                for rad in p.hybrid_radii:
                    hl_, hw_ = P.rect(rad)
                    print(' rad', rad, 'start clearance', round(float(fp.footprint_clearance(np.zeros((1, 3)), allp, hl_, hw_).min()), 3))
                    g_, gy_ = P.ha_goals(idx, mp, None)
                    r0 = P.ha.plan(allp, g_, gy_, hl_, hw_, st, 2.0)
                    print('  plan (no block) ', P.ha.stats)
                near_ = allp[np.hypot(allp[:, 0], allp[:, 1]) < 0.45]
                print(' nearest pts', np.round(near_[np.argsort(np.hypot(near_[:, 0], near_[:, 1]))[:8]], 3).tolist(), 'stuck', stuck)
                print(' ha', P.ha_stats, 'mode', info['mode'], 'rej', info.get('rej'), 'cv', cv)
            break
    else: zero_time = 0.0
arr = np.array([(h[0], h[4]) for h in hx]); step_ds = np.abs(arr[:, 1]) * 0.05
hmask = np.array([h[8] in ('HA', 'DWA') for h in hx])
slow = dict(ha_s=round(0.05 * float(hmask.sum()), 1), follow_s=round(0.05 * float((~hmask).sum()), 1),
            below_0p12_s=round(0.05 * float((arr[:, 1] < 0.12).sum()), 1), mean_v=round(float(arr[:, 1].mean()), 3), max_v=round(float(arr[:, 1].max()), 2))
r = dict(worst_view_box=round(worst_view_box, 3), worst_view_wall=round(worst_view_wall, 3), max_alat=round(max_alat, 3), max_steer_step=round(max_dst, 3), time_use=slow, first_move=first_move, worst_view=round(worst_view, 3), blind_ticks=blind_ticks, stop=stop, laps=laps, t=round(t, 1), progress=round(prog, 2), min_gap={k: round(z, 3) for k, z in mins.items()},
         min_wall_est=round(min_w, 3), stuck_ticks=stuck, ha_ticks=plan_ticks, dwa=P.dwa2.stats, linear_rule=linear, ha=P.ha_stats,
         tick_ms=dict(mean=round(float(np.mean(ms)), 1), p95=round(float(np.percentile(ms, 95)), 1), max=round(float(np.max(ms)), 1)))
r['refine_fail'] = sys.modules['hybrid_astar_local'].REFINE_FAIL if 'hybrid_astar_local' in sys.modules else None
print(json.dumps(r))
if a.debug:
    for e in P.ha_events: print('EV', e)
if a.out:
    with open(a.out, 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow('t x y yaw v steer prog mode planner rad seen true mem track_err path_y_at_o2'.split()); wr.writerows(hx)
