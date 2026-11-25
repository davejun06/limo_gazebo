#!/usr/bin/env python3
"""With no path, the planner must hold still and report STUCK every tick after 5 s, pulses or not."""
import importlib.util, math, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('fp', 'src/frenet_planner_optimized.py')
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
th = np.linspace(0, 2 * math.pi, 400, endpoint=False)
route = np.c_[20 * np.cos(th), 20 * np.sin(th)]
POSE = (20.0, 0.0, math.pi / 2)

def case(name, pts, drift):
    p = fp.default_params(); p.use_hybrid = True; p.hybrid_budget = 0.2
    P = fp.FrenetPlanner(route, p)
    x, y, yaw = POSE; t = 0.0; late_moves = 0; late_notstuck = 0; crawl = 0
    while t < 20.0:
        v, st, info = P.step((x, y, yaw), pts, 0.5, 0.0, t, None, (x, y, yaw))
        crawl += info['mode'] == 'CRAWL'
        if t > 5.5:
            late_moves += v != 0.0
            late_notstuck += not info.get('stuck')
        if drift:
            y += v * 0.05      # the car creeps forward on a pulse
        t += 0.05
    ok = late_moves == 0 and late_notstuck == 0
    print(f"{name}: crawl ticks {crawl}, moved after 5 s {late_moves}, not STUCK after 5 s {late_notstuck}, travelled {y:.3f} -> {'PASS' if ok else 'FAIL'}")
    return ok

ang = np.linspace(-math.pi, math.pi, 400)
tight = np.c_[0.2 * np.cos(ang), 0.2 * np.sin(ang)]             # hard against the car all round: no arc
wall = np.c_[np.full(120, 0.75), np.linspace(-2.4, 2.4, 120)]    # wall ahead: a short escape arc passes
side = np.r_[np.c_[np.linspace(-1, 1, 60), np.full(60, 0.5)], np.c_[np.linspace(-1, 1, 60), np.full(60, -0.5)]]
r = [case('no arc at all', tight, False), case('escape arc possible', np.vstack([wall, side]), True)]
sys.exit(0 if all(r) else 1)
