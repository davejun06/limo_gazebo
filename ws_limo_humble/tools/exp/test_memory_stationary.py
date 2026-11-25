#!/usr/bin/env python3
"""Remembered points must outlast a stop, yet still age while the car moves."""
import importlib.util, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('fp', 'src/frenet_planner_optimized.py')
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
POINT = np.array([[0.0, -0.17]])   # beside the car: nearer than range_min, so blind
args = ((0.124, 0.0), 0.2, 2.0943951)


def run(speed, seconds, hold=15.0):
    m = fp.UnseenMemory(hold, 1.0, 0.02, 0.03)
    m.add(0.0, POINT, (0.0, 0.0, 0.0))
    t, x, last = 0.0, 0.0, None
    while t < seconds:
        t += 0.05; x += speed * 0.05
        n = len(m.unseen(t, (x, 0.0, 0.0), *args))
        last = t if n else last
    return last


stopped = run(0.0, 60.0)
assert stopped is not None and stopped > 59.0, f'stopped car lost the point at {stopped}'
moving = run(0.06, 20.0)   # 0.06 m/s: 1 m keep radius not reached in 15 s
assert moving is not None and moving < 15.2, f'moving car kept the point until {moving}'
print('ok: stopped keeps it past', round(stopped, 1), 's; moving at 0.06 m/s drops it at', round(moving, 1), 's')
