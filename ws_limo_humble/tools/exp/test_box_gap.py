#!/usr/bin/env python3
"""Checks run_monitor.box_clearance against dense sampling of both outlines. Needs ROS sourced."""
import math
import random
import sys

sys.path.insert(0, 'tools/exp')
from run_monitor import FOOT_HALF, box_clearance  # noqa: E402


def outline(hx, hy, n=120):
    for i in range(n + 1):
        t = -1 + 2 * i / n
        yield (hx * t, -hy)
        yield (hx * t, hy)
        yield (-hx, hy * t)
        yield (hx, hy * t)


random.seed(1)
box = (0.0, 0.0, 0.15, 0.4)
worst = 0.0
for _ in range(400):
    x, y, yaw = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-3.14, 3.14)
    c, s = math.cos(yaw), math.sin(yaw)
    car = [(x + c * a - s * b, y + s * a + c * b) for a, b in outline(*FOOT_HALF)]
    brute = min(math.hypot(max(abs(px) - 0.15, 0), max(abs(py) - 0.4, 0)) for px, py in car)
    got = box_clearance(x, y, yaw, box)
    if brute > 0.003:
        worst = max(worst, abs(got - brute))
print('max error against sampled outline', round(worst, 4))
tests = [((0.5, 0, 0), 0.5 - 0.165 - 0.15), ((0, 0, 0), 0.0)]
ok = worst < 0.004
for (x, y, yaw), want in tests:
    got = box_clearance(x, y, yaw, box)
    print((x, y, yaw), 'gap', round(got, 4), 'want', round(want, 4))
    ok = ok and abs(got - want) < 1e-6
# Car corner hits between the old 5x5 grid samples: a box corner against the car's side.
got = box_clearance(0.0, -0.4 - 0.155 - 0.01, 0.0, box)
print('flat side gap', round(got, 4), 'want 0.01')
ok = ok and abs(got - 0.01) < 1e-6
print('PASS' if ok else 'FAIL')
