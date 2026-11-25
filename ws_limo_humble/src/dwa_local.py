#!/usr/bin/env python3
"""Dynamic Window Approach for a car-like robot, in the base_link frame. numpy only.

Samples speed and steering, rolls each out as one arc, and drops arcs that hit a point with the
same inflated-rectangle test as the other planners. A speed is allowed only if the car can stop
short of every point along that arc.
"""
import math

import numpy as np

from hybrid_astar_local import arc, closing_speed_limit, footprint_clearance


class DWA:
    def __init__(self, wheelbase, max_steer, n_steer=15, n_speed=6, horizon=1.5, min_len=0.35,
                 w_target=1.0, w_clear=0.6, w_speed=0.35, w_change=0.08, clear_cap=0.30):
        self.steers = np.linspace(-max_steer, max_steer, n_steer)
        self.curvs = np.tan(self.steers) / wheelbase
        self.max_steer = max_steer
        self.n_speed, self.horizon, self.min_len = n_speed, horizon, min_len
        self.w_target, self.w_clear, self.w_speed, self.w_change, self.clear_cap = \
            w_target, w_clear, w_speed, w_change, clear_cap
        self.stats = dict(calls=0, none=0)

    def plan(self, pts, target, v_min, v_max, steer0, half_length, half_width,
             keep=0.02, stop_time=1.2, drift_time=0.25):
        """Best (speed, steer, poses, bare clearance), or None if no arc is free."""
        self.stats['calls'] += 1
        best = None
        for v in np.linspace(v_min, max(v_min, v_max), self.n_speed):
            length = max(self.min_len, float(v) * self.horizon)
            n = max(4, int(length / 0.05))
            for s, k in zip(self.steers, self.curvs):
                poses = arc(float(k), length, n)
                if len(pts):
                    if footprint_clearance(poses, pts, half_length, half_width).min() <= 0.0:
                        continue
                    bare = float(footprint_clearance(poses, pts, half_length - 0.045 if half_length > 0.165 else half_length,
                                                     min(half_width, 0.155)).min())
                    # The lowest speed stays allowed: the rectangle test above already bounds it,
                    # and refusing it would turn every narrow pass into a stop.
                    if v > v_min and float(closing_speed_limit(poses[:n // 2 + 1], pts, 0.165, 0.155, keep,
                                                               stop_time, drift_time).min()) < v:
                        continue
                else:
                    bare = self.clear_cap
                end = poses[-1]
                to_target = math.hypot(target[0] - end[0], target[1] - end[1]) / max(1e-3, math.hypot(*target))
                score = (-self.w_target * to_target + self.w_clear * min(bare, self.clear_cap) / self.clear_cap
                         + self.w_speed * v / max(v_max, 1e-3)
                         - self.w_change * abs(float(s) - steer0) / self.max_steer)
                if best is None or score > best[0]:
                    best = (score, float(v), float(s), poses, bare)
        if best is None:
            self.stats['none'] += 1
            return None
        return best[1:]


class TwoStageDWA:
    """Two arcs in a row: enough to line up and then pass, which one arc cannot see.

    A shape counts up to its first contact, and needs at least min_free metres clear. Picks a shape
    only; the caller sets the speed from how close it comes to the points. Forward only.
    """

    def __init__(self, wheelbase, max_steer, n_first=9, n_second=7, first=0.40, second=1.00, spacing=0.10,
                 min_free=0.50, w_target=1.0, w_yaw=0.25, w_clear=0.8, w_change=0.10, w_free=0.5, clear_cap=0.25,
                 w_path=1.5, path_scale=0.5, clear_skip=0, rear_lr=0.0):
        self.clear_skip = clear_skip              # poses left out of the clearance score, not the contact test
        self.max_steer, self.spacing, self.min_free = max_steer, spacing, min_free
        self.w_target, self.w_yaw, self.w_clear, self.w_change, self.w_free, self.clear_cap = \
            w_target, w_yaw, w_clear, w_change, w_free, clear_cap
        self.w_path, self.path_scale = w_path, path_scale
        s1 = np.linspace(-max_steer, max_steer, n_first)
        s2 = np.linspace(-max_steer, max_steer, n_second)
        n1, n2 = int(round(first / spacing)), int(round(second / spacing))
        shapes, firsts = [], []
        for a in s1:
            head = arc(math.tan(a) / wheelbase, first, n1)
            x, y, yaw = head[-1]
            c, s = math.cos(yaw), math.sin(yaw)
            for b in s2:
                tail = arc(math.tan(b) / wheelbase, second, n2)
                shapes.append(np.vstack([head, np.c_[x + c * tail[:, 0] - s * tail[:, 1],
                                                     y + s * tail[:, 0] + c * tail[:, 1], yaw + tail[:, 2]]]))
                firsts.append(a)
        self.shapes = np.stack(shapes)            # (M, n, 3)
        if rear_lr > 0.0:                         # the arcs are the rear axle's; poses wanted are base_link's, rear_lr ahead of it
            yaw_ = self.shapes[:, :, 2]
            self.shapes = np.stack([self.shapes[:, :, 0] - rear_lr + rear_lr * np.cos(yaw_),
                                    self.shapes[:, :, 1] + rear_lr * np.sin(yaw_), yaw_], axis=2)
        self.firsts = np.array(firsts)
        self.stats = dict(calls=0, none=0)

    def plan(self, pts, target, target_yaw, steer0, half_length, half_width, route=None, path_gain=1.0):
        """(first steering, poses up to the first contact) of the best shape, or None."""
        self.stats['calls'] += 1
        M, n, _ = self.shapes.shape
        idx = np.arange(M)
        if len(pts):
            near = pts[(np.abs(pts[:, 0]) < 2.0) & (np.abs(pts[:, 1]) < 2.0)]
        else:
            near = pts
        if len(near):
            clr = footprint_clearance(self.shapes.reshape(-1, 3), near, half_length, half_width).reshape(M, n)
        else:
            clr = np.full((M, n), self.clear_cap)
        hit = clr <= 0.0
        free = np.where(hit.any(axis=1), hit.argmax(axis=1), n)     # poses clear before the first contact
        ok = free * self.spacing >= self.min_free
        if not ok.any():
            self.stats['none'] += 1
            return None
        last = np.maximum(free - 1, 0)
        ends = self.shapes[idx, last, :]
        dist = np.hypot(target[0] - ends[:, 0], target[1] - ends[:, 1]) / max(1e-3, math.hypot(*target))
        dyaw = np.abs((target_yaw - ends[:, 2] + math.pi) % (2.0 * math.pi) - math.pi) / math.pi
        # lowest clearance over the clear part of each shape
        # Hypothesis, unmeasured: with clear_skip 0 a car beside something scores about the same on every shape.
        masked = np.where((np.arange(n)[None, :] < free[:, None]) & (np.arange(n)[None, :] >= self.clear_skip), clr, np.inf)
        low = masked.min(axis=1)
        score = (-self.w_target * dist - self.w_yaw * dyaw
                 + self.w_clear * np.minimum(low, self.clear_cap) / self.clear_cap
                 + self.w_free * free / n
                 - self.w_change * np.abs(self.firsts - steer0) / self.max_steer)
        if route is not None and len(route):       # stay on the route unless something is in the way
            d = np.hypot(ends[:, 0, None] - route[None, :, 0], ends[:, 1, None] - route[None, :, 1]).min(axis=1)
            score -= path_gain * self.w_path * np.minimum(d, 3.0 * self.path_scale) / self.path_scale
        score[~ok] = -np.inf
        i = int(score.argmax())
        return float(self.firsts[i]), self.shapes[i, :int(free[i])]
