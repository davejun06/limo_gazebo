#!/usr/bin/env python3
"""Short-range forward-only Hybrid A* for a car-like robot, in the base_link frame.

Searches (x, y, heading) with Ackermann arcs, so it can choose the pose in which the car goes
through a gap. Collision uses the same inflated-rectangle test as the Frenet lattice. numpy only.
"""
import heapq
import math
import time

import numpy as np


REFINE_FAIL = dict(curv=0, coll=0)   # why center_and_smooth kept the input (log only)


def footprint_clearance(poses, pts, half_length, half_width):
    """Distance from each pose's rectangle to the nearest point. 0 means a point is inside."""
    if not len(pts):
        return np.full(len(poses), np.inf)
    dx = pts[None, :, 0] - poses[:, None, 0]
    dy = pts[None, :, 1] - poses[:, None, 1]
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    ox = np.maximum(np.abs(c * dx + s * dy) - half_length, 0.0)
    oy = np.maximum(np.abs(-s * dx + c * dy) - half_width, 0.0)
    return np.hypot(ox, oy).min(axis=1)


def closing_speed_limit(poses, pts, half_length, half_width, keep, stop_time, drift_time):
    """Highest speed per pose that can stop short of every point, keeping `keep` metres.

    Each point counts by how fast the gap to it shrinks when driving ahead: fully when straight
    ahead, not at all when beside the car. A farther point ahead can be the limit, not the nearest
    point beside. Clearance alone also limits speed through drift_time, for tracking error.
    """
    dx = pts[None, :, 0] - poses[:, None, 0]
    dy = pts[None, :, 1] - poses[:, None, 1]
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    u = c * dx + s * dy
    ox = np.maximum(np.abs(u) - half_length, 0.0)
    oy = np.maximum(np.abs(-s * dx + c * dy) - half_width, 0.0)
    dist = np.hypot(ox, oy)
    closing = np.where(u > 0.0, ox / np.maximum(dist, 1e-6), 0.0)
    room = np.maximum(dist - keep, 0.0)
    by_point = room / np.maximum(stop_time * closing, 1e-3)
    return np.minimum(by_point.min(axis=1), room.min(axis=1) / drift_time)


def arc(steer_curv, length, samples):
    """Poses along an arc from the origin, heading +x: samples x (x, y, yaw), excluding the start."""
    d = length * np.arange(1, samples + 1) / samples
    if abs(steer_curv) < 1e-9:
        return np.c_[d, np.zeros(samples), np.zeros(samples)]
    h = steer_curv * d
    return np.c_[np.sin(h) / steer_curv, (1.0 - np.cos(h)) / steer_curv, h]


def obstacle_distance_field(pts, goal_xy, half_width, res=0.10):
    """Shortest obstacle-aware distance to the goal on a coarse grid, ignoring heading.

    Walls are grown by less than the car's half width, so a real gap is never sealed by the grid:
    this stays a guide for the search, never a collision test. Returns (field, origin_xy, res).
    """
    extent = min(6.0, float(np.abs(goal_xy).max()) + 1.5)
    n = int(2.0 * extent / res) + 1
    org = np.array([-extent, -extent])
    blocked = np.zeros((n, n), bool)
    if len(pts):
        ij = np.floor((pts - org) / res).astype(int)
        ok = (ij >= 0).all(axis=1) & (ij < n).all(axis=1)
        blocked[ij[ok, 1], ij[ok, 0]] = True
        for _ in range(int(max(0.0, half_width - 0.07) / res)):
            g = blocked.copy()
            g[1:] |= blocked[:-1]; g[:-1] |= blocked[1:]
            g[:, 1:] |= blocked[:, :-1]; g[:, :-1] |= blocked[:, 1:]
            blocked = g
    field = np.full((n, n), np.inf)
    gij = np.floor((goal_xy - org) / res).astype(int)
    ok = (gij >= 0).all(axis=1) & (gij < n).all(axis=1)
    if not ok.any():
        return None, org, res
    blocked[gij[ok, 1], gij[ok, 0]] = False
    field[gij[ok, 1], gij[ok, 0]] = 0.0
    diag = res * math.sqrt(2.0)
    for _ in range(3 * n):
        f = field.copy()
        f[1:, :] = np.minimum(f[1:, :], field[:-1, :] + res)
        f[:-1, :] = np.minimum(f[:-1, :], field[1:, :] + res)
        f[:, 1:] = np.minimum(f[:, 1:], field[:, :-1] + res)
        f[:, :-1] = np.minimum(f[:, :-1], field[:, 1:] + res)
        f[1:, 1:] = np.minimum(f[1:, 1:], field[:-1, :-1] + diag)
        f[:-1, :-1] = np.minimum(f[:-1, :-1], field[1:, 1:] + diag)
        f[1:, :-1] = np.minimum(f[1:, :-1], field[:-1, 1:] + diag)
        f[:-1, 1:] = np.minimum(f[:-1, 1:], field[1:, :-1] + diag)
        f[blocked] = np.inf
        if np.array_equal(np.nan_to_num(f, posinf=1e9), np.nan_to_num(field, posinf=1e9)):
            break
        field = f
    return field, org, res


class HybridAStar:
    def __init__(self, wheelbase, max_steer, step=0.20, samples=4, n_steer=7, xy_res=0.08,
                 yaw_res=math.radians(15.0), w_clear=1.5, clear_cap=0.25, w_steer=0.05,
                 w_dsteer=0.10, h_weight=1.2, goal_tol=0.25, goal_yaw_tol=0.6):
        self.step, self.samples, self.xy_res, self.yaw_res = step, samples, xy_res, yaw_res
        self.w_clear, self.clear_cap, self.w_steer, self.w_dsteer = w_clear, clear_cap, w_steer, w_dsteer
        self.h_weight, self.goal_tol, self.goal_yaw_tol = h_weight, goal_tol, goal_yaw_tol
        self.steers = np.linspace(-max_steer, max_steer, n_steer)
        self.max_steer = max_steer
        self.curvs = np.tan(self.steers) / wheelbase
        self.prims = np.stack([arc(k, step, samples) for k in self.curvs])   # n_steer, samples, 3
        self.stats = dict(expanded=0, result='', ms=0.0)

    def plan(self, pts, goal_xy, goal_yaw, half_length, half_width, steer0=0.0, budget=0.06):
        """One-shot search; 'budget' in self.stats if it did not finish in time."""
        gen = self.search(pts, goal_xy, goal_yaw, half_length, half_width, steer0, budget)
        try:
            next(gen)
        except StopIteration as done:
            return done.value
        gen.close()
        self.stats['result'] = 'budget'
        self.stats['ms'] = budget * 1e3
        return None

    def search(self, pts, goal_xy, goal_yaw, half_length, half_width, steer0=0.0, budget=0.04):
        """Generator: yields whenever `budget` seconds of work are used, so a search can span ticks.

        Its return value is Nx3 poses from the origin to a goal, or None if the space is exhausted.
        goal_xy, goal_yaw: acceptable goal points on the route (base frame), nearest first.
        """
        t0 = time.perf_counter()
        resumed = t0
        used = 0.0
        n_s, m = len(self.steers), self.samples
        reach = self.step + math.hypot(half_length, half_width) + self.clear_cap
        target = goal_xy[0]
        field, org, gres = obstacle_distance_field(pts, goal_xy, half_width)
        ng_max = field.shape[0] - 1 if field is not None else 0

        def heur(px_, py_):
            if field is not None:
                gi, gj = int((py_ - org[1]) / gres), int((px_ - org[0]) / gres)
                if 0 <= gi <= ng_max and 0 <= gj <= ng_max:
                    fv = field[gi, gj]
                    if fv != np.inf:
                        return float(fv)
                    return 20.0 + math.hypot(target[0] - px_, target[1] - py_)  # walled off on the grid
            return math.hypot(target[0] - px_, target[1] - py_)

        nodes = [(0.0, 0.0, 0.0, 0.0, -1, float(steer0))]     # x, y, yaw, g, parent, steer
        heap = [(self.h_weight * heur(0.0, 0.0), 0)]
        closed = set()
        n_exp = 0
        result = 'no_path'
        goal_node = None
        while heap:
            if n_exp % 8 == 0 and time.perf_counter() - resumed > budget:
                used += time.perf_counter() - resumed
                self.stats = dict(expanded=n_exp, result='pending', ms=used * 1e3)
                yield None
                resumed = time.perf_counter()
            _, i = heapq.heappop(heap)
            x, y, yaw, g, _, st = nodes[i]
            key = (round(x / self.xy_res), round(y / self.xy_res), round(yaw / self.yaw_res))
            if key in closed:
                continue
            closed.add(key)
            n_exp += 1
            d = np.hypot(goal_xy[:, 0] - x, goal_xy[:, 1] - y)
            j = int(d.argmin())
            if i and d[j] < self.goal_tol:
                dy = (goal_yaw[j] - yaw + math.pi) % (2.0 * math.pi) - math.pi
                if abs(dy) < self.goal_yaw_tol:
                    result, goal_node = 'found', i
                    break
            c, s = math.cos(yaw), math.sin(yaw)
            px = x + c * self.prims[:, :, 0] - s * self.prims[:, :, 1]
            py = y + s * self.prims[:, :, 0] + c * self.prims[:, :, 1]
            pyaw = yaw + self.prims[:, :, 2]
            poses = np.stack([px.ravel(), py.ravel(), pyaw.ravel()], axis=1)
            near = pts[(np.abs(pts[:, 0] - x) < reach) & (np.abs(pts[:, 1] - y) < reach)] if len(pts) else pts
            clr = footprint_clearance(poses, near, half_length, half_width).reshape(n_s, m).min(axis=1)
            for k in range(n_s):
                if clr[k] <= 0.0:
                    continue
                nx, ny, ny_aw = float(px[k, -1]), float(py[k, -1]), float(pyaw[k, -1])
                pen = max(0.0, self.clear_cap - float(clr[k])) / self.clear_cap
                ng = g + self.step * (1.0 + self.w_clear * pen
                                      + self.w_steer * abs(self.steers[k]) / self.max_steer) \
                    + self.w_dsteer * abs(self.steers[k] - st) / self.max_steer
                nodes.append((nx, ny, ny_aw, ng, i, float(self.steers[k])))
                heapq.heappush(heap, (ng + self.h_weight * heur(nx, ny), len(nodes) - 1))
        self.stats = dict(expanded=n_exp, result=result, ms=(used + time.perf_counter() - resumed) * 1e3)
        if goal_node is None:
            return None
        chain = []
        i = goal_node
        while i > 0:
            chain.append(i)
            i = nodes[i][4]
        out = [np.zeros((1, 3))]
        for i in reversed(chain):
            px, py, pyaw, _, parent, st = nodes[i]
            bx, by, byaw = nodes[parent][:3]
            a = arc(math.tan(st) * self.curvs[-1] / math.tan(self.max_steer), self.step, self.samples)
            c, s = math.cos(byaw), math.sin(byaw)
            out.append(np.c_[bx + c * a[:, 0] - s * a[:, 1], by + s * a[:, 0] + c * a[:, 1], byaw + a[:, 2]])
        return np.vstack(out)


def narrowest_gap(path, pts, half_length):
    """Width of the tightest gap along a path, and the pose index where it is.

    Width is the sum of the nearest distances to the left and right, beside the footprint. It does
    not depend on how well the path is centred. A side with nothing seen does not count as a gap.
    """
    if not len(pts) or len(path) < 2:
        return math.inf, 0
    c, s = np.cos(path[:, 2])[:, None], np.sin(path[:, 2])[:, None]
    dx = pts[None, :, 0] - path[:, None, 0]
    dy = pts[None, :, 1] - path[:, None, 1]
    u = c * dx + s * dy
    v = -s * dx + c * dy
    beside = np.abs(u) <= half_length
    left = np.where(beside & (v > 0), v, np.inf).min(axis=1)
    right = np.where(beside & (v <= 0), -v, np.inf).min(axis=1)
    width = left + right
    i = int(width.argmin())
    return float(width[i]), i


def resample(xy, spacing):
    d = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.r_[True, d > 1e-6]
    xy = xy[keep]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    n = max(2, int(round(s[-1] / spacing)) + 1)
    q = np.linspace(0.0, s[-1], n)
    return np.c_[np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])]


def catmull_rom(xy, spacing):
    """A smooth curve through the points, sampled about `spacing` apart: no kinks at the points."""
    p = np.vstack([2.0 * xy[0] - xy[1], xy, 2.0 * xy[-1] - xy[-2]])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    out = [xy[:1]]
    for i in range(len(xy) - 1):
        m = max(1, int(round(seg[i] / spacing)))
        t = (np.arange(1, m + 1) / m)[:, None]
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        out.append(0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
                          + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t ** 3))
    return np.vstack(out)


def center_and_smooth(path, pts, half_length, half_width, max_curv, clear=0.35, iters=10,
                      coarse=0.15, fine=0.05, pin=1):
    """Push a path toward the middle of whatever is beside it, then smooth it.

    Works on a coarse copy, so smoothing reaches about a metre and removes needless dips. The first
    `pin` coarse poses stay fixed. Curvature is measured between adjacent points, as for lattice
    candidates. Returns Nx3 at `fine` spacing, or None if the result collides or exceeds the limit;
    callers keep the input then.
    """
    xy = resample(path[:, :2], coarse)
    if len(xy) < pin + 4:
        return None
    # keep the car's present heading: the second point stays on the present tangent
    xy[1] = path[0, :2] + coarse * np.array([math.cos(path[0, 2]), math.sin(path[0, 2])])
    weight = np.clip((np.arange(len(xy)) - pin) / 2.0, 0.0, 1.0)
    for _ in range(iters):
        t = np.gradient(xy, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        if len(pts):
            dx = pts[None, :, 0] - xy[:, None, 0]
            dy = pts[None, :, 1] - xy[:, None, 1]
            u = t[:, 0, None] * dx + t[:, 1, None] * dy
            v = -t[:, 1, None] * dx + t[:, 0, None] * dy
            beside = np.abs(u) <= half_length + 0.15    # line up before the gap, not inside it
            left = np.where(beside & (v > 0), v, np.inf).min(axis=1)
            right = np.where(beside & (v <= 0), -v, np.inf).min(axis=1)
            shift = np.clip((np.minimum(left, clear) - np.minimum(right, clear)) / 2.0, -0.03, 0.03)
            xy[:, 0] += -t[:, 1] * shift * weight
            xy[:, 1] += t[:, 0] * shift * weight
            # in a gap, centring beats smoothing; elsewhere smoothing removes needless dips
            narrow = (left < clear) & (right < clear)
            sw = weight * np.where(narrow, 0.15, 1.0)
        else:
            sw = weight
        for _ in range(3):
            mid = 0.5 * (xy[:-2] + xy[2:])
            xy[1:-1] += 0.5 * (mid - xy[1:-1]) * sw[1:-1, None]
    for _ in range(5):        # too sharp after centring: smooth more before giving up
        fine_xy = catmull_rom(xy, fine)
        d = np.diff(fine_xy, axis=0)
        yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        yaw = np.r_[yaw, yaw[-1]]
        yaw += round((path[0, 2] - yaw[0]) / (2.0 * math.pi)) * 2.0 * math.pi
        seg = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        k = np.abs(np.diff(yaw[:-1])) / (0.5 * (seg[:-1] + seg[1:]))
        if not len(k) or k.max() <= max_curv:
            break
        for _ in range(4):
            mid = 0.5 * (xy[:-2] + xy[2:])
            xy[1:-1] += 0.5 * (mid - xy[1:-1]) * weight[1:-1, None]
    xy = fine_xy
    if len(k) and k.max() > max_curv:
        REFINE_FAIL['curv'] += 1
        REFINE_FAIL['curv_val'] = round(float(k.max()), 2)
        return None
    out = np.c_[xy, yaw]
    if len(pts):
        c = footprint_clearance(out, pts, half_length, half_width)
        if c.min() <= 0.0:
            REFINE_FAIL['coll'] += 1
            REFINE_FAIL['coll_at'] = int(c.argmin())
            return None
    return out
