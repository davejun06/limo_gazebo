#!/usr/bin/env python3
"""Obstacle detector + Frenet lattice local planner for the LIMO (Ackermann).

Global path  : /pp_path (the raceline, map frame)      -> reference line
Obstacles    : /scan (LaserScan) + TF (map->base_link)
Normal drive : Pure Pursuit command is passed through unchanged (FOLLOW)
Obstacle     : lattice of lateral-offset paths d(s) (quintic) around the
               global path -> drop colliding / infeasible ones -> pick best ->
               track it with pure pursuit (AVOID)
Output       : /final_ackermann_cmd (AckermannDriveStamped)
Visual       : /obstacles, /local_path, /local_candidates, /obstacle_detected

Run:
  python3 frenet_planner.py --ros-args -p use_sim_time:=true
Bridge must read the final topic:
  python3 ackermann_to_twist.py --ros-args \
      -r /pure_pursuit/ackermann_cmd:=/final_ackermann_cmd -p use_sim_time:=true
"""
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_astar_local import (REFINE_FAIL, HybridAStar, arc, center_and_smooth,  # noqa: E402
                                closing_speed_limit, footprint_clearance, narrowest_gap)

from dwa_local import DWA, TwoStageDWA  # noqa: E402

WHEELBASE = 0.24            # m
MAX_STEER = 0.42            # rad
MAX_CURV = math.tan(MAX_STEER) / WHEELBASE      # ~1.86 1/m


PRESETS = {
    'safe': {},                                   # the tested defaults below
    # 미검증, offline simulation only. Same speed caps and radii as 'safe'. See EXPERIMENT_CHANGELOG.md.
    # 미검증, offline simulation only. Same speed caps and radii as 'safe'.
    'dwa_lap': dict(use_dwa=True, hybrid_obs_ramp=False, narrow_look=2.0, narrow_stop_time=1.2),
    'hybrid_lap': dict(use_hybrid=True, hybrid_obs_ramp=False, narrow_closing=True,
                       narrow_look=2.0, narrow_stop_time=1.2, track_ld_min=0.3),
    'fast': dict(max_speed=1.0, avoid_max_speed=0.55, obs_speed=0.30, free_speed=0.80,
                 wall_speed=0.40, wall_near=0.22, a_lat=0.50, curve_min=0.40,
                 steer_slow=0.35, squeeze_speed=0.25, stop_front=0.55),
    'race': dict(max_speed=1.3, avoid_max_speed=0.70, obs_speed=0.40, free_speed=1.00,
                 wall_speed=0.50, wall_near=0.20, a_lat=0.70, curve_min=0.50,
                 steer_slow=0.40, squeeze_speed=0.30, stop_front=0.70),
}


def apply_profile(p, name):
    """Overwrite the speed-related defaults with a preset (explicit -p values still win)."""
    if name not in PRESETS:
        raise ValueError('profile must be one of %s' % sorted(PRESETS))
    for k, v in PRESETS[name].items():
        setattr(p, k, v)
    return p


def default_params():
    return SimpleNamespace(
        corridor=0.28,          # [m] scan point this close to the path ahead = blocked
        detect_dist=4.0,        # [m] look-ahead distance for obstacle detection
        plan_range=5.0,         # [m] scan points farther than this are ignored
        r_robot=0.20,           # [m] circumscribed radius of the centered LIMO chassis and wheels
        footprint_length=0.33,  # [m] Ackermann URDF body plus wheel longitudinal extent
        footprint_width=0.31,   # [m] Ackermann URDF collision-box width
        lat_max=1.20,           # [m] max lateral offset of the lattice
        n_lat=13,               # number of lateral offsets
        horizons=(0.8, 1.2, 2.0),   # [m] include short horizons for close obstacles
        fallback_lat_max=2.40,  # [m] wider search used only if the normal lattice has no safe path
        fallback_n_lat=25,
        fallback_horizons=(0.8, 3.2),
        pass_margin=0.80,       # [m] clear the projected obstacle length and rear footprint before rejoining
        rejoin_lengths=(1.6, 2.4, 3.2),  # [m] smooth transitions back to the global path
        tail=1.2,               # [m] path continues parallel after the lattice end
                                #     (only used to check for collisions)
        avoid_max_speed=0.48,    # [m/s]
        min_speed=0.12,         # [m/s]
        max_speed=0.8,          # [m/s] global cap
        slow_dist=1.0,          # [m] look this far along the path for walls
        wall_extent=0.0,        # [m] OPT-IN: scan segments longer than this are walls (0 = off, safest)
        wall_near=0.26,         # [m] wall closer than this to the path -> crawl
        wall_far=0.45,          # [m] wall farther than this -> no slowdown
        obs_speed=0.18,         # [m/s] speed right in front of an obstacle
        obs_slow_dist=3.0,      # [m] start slowing this far before an obstacle
        free_speed=0.55,        # [m/s] max speed while returning to the path (obstacle passed)
        body_r=0.12,            # [m] scan points this close to base_link = own body, ignored
        recover_off=0.40,       # [m] farther than this from the path -> lattice drives it back
        crawl_speed=0.16,       # [m/s] crawl speed for a verified short forward escape
        a_lat=0.32,             # [m/s^2] corner speed limit v = sqrt(a_lat / curvature)
        curve_min=0.28,         # [m/s] never slower than this because of a corner
        curve_look=2.0,         # [m] look this far along the path for corners
        steer_slow=0.30,        # [rad] |steer| at which speed has dropped to curve_min
        front_offset=0.0,        # [m] base_link is centered in the active Ackermann URDF
                                #     Use ~0.15 if base_link is at the rear axle / bumper sticks out
        # Candidate search stops at the first radius that fits, so a fine ladder picks the
        # largest margin the gap allows. 0.0 maps to r_min, the smallest ever accepted.
        radius_factors=(1.0, 0.95, 0.9, 0.85, 0.8, 0.0),
        r_min=0.17,              # [m] smallest radius: half width 0.155 + 0.015 m. No path below this means stop
        squeeze_speed=0.15,       # [m/s] max speed while squeezing through a tight gap
        escape_dist=0.50,          # [m] short collision-checked arc; replan before the gap tightens
        escape_margin=0.05,        # [m] minimum clearance required for a crawl escape
        escape_steer_samples=17,   # steering candidates across the physical Ackermann range
        stop_front=0.40,        # [m] emergency stop: scan point this close ahead ...
        stop_half=0.12,         # [m] ... and this close to the centre line
        crawl_half=0.16,        # [m] half width of the 'way ahead is clear' check for CRAWL
        tight_stop_front=0.30,  # [m] stop box length while squeezing through a tight gap
        tight_stop_half=0.105,  # [m] stop box half width while squeezing (robot half width ~0.11)
        tight_hold=1.0,         # [s] keep the squeeze stop box this long after the last squeeze
        wall_speed=0.30,        # [m/s] minimum speed cap near walls
        w_off=1.0,              # cost: |lateral offset|
        w_clear=2.0,            # cost: closeness to obstacles
        w_curv=0.6,             # cost: curvature
        w_prev=0.8,             # cost: change of offset between cycles
        clear_cap=0.35,          # [m] clearance above this costs nothing
        clear_hold=0.3,         # [s] keep AVOID this long after obstacle is gone
        return_off=0.30,        # [m] back on the path -> FOLLOW again
        cluster_gap=0.30,       # [m] scan points closer than this = same object
        track_ld_min=0.4,       # [m] pure pursuit on the local path
        track_ld_gain=0.5,
        # second-stage fallback: used ONLY when the normal and the wide lattice both find nothing
        # (e.g. the next obstacle sits where the first avoidance path would rejoin the line).
        # The collision check is identical; only the manoeuvre shape is more flexible.
        fb2_horizons=(0.6, 1.0, 1.6, 2.4),
        fb2_rejoin_lengths=(0.8, 1.2, 1.6),   # shorter return, fits between close obstacles
        fb2_pass_margin=0.40,
        fb2_lat_max=1.60,
        fb2_n_lat=21,
        # Points the LiDAR saw but can no longer confirm: nearer than range_min, or outside its
        # field of view. Without them a corner beside the car drops out of the collision check.
        use_unseen_memory=True,
        memory_hold=15.0,       # [s] forget after this long; visible areas never use memory
        memory_keep_r=1.0,      # [m] only points this close to base_link are kept or used
        memory_cell=0.02,       # [m] one remembered point per cell
        memory_rmin_margin=0.03,  # [m] treat just beyond range_min as unconfirmed too
        memory_average=False,     # 미검증, offline test only: mean point per memory_avg_cell
        memory_avg_cell=0.04,     # [m]
        # Short-range Hybrid A*: while blocked, search (x, y, heading) instead of parallel offsets.
        # Same inflated-rectangle collision test and same radius limits as the lattice.
        use_hybrid=False,         # 미검증: offline simulation only; see EXPERIMENT_CHANGELOG.md
        hybrid_radii=(0.20, 0.19, 0.18, 0.17),  # [m] tried largest first, one per tick
        hybrid_budget=0.035,      # [s] search time per tick; a search continues over ticks
        hybrid_max_ticks=40,      # ticks before giving up on one radius (2 s at 20 Hz)
        hybrid_pass_margin=0.8,   # [m] goal this far past the farthest blocking point
        hybrid_goal_min=2.0,      # [m] nearest goal along the route
        hybrid_goal_max=5.0,      # [m] farthest goal along the route
        hybrid_goal_span=1.5,     # [m] any route point over this length is an accepted goal
        hybrid_keep_len=1.2,      # [m] look for the next path when less than this remains and blocked
        hybrid_end_len=0.3,       # [m] a path shorter than this is finished
        hybrid_arc_check=0.25,    # [m] shortest executed-arc collision check
        hybrid_refine_dt=0.5,     # [s] re-center the kept path on the latest scan this often
        hybrid_retry_wait=3.0,    # [s] stopped with no path at any radius: search again after this
        hybrid_wait_stuck=5.0,    # [s] no path this long: hold still, report STUCK, no fallback or escape
        hybrid_wait_move=0.03,    # [m] the no-path timer restarts only after moving this far
        hybrid_block_far=4.0,     # [m] forget a closed side once the car is this far from it
        hybrid_side_forget=8.0,   # [m] an object this far away may be decided again (next lap)
        hybrid_alt_width=0.0,     # [m] gap narrower than this: search again around it. 0 = off
        hybrid_side_gain=0.02,    # [m] one side of an obstacle must be this much wider to be chosen
        hybrid_side_far=2.0,      # [m] decide this close: farther away, occlusion overstates a gap
        hybrid_side_near=0.8,     # [m] ... and not closer than this
        hybrid_side_size=1.6,     # [m] larger than this is a wall, not an obstacle to go round
        hybrid_alt_gain=0.02,     # [m] the other way must be this much wider to be taken
        hybrid_alt_ahead=0.5,     # [m] only for a gap at least this far ahead of the car
        hybrid_alt_block=0.35,    # [m] half length of the temporary block across the pinch
        hybrid_alt_every=0.7,     # [m] compare ways round at most once per this distance driven
        hybrid_alt_span=1.5,      # [m] compare widths up to this far past the first gap
        hybrid_block_hold=60.0,   # [s] upper limit; a slow pass must not outlive the choice
        # DWA local planner, 미검증, offline only. Same rectangle test, radii and stop box.
        use_dwa=False,
        dwa_w_clear=1.2,
        dwa_clear_cap=0.35,       # [m] clearance above this earns nothing
        dwa_rear_lr=0.0,          # [m] base_link ahead of the rear axle, as measured 0.12; 0 keeps the old form
        dwa_clear_skip=0,         # poses (0.1 m each) left out of the clearance score only; contact test unchanged
        dwa_through=0.10,         # [m] route this close to a point counts as running through it
        dwa_two_stage=True,       # two arcs in a row; False is the one-arc form that failed
        dwa_turn_far=2.2,         # [m] route blocked this close: start easing off the route
        dwa_turn_near=0.9,        # [m] route term at its lowest from here in
        dwa_path_min=0.15,        # lowest share of the route term
        dwa_past_block=0.8,       # [m] aim this far past the farthest blocking point
        dwa_retry_wait=0.5,       # [s] stopped with no free shape: look again after this
        dwa_target_dist=1.5,      # [m] route point the arcs aim for
        dwa_min_speed=0.06,       # [m/s]
        dwa_stop_time=1.2,        # [s]
        # Lap-time options, all off by default and 미검증. See EXPERIMENT_CHANGELOG.md.
        hybrid_obs_ramp=True,     # False: no fixed slowdown near obstacles; narrow_speed still limits speed
        steer_filter_tau=0.0,     # [s] > 0: corner slowdown from smoothed, clipped PP steering
        # Slow down by the real footprint clearance on the path ahead. Only ever lowers speed.
        narrow_look=1.0,          # [m] look this far along the path
        narrow_min_speed=0.06,    # [m/s] lowest speed this limit asks for
        narrow_keep=0.02,         # [m] clearance to keep after stopping
        narrow_stop_time=0.8,     # [s] stop request to standstill, measured once in the exp-007 run
        narrow_closing=False,     # True: limit by how fast clearance shrinks along the path. 미검증
        narrow_decel=0.15,        # [m/s^2] braking assumed ahead of a lower limit; one stop measured about 0.135
        narrow_drift_time=0.25,   # [s] allowed tracking drift at this speed; an estimate
    )


# ----------------------------------------------------------------- helpers
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def periodic_cubic(s, y, sn):
    """Closed C2 cubic spline through (s, y), evaluated at sn. y[0] must equal y[-1]."""
    n = len(s) - 1
    h = np.diff(s)
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(n):
        before, after = h[i - 1], h[i]
        A[i, (i - 1) % n] += before
        A[i, i] += 2.0 * (before + after)
        A[i, (i + 1) % n] += after
        rhs[i] = 6.0 * ((y[(i + 1) % n] - y[i]) / after - (y[i] - y[(i - 1) % n]) / before)
    m = np.linalg.solve(A, rhs)
    m = np.r_[m, m[0]]
    j = np.clip(np.searchsorted(s, sn, side='right') - 1, 0, n - 1)
    d = sn - s[j]
    e = s[j + 1] - sn
    hj = h[j]
    return (m[j] * e ** 3 + m[j + 1] * d ** 3) / (6.0 * hj) \
        + (y[j] / hj - m[j] * hj / 6.0) * e + (y[j + 1] / hj - m[j + 1] * hj / 6.0) * d


def resample_closed(xy, step):
    """Closed path resampled to ~uniform spacing on a spline through its points.

    Straight-line interpolation of waypoints leaves a heading jump at each one, which
    measures as false curvature at this sampling step (2.14 1/m on a route whose real
    maximum is about 0.9) and made candidates fail the curvature limit on open track.
    """
    xy = np.asarray(xy, float)[:, :2]
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:
        xy = xy[:-1]
    c = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(round(s[-1] / step)), 8)
    sn = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.c_[periodic_cubic(s, c[:, 0], sn), periodic_cubic(s, c[:, 1], sn)], s[-1]


def to_base(pts, pose):
    x, y, yaw = pose
    d = np.asarray(pts, float) - np.array([x, y])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.c_[c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]]


def to_map(pts, pose):
    x, y, yaw = pose
    p = np.asarray(pts, float)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.c_[c * p[..., 0] - s * p[..., 1] + x, s * p[..., 0] + c * p[..., 1] + y]


def scan_to_points(ranges, angle_min, angle_inc, range_min, range_max,
                   max_range, laser_tf=(0.0, 0.0, 0.0)):
    """LaserScan -> Nx2 points in base_link (order kept for clustering)."""
    r = np.asarray(ranges, float)
    ang = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r > range_min) & (r < min(range_max, max_range))
    x, y = r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
    tx, ty, tyaw = laser_tf
    c, s = math.cos(tyaw), math.sin(tyaw)
    return np.c_[c * x - s * y + tx, s * x + c * y + ty]


class UnseenMemory:
    """Recent scan points in a fixed frame, returned only where the sensor cannot check them."""

    STOP_WINDOW = 0.5   # [s]
    STOP_MOVE = 0.01    # [m] less than this over the window counts as stopped (0.02 m/s)

    def __init__(self, hold, keep_r, cell, rmin_margin, average=False):
        self.hold, self.keep_r, self.cell, self.rmin_margin = hold, keep_r, cell, rmin_margin
        self.average = average                # one mean point per cell, instead of the newest
        self.clear()

    def clear(self):
        self.pts = np.zeros((0, 2))
        self.stamp = np.zeros(0)
        self.count = np.zeros(0)
        self.anchor = None

    def hold_while_stopped(self, now, pose):
        """A stopped car learns nothing new about a blind corner, so its memory must not age.

        Otherwise a corner beside the car expires and a path through it appears.
        """
        if self.anchor is None:
            self.anchor = (now, pose[0], pose[1])
            return
        t0, x0, y0 = self.anchor
        if now - t0 < self.STOP_WINDOW:
            return
        if math.hypot(pose[0] - x0, pose[1] - y0) < self.STOP_MOVE:
            self.stamp = self.stamp + (now - t0)
        self.anchor = (now, pose[0], pose[1])

    def add(self, now, pts_base, pose):
        """Store points given in the base frame at pose (x, y, yaw) of the fixed frame."""
        fresh = pts_base[np.hypot(pts_base[:, 0], pts_base[:, 1]) < self.keep_r] \
            if len(pts_base) else pts_base
        keep = now - self.stamp < self.hold
        if self.average:
            pts = np.vstack([to_map(fresh, pose), self.pts[keep]]) if len(fresh) else self.pts[keep]
            if len(pts):
                w = np.r_[np.ones(len(fresh)), self.count[keep]]
                st = np.r_[np.full(len(fresh), now), self.stamp[keep]]
                _, inv = np.unique(np.floor(pts / self.cell).astype(int), axis=0, return_inverse=True)
                inv = inv.ravel()
                n = inv.max() + 1
                total = np.bincount(inv, w, n)
                self.pts = np.c_[np.bincount(inv, w * pts[:, 0], n), np.bincount(inv, w * pts[:, 1], n)] / total[:, None]
                self.count = total
                self.stamp = np.zeros(n)
                np.maximum.at(self.stamp, inv, st)
            else:
                self.pts, self.stamp, self.count = pts, np.zeros(0), np.zeros(0)
            return
        pts = np.vstack([to_map(fresh, pose), self.pts[keep]]) if len(fresh) else self.pts[keep]
        stamp = np.r_[np.full(len(fresh), now), self.stamp[keep]]
        if len(pts):                         # first occurrence is the newest
            _, first = np.unique(np.floor(pts / self.cell).astype(int), axis=0, return_index=True)
            pts, stamp = pts[first], stamp[first]
        self.pts, self.stamp = pts, stamp

    def unseen(self, now, pose, laser_xy, range_min, half_fov):
        """Base-frame points now nearer than range_min to the laser or outside its view."""
        self.hold_while_stopped(now, pose)
        ok = now - self.stamp < self.hold
        if not ok.any():
            return np.zeros((0, 2))
        base = to_base(self.pts[ok], pose)
        rx, ry = base[:, 0] - laser_xy[0], base[:, 1] - laser_xy[1]
        blind = (np.hypot(rx, ry) < range_min + self.rmin_margin) | (np.abs(np.arctan2(ry, rx)) > half_fov)
        return base[blind & (np.hypot(base[:, 0], base[:, 1]) < self.keep_r)]


def clusters_of(pts, gap):
    """Split ordered scan points into objects; return centroids + sizes."""
    if len(pts) == 0:
        return []
    cut = np.where(np.hypot(*np.diff(pts, axis=0).T) > gap)[0] + 1
    out = []
    for g in np.split(pts, cut):
        out.append((g.mean(axis=0), len(g),
                    float(np.hypot(*(g.max(axis=0) - g.min(axis=0))))))
    return out


def segment_extent(pts, gap):
    """For ordered scan points: length of the connected segment each point belongs to."""
    n = len(pts)
    if n == 0:
        return np.zeros(0)
    step = np.hypot(*np.diff(pts, axis=0).T) if n > 1 else np.zeros(0)
    cut = np.where(step > gap)[0] + 1
    ext = np.zeros(n)
    lo = 0
    for hi in list(cut) + [n]:
        ext[lo:hi] = float(step[lo:hi - 1].sum()) if hi - lo > 1 else 0.0
        lo = hi
    return ext


def quintic(d0, dp0, d1, L):
    """d(s)=a0+a1 s+...+a5 s^5, d(0)=d0,d'(0)=dp0,d''(0)=0,d(L)=d1,d'(L)=d''(L)=0."""
    A = np.array([[L ** 3, L ** 4, L ** 5],
                  [3 * L ** 2, 4 * L ** 3, 5 * L ** 4],
                  [6 * L, 12 * L ** 2, 20 * L ** 3]])
    b = np.array([d1 - d0 - dp0 * L, -dp0, 0.0])
    a3, a4, a5 = np.linalg.solve(A, b)
    return d0, dp0, 0.0, a3, a4, a5


# ------------------------------------------------------------------ planner
class FrenetPlanner:
    STEP = 0.05

    def __init__(self, path_xy, params=None):
        self.p = params or default_params()
        self.ref, self.length = resample_closed(path_xy, self.STEP)
        self.n = len(self.ref)
        self.ds = self.length / self.n
        nxt = np.roll(self.ref, -1, axis=0)
        prv = np.roll(self.ref, 1, axis=0)
        self.theta = np.arctan2(nxt[:, 1] - prv[:, 1], nxt[:, 0] - prv[:, 0])
        self.normal = np.c_[-np.sin(self.theta), np.cos(self.theta)]   # left
        self.idx = None
        self.mode = 'FOLLOW'
        self.last_blocked = -1e9
        self.prev_off = 0.0
        self.steer = 0.0
        self.tight_until = -1e9
        self.rej = dict(curv=0, back=0, coll=0, total=0)
        self.last_rad = self.p.r_robot
        self.min_rad = self.p.r_robot           # smallest radius used since last read (log only)
        # Primitives stay within the lattice's limit, 0.9 of the physical curvature.
        self.ha = HybridAStar(WHEELBASE, math.atan(0.9 * MAX_CURV * WHEELBASE))
        self.dwa = DWA(WHEELBASE, math.atan(0.9 * MAX_CURV * WHEELBASE))
        self.dwa2 = TwoStageDWA(WHEELBASE, math.atan(0.9 * MAX_CURV * WHEELBASE),
                                w_clear=self.p.dwa_w_clear, clear_cap=self.p.dwa_clear_cap,
                                clear_skip=int(self.p.dwa_clear_skip), rear_lr=float(self.p.dwa_rear_lr))
        self.dwa_none_since = None
        self.dwa_none_xy = (1e9, 1e9)
        self.ha_path = None                     # Nx3 poses in the odometry frame
        self.ha_rad = self.p.r_robot
        self.ha_try = 0
        self.ha_refined = -1e9
        self.ha_search = None
        self.ha_alt_xy = (1e9, 1e9)
        self.ha_retry_at = -1e9
        self.ha_retry_xy = (1e9, 1e9)
        self.steer_filter = 0.0
        self.steer_filter_t = -1e9
        self.ha_wait_since = None
        self.ha_wait_xy = (1e9, 1e9)
        self.ha_blocks = []                     # (expiry, Nx2 odom points): gaps decided against
        self.ha_sides = []                      # odom centres of objects already decided
        self.ha_events = []                     # recent search results (log only)
        self.ha_stats = dict(plans=0, found=0, no_path=0, budget=0, max_ms=0.0, stops=0,
                             refined=0, refine_kept=0, max_ticks=0, wait_ticks=0, alts=0, alt_used=0,
                             blocks_dropped=0, steer_swaps=0, steer_swap_max=0.0, sides=0, exp_by_rad={}, crawl_ticks=0)

    def reset(self):
        self.ha_retry_at = -1e9
        self.ha_wait_since = None
        self.ha_blocks = []
        self.ha_sides = []
        self.ha_path = None
        self.ha_search = None
        self.ha_try = 0
        self.mode = 'FOLLOW'
        self.prev_off = 0.0
        self.steer = 0.0
        self.tight_until = -1e9

    # -- reference line ----------------------------------------------------
    def nearest(self, pos):
        if self.idx is not None:
            ids = (self.idx + np.arange(-20, 140)) % self.n
            d = np.hypot(*(self.ref[ids] - pos).T)
            if d.min() < 1.5:
                self.idx = int(ids[np.argmin(d)])
                return self.idx
        self.idx = int(np.argmin(np.hypot(*(self.ref - pos).T)))
        return self.idx

    def ahead(self, idx, length):
        ids = (idx + np.arange(int(length / self.ds))) % self.n
        return self.ref[ids]

    # -- one control cycle -------------------------------------------------
    def step(self, pose, pts, pp_v, pp_steer, now, unseen=None, odom_pose=None):
        """Plan forward-only motion; remain stopped if no safe path exists.

        odom_pose holds a Hybrid A* path still while AMCL corrects; the map pose if not given.
        """
        self.odom_pose = pose if odom_pose is None else odom_pose
        self.now = now
        return self._step(pose, pts, pp_v, pp_steer, now, unseen)

    # -- Hybrid A* ---------------------------------------------------------
    def rect(self, rad):
        margin = max(0.0, rad - self.p.footprint_width / 2.0)
        return self.p.footprint_length / 2.0 + margin, self.p.footprint_width / 2.0 + margin

    def ha_in_base(self):
        """Remaining kept path in base_link, from the pose nearest the car."""
        x, y, yaw = self.odom_pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.ha_path[:, 0] - x, self.ha_path[:, 1] - y
        b = np.c_[c * dx + s * dy, -s * dx + c * dy, self.ha_path[:, 2] - yaw]
        return b[int(np.hypot(b[:, 0], b[:, 1]).argmin()):]

    def ha_goals(self, idx, pose, block_far):
        p = self.p
        s0 = float(np.clip((block_far or 0.0) + p.hybrid_pass_margin, p.hybrid_goal_min, p.hybrid_goal_max))
        i0, i1 = int(s0 / self.ds), int((s0 + p.hybrid_goal_span) / self.ds) + 1
        ids = (idx + np.arange(i0, i1)) % self.n
        return to_base(self.ref[ids], pose), self.theta[ids] - pose[2]

    def hybrid(self, idx, pose, pts, block_far, blocked):
        """Path in base_link (Nx3) to follow, 'wait' while searching with nothing to follow, or None."""
        p = self.p
        kept, left = None, 0.0
        if self.ha_path is not None:
            b = self.ha_in_base()
            hl, hw = self.rect(self.ha_rad)
            left = len(b) * self.ha.step / self.ha.samples
            if left >= p.hybrid_end_len and footprint_clearance(b, pts, hl, hw).min() > 0.0:
                kept = b
                if self.now - self.ha_refined >= p.hybrid_refine_dt:
                    self.ha_refined = self.now
                    better = center_and_smooth(b, pts, hl, hw, 0.9 * MAX_CURV)
                    self.ha_stats['refined' if better is not None else 'refine_kept'] += 1
                    if better is not None:
                        self.ha_store(better, self.odom_pose)
                        kept = better
            else:
                self.ha_path = None
                self.ha_search = None             # its snapshot may have used the same bad data
        if (kept is not None and self.ha_search is None
                and (not blocked or left >= p.hybrid_keep_len)):
            return kept
        if kept is None and not blocked:
            return None
        st = self.ha_stats
        if kept is None and self.now < self.ha_retry_at and math.hypot(
                self.odom_pose[0] - self.ha_retry_xy[0], self.odom_pose[1] - self.ha_retry_xy[1]) < 0.01:
            return 'wait'                         # stopped, nothing changed: do not search every tick
        if self.ha_search is None:
            rad = p.hybrid_radii[self.ha_try % len(p.hybrid_radii)]
            hl, hw = self.rect(rad)
            goals, gyaw = self.ha_goals(idx, pose, block_far)
            snap = pts.copy()
            ox, oy = self.odom_pose[:2]
            self.ha_blocks = [b for b in self.ha_blocks if b[0] > self.now and float(
                np.hypot(b[1][:, 0] - ox, b[1][:, 1] - oy).min()) < p.hybrid_block_far]
            held = [self.from_odom(b[1]) for b in self.ha_blocks]
            search_pts = np.vstack([snap] + held) if held else snap
            self.ha_search = dict(
                gen=self.ha.search(search_pts, goals, gyaw, hl, hw, self.steer, p.hybrid_budget),
                rad=rad, real=snap, pose=self.odom_pose, ticks=0, hl=hl, hw=hw, goals=goals, gyaw=gyaw,
                search_pts=search_pts)
            st['plans'] += 1
        s = self.ha_search
        s['ticks'] += 1
        st['max_ticks'] = max(st['max_ticks'], s['ticks'])
        try:
            next(s['gen'])
        except StopIteration as done:
            path = done.value
            self.ha_search = None
            st['max_ms'] = max(st['max_ms'], self.ha.stats['ms'])
            if path is None:
                if s.get('first') is not None:       # no other way round: use the one we have
                    return self.ha_commit(s['first'], s['rad'], s['pose'])
                if s.get('first_width') is not None:  # recheck found no other way: keep going
                    return kept if kept is not None else None
                st['no_path'] += 1
                self.ha_try += 1
                st['exp_by_rad'][str(s['rad'])] = self.ha.stats['expanded']
                if self.ha_try % len(p.hybrid_radii) == 0:   # every radius failed
                    self.ha_log_failure()
                    self.ha_retry_at = self.now + p.hybrid_retry_wait
                    self.ha_retry_xy = self.odom_pose[:2]
                    if self.ha_blocks:               # a closed side must never cause a dead end
                        st['blocks_dropped'] += 1
                        self.ha_blocks = []
                        return kept if kept is not None else 'wait'
                    return kept
                return kept if kept is not None else 'wait'
            st['found'] += 1
            better = center_and_smooth(path, s['real'], s['hl'], s['hw'], 0.9 * MAX_CURV)
            path = path if better is None else better
            # Compare the same stretch: a pinch far past the one in question says nothing about it.
            n_cmp = len(path)
            if s.get('first_width') is not None:
                n_cmp = int((s['first_at'] + p.hybrid_alt_span) * self.ha.samples / self.ha.step)
            width, at = narrowest_gap(path[:n_cmp], s['real'], s['hl'])
            self.ha_events.append((round(self.now, 1), round(self.odom_pose[0], 2), round(self.odom_pose[1], 2),
                                   'alt' if s.get('first_width') is not None else 'first', round(width, 3),
                                   round(at * self.ha.step / self.ha.samples, 2), round(s['rad'], 2),
                                   tuple(np.round(path[at, :2], 2).tolist())))
            del self.ha_events[:-60]
            if s.get('first_width') is not None:     # compare the two ways round
                if width > s['first_width'] + p.hybrid_alt_gain:
                    st['alt_used'] += 1
                    # Keep the choice: later searches must not go back through the blocked gap.
                    self.ha_blocks.append((self.now + p.hybrid_block_hold, self.to_odom(s['block'], s['pose'])))
                    return self.ha_commit(path, s['rad'], s['pose'])
                if s.get('first') is None:           # recheck: the kept path stays
                    return kept if kept is not None else None
                return self.ha_commit(s['first'], s['rad'], s['pose'])
            if self.ha_start_alt(path, width, at, s, kept):
                return kept if kept is not None else 'wait'
            return self.ha_commit(path, s['rad'], s['pose'])
        if s['ticks'] >= p.hybrid_max_ticks:          # too slow to trust
            s['gen'].close()
            self.ha_search = None
            st['budget'] += 1
            if s.get('first') is not None:
                return self.ha_commit(s['first'], s['rad'], s['pose'])
            if s.get('first_width') is None:
                self.ha_try += 1
        return kept if kept is not None else 'wait'

    def ha_start_alt(self, path, width, at, s, kept, keep_first=True):
        """Block the gap at pose `at` and search again from the same snapshot. True if started."""
        p = self.p
        ahead = at * self.ha.step / self.ha.samples
        moved = math.hypot(self.odom_pose[0] - self.ha_alt_xy[0], self.odom_pose[1] - self.ha_alt_xy[1])
        if width >= p.hybrid_alt_width or ahead < p.hybrid_alt_ahead or moved < p.hybrid_alt_every:
            return False
        self.ha_alt_xy = self.odom_pose[:2]
        c, sn = math.cos(path[at, 2]), math.sin(path[at, 2])
        off = np.arange(-p.hybrid_alt_block, p.hybrid_alt_block + 1e-6, 0.04)
        block = np.c_[path[at, 0] - sn * off, path[at, 1] + c * off]
        self.ha_search = dict(
            gen=self.ha.search(np.vstack([s['search_pts'], block]), s['goals'], s['gyaw'], s['hl'], s['hw'],
                               self.steer, p.hybrid_budget),
            rad=s['rad'], real=s['real'], pose=s['pose'], ticks=0, hl=s['hl'], hw=s['hw'],
            goals=s['goals'], gyaw=s['gyaw'], first=path if keep_first else None, first_width=width,
            first_at=ahead, block=block, search_pts=s['search_pts'])
        self.ha_stats['alts'] += 1
        return True

    @staticmethod
    def to_odom(xy, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.c_[pose[0] + c * xy[:, 0] - s * xy[:, 1], pose[1] + s * xy[:, 0] + c * xy[:, 1]]

    def from_odom(self, xy):
        x, y, yaw = self.odom_pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = xy[:, 0] - x, xy[:, 1] - y
        return np.c_[c * dx + s * dy, -s * dx + c * dy]

    def choose_side(self, pts, near, ahead_b):
        """Measure the gap on both sides of a small blocking object and close the narrower one.

        Decided once per object, from the scan. A closed side is only a guide for the search: it
        is dropped if it causes a dead end, and is never part of a collision check.
        """
        p = self.p
        if not near.any() or len(ahead_b) < 2:
            return
        seeds = pts[near]
        centre = seeds.mean(axis=0)
        dist = float(np.hypot(*centre))
        if not p.hybrid_side_near <= dist <= p.hybrid_side_far:
            return
        odom_c = self.to_odom(centre[None, :], self.odom_pose)[0]
        self.ha_sides = [c for c in self.ha_sides if math.hypot(c[0] - self.odom_pose[0], c[1] - self.odom_pose[1])
                         < p.hybrid_side_forget]
        if any(math.hypot(*(odom_c - c)) < 0.8 for c in self.ha_sides):
            return
        # the object: everything connected to the blocking points
        member = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) < 0.1
        member |= np.min(np.hypot(pts[:, None, 0] - seeds[None, :, 0], pts[:, None, 1] - seeds[None, :, 1]), axis=1) < 0.05
        for _ in range(40):
            grown = member.copy()
            core = pts[member]
            if len(core):
                grown |= np.min(np.hypot(pts[:, None, 0] - core[None, :, 0], pts[:, None, 1] - core[None, :, 1]), axis=1) < 0.15
            if (grown == member).all():
                break
            member = grown
        obj = pts[member]
        if not len(obj) or max(np.ptp(obj[:, 0]), np.ptp(obj[:, 1])) > p.hybrid_side_size:
            return                                  # a wall: there is no way round it to choose
        i = int(np.hypot(ahead_b[:, 0] - centre[0], ahead_b[:, 1] - centre[1]).argmin())
        i = max(1, min(i, len(ahead_b) - 1))
        t = ahead_b[i] - ahead_b[i - 1]
        t = t / max(float(np.hypot(*t)), 1e-6)
        n = np.array([-t[1], t[0]])
        s_obj, n_obj = obj @ t, obj @ n
        rest = pts[~member]
        if not len(rest):
            return
        # Shortest real distance from the object to what is on each side of it.
        d = np.hypot(rest[:, None, 0] - obj[None, :, 0], rest[:, None, 1] - obj[None, :, 1])
        dmin = d.min(axis=1)
        side = rest @ n - 0.5 * (n_obj.min() + n_obj.max())
        up = dmin[(side > 0) & (dmin < p.hybrid_side_size)]
        down = dmin[(side <= 0) & (dmin < p.hybrid_side_size)]
        if not len(up) or not len(down):
            return                                  # a side not seen yet is unknown, not wide: wait
        gap_up, gap_down = float(up.min()), float(down.min())
        self.ha_sides.append(odom_c)
        self.ha_events.append((round(self.now, 1), 'side', tuple(np.round(self.odom_pose[:2], 2).tolist()),
                               'obj', tuple(np.round(self.to_odom(obj, self.odom_pose).min(axis=0), 2).tolist()),
                               tuple(np.round(self.to_odom(obj, self.odom_pose).max(axis=0), 2).tolist()),
                               'gaps', round(gap_up, 3), round(gap_down, 3), 'n', len(obj)))
        if abs(gap_up - gap_down) < p.hybrid_side_gain:
            return
        # Close the narrower side along the line between the object and what it faces there.
        pick = (side > 0) if gap_up < gap_down else (side <= 0)
        k = np.where(pick)[0][dmin[pick].argmin()]
        o = obj[d[k].argmin()]
        q = rest[k]
        m = max(2, int(np.hypot(*(q - o)) / 0.04) + 1)
        block = o[None, :] + np.linspace(0.0, 1.0, m)[:, None] * (q - o)[None, :]
        self.ha_blocks.append((self.now + p.hybrid_block_hold, self.to_odom(block, self.odom_pose)))
        self.ha_path = None                         # the kept path may use the closed side
        self.ha_search = None
        self.ha_stats['sides'] += 1

    def ha_log_failure(self):
        """Log only: which points stop the start pose at the smallest radius."""
        hl, hw = self.rect(self.p.r_min)
        o = np.zeros((1, 3))
        live, mem = self.pts_live, self.pts_mem
        lc = float(footprint_clearance(o, live, hl, hw)[0]) if len(live) else -1.0
        mc = float(footprint_clearance(o, mem, hl, hw)[0]) if len(mem) else -1.0
        self.ha_events.append((round(self.now, 1), 'fail', tuple(np.round(self.odom_pose[:2], 3).tolist()),
                               'start_clear_live', round(lc, 3), 'start_clear_mem', round(mc, 3),
                               'expanded', dict(self.ha_stats['exp_by_rad'])))
        del self.ha_events[:-60]

    def ha_commit(self, path, rad, pose):
        bare = footprint_clearance(path, self.pts_live, *self.rect(0.0)) if len(self.pts_live) else np.zeros(1)
        self.ha_events.append((round(self.now, 1), 'plan', round(rad, 2), tuple(np.round(pose[:2], 3).tolist()),
                               'end', tuple(np.round(path[-1, :2], 2).tolist()), 'min_clear', round(float(bare.min()), 3)))
        del self.ha_events[:-60]
        self.ha_try = 0
        self.ha_rad = rad
        self.ha_store(path, pose)
        self.ha_refined = self.now
        return self.ha_in_base()

    def ha_store(self, path, base_pose):
        x, y, yaw = base_pose
        c, s = math.cos(yaw), math.sin(yaw)
        self.ha_path = np.c_[x + c * path[:, 0] - s * path[:, 1],
                             y + s * path[:, 0] + c * path[:, 1], yaw + path[:, 2]]

    def narrow_speed(self, b, pts):
        """Speed limit from the bare footprint's clearance over the next narrow_look metres."""
        p = self.p
        spacing = self.ha.step / self.ha.samples
        look = b[:max(2, int(p.narrow_look / spacing))]
        look = np.vstack([np.zeros((1, 3)), look])   # the car's own pose: it may be off the path
        if not len(pts):
            return p.avoid_max_speed
        if not p.narrow_closing:
            clr = footprint_clearance(look, pts, *self.rect(0.0))
            return float(np.clip((float(clr.min()) - p.narrow_keep) / p.narrow_stop_time,
                                 p.narrow_min_speed, p.avoid_max_speed))
        # Stopping distance matters where clearance shrinks: toward a corner, not beside a parallel
        # wall. Beside a wall only tracking drift can use up the clearance. Each pose ahead gets
        # its own limit, and the car brakes for it in time.
        hl, hw = self.rect(0.0)
        d = spacing * np.arange(len(look))
        limit = closing_speed_limit(look, pts, hl, hw, p.narrow_keep, p.narrow_stop_time, p.narrow_drift_time)
        limit = np.clip(limit, p.narrow_min_speed, p.avoid_max_speed)
        return float(np.sqrt(limit ** 2 + 2.0 * p.narrow_decel * d).min())

    def clear_steer(self, steer, v, pts):
        """The steering nearest `steer` whose arc is clear until the car could stop, or None.

        The arc is checked at the smallest radius; tracking may not match the planned path exactly.
        """
        p = self.p
        if not len(pts):
            return steer
        length = max(p.hybrid_arc_check, v * 0.8)
        n = max(2, int(length / 0.05))
        hl, hw = self.rect(p.r_min)
        for s in sorted(np.r_[steer, np.linspace(-MAX_STEER, MAX_STEER, 17)], key=lambda z: abs(z - steer)):
            if footprint_clearance(arc(math.tan(s) / WHEELBASE, length, n), pts, hl, hw).min() > 0.0:
                return float(s)
        return None

    def forward_escape(self, pts):
        """Find a short, collision-free Ackermann arc when the lattice has no route."""
        p = self.p
        if not len(pts):
            return None
        distance = np.arange(self.STEP, p.escape_dist + self.STEP / 2.0, self.STEP)
        best = None
        for steer in np.linspace(-MAX_STEER, MAX_STEER, p.escape_steer_samples):
            curvature = math.tan(float(steer)) / WHEELBASE
            if abs(curvature) < 1e-8:
                path = np.c_[distance, np.zeros_like(distance)]
                heading = np.zeros_like(distance)
            else:
                heading = curvature * distance
                path = np.c_[np.sin(heading) / curvature,
                             (1.0 - np.cos(heading)) / curvature]
            dx = pts[None, :, 0] - path[:, None, 0]
            dy = pts[None, :, 1] - path[:, None, 1]
            local_x = np.cos(heading)[:, None] * dx + np.sin(heading)[:, None] * dy
            local_y = -np.sin(heading)[:, None] * dx + np.cos(heading)[:, None] * dy
            outside_x = np.maximum(np.abs(local_x) - p.footprint_length / 2.0, 0.0)
            outside_y = np.maximum(np.abs(local_y) - p.footprint_width / 2.0, 0.0)
            clearance = float(np.hypot(outside_x, outside_y).min())
            if clearance < p.escape_margin:
                continue
            score = clearance - 0.05 * abs(float(steer)) / MAX_STEER
            if best is None or score > best[0]:
                best = (score, path, float(steer), clearance)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _step(self, pose, pts, pp_v, pp_steer, now, unseen=None):
        """pose=(x,y,yaw) in map, pts=Nx2 scan points in base_link.
        Returns (speed, steer, info)."""
        p = self.p
        x, y, yaw = pose
        idx = self.nearest(np.array([x, y]))
        off0 = float((np.array([x, y]) - self.ref[idx]) @ self.normal[idx])  # +left
        dpsi = wrap(yaw - self.theta[idx])

        if len(pts):
            rr = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[(rr < p.plan_range) & (rr > p.body_r)]
            # segment length of every scan point, measured on the full-resolution scan
            ext = (segment_extent(pts, max(p.cluster_gap, 0.35))
                   if p.wall_extent > 0.0 else np.zeros(len(pts)))
            if len(pts) > 220:                  # keep the step fast (20 Hz)
                k = int(math.ceil(len(pts) / 220.0))
                pts, ext = pts[::k], ext[::k]
        else:
            ext = np.zeros(0)
        self.pts_live, self.pts_mem = pts, np.zeros((0, 2))   # log only
        if unseen is not None and len(unseen):  # after thinning: every remembered point counts
            self.pts_mem = unseen[np.hypot(unseen[:, 0], unseen[:, 1]) > p.body_r]
            unseen = unseen[np.hypot(unseen[:, 0], unseen[:, 1]) > p.body_r]
            pts = np.vstack([pts, unseen]) if len(pts) else unseen
            ext = np.r_[ext, np.zeros(len(unseen))]
        stop_box = np.zeros(len(pts), dtype=bool)
        if len(pts):                           # emergency stop box in front of the car
            sf, sh = p.stop_front, p.stop_half
            if now < self.tight_until:
                sf, sh = min(sf, p.tight_stop_front), min(sh, p.tight_stop_half)
            stop_box = ((pts[:, 0] > 0.05) & (pts[:, 0] < sf)
                        & (np.abs(pts[:, 1]) < sh))
        ahead_b = to_base(self.ahead(idx, p.detect_dist + 1.5), pose)
        det_b = ahead_b[:int(p.detect_dist / self.ds)]

        near = np.zeros(len(pts), bool)
        near_s = np.full(len(pts), p.detect_dist, dtype=float)
        route_gap = math.inf
        if len(pts) and len(det_b):
            d2 = ((pts[:, None, :] - det_b[None, :, :]) ** 2).sum(-1)
            near = d2.min(axis=1) < p.corridor ** 2
            near_s = d2.argmin(axis=1) * self.ds
            route_gap = float(np.sqrt(d2.min()))     # how close the route itself comes to any point
            if p.wall_extent > 0.0:                # long connected segments = walls
                near &= ext <= p.wall_extent
        near |= stop_box
        block_s = float(near_s[near].min()) if near.any() else None
        block_far = float(near_s[near].max()) if near.any() else None
        # wall proximity speed limit: clearance between scan and the path ahead
        cap = p.max_speed
        if len(pts):
            sl = ahead_b[:max(2, int(p.slow_dist / self.ds)):2]
            clr = float(np.sqrt(((pts[:, None, :] - sl[None, :, :]) ** 2)
                                .sum(-1).min()))
            clr = min(clr, float(np.hypot(pts[:, 0], pts[:, 1]).min()) - p.r_robot + p.wall_near)
            f = (clr - p.wall_near) / max(1e-6, p.wall_far - p.wall_near)
            cap = p.wall_speed + float(np.clip(f, 0.0, 1.0)) * (p.max_speed - p.wall_speed)
        # corner speed limit from the global path curvature ahead + current steering
        w = 10                                  # 0.5 m windows
        ids = (idx + np.arange(0, int(p.curve_look / self.ds))) % self.n
        dth = np.abs(np.array([wrap(self.theta[(i + w) % self.n] - self.theta[i]) for i in ids[::5]]))
        kmax = float(dth.max()) / (w * self.ds) if len(dth) else 0.0
        v_curve = math.sqrt(p.a_lat / max(kmax, 1e-3))
        steer_for_cap = pp_steer
        if p.steer_filter_tau > 0.0:
            dt = max(0.0, min(0.5, now - self.steer_filter_t))
            a = dt / (p.steer_filter_tau + dt) if dt > 0.0 else 0.0
            self.steer_filter += a * (float(np.clip(pp_steer, -MAX_STEER, MAX_STEER)) - self.steer_filter)
            self.steer_filter_t = now
            steer_for_cap = self.steer_filter
        v_steer = p.max_speed - (p.max_speed - p.curve_min) * min(1.0, abs(steer_for_cap) / p.steer_slow)
        cap = min(cap, max(p.curve_min, min(v_curve, v_steer)))
        blocked = bool(near.any())
        if blocked:
            self.last_blocked = now
        clusters = clusters_of(pts[near], p.cluster_gap) if blocked else []

        if self.mode == 'FOLLOW':
            if blocked or abs(off0) > p.recover_off:
                self.mode = 'AVOID'
        elif (not blocked and now - self.last_blocked > p.clear_hold
              and abs(off0) < p.return_off and self.ha_path is None):
            self.mode = 'FOLLOW'

        info = dict(mode=self.mode, blocked=blocked, clusters=clusters,
                    cands=[], best=None, off=off0, cap=cap)
        if self.mode == 'FOLLOW':
            self.steer = float(np.clip(pp_steer, -MAX_STEER, MAX_STEER))
            self.prev_off = off0
            return float(min(pp_v, cap)), self.steer, info

        v = float(np.clip(pp_v, p.min_speed, p.free_speed if not blocked else p.avoid_max_speed))
        v = min(v, max(cap, p.wall_speed))
        if blocked and (p.hybrid_obs_ramp or not (p.use_hybrid or p.use_dwa)):   # ease off while approaching an obstacle
            dmin = float(np.hypot(pts[near][:, 0], pts[near][:, 1]).min())
            f = float(np.clip((dmin - 0.4) / max(1e-6, p.obs_slow_dist - 0.4), 0.0, 1.0))
            v = min(v, p.obs_speed + f * (p.avoid_max_speed - p.obs_speed))
        if p.use_dwa:
            i_t = min(len(ahead_b) - 1, int(p.dwa_target_dist / self.ds))
            res = None
            if p.dwa_two_stage:
                dist_t = p.dwa_target_dist
                path_gain = 1.0
                # The route runs through something: turn off it in time. Merely passing close to
                # something, it stays the best line and gets centred by the clearance term.
                if blocked and block_s is not None and route_gap < p.dwa_through:
                    path_gain = float(np.clip((block_s - p.dwa_turn_near) / (p.dwa_turn_far - p.dwa_turn_near),
                                              p.dwa_path_min, 1.0))
                    if block_far is not None and path_gain < 1.0:
                        dist_t = max(dist_t, block_far + p.dwa_past_block)
                if self.dwa_none_since is not None and now - self.dwa_none_since < p.dwa_retry_wait \
                        and math.hypot(self.odom_pose[0] - self.dwa_none_xy[0], self.odom_pose[1] - self.dwa_none_xy[1]) < 0.01:
                    info['planner'] = 'DWA'
                    info['stuck'] = True
                    info['rej'] = dict(self.rej)
                    return 0.0, self.steer, info
                i_t = min(len(ahead_b) - 1, int(dist_t / self.ds))
                j_t = (idx + i_t) % self.n
                for rad in dict.fromkeys(max(p.r_min, f * p.r_robot) for f in p.radius_factors):
                    res = self.dwa2.plan(pts, ahead_b[i_t], self.theta[j_t] - yaw, self.steer, *self.rect(rad), route=ahead_b[::2],
                                         path_gain=path_gain)
                    if res is not None:
                        self.last_rad = rad
                        self.min_rad = min(self.min_rad, rad)
                        break
                info['planner'] = 'DWA'
                info['rej'] = dict(self.rej)
                if res is None:
                    info['stuck'] = True         # no free shape: stop, no creep
                    self.dwa_none_since, self.dwa_none_xy = now, self.odom_pose[:2]
                    return 0.0, self.steer, info
                self.dwa_none_since = None
                saved_stop = p.narrow_closing
                p.narrow_closing = True
                vd = min(v, self.narrow_speed(res[1], pts))
                p.narrow_closing = saved_stop
                # The same corner rules as elsewhere, on the steering actually commanded.
                k_cmd = abs(math.tan(res[0])) / WHEELBASE
                vd = min(vd, max(p.curve_min, math.sqrt(p.a_lat / max(k_cmd, 1e-3))))
                vd = min(vd, p.max_speed - (p.max_speed - p.curve_min) * min(1.0, abs(res[0]) / p.steer_slow))
                if self.last_rad < 0.9 * p.r_robot:
                    vd = min(vd, p.squeeze_speed)
                info['cands'], info['best'] = [res[1][:, :2]], res[1][:, :2]
                self.prev_off = off0
                self.steer = res[0]
                return vd, self.steer, info
            for rad in dict.fromkeys(max(p.r_min, f * p.r_robot) for f in p.radius_factors):
                res = self.dwa.plan(pts, ahead_b[i_t], p.dwa_min_speed, v, self.steer, *self.rect(rad),
                                    stop_time=p.dwa_stop_time)
                if res is not None:
                    self.last_rad = rad
                    self.min_rad = min(self.min_rad, rad)
                    break
            info['planner'] = 'DWA'
            info['rej'] = dict(self.rej)
            if res is None:
                info['stuck'] = True             # no free arc: stop, no creep
                return 0.0, self.steer, info
            info['cands'], info['best'] = [res[2][:, :2]], res[2][:, :2]
            self.prev_off = off0
            self.steer = res[1]
            return res[0], self.steer, info
        if p.use_hybrid and (blocked or self.ha_path is not None):
            self.choose_side(pts, near, ahead_b)
            b = self.hybrid(idx, pose, pts, block_far, blocked)
            ox, oy = self.odom_pose[:2]
            if b is None or isinstance(b, str):  # no path to follow: time it until the car really moves
                if (self.ha_wait_since is None
                        or math.hypot(ox - self.ha_wait_xy[0], oy - self.ha_wait_xy[1]) > p.hybrid_wait_move):
                    self.ha_wait_since, self.ha_wait_xy = now, (ox, oy)
                if now - self.ha_wait_since > p.hybrid_wait_stuck:
                    # Long enough: hold still and say so on every tick. No fallback or escape pulses,
                    # which would hide the stop from the monitor.
                    self.ha_stats['wait_ticks'] += 1
                    info['planner'] = 'HA'
                    info['searching'] = True
                    info['stuck'] = True
                    info['rej'] = dict(self.rej)
                    return 0.0, self.steer, info
            else:
                self.ha_wait_since = None
            if isinstance(b, str):               # searching with nothing checked to follow: hold
                self.ha_stats['wait_ticks'] += 1
                info['planner'] = 'HA'
                info['searching'] = True
                info['rej'] = dict(self.rej)
                return 0.0, self.steer, info
            if b is not None:
                self.last_rad = self.ha_rad
                self.min_rad = min(self.min_rad, self.ha_rad)
                if self.ha_rad < 0.9 * p.r_robot:
                    v = min(v, p.squeeze_speed)
                v = min(v, self.narrow_speed(b, pts))
                self.prev_off = off0
                self.steer = self.track(b[:, :2], v)
                info['planner'] = 'HA'
                info['cands'], info['best'] = [b[:, :2]], b[:, :2]
                info['rej'] = dict(self.rej)
                checked = self.clear_steer(self.steer, v, pts)
                if checked is None:
                    self.ha_path = None
                    self.ha_stats['stops'] += 1
                    info['stuck'] = True
                    return 0.0, self.steer, info
                if checked != self.steer:
                    self.ha_stats['steer_swaps'] += 1
                    self.ha_stats['steer_swap_max'] = max(self.ha_stats['steer_swap_max'],
                                                          round(abs(checked - self.steer), 3))
                self.steer = checked
                return v, self.steer, info
        best, cands, off1 = self.lattice(
            idx, off0, dpsi, pose, pts, free=not blocked, block_s=block_s)
        if best is None and blocked:
            best, cands, off1 = self.lattice(
                idx, off0, dpsi, pose, pts, free=False,
                lat_max=p.fallback_lat_max, n_lat=p.fallback_n_lat,
                horizons=p.fallback_horizons, block_s=block_s)
        info['fallback'] = 0
        if best is None and blocked:
            # stage 2a: shorter return so the path can come back between close obstacles
            best, cands, off1 = self.lattice(
                idx, off0, dpsi, pose, pts, free=False,
                lat_max=p.fb2_lat_max, n_lat=p.fb2_n_lat, horizons=p.fb2_horizons,
                block_s=block_s, rejoin_lengths=p.fb2_rejoin_lengths, pass_margin=p.fb2_pass_margin)
            info['fallback'] = 1
            if best is None and block_far is not None and block_far > block_s + 1e-6:
                # stage 2b: stay offset until the farthest blocking obstacle is passed
                best, cands, off1 = self.lattice(
                    idx, off0, dpsi, pose, pts, free=False,
                    lat_max=p.fb2_lat_max, n_lat=p.fb2_n_lat, horizons=p.fb2_horizons,
                    block_s=block_far, rejoin_lengths=p.fb2_rejoin_lengths,
                    pass_margin=p.fb2_pass_margin)
                info['fallback'] = 2
            if best is None:
                info['fallback'] = 0
        if best is None and p.use_hybrid:
            self.ha_stats['lattice_none'] = self.ha_stats.get('lattice_none', 0) + 1
        info['cands'], info['best'] = cands, best
        info['rej'] = dict(self.rej)
        if best is None:                       # no lattice path
            escape = self.forward_escape(pts)
            if escape is not None:
                path, self.steer, clearance = escape
                info['mode'] = 'CRAWL'
                info['best'] = path
                info['escape_clearance'] = clearance
                self.tight_until = now + p.tight_hold
                if p.use_hybrid:
                    self.ha_stats['crawl_ticks'] += 1
                    self.ha_events.append((round(now, 2), 'crawl', round(float(clearance), 3)))
                    del self.ha_events[:-60]
                return float(p.crawl_speed), self.steer, info
            if stop_box.any():
                self.steer = 0.0
                info['mode'] = 'ESTOP'
                info['stuck'] = True
                return 0.0, self.steer, info
            # No verified path and no checked escape arc: stay stopped. Driving on an unchecked
            # Pure Pursuit arc would be motion with unknown clearance.
            info['stuck'] = True
            return 0.0, self.steer, info
        if self.last_rad < 0.9 * p.r_robot:     # little margin -> slow down; stop box stays standard
            v = min(v, p.squeeze_speed)
        if p.use_hybrid:                        # same limit when the lattice is the fallback
            look = np.c_[best[:, :2], np.arctan2(np.gradient(best[:, 1]), np.gradient(best[:, 0]))]
            v = min(v, self.narrow_speed(look, pts))
        self.prev_off = off1
        self.steer = self.track(best, v)
        return v, self.steer, info

    # -- lattice -------------------------------------------------------------
    def _geometry(self, idx, off0, dpsi, pose, L, rejoin_length, hold_end, offs):
        """Candidate paths for one (horizon, rejoin) pair, all lateral offsets at once.

        Everything here is independent of the robot radius ``rad``, so it is computed
        once per call and reused by every squeeze pass.  Candidates that violate the
        curvature limit or start behind the car are dropped exactly as before.
        Returns a list of dicts in the same order as ``offs``.
        """
        p = self.p
        route_end = hold_end + rejoin_length
        m = int(route_end / self.ds) + 1
        check_tail = max(p.tail, p.detect_dist - route_end)
        mt = m + int(math.ceil(check_tail / self.ds))
        ids = (idx + np.arange(mt)) % self.n
        s = np.arange(mt) * self.ds
        dp0 = math.tan(float(np.clip(dpsi, -1.0, 1.0)))
        offs = np.asarray(offs, float)
        # quintic coefficients for every offset: one 3x3 solve with many right-hand sides
        A1 = np.array([[L ** 3, L ** 4, L ** 5],
                       [3 * L ** 2, 4 * L ** 3, 5 * L ** 4],
                       [6 * L, 12 * L ** 2, 20 * L ** 3]])
        B1 = np.vstack([offs - off0 - dp0 * L, np.full(len(offs), -dp0), np.zeros(len(offs))])
        a_entry = np.linalg.solve(A1, B1)
        R = rejoin_length
        A2 = np.array([[R ** 3, R ** 4, R ** 5],
                       [3 * R ** 2, 4 * R ** 3, 5 * R ** 4],
                       [6 * R, 12 * R ** 2, 20 * R ** 3]])
        B2 = np.vstack([-offs, np.zeros(len(offs)), np.zeros(len(offs))])
        a_ret = np.linalg.solve(A2, B2)
        entering = s <= L
        holding = (s > L) & (s <= hold_end)
        rejoining = (s > hold_end) & (s <= route_end)
        se = s[entering]
        sr = s[rejoining] - hold_end
        d = np.zeros((len(offs), mt))
        entry = np.vstack([np.full(len(offs), off0), np.full(len(offs), dp0),
                           np.zeros(len(offs)), a_entry])          # (6, n_off)
        ret = np.vstack([offs, np.zeros(len(offs)), np.zeros(len(offs)), a_ret])
        for k in range(6):
            d[:, entering] += entry[k][:, None] * se[None, :] ** k
            d[:, rejoining] += ret[k][:, None] * sr[None, :] ** k
        d[:, holding] = offs[:, None]
        xy = self.ref[ids][None, :, :] + d[:, :, None] * self.normal[ids][None, :, :]
        h = np.unwrap(np.arctan2(np.diff(xy[:, :m, 1], axis=1), np.diff(xy[:, :m, 0], axis=1)), axis=1)
        # Divide by the candidate's own arc length: an offset path's points are not self.ds apart,
        # and using self.ds understates the curvature on the inside of a corner.
        seg = np.hypot(np.diff(xy[:, :m, 0], axis=1), np.diff(xy[:, :m, 1], axis=1))
        kap = np.diff(h, axis=1) / np.maximum(0.5 * (seg[:, :-1] + seg[:, 1:]), 1e-6)
        kmax = np.abs(kap).max(axis=1) if kap.shape[1] else np.zeros(len(offs))
        x0, y0, yaw = pose
        c, sn = math.cos(yaw), math.sin(yaw)
        dx = xy[:, :, 0] - x0
        dy = xy[:, :, 1] - y0
        bt = np.stack([c * dx + sn * dy, -sn * dx + c * dy], axis=-1)    # (n_off, mt, 2)
        out = []
        for i, d1 in enumerate(offs):
            if kmax[i] > 0.9 * MAX_CURV:
                self.rej['curv'] += 1
                continue
            b = bt[i, :m]
            if len(b) == 0 or b[0, 0] < -0.3:
                self.rej['back'] += 1
                continue
            chk = bt[i]
            hd = np.arctan2(np.gradient(chk[:, 1]), np.gradient(chk[:, 0]))
            if p.front_offset > 0.0 and len(chk) > 2:
                fr = chk + p.front_offset * np.c_[np.cos(hd), np.sin(hd)]
                chk = np.vstack([chk, fr])
                hd = np.r_[hd, hd]
            coarse = chk[::3]
            ch = hd[::3]
            out.append(dict(d1=float(d1), kmax=float(kmax[i]), b=b, chk=chk, hd=hd,
                            coarse=coarse, ccos=np.cos(ch), csin=np.sin(ch),
                            fcos=np.cos(hd), fsin=np.sin(hd),
                            cmin=chk.min(axis=0), cmax=chk.max(axis=0), L=L, rl=rejoin_length))
        return out

    def lattice(self, idx, off0, dpsi, pose, pts, free=False,
                lat_max=None, n_lat=None, horizons=None, block_s=None,
                rejoin_lengths=None, pass_margin=None):
        p = self.p
        rejoin_lengths = p.rejoin_lengths if rejoin_lengths is None else rejoin_lengths
        pass_margin = p.pass_margin if pass_margin is None else pass_margin
        lat_max = p.lat_max if lat_max is None else lat_max
        n_lat = p.n_lat if n_lat is None else n_lat
        horizons = p.horizons if horizons is None else horizons
        w_off = p.w_off * (4.0 if free else 1.0)     # obstacle passed -> return strongly
        w_prev = p.w_prev * (0.25 if free else 1.0)
        offs = np.linspace(-lat_max, lat_max, n_lat)
        feasible, scored = [], []
        self.last_rad = p.r_robot
        self.rej = dict(curv=0, back=0, coll=0, total=0)
        geo = {}                                    # (L, rejoin) -> candidate geometry, reused by every pass
        first_pass = True
        for rad in dict.fromkeys(max(p.r_min, f * p.r_robot) for f in p.radius_factors):  # squeeze passes
            margin = max(0.0, rad - p.footprint_width / 2.0)
            half_length = p.footprint_length / 2.0 + margin
            half_width = p.footprint_width / 2.0 + margin
            bbox_radius = math.hypot(half_length, half_width)
            for L in horizons:
                hold_end = max(L, (block_s or 0.0) + pass_margin)
                for rejoin_length in rejoin_lengths:
                    key = (L, rejoin_length)
                    if key not in geo:
                        geo[key] = self._geometry(idx, off0, dpsi, pose, L, rejoin_length, hold_end, offs)
                    for g in geo[key]:
                        if len(pts):
                            lower = g['cmin'] - bbox_radius
                            upper = g['cmax'] + bbox_radius
                            nearby = pts[np.all((pts >= lower) & (pts <= upper), axis=1)]
                            if len(nearby):
                                coarse = g['coarse']
                                coarse_delta = nearby[None, :, :] - coarse[:, None, :]
                                coarse_x = (g['ccos'][:, None] * coarse_delta[:, :, 0]
                                            + g['csin'][:, None] * coarse_delta[:, :, 1])
                                coarse_y = (-g['csin'][:, None] * coarse_delta[:, :, 0]
                                            + g['ccos'][:, None] * coarse_delta[:, :, 1])
                                coarse_dx = np.maximum(np.abs(coarse_x) - half_length, 0.0)
                                coarse_dy = np.maximum(np.abs(coarse_y) - half_width, 0.0)
                                if np.hypot(coarse_dx, coarse_dy).min() <= 0.0:
                                    if first_pass:
                                        self.rej['coll'] += 1
                                    continue
                                chk = g['chk']
                                delta = nearby[None, :, :] - chk[:, None, :]
                                local_x = (g['fcos'][:, None] * delta[:, :, 0]
                                           + g['fsin'][:, None] * delta[:, :, 1])
                                local_y = (-g['fsin'][:, None] * delta[:, :, 0]
                                           + g['fcos'][:, None] * delta[:, :, 1])
                                dx = np.maximum(np.abs(local_x) - half_length, 0.0)
                                dy = np.maximum(np.abs(local_y) - half_width, 0.0)
                                clear = float(np.hypot(dx, dy).min())
                            else:
                                clear = p.clear_cap
                        else:
                            clear = p.clear_cap
                        if clear <= 0.0:
                            if first_pass:
                                self.rej['coll'] += 1
                            continue
                        if first_pass:
                            self.rej['total'] += 1
                        d1 = g['d1']
                        cost = (w_off * abs(d1) / lat_max
                                + p.w_clear * (1.0 - min(clear, p.clear_cap) / p.clear_cap)
                                + p.w_curv * g['kmax'] / MAX_CURV
                                + w_prev * abs(d1 - self.prev_off) / lat_max
                                + 0.05 * rejoin_length
                                + (0.35 if free else -0.15) * L)
                        feasible.append(g['b'])
                        scored.append((cost, g['b'], d1))
            first_pass = False
            if scored:
                self.last_rad = rad
                self.min_rad = min(self.min_rad, rad)
                break
        if not scored:
            return None, [], 0.0
        cost, best, d1 = min(scored, key=lambda t: t[0])
        return best, feasible, d1

    def track(self, local_b, v):
        """Pure pursuit on the chosen local path (base frame) -> steering."""
        p = self.p
        ld = p.track_ld_min + p.track_ld_gain * v
        dist = np.hypot(local_b[:, 0], local_b[:, 1])
        ok = np.where((dist >= ld) & (local_b[:, 0] > 0.0))[0]
        tgt = local_b[ok[0]] if len(ok) else local_b[-1]
        ld_eff = max(float(np.hypot(*tgt)), 0.2)
        alpha = math.atan2(tgt[1], tgt[0])
        steer = math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld_eff)
        return float(np.clip(steer, -MAX_STEER, MAX_STEER))


# ------------------------------------------------------------------- ROS 2
try:
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Path
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray
    HAVE_ROS = True
except ImportError:          # lets the algorithm be tested without ROS
    HAVE_ROS = False


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


if HAVE_ROS:
    class FrenetNode(Node):
        def __init__(self):
            super().__init__('frenet_local_planner')
            self.p = default_params()
            self._sides_logged = []
            self.declare_parameter('profile', 'dwa_lap')   # fixed local path; safe | dwa_lap | hybrid_lap | fast | race
            apply_profile(self.p, str(self.get_parameter('profile').value))
            for k, v in vars(self.p).items():
                self.declare_parameter(k, list(v) if isinstance(v, tuple) else v)
                val = self.get_parameter(k).value
                setattr(self.p, k, tuple(val) if isinstance(v, tuple) else val)
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('pp_cmd_topic', '/pure_pursuit/ackermann_cmd')
            self.declare_parameter('out_topic', '/final_ackermann_cmd')
            self.declare_parameter('path_topic', '/pp_path')
            self.declare_parameter('map_frame', 'map')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('cmd_timeout', 0.5)
            g = self.get_parameter
            self.map_frame = g('map_frame').value
            self.base_frame = g('base_frame').value
            self.cmd_timeout = g('cmd_timeout').value

            self.planner = None
            self.scan = None
            self.pp = None                     # (speed, steer, time)
            self.last_zero = -1e9
            self.tf_buffer = Buffer()
            # own thread for /tf: a long planning cycle must not starve the TF buffer
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
            self._step_ms = []
            self.declare_parameter('odom_frame', 'odom')
            self.odom_frame = g('odom_frame').value
            self.memory = UnseenMemory(
                self.p.memory_hold, self.p.memory_keep_r,
                self.p.memory_avg_cell if self.p.memory_average else self.p.memory_cell,
                self.p.memory_rmin_margin, self.p.memory_average)
            self._mem_stamp = None
            self._mem_n = 0

            self.create_subscription(Path, g('path_topic').value, self.on_path, 1)
            self.create_subscription(LaserScan, g('scan_topic').value,
                                     self.on_scan, qos_profile_sensor_data)
            self.create_subscription(AckermannDriveStamped,
                                     g('pp_cmd_topic').value, self.on_pp, 1)
            self.pub_cmd = self.create_publisher(
                AckermannDriveStamped, g('out_topic').value, 1)
            self.pub_obs = self.create_publisher(MarkerArray, '/obstacles', 1)
            self.pub_path = self.create_publisher(Path, '/local_path', 1)
            self.pub_cand = self.create_publisher(MarkerArray, '/local_candidates', 1)
            self.pub_flag = self.create_publisher(Bool, '/obstacle_detected', 1)
            self.create_timer(0.05, self.tick)
            self.get_logger().info(
                f"frenet planner: {g('pp_cmd_topic').value} -> {g('out_topic').value}")

        # -- inputs ---------------------------------------------------------
        def now_s(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_path(self, msg):
            xy = np.array([[q.pose.position.x, q.pose.position.y] for q in msg.poses])
            if len(xy) < 8:
                return
            if self.planner is not None and len(self.planner.ref) and \
                    getattr(self, '_sig', None) == (len(xy), float(xy[0, 0]), float(xy[-1, 1])):
                return
            self._sig = (len(xy), float(xy[0, 0]), float(xy[-1, 1]))
            self.planner = FrenetPlanner(xy, self.p)
            self.get_logger().info(f'global path: {len(xy)} pts, '
                                   f'{self.planner.length:.1f} m')

        def on_scan(self, msg):
            self.scan = msg

        def on_pp(self, msg):
            self.pp = (msg.drive.speed, msg.drive.steering_angle, self.now_s())

        def pose(self):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, Time(), Duration(seconds=0.05))
            except Exception:                  # noqa: BLE001
                return None
            return (t.transform.translation.x, t.transform.translation.y,
                    yaw_of(t.transform.rotation))

        def scan_points(self):
            s = self.scan
            if s is None:
                return np.zeros((0, 2))
            tf = (0.0, 0.0, 0.0)
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, s.header.frame_id, Time(), Duration(seconds=0.05))
                tf = (t.transform.translation.x, t.transform.translation.y,
                      yaw_of(t.transform.rotation))
                self._laser_tf = tf
            except Exception:                  # noqa: BLE001
                if self.get_clock().now().nanoseconds * 1e-9 - getattr(self, '_tfw', -1e9) > 3.0:
                    self._tfw = self.get_clock().now().nanoseconds * 1e-9
                    self.get_logger().warn(
                        f'no TF {self.base_frame} <- {s.header.frame_id}: scan used WITHOUT mounting offset')
            return scan_to_points(s.ranges, s.angle_min, s.angle_increment,
                                  s.range_min, s.range_max, self.p.plan_range, tf)

        def odom_pose(self, stamp=None):
            t = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, Time() if stamp is None else Time.from_msg(stamp))
            return (t.transform.translation.x, t.transform.translation.y,
                    yaw_of(t.transform.rotation))

        def odom_or_drop(self):
            """Odometry pose; without it a kept Hybrid A* path cannot be held still, so drop it."""
            try:
                return self.odom_pose()
            except Exception:
                self.planner.ha_path = None
                self.planner.ha_search = None
                return None

        def with_unseen(self, pts, now):
            """Remembered points the sensor cannot confirm. Odometry frame: AMCL jumps must
            not move them. Without odometry or the laser mount the memory is dropped."""
            self._mem_n = 0
            none = np.zeros((0, 2))
            if not self.p.use_unseen_memory:
                return none
            laser = getattr(self, '_laser_tf', None)
            try:
                if laser is None:
                    raise ValueError('laser mount unknown')
                pose_now = self.odom_pose()
                s = self.scan
                if self._mem_stamp != (s.header.stamp.sec, s.header.stamp.nanosec):
                    self._mem_stamp = (s.header.stamp.sec, s.header.stamp.nanosec)
                    try:
                        pose_scan = self.odom_pose(s.header.stamp)
                    except Exception:  # noqa: BLE001
                        pose_scan = pose_now
                    self.memory.add(now, pts, pose_scan)
                extra = self.memory.unseen(now, pose_now, laser[:2], s.range_min, s.angle_max)
            except Exception as e:  # noqa: BLE001
                self.memory.clear()
                if now - getattr(self, '_mem_warn', -1e9) > 3.0:
                    self._mem_warn = now
                    self.get_logger().warn(f'unseen-point memory off: {e}')
                return none
            self._mem_n = len(extra)
            return extra

        # -- main loop ------------------------------------------------------
        def tick(self):
            t = self.now_s()
            pp = self.pp
            if self.planner is None:           # no global path yet: pass through
                if pp is not None and t - pp[2] <= self.cmd_timeout:
                    self.send(pp[0], pp[1])
                    if t - getattr(self, '_warn_t', -1e9) > 3.0:
                        self._warn_t = t
                        self.get_logger().warn(
                            'no /pp_path received yet -> passing Pure Pursuit through')
                return
            if pp is None or t - pp[2] > self.cmd_timeout:      # PP not running
                self.planner.reset()
                self.memory.clear()               # the car may be moved while stopped
                if t - self.last_zero > 1.0:
                    self.send(0.0, 0.0)
                    self.last_zero = t
                return
            pose = self.pose()
            sc = self.scan
            stale = (sc is None or
                     t - (sc.header.stamp.sec + sc.header.stamp.nanosec * 1e-9) > 0.5)
            if pose is None or stale:          # blind -> stop (fail safe)
                self.send(0.0, 0.0)
                if t - getattr(self, '_warn2_t', -1e9) > 2.0:
                    self._warn2_t = t
                    self.get_logger().warn(
                        'no TF map->base_link' if pose is None else 'no fresh /scan')
                return
            t_step = time.perf_counter()
            pts = self.scan_points()
            odom = self.odom_or_drop()
            if odom is None and self.p.use_hybrid:     # a kept path cannot be placed: hold still
                self.send(0.0, 0.0)
                self.get_logger().warn('Hybrid A*: no TF odom->base_link, holding still',
                                       throttle_duration_sec=1.0)
                return
            v, steer, info = self.planner.step(
                pose, pts, pp[0], pp[1], t, self.with_unseen(pts, t), odom)
            self._step_ms.append((time.perf_counter() - t_step) * 1e3)
            self.send(v, steer)
            if t - getattr(self, '_log_t', -1e9) > 1.0:
                last = getattr(self, '_last_pose', None)
                meas = ''
                if last is not None and t > last[2]:
                    dx, dy = pose[0] - last[0], pose[1] - last[1]
                    mv = (dx * math.cos(pose[2]) + dy * math.sin(pose[2])) / (t - last[2])
                    meas = f' real_v={mv:.2f}'
                    speed_mismatch = abs(mv) > 2.0 * abs(v) + 0.25
                    direction_mismatch = abs(mv) > 0.08 and abs(v) > 0.05 and mv * v < 0.0
                    if speed_mismatch or direction_mismatch:
                        self.get_logger().warn(
                            f'odometry speed={mv:.2f} m/s disagrees with out_v={v:.2f}; '
                            'check command routing and controller response')
                self._last_pose = (pose[0], pose[1], t)
                self._log_t = t
                tick = ''
                if self._step_ms:
                    tick = (f" tick_ms={sum(self._step_ms) / len(self._step_ms):.0f}"
                            f"/{max(self._step_ms):.0f}(avg/max)")
                    if max(self._step_ms) > 100.0:
                        self.get_logger().warn(
                            f'planning cycle took {max(self._step_ms):.0f} ms (> 100 ms): '
                            'scan/TF may be stale; reduce n_lat / horizons if this repeats')
                    self._step_ms = []
                rj = info.get('rej', {})
                why = ''
                if info.get('blocked'):
                    why = (f" blk n_ok={len(info['cands'])} rej(curv/back/coll)="
                           f"{rj.get('curv', 0)}/{rj.get('back', 0)}/{rj.get('coll', 0)}"
                           f" fb={info.get('fallback', 0)}"
                           f"{' STUCK' if info.get('stuck') else ''}")
                elif info.get('stuck'):              # the monitor must see a stop off the route too
                    why = ' STUCK'
                why += f' mem={self._mem_n} min_rad={self.planner.min_rad:.3f}'
                if self.p.use_hybrid:
                    why += f" ha={self.planner.ha_stats} refine_fail={REFINE_FAIL}"
                    for e in self.planner.ha_events:     # one line per side decision
                        if e[1] in ('side', 'fail', 'plan', 'crawl') and e not in self._sides_logged:
                            self._sides_logged.append(e)
                            del self._sides_logged[:-200]
                            self.get_logger().info(f'ha_{e[1]} {e}')
                self.planner.min_rad = self.p.r_robot
                self.get_logger().info(
                    f"mode={info['mode']} off={info['off']:+.2f} "
                    f"cap={info['cap']:.2f} pp_v={pp[0]:.2f} out_v={v:.2f}{meas}{tick}{why}")
            if t - getattr(self, '_chk_t', -1e9) > 5.0:
                self._chk_t = t
                n = self.count_subscribers(self.get_parameter('pp_cmd_topic').value)
                if n != 1:
                    self.get_logger().warn(
                        f'{n} nodes subscribe to the Pure Pursuit command topic '
                        '(expected only this planner): a bridge bypasses the planner')
            self.visualize(pose, info)

        def send(self, v, steer):
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.base_frame
            m.drive.speed = float(v)
            m.drive.steering_angle = float(steer)
            self.pub_cmd.publish(m)

        # -- RViz -----------------------------------------------------------
        def marker(self, mid, ns, mtype, rgba, scale):
            mk = Marker()
            mk.header.frame_id = self.map_frame
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns, mk.id, mk.type, mk.action = ns, mid, mtype, Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x, mk.scale.y, mk.scale.z = scale
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = rgba
            mk.lifetime = Duration(seconds=0.3).to_msg()
            return mk

        def visualize(self, pose, info):
            self.pub_flag.publish(Bool(data=bool(info['blocked'])))
            obs = MarkerArray()
            for i, (c, n, size) in enumerate(info['clusters']):
                mk = self.marker(i, 'obstacle', Marker.CYLINDER,
                                 (1.0, 0.1, 0.1, 0.8), (max(size, 0.15), max(size, 0.15), 0.3))
                cm = to_map(c[None, :], pose)[0]
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = \
                    float(cm[0]), float(cm[1]), 0.15
                obs.markers.append(mk)
            self.pub_obs.publish(obs)

            arr = MarkerArray()
            now_v = self.now_s()
            if info['cands'] and now_v - getattr(self, '_cand_t', -1e9) < 0.25:
                info = dict(info, cands=None)            # RViz only: refresh candidates at 4 Hz
            else:
                self._cand_t = now_v
            if info['cands'] is None:
                pass
            elif info['cands']:
                lines = self.marker(0, 'candidates', Marker.LINE_LIST,
                                    (0.2, 0.8, 0.3, 0.35), (0.01, 0.0, 0.0))
                for b in info['cands']:
                    mp = to_map(b, pose)
                    for a, c in zip(mp[:-1:2], mp[1::2]):
                        lines.points.append(Point(x=float(a[0]), y=float(a[1]), z=0.05))
                        lines.points.append(Point(x=float(c[0]), y=float(c[1]), z=0.05))
                arr.markers.append(lines)
            if info['cands'] is not None:
                self.pub_cand.publish(arr)

            path = Path()
            path.header.frame_id = self.map_frame
            path.header.stamp = self.get_clock().now().to_msg()
            if info['best'] is not None:
                for q in to_map(info['best'], pose):
                    ps = PoseStamped()
                    ps.header = path.header
                    ps.pose.position.x, ps.pose.position.y = float(q[0]), float(q[1])
                    ps.pose.orientation.w = 1.0
                    path.poses.append(ps)
            self.pub_path.publish(path)
            if info['mode'] != getattr(self, '_last_mode', None):
                self._last_mode = info['mode']
                self.get_logger().info(f"mode -> {info['mode']}")


def main():
    if not HAVE_ROS:
        raise SystemExit('rclpy not found: source /opt/ros/humble/setup.bash')
    rclpy.init()
    node = FrenetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send(0.0, 0.0)
        except Exception:                      # noqa: BLE001  (context already shut down)
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
