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
  Add -p profile:=balanced (or safe, race, legacy) to choose a speed profile.
Bridge must read the final topic:
  python3 ackermann_to_twist.py --ros-args \
      -r /pure_pursuit/ackermann_cmd:=/final_ackermann_cmd -p use_sim_time:=true
"""
import math
import time
from types import SimpleNamespace

import numpy as np

WHEELBASE = 0.24            # m
MAX_STEER = 0.42            # rad
MAX_CURV = math.tan(MAX_STEER) / WHEELBASE      # ~1.86 1/m


def curvature_histogram_edges(limit):
    fixed_edges = (2.0, 2.5, 3.0, 4.0)
    return tuple([limit] + [edge for edge in fixed_edges if edge > limit]
                 + [math.inf])


PRESETS = {
    'legacy': {},  # Original source defaults; opt-in with -p profile:=legacy.
    'safe': dict(
        max_speed=0.45, avoid_max_speed=0.32, obs_speed=0.18, free_speed=0.35,
        wall_speed=0.22, a_lat=0.24, curve_min=0.18, squeeze_speed=0.12,
        brake_decel=0.45, brake_reaction_time=0.15,
        stop_decel=0.45, stop_reaction_time=0.15,
        min_obstacle_ttc=1.5, use_adaptive_sampling=True,
        use_obstacle_braking=True, use_dynamic_stop_zone=True,
        strict_speed_caps=True),
    'balanced': dict(
        max_speed=0.60, avoid_max_speed=0.42, obs_speed=0.22, free_speed=0.48,
        wall_speed=0.28, a_lat=0.30, curve_min=0.22, squeeze_speed=0.14,
        brake_decel=0.55, stop_decel=0.55, stop_reaction_time=0.12,
        min_obstacle_ttc=1.3,
        use_adaptive_sampling=True, use_obstacle_braking=True,
        use_dynamic_stop_zone=True, strict_speed_caps=True),
    'race': dict(
        max_speed=0.70, avoid_max_speed=0.48, obs_speed=0.24, free_speed=0.55,
        wall_speed=0.30, a_lat=0.32, curve_min=0.28, squeeze_speed=0.15,
        brake_decel=0.60, stop_decel=0.60, stop_reaction_time=0.10,
        min_obstacle_ttc=1.2,
        use_adaptive_sampling=True, use_obstacle_braking=True,
        use_dynamic_stop_zone=True, strict_speed_caps=True),
    # Retain the former aggressive presets without making them the new race profile.
    'fast': dict(max_speed=1.0, avoid_max_speed=0.55, obs_speed=0.30, free_speed=0.80,
                 wall_speed=0.40, wall_near=0.22, a_lat=0.50, curve_min=0.40,
                 steer_slow=0.35, squeeze_speed=0.25, stop_front=0.55),
    'legacy_race': dict(
        max_speed=1.3, avoid_max_speed=0.70, obs_speed=0.40, free_speed=1.00,
        wall_speed=0.50, wall_near=0.20, a_lat=0.70, curve_min=0.50,
        steer_slow=0.40, squeeze_speed=0.30, stop_front=0.70),
}


def apply_profile(p, name):
    """Apply a profile before ROS parameter overrides are loaded."""
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
        recovery_horizons=(3.0, 4.5, 6.0),  # [m] long transitions used only when recovering off path
        fallback_lat_max=2.40,  # [m] wider search used only if the normal lattice has no safe path
        fallback_n_lat=25,
        fallback_horizons=(0.8, 3.2),
        pass_margin=0.80,       # [m] clear the projected obstacle length and rear footprint before rejoining
        rejoin_lengths=(1.6, 2.4, 3.2),  # [m] smooth transitions back to the global path
        recovery_rejoin_lengths=(3.2, 4.8, 6.4),  # [m] smooth off-path return without relaxing curvature limits
        recovery_arc_distance=2.0,  # [m] collision-checked steering-arc horizon for severe pose errors
        recovery_heading_weight=0.35,  # [m/rad] reward alignment with the forward global path
        min_path_clearance=0.05,  # [m] reject lattice paths with less footprint clearance
        tail=1.2,               # [m] path continues parallel after the lattice end
                                #     (only used to check for collisions)
        avoid_max_speed=0.48,    # [m/s]
        min_speed=0.12,         # [m/s]
        max_speed=0.8,          # [m/s] global cap
        slow_dist=1.0,          # [m] look this far along the path for walls
        wall_extent=0.0,        # [m] OPT-IN: scan segments longer than this are walls (0 = off, safest)
        wall_near=0.26,         # [m] wall closer than this to the path -> crawl
        wall_far=0.45,          # [m] wall farther than this -> no slowdown
        obs_speed=0.24,         # [m/s] speed right in front of an obstacle
        obs_slow_dist=3.0,      # [m] start slowing this far before an obstacle
        detour_speed=0.08,      # [m/s] crawl speed on a collision-checked detour
        detour_min_clearance=0.05,  # [m] minimum footprint clearance to use detour crawl
        free_speed=0.55,        # [m/s] max speed while returning to the path (obstacle passed)
        body_r=0.12,            # [m] scan points this close to base_link = own body, ignored
        recover_off=0.40,       # [m] farther than this from the path -> lattice drives it back
        recover_heading=0.55,  # [rad] heading error that enters recovery planning
        return_heading=0.22,   # [rad] heading alignment required before FOLLOW resumes
        crawl_speed=0.16,       # [m/s] crawl speed for a verified short forward escape
        a_lat=0.32,             # [m/s^2] corner speed limit v = sqrt(a_lat / curvature)
        curve_min=0.28,         # [m/s] never slower than this because of a corner
        curve_look=2.0,         # [m] look this far along the path for corners
        steer_slow=0.30,        # [rad] |steer| at which speed has dropped to curve_min
        front_offset=0.0,        # [m] base_link is centered in the active Ackermann URDF
                                #     Use ~0.15 if base_link is at the rear axle / bumper sticks out
        r_min=0.125,             # [m] last-resort squeeze radius (robot half-width + margin); gap >= 2*r_min passes
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
        use_adaptive_sampling=False,  # add scan-derived offsets after the coarse lattice fails
        adaptive_clearance=0.02,      # [m] extra width when estimating free lateral intervals
        adaptive_step=0.04,           # [m] offsets sampled around a free interval center
        adaptive_max_offsets=12,      # maximum additional offsets evaluated per planning cycle
        use_obstacle_braking=False,   # cap approach speed using distance and assumed braking
        brake_decel=0.55,             # [m/s^2] conservative assumed deceleration
        brake_margin=0.10,            # [m] extra distance before the obstacle
        brake_reaction_time=0.10,     # [s] assumed sensing/control delay
        min_obstacle_ttc=1.5,         # [s] minimum nominal time-to-collision while approaching
        use_dynamic_stop_zone=False,  # adapt the raw-scan stop zone to speed and footprint
        stop_decel=0.55,              # [m/s^2] conservative assumed deceleration
        stop_reaction_time=0.10,      # [s] assumed sensing/control delay
        stop_margin=0.02,             # [m] extra footprint clearance for emergency stopping
        strict_speed_caps=False,      # never raise a curvature/wall cap to a configured speed floor
    )


# ----------------------------------------------------------------- helpers
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def resample_closed(xy, step):
    """Closed path resampled to ~uniform spacing."""
    xy = np.asarray(xy, float)[:, :2]
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:
        xy = xy[:-1]
    c = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(round(s[-1] / step)), 8)
    sn = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.c_[np.interp(sn, s, c[:, 0]), np.interp(sn, s, c[:, 1])], s[-1]


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
        self.curvature = np.array([
            wrap(self.theta[(i + 1) % self.n] - self.theta[(i - 1) % self.n])
            / (2.0 * self.ds)
            for i in range(self.n)
        ])
        self.idx = None
        self.mode = 'FOLLOW'
        self.last_blocked = -1e9
        self.prev_off = 0.0
        self.steer = 0.0
        self.tight_until = -1e9

    def reset(self):
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
    def step(self, pose, pts, pp_v, pp_steer, now):
        """Plan forward-only motion; remain stopped if no safe path exists."""
        return self._step(pose, pts, pp_v, pp_steer, now)

    def forward_escape(self, pts, pose=None, idx=None, off0=None, dpsi=None):
        """Find a collision-free Ackermann arc, preferring forward path recovery."""
        p = self.p
        if not len(pts):
            return None
        arc_distance = (
            p.recovery_arc_distance
            if pose is not None and off0 is not None
            and abs(off0) > p.recover_off
            else p.escape_dist)
        distance = np.arange(self.STEP, arc_distance + self.STEP / 2.0, self.STEP)
        best = None
        goal_ids = None
        start_error = None
        if pose is not None and idx is not None and off0 is not None:
            goal_ids = (idx + np.arange(
                max(2, int(p.detect_dist / self.ds)))) % self.n
            start_error = (
                abs(off0) + p.recovery_heading_weight * abs(dpsi or 0.0))
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
            point_clearance = np.hypot(outside_x, outside_y).min(axis=1)
            prefix_clearance = np.minimum.accumulate(point_clearance)
            safe_ends = np.flatnonzero(prefix_clearance >= p.escape_margin)
            if not len(safe_ends):
                continue
            if goal_ids is not None:
                for end in safe_ends:
                    endpoint = to_map(path[end], pose).reshape(2)
                    goal_index = int(np.argmin(np.sum(
                        (self.ref[goal_ids] - endpoint) ** 2, axis=1)))
                    goal_point = self.ref[goal_ids[goal_index]]
                    position_error = float(np.hypot(*(goal_point - endpoint)))
                    endpoint_yaw = pose[2] + curvature * float(distance[end])
                    heading_error = abs(wrap(
                        endpoint_yaw - self.theta[goal_ids[goal_index]]))
                    score = (
                        position_error
                        + p.recovery_heading_weight * heading_error
                        + 0.03 * float(distance[end])
                        - 0.02 * min(prefix_clearance[end], p.clear_cap))
                    if score >= start_error - 0.02:
                        continue
                    if best is None or score < best[0]:
                        best = (
                            score, path[:end + 1], float(steer),
                            float(prefix_clearance[end]))
            else:
                clearance = float(prefix_clearance[-1])
                score = -clearance + 0.05 * abs(float(steer)) / MAX_STEER
                if best is None or score < best[0]:
                    best = (score, path, float(steer), clearance)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def adaptive_offsets(self, idx, pose, pts, lat_max, coarse_offsets):
        """Sample extra lateral targets at LiDAR-derived free-gap centers."""
        p = self.p
        if not p.use_adaptive_sampling or not len(pts):
            return []

        count = max(2, int(math.ceil(p.detect_dist / self.ds)))
        ids = (idx + np.arange(count)) % self.n
        path_b = to_base(self.ref[ids], pose)
        yaw = pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        normals_b = np.c_[c * self.normal[ids, 0] + s * self.normal[ids, 1],
                          -s * self.normal[ids, 0] + c * self.normal[ids, 1]]
        delta = pts[:, None, :] - path_b[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        nearest = np.argmin(dist2, axis=1)
        lateral = np.sum(
            delta[np.arange(len(pts)), nearest] * normals_b[nearest], axis=1)
        along = nearest * self.ds
        valid = ((pts[:, 0] > 0.05) & (along <= p.detect_dist)
                 & (np.abs(lateral) <= lat_max + p.footprint_width))
        if not np.any(valid):
            return []

        bands = np.floor(along[valid] / 0.25).astype(int)
        valid_lateral = lateral[valid]
        valid_along = along[valid]
        band_order = sorted(
            np.unique(bands),
            key=lambda band: float(
                np.min(valid_along[bands == band])))
        coarse = np.asarray(coarse_offsets, dtype=float)
        inflation = p.footprint_width / 2.0 + p.adaptive_clearance
        proposed = []

        for band in band_order[:6]:
            obstacle_intervals = sorted(
                (max(-lat_max, float(value) - inflation),
                 min(lat_max, float(value) + inflation))
                for value in valid_lateral[bands == band])
            merged = []
            for lo, hi in obstacle_intervals:
                if not merged or lo > merged[-1][1]:
                    merged.append([lo, hi])
                else:
                    merged[-1][1] = max(merged[-1][1], hi)

            gaps = []
            cursor = -lat_max
            for lo, hi in merged:
                if lo - cursor > 0.04:
                    gaps.append((cursor, lo))
                cursor = max(cursor, hi)
            if lat_max - cursor > 0.04:
                gaps.append((cursor, lat_max))

            for lo, hi in sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True):
                center = 0.5 * (lo + hi)
                for candidate in (center, center - p.adaptive_step,
                                  center + p.adaptive_step):
                    is_new = np.min(np.abs(coarse - candidate)) > 0.015
                    if lo <= candidate <= hi and is_new:
                        proposed.append(float(candidate))
                if len(proposed) >= p.adaptive_max_offsets:
                    break
            if len(proposed) >= p.adaptive_max_offsets:
                break

        unique = []
        for candidate in proposed:
            is_unique = not any(
                abs(candidate - old) < 0.015 for old in unique)
            if is_unique:
                unique.append(candidate)
            if len(unique) >= p.adaptive_max_offsets:
                break
        return unique

    @staticmethod
    def braking_speed_cap(distance, deceleration, reaction_time):
        """Maximum speed whose idealized stopping distance fits in distance."""
        a = max(float(deceleration), 1e-3)
        t = max(float(reaction_time), 0.0)
        d = max(float(distance), 0.0)
        return max(0.0, math.sqrt((a * t) ** 2 + 2.0 * a * d) - a * t)

    def _step(self, pose, pts, pp_v, pp_steer, now):
        """pose=(x,y,yaw) in map, pts=Nx2 scan points in base_link.
        Returns (speed, steer, info)."""
        p = self.p
        x, y, yaw = pose
        idx = self.nearest(np.array([x, y]))
        off0 = float((np.array([x, y]) - self.ref[idx]) @ self.normal[idx])  # +left
        dpsi = wrap(yaw - self.theta[idx])
        initial_tangent_scale = 1.0 - self.curvature[idx] * off0
        dp0 = initial_tangent_scale * math.tan(dpsi)

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
        speed_now = abs(float(pp_v))
        if p.use_dynamic_stop_zone:
            sf = (p.footprint_length / 2.0 + p.stop_margin
                  + speed_now * p.stop_reaction_time
                  + speed_now * speed_now / (2.0 * max(p.stop_decel, 1e-3)))
            sf = max(p.stop_front, sf)
            sh = max(p.stop_half, p.footprint_width / 2.0 + p.stop_margin)
        else:
            sf, sh = p.stop_front, p.stop_half
        stop_box = np.zeros(len(pts), dtype=bool)
        if len(pts):                           # emergency stop box in front of the car
            if now < self.tight_until and not p.use_dynamic_stop_zone:
                sf, sh = min(sf, p.tight_stop_front), min(sh, p.tight_stop_half)
            stop_box = ((pts[:, 0] > 0.05) & (pts[:, 0] < sf)
                        & (np.abs(pts[:, 1]) < sh))
        ahead_b = to_base(self.ahead(idx, p.detect_dist + 1.5), pose)
        det_b = ahead_b[:int(p.detect_dist / self.ds)]

        near = np.zeros(len(pts), bool)
        near_s = np.full(len(pts), p.detect_dist, dtype=float)
        if len(pts) and len(det_b):
            d2 = ((pts[:, None, :] - det_b[None, :, :]) ** 2).sum(-1)
            near = d2.min(axis=1) < p.corridor ** 2
            near_s = d2.argmin(axis=1) * self.ds
            if p.wall_extent > 0.0:                # long connected segments = walls
                near &= ext <= p.wall_extent
        near |= stop_box
        block_s = float(near_s[near].min()) if near.any() else None
        # wall proximity speed limit: clearance between scan and the path ahead
        cap = p.max_speed
        wall_pts = pts[~near] if len(pts) else pts
        if len(pts) and len(wall_pts):
            sl = ahead_b[:max(2, int(p.slow_dist / self.ds)):2]
            clr = float(np.sqrt(((wall_pts[:, None, :] - sl[None, :, :]) ** 2)
                                .sum(-1).min()))
            clr = min(
                clr,
                float(np.hypot(wall_pts[:, 0], wall_pts[:, 1]).min())
                - p.r_robot + p.wall_near)
            f = (clr - p.wall_near) / max(1e-6, p.wall_far - p.wall_near)
            wall_floor = 0.0 if p.strict_speed_caps else p.wall_speed
            cap = wall_floor + float(np.clip(f, 0.0, 1.0)) * (
                p.max_speed - wall_floor)
        # corner speed limit from the global path curvature ahead + current steering
        w = 10                                  # 0.5 m windows
        ids = (idx + np.arange(0, int(p.curve_look / self.ds))) % self.n
        dth = np.abs(np.array([wrap(self.theta[(i + w) % self.n] - self.theta[i]) for i in ids[::5]]))
        kmax = float(dth.max()) / (w * self.ds) if len(dth) else 0.0
        v_curve = math.sqrt(p.a_lat / max(kmax, 1e-3))
        v_steer = p.max_speed - (p.max_speed - p.curve_min) * min(1.0, abs(pp_steer) / p.steer_slow)
        curve_cap = min(v_curve, v_steer)
        if not p.strict_speed_caps:
            curve_cap = max(p.curve_min, curve_cap)
        cap = min(cap, curve_cap)
        blocked = bool(near.any())
        if blocked:
            self.last_blocked = now
        clusters = clusters_of(pts[near], p.cluster_gap) if blocked else []

        if self.mode == 'FOLLOW':
            if (blocked or abs(off0) > p.recover_off
                    or abs(dpsi) > p.recover_heading):
                self.mode = 'AVOID'
        elif (not blocked and now - self.last_blocked > p.clear_hold
              and abs(off0) < p.return_off
              and abs(dpsi) < p.return_heading):
            self.mode = 'FOLLOW'

        info = dict(
            mode=self.mode, blocked=blocked, clusters=clusters,
            cands=[], best=None, off=off0, cap=cap,
            heading_error=dpsi, reference_curvature=float(self.curvature[idx]),
            reason='clear_path', stop_points=int(stop_box.sum()),
            stop_distance=sf, candidate_count=0, adaptive_count=0,
            fallback=False, brake_cap=None)
        if self.mode == 'FOLLOW':
            info['reason'] = (
                'off_global_path' if abs(off0) > p.recover_off else 'clear_path')
            self.steer = float(np.clip(pp_steer, -MAX_STEER, MAX_STEER))
            self.prev_off = off0
            return float(min(pp_v, cap)), self.steer, info

        max_avoid_speed = p.free_speed if not blocked else p.avoid_max_speed
        v = float(np.clip(pp_v, p.min_speed, max_avoid_speed))
        v = min(v, cap if p.strict_speed_caps else max(cap, p.wall_speed))
        if blocked:                            # ease off while approaching an obstacle
            dmin = float(np.hypot(pts[near][:, 0], pts[near][:, 1]).min())
            f = float(np.clip(
                (dmin - 0.4) / max(1e-6, p.obs_slow_dist - 0.4), 0.0, 1.0))
            v = min(v, p.obs_speed + f * (p.avoid_max_speed - p.obs_speed))
            info['reason'] = 'obstacle_on_global_path'
            if p.use_obstacle_braking:
                front_clearance = dmin - p.footprint_length / 2.0 - p.brake_margin
                brake_cap = self.braking_speed_cap(
                    front_clearance, p.brake_decel, p.brake_reaction_time)
                ttc_cap = max(0.0, front_clearance) / max(
                    p.min_obstacle_ttc, 1e-3)
                brake_cap = min(brake_cap, ttc_cap)
                info['brake_cap'] = brake_cap
                v = min(v, brake_cap)
        elif self.mode == 'AVOID':
            info['reason'] = 'recovering_to_global_path'
        self._lattice_diagnostics = []
        best, cands, off1 = self.lattice(
            idx, off0, dpsi, pose, pts, free=not blocked, block_s=block_s,
            initial_slope=dp0, stage='normal')
        adaptive_offsets = []
        if best is None and blocked and p.use_adaptive_sampling:
            coarse_offsets = np.linspace(-p.lat_max, p.lat_max, p.n_lat)
            adaptive_offsets = self.adaptive_offsets(
                idx, pose, pts, p.lat_max, coarse_offsets)
            if adaptive_offsets:
                best, cands, off1 = self.lattice(
                    idx, off0, dpsi, pose, pts, free=False,
                    offsets=np.r_[coarse_offsets, adaptive_offsets],
                    block_s=block_s, initial_slope=dp0, stage='adaptive')
        if best is None and blocked:
            info['fallback'] = True
            best, cands, off1 = self.lattice(
                idx, off0, dpsi, pose, pts, free=False,
                lat_max=p.fallback_lat_max, n_lat=p.fallback_n_lat,
                horizons=p.fallback_horizons, block_s=block_s,
                initial_slope=dp0, stage='fallback')
        if (best is None and abs(off0) > p.recover_off
                and len(p.recovery_horizons)
                and len(p.recovery_rejoin_lengths)):
            recovery_start = len(self._lattice_diagnostics)
            best, cands, off1 = self.lattice(
                idx, off0, dpsi, pose, pts, free=not blocked,
                horizons=p.recovery_horizons,
                rejoin_lengths=p.recovery_rejoin_lengths,
                block_s=block_s, initial_slope=dp0, stage='recovery')
            info['recovery_attempted'] = True
            info['recovery_candidates_tested'] = sum(
                item['tested'] for item in
                self._lattice_diagnostics[recovery_start:])
        info['cands'], info['best'] = cands, best
        info['candidate_count'] = len(cands)
        info['path_clearance'] = self.last_clearance
        info['adaptive_count'] = len(adaptive_offsets)
        info['lattice_stages'] = list(self._lattice_diagnostics)
        info['max_curvature_limit'] = MAX_CURV
        info['candidate_curvature_limit'] = 0.9 * MAX_CURV
        reject_totals = {'tested': 0, 'curvature': 0, 'behind': 0,
                         'collision': 0, 'feasible': 0,
                         'min_curvature': math.inf,
                         'max_curvature': 0.0}
        for attempt in self._lattice_diagnostics:
            for key in reject_totals:
                if key == 'min_curvature':
                    reject_totals[key] = min(reject_totals[key], attempt[key])
                elif key == 'max_curvature':
                    reject_totals[key] = max(reject_totals[key], attempt[key])
                else:
                    reject_totals[key] += attempt[key]
        info['reject_totals'] = reject_totals
        if best is None:                       # no lattice path
            escape = self.forward_escape(pts, pose, idx, off0, dpsi)
            if escape is not None:
                path, self.steer, clearance = escape
                info['mode'] = 'CRAWL'
                info['best'] = path
                info['escape_clearance'] = clearance
                info['creep_decision'] = 'use_collision_checked_escape'
                info['reason'] = (
                    'recovery_arc_toward_global_path'
                    if abs(off0) > p.recover_off
                    else 'collision_checked_escape_arc')
                self.tight_until = now + p.tight_hold
                crawl_v = p.crawl_speed
                if info['brake_cap'] is not None:
                    crawl_v = min(crawl_v, info['brake_cap'])
                return float(crawl_v), self.steer, info
            if stop_box.any():
                self.steer = 0.0
                info['mode'] = 'ESTOP'
                info['stuck'] = True
                info['reason'] = 'stop_zone_occupied_no_safe_path'
                info['creep_decision'] = 'stop_stop_zone_occupied'
                return 0.0, self.steer, info
            forward_strip = (
                (pts[:, 0] > 0.05)
                & (pts[:, 0] < 0.70 + p.front_offset)
                & (np.abs(pts[:, 1]) < max(p.crawl_half, p.r_robot))
            ) if len(pts) else np.zeros(0, dtype=bool)
            info['forward_strip_points'] = int(forward_strip.sum())
            info['forward_strip_clear'] = not bool(forward_strip.any())
            info['creep_decision'] = 'stop_no_verified_path'
            info['stuck'] = True
            info['reason'] = (
                'no_safe_lattice_or_escape_path_forward_strip_clear'
                if info['forward_strip_clear']
                else 'no_safe_lattice_or_escape_path_forward_strip_blocked')
            return 0.0, self.steer, info
        if (blocked and info['brake_cap'] is not None
                and info['brake_cap'] <= 0.0
                and self.last_clearance >= p.detour_min_clearance):
            v = min(p.detour_speed, cap)
            info['reason'] = 'collision_checked_detour_crawl'
        if self.last_rad < 0.7 * p.r_robot:     # tight squeeze -> slow down
            v = min(v, p.squeeze_speed)
            self.tight_until = now + p.tight_hold
        self.prev_off = off1
        self.steer = self.track(best, v)
        return v, self.steer, info

    # -- lattice -------------------------------------------------------------
    def lattice(self, idx, off0, dpsi, pose, pts, free=False,
                lat_max=None, n_lat=None, horizons=None, block_s=None,
                offsets=None, rejoin_lengths=None, initial_slope=None,
                stage='normal'):
        p = self.p
        lat_max = p.lat_max if lat_max is None else lat_max
        n_lat = p.n_lat if n_lat is None else n_lat
        horizons = p.horizons if horizons is None else horizons
        rejoin_lengths = (
            p.rejoin_lengths if rejoin_lengths is None else rejoin_lengths)
        w_off = p.w_off * (4.0 if free else 1.0)     # obstacle passed -> return strongly
        w_prev = p.w_prev * (0.25 if free else 1.0)
        dp0 = (math.tan(float(np.clip(dpsi, -1.0, 1.0)))
               if initial_slope is None else float(initial_slope))
        offs = (np.linspace(-lat_max, lat_max, n_lat) if offsets is None
                else np.unique(np.clip(np.asarray(offsets, dtype=float), -lat_max, lat_max)))
        feasible, scored = [], []
        diagnostics = {'tested': 0, 'curvature': 0, 'behind': 0,
                       'collision': 0, 'feasible': 0,
                       'min_curvature': math.inf,
                       'max_curvature': 0.0,
                       'curvature_values': [],
                       'stage': stage,
                       'off0': float(off0),
                       'offset_count': len(offs),
                       'horizons': tuple(float(value) for value in horizons)}
        self.last_rad = p.r_robot
        self.last_clearance = None
        stride = 1                                  # check at the path's 0.05 m resolution
        for rad in dict.fromkeys(max(p.r_min, f * p.r_robot) for f in (1.0, 0.8, 0.6, 0.45, 0.0)):  # squeeze passes
            for L in horizons:
                hold_end = max(L, (block_s or 0.0) + p.pass_margin)
                for rejoin_length in rejoin_lengths:
                    route_end = hold_end + rejoin_length
                    m = int(route_end / self.ds) + 1
                    check_tail = max(p.tail, p.detect_dist - route_end)
                    mt = m + int(math.ceil(check_tail / self.ds))
                    ids = (idx + np.arange(mt)) % self.n
                    s = np.arange(mt) * self.ds
                    for d1 in offs:
                        diagnostics['tested'] += 1
                        entry = quintic(off0, dp0, d1, L)
                        returning = quintic(d1, 0.0, 0.0, rejoin_length)
                        d = np.zeros(mt)
                        entering = s <= L
                        holding = (s > L) & (s <= hold_end)
                        rejoining = (s > hold_end) & (s <= route_end)
                        for k in range(6):
                            d[entering] += entry[k] * s[entering] ** k
                            d[rejoining] += returning[k] * (s[rejoining] - hold_end) ** k
                        d[holding] = d1
                        xy = self.ref[ids] + d[:, None] * self.normal[ids]
                        h = np.unwrap(np.arctan2(np.diff(xy[:m, 1]), np.diff(xy[:m, 0])))
                        segment_lengths = np.hypot(
                            np.diff(xy[:m, 0]), np.diff(xy[:m, 1]))
                        ds_local = 0.5 * (segment_lengths[:-1] + segment_lengths[1:])
                        kap = np.diff(h) / np.maximum(ds_local, 1e-6)
                        kmax = float(np.abs(kap).max()) if len(kap) else 0.0
                        diagnostics['min_curvature'] = min(
                            diagnostics['min_curvature'], kmax)
                        diagnostics['max_curvature'] = max(
                            diagnostics['max_curvature'], kmax)
                        if kmax > 0.9 * MAX_CURV:
                            diagnostics['curvature'] += 1
                            diagnostics['curvature_values'].append(
                                (kmax, float(d1), float(L)))
                            continue
                        bt = to_base(xy, pose)              # detour, rejoin, then global path
                        b = bt[:m]
                        if len(b) == 0 or b[0, 0] < -0.3:
                            diagnostics['behind'] += 1
                            continue
                        if len(pts):
                            chk = bt[::stride]
                            hd = np.arctan2(np.gradient(chk[:, 1]), np.gradient(chk[:, 0]))
                            if p.front_offset > 0.0 and len(chk) > 2:      # footprint front circle
                                fr = chk + p.front_offset * np.c_[np.cos(hd), np.sin(hd)]
                                chk = np.vstack([chk, fr])
                                hd = np.r_[hd, hd]
                            margin = max(0.0, rad - p.footprint_width / 2.0)
                            half_length = p.footprint_length / 2.0 + margin
                            half_width = p.footprint_width / 2.0 + margin
                            bbox_radius = math.hypot(half_length, half_width)
                            lower = chk.min(axis=0) - bbox_radius
                            upper = chk.max(axis=0) + bbox_radius
                            nearby = pts[np.all((pts >= lower) & (pts <= upper), axis=1)]
                            if len(nearby):
                                coarse = chk[::3]
                                coarse_heading = hd[::3]
                                coarse_delta = nearby[None, :, :] - coarse[:, None, :]
                                coarse_x = (np.cos(coarse_heading)[:, None] * coarse_delta[:, :, 0]
                                            + np.sin(coarse_heading)[:, None] * coarse_delta[:, :, 1])
                                coarse_y = (-np.sin(coarse_heading)[:, None] * coarse_delta[:, :, 0]
                                            + np.cos(coarse_heading)[:, None] * coarse_delta[:, :, 1])
                                coarse_dx = np.maximum(np.abs(coarse_x) - half_length, 0.0)
                                coarse_dy = np.maximum(np.abs(coarse_y) - half_width, 0.0)
                                if np.hypot(coarse_dx, coarse_dy).min() <= 0.0:
                                    diagnostics['collision'] += 1
                                    continue
                                delta = nearby[None, :, :] - chk[:, None, :]
                                local_x = (np.cos(hd)[:, None] * delta[:, :, 0]
                                           + np.sin(hd)[:, None] * delta[:, :, 1])
                                local_y = (-np.sin(hd)[:, None] * delta[:, :, 0]
                                           + np.cos(hd)[:, None] * delta[:, :, 1])
                                dx = np.maximum(np.abs(local_x) - half_length, 0.0)
                                dy = np.maximum(np.abs(local_y) - half_width, 0.0)
                                clear = float(np.hypot(dx, dy).min())
                            else:
                                clear = p.clear_cap
                        else:
                            clear = p.clear_cap
                        if clear < p.min_path_clearance:
                            diagnostics['collision'] += 1
                            continue
                        cost = (w_off * abs(d1) / lat_max
                                + p.w_clear * (1.0 - min(clear, p.clear_cap) / p.clear_cap)
                                + p.w_curv * kmax / MAX_CURV
                                + w_prev * abs(d1 - self.prev_off) / lat_max
                                + 0.05 * rejoin_length
                                + (0.35 if free else -0.15) * L)
                        feasible.append(b)
                        scored.append((cost, b, d1, clear))
                        diagnostics['feasible'] += 1
            if scored:
                self.last_rad = rad
                break
        curvature_values = diagnostics.pop('curvature_values')
        diagnostics['curvature_distribution'] = {}
        diagnostics['curvature_examples'] = []
        if curvature_values:
            values = np.asarray([sample[0] for sample in curvature_values])
            edges = curvature_histogram_edges(0.9 * MAX_CURV)
            counts, _ = np.histogram(values, bins=edges)
            diagnostics['curvature_distribution'] = {
                'p50': float(np.percentile(values, 50)),
                'p90': float(np.percentile(values, 90)),
                'max': float(values.max()),
                'bins': tuple(int(count) for count in counts),
                'edges': edges,
            }
            diagnostics['curvature_examples'] = [
                {'curvature': curvature, 'offset': offset, 'horizon': horizon}
                for curvature, offset, horizon in sorted(
                    curvature_values, reverse=True)[:3]
            ]
        self._lattice_diagnostics.append(diagnostics)
        if not scored:
            return None, [], 0.0
        cost, best, d1, self.last_clearance = min(
            scored, key=lambda t: t[0])
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
    from nav_msgs.msg import Odometry, Path
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool, String
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
            self.declare_parameter('profile', 'safe')      # legacy | safe | balanced | race
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
            self.declare_parameter('require_start_signal', True)
            g = self.get_parameter
            self.map_frame = g('map_frame').value
            self.base_frame = g('base_frame').value
            self.cmd_timeout = g('cmd_timeout').value

            self.planner = None
            self.scan = None
            self.scan_received_ros = None
            self.scan_received_wall = None
            self.scan_received_monotonic = None
            self.scan_period_wall_ms = None
            self.scan_received_ros_age = None
            self.pose_tf_stamp = None
            self.pose_tf_age = None
            self.pose_tf_lookup_ms = None
            self.scan_tf_stamp = None
            self.scan_tf_age = None
            self.scan_tf_lookup_ms = None
            self.pp = None                     # (speed, steer, time)
            self.odom_speed = None
            self.started = not g('require_start_signal').value
            self.last_zero = -1e9
            self._tick_log_monotonic = -1e9
            self._last_tick_wall = None
            self._last_tick_ros = None
            self._tf_failure_log_wall = {}
            self.tf_buffer = Buffer()
            TransformListener(self.tf_buffer, self)

            self.create_subscription(Path, g('path_topic').value, self.on_path, 1)
            self.create_subscription(LaserScan, g('scan_topic').value,
                                     self.on_scan, qos_profile_sensor_data)
            self.create_subscription(AckermannDriveStamped,
                                     g('pp_cmd_topic').value, self.on_pp, 1)
            self.create_subscription(Odometry, '/odom', self.on_odom, 1)
            self.create_subscription(String, '/control_selector',
                                     self.on_control_selector, 1)
            self.pub_cmd = self.create_publisher(
                AckermannDriveStamped, g('out_topic').value, 1)
            self.pub_obs = self.create_publisher(MarkerArray, '/obstacles', 1)
            self.pub_path = self.create_publisher(Path, '/local_path', 1)
            self.pub_cand = self.create_publisher(MarkerArray, '/local_candidates', 1)
            self.pub_flag = self.create_publisher(Bool, '/obstacle_detected', 1)
            self.create_timer(0.05, self.tick)
            self.get_logger().info(
                f"frenet planner: {g('pp_cmd_topic').value} -> {g('out_topic').value}")
            if not self.started:
                self.get_logger().info(
                    'waiting for /control_selector: pure_pursuit before motion')

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
            mono_now = time.monotonic()
            if self.scan_received_monotonic is not None:
                self.scan_period_wall_ms = (
                    mono_now - self.scan_received_monotonic) * 1000.0
            self.scan = msg
            self.scan_received_ros = self.now_s()
            self.scan_received_wall = time.time()
            self.scan_received_monotonic = mono_now
            header_s = (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            self.scan_received_ros_age = self.scan_received_ros - header_s

        def on_pp(self, msg):
            self.pp = (msg.drive.speed, msg.drive.steering_angle, self.now_s())

        def on_odom(self, msg):
            self.odom_speed = msg.twist.twist.linear.x

        def on_control_selector(self, msg):
            started = msg.data == 'pure_pursuit'
            if started != self.started:
                self.started = started
                self.get_logger().info(
                    'motion enabled by /control_selector' if started
                    else 'motion disabled by /control_selector')
            if not started:
                self.send(0.0, 0.0)

        def pose(self):
            lookup_start = time.perf_counter()
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, Time(), Duration(seconds=0.05))
            except Exception as exc:            # noqa: BLE001
                self.pose_tf_lookup_ms = (
                    time.perf_counter() - lookup_start) * 1000.0
                self.log_tf_failure(
                    'pose', self.map_frame, self.base_frame, exc,
                    self.pose_tf_lookup_ms)
                return None
            self.pose_tf_lookup_ms = (
                time.perf_counter() - lookup_start) * 1000.0
            self.pose_tf_stamp = (
                t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)
            self.pose_tf_age = self.now_s() - self.pose_tf_stamp
            return (t.transform.translation.x, t.transform.translation.y,
                    yaw_of(t.transform.rotation))

        def scan_points(self):
            s = self.scan
            if s is None:
                return np.zeros((0, 2))
            tf = (0.0, 0.0, 0.0)
            lookup_start = time.perf_counter()
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, s.header.frame_id, Time(), Duration(seconds=0.05))
                self.scan_tf_lookup_ms = (
                    time.perf_counter() - lookup_start) * 1000.0
                self.scan_tf_stamp = (
                    t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)
                self.scan_tf_age = self.now_s() - self.scan_tf_stamp
                tf = (t.transform.translation.x, t.transform.translation.y,
                      yaw_of(t.transform.rotation))
            except Exception as exc:            # noqa: BLE001
                self.scan_tf_lookup_ms = (
                    time.perf_counter() - lookup_start) * 1000.0
                self.log_tf_failure(
                    'scan_mount', self.base_frame, s.header.frame_id, exc,
                    self.scan_tf_lookup_ms)
            return scan_to_points(s.ranges, s.angle_min, s.angle_increment,
                                  s.range_min, s.range_max, self.p.plan_range, tf)

        def log_tf_failure(self, kind, target, source, exc, lookup_ms):
            wall_now = time.monotonic()
            if wall_now - self._tf_failure_log_wall.get(kind, -1e9) < 1.0:
                return
            self._tf_failure_log_wall[kind] = wall_now
            self.get_logger().warn(
                f'tf_failure kind={kind} target={target} source={source} '
                f'lookup_ms={lookup_ms:.3f} '
                f'ros_now={self.now_s():.6f} exception_type={type(exc).__name__} '
                f'exception={str(exc)} '
                f'fallback={"zero_mount_offset" if kind == "scan_mount" else "stop"}')

        def stage_diagnostics_text(self, info):
            result = []
            for stage in info.get('lattice_stages', []):
                distribution = stage.get('curvature_distribution', {})
                bins = distribution.get('bins', ())
                bin_text = ','.join(str(count) for count in bins)
                edge_text = ','.join(
                    f'{edge:.2f}' if math.isfinite(edge) else 'inf'
                    for edge in distribution.get('edges', ()))
                examples = ','.join(
                    f"(off={sample['offset']:+.2f},L={sample['horizon']:.2f},"
                    f"k={sample['curvature']:.2f})"
                    for sample in stage.get('curvature_examples', []))
                horizons = ','.join(
                    f'{value:.2f}' for value in stage.get('horizons', ()))
                result.append(
                    f"{stage['stage']}[tested={stage['tested']},"
                    f"off0={stage['off0']:+.2f},"
                    f"curv={stage['curvature']},coll={stage['collision']},"
                    f"behind={stage['behind']},ok={stage['feasible']},"
                    f"offsets={stage['offset_count']},L={horizons},"
                    f"k50={distribution.get('p50', 0.0):.2f},"
                    f"k90={distribution.get('p90', 0.0):.2f},"
                    f"kmax={distribution.get('max', 0.0):.2f},"
                    f"kbins=[{edge_text}]={bin_text},"
                    f"rejected={examples}]")
            return ';'.join(result) if result else 'none'

        # -- main loop ------------------------------------------------------
        def tick(self):
            wall_start = time.perf_counter()
            wall_now = time.time()
            mono_now = time.monotonic()
            ros_now = self.now_s()
            wall_gap_ms = (
                (mono_now - self._last_tick_wall) * 1000.0
                if self._last_tick_wall is not None else 0.0)
            ros_gap_ms = (
                (ros_now - self._last_tick_ros) * 1000.0
                if self._last_tick_ros is not None else 0.0)
            self._last_tick_wall = mono_now
            self._last_tick_ros = ros_now
            try:
                self.tick_once()
            finally:
                tick_ms = (time.perf_counter() - wall_start) * 1000.0
                if mono_now - self._tick_log_monotonic >= 1.0:
                    self._tick_log_monotonic = mono_now
                    scan = self.scan
                    if scan is None:
                        scan_fields = (
                            'scan_header=none scan_rx_ros=none '
                            'scan_rx_wall=none scan_ros_age=none '
                            'scan_wall_age=none')
                    else:
                        header_s = (
                            scan.header.stamp.sec
                            + scan.header.stamp.nanosec * 1e-9)
                        ros_rx = (
                            f'{self.scan_received_ros:.6f}'
                            if self.scan_received_ros is not None else 'none')
                        wall_rx = (
                            f'{self.scan_received_wall:.6f}'
                            if self.scan_received_wall is not None else 'none')
                        ros_age = ros_now - header_s
                        wall_age = (
                            mono_now - self.scan_received_monotonic
                            if self.scan_received_monotonic is not None else math.inf)
                        scan_fields = (
                            f'scan_header={header_s:.6f} '
                            f'scan_rx_ros={ros_rx} scan_rx_wall={wall_rx} '
                            f'scan_ros_age={ros_age:.6f} '
                            f'scan_rx_ros_age={self.scan_received_ros_age:.6f} '
                            f'scan_wall_age={wall_age:.6f} '
                            f'scan_period_wall_ms={self.scan_period_wall_ms if self.scan_period_wall_ms is not None else -1.0:.3f} '
                            f'tf_pose_stamp={self.pose_tf_stamp if self.pose_tf_stamp is not None else -1.0:.6f} '
                            f'tf_pose_age={self.pose_tf_age if self.pose_tf_age is not None else -1.0:.6f} '
                            f'tf_pose_lookup_ms={self.pose_tf_lookup_ms if self.pose_tf_lookup_ms is not None else -1.0:.3f} '
                            f'tf_scan_stamp={self.scan_tf_stamp if self.scan_tf_stamp is not None else -1.0:.6f} '
                            f'tf_scan_age={self.scan_tf_age if self.scan_tf_age is not None else -1.0:.6f} '
                            f'tf_scan_lookup_ms={self.scan_tf_lookup_ms if self.scan_tf_lookup_ms is not None else -1.0:.3f}')
                    self.get_logger().info(
                        f'timing tick_ms={tick_ms:.3f} '
                        f'tick_wall_gap_ms={wall_gap_ms:.3f} '
                        f'tick_ros_gap_ms={ros_gap_ms:.3f} '
                        f'ros_now={ros_now:.6f} wall_now={wall_now:.6f} '
                        f'use_sim_time={self.get_parameter("use_sim_time").value} '
                        f'{scan_fields}')

        def tick_once(self):
            t = self.now_s()
            pp = self.pp
            if not self.started:
                self.send(0.0, 0.0)
                if self.planner is not None:
                    pose = self.pose()
                    sc = self.scan
                    stale = (sc is None or
                             t - (sc.header.stamp.sec +
                                  sc.header.stamp.nanosec * 1e-9) > 0.5)
                    if pose is not None and not stale:
                        _, _, info = self.planner.step(
                            pose, self.scan_points(), 0.0, 0.0, t)
                        self.visualize(pose, info)
                return
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
            v, steer, info = self.planner.step(
                pose, self.scan_points(), pp[0], pp[1], t)
            self.send(v, steer)
            if t - getattr(self, '_log_t', -1e9) > 1.0:
                meas = ''
                if self.odom_speed is not None:
                    mv = self.odom_speed
                    meas = f' real_v={mv:.2f}'
                    speed_mismatch = abs(mv) > 2.0 * abs(v) + 0.25
                    direction_mismatch = abs(mv) > 0.08 and abs(v) > 0.05 and mv * v < 0.0
                    if speed_mismatch or direction_mismatch:
                        self.get_logger().warn(
                            f'odometry speed={mv:.2f} m/s disagrees with out_v={v:.2f}; '
                            'check command routing and controller response')
                self._log_t = t
                brake_cap = info.get('brake_cap')
                brake_text = f'{brake_cap:.2f}' if brake_cap is not None else '-1.00'
                rejects = info.get('reject_totals', {})
                self.get_logger().info(
                    f"mode={info['mode']} off0={info['off']:+.2f} "
                    f"cap={info['cap']:.2f} pp_v={pp[0]:.2f} out_v={v:.2f}"
                    f" dpsi={info.get('heading_error', 0.0):+.2f}"
                    f" heading_error_deg={math.degrees(info.get('heading_error', 0.0)):+.1f}"
                    f" ref_k={info.get('reference_curvature', 0.0):+.2f}"
                    f" reason={info.get('reason', 'unknown')}"
                    f" blocked={int(info.get('blocked', False))}"
                    f" stop_pts={info.get('stop_points', 0)}"
                    f" stop_dist={info.get('stop_distance', 0.0):.2f}"
                    f" cands={info.get('candidate_count', 0)}"
                    f" adaptive={info.get('adaptive_count', 0)}"
                    f" fallback={int(info.get('fallback', False))}"
                    f" path_clear={info.get('path_clearance') if info.get('path_clearance') is not None else -1.0:.2f}"
                    f" tested={rejects.get('tested', 0)}"
                    f" rej_curv={rejects.get('curvature', 0)}"
                    f" rej_behind={rejects.get('behind', 0)}"
                    f" rej_collision={rejects.get('collision', 0)}"
                    f" feasible={rejects.get('feasible', 0)}"
                    f" min_k={rejects.get('min_curvature', math.inf):.2f}"
                    f" recovery={int(info.get('recovery_attempted', False))}"
                    f" rec_tested={info.get('recovery_candidates_tested', 0)}"
                    f" curv_limit={info.get('max_curvature_limit', MAX_CURV):.3f}"
                    f" curv_accept={info.get('candidate_curvature_limit', 0.9 * MAX_CURV):.3f}"
                    f" wheelbase={WHEELBASE:.3f} max_steer={MAX_STEER:.3f}"
                    f" max_curv_formula=tan(steer)/wheelbase"
                    f" creep={info.get('creep_decision', 'not_applicable')}"
                    f" forward_strip_clear={info.get('forward_strip_clear', 'n/a')}"
                    f" forward_strip_points={info.get('forward_strip_points', 'n/a')}"
                    f" stages={self.stage_diagnostics_text(info)}"
                    f" brake_cap={brake_text}"
                    f"{meas}")
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
            if info['cands']:
                lines = self.marker(0, 'candidates', Marker.LINE_LIST,
                                    (0.2, 0.8, 0.3, 0.35), (0.01, 0.0, 0.0))
                for b in info['cands']:
                    mp = to_map(b, pose)
                    for a, c in zip(mp[:-1:2], mp[1::2]):
                        lines.points.append(Point(x=float(a[0]), y=float(a[1]), z=0.05))
                        lines.points.append(Point(x=float(c[0]), y=float(c[1]), z=0.05))
                arr.markers.append(lines)
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
