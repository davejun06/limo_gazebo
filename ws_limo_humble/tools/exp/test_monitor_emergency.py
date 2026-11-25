#!/usr/bin/env python3
"""Fake world for the monitor's bridge-loss path. Run on a private ROS_DOMAIN_ID only.

Publishes /clock, /odom, TF, /scan and /final_ackermann_cmd, plus a /cmd_vel publisher that
disappears 6 s after the monitor's start signal. Passes if the monitor keeps sending zero on /cmd_vel after that.
"""
import math
import os
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

if os.environ.get('ROS_DOMAIN_ID') == '77':
    sys.exit('refusing to run on the simulator domain')
rclpy.init()
n = Node('fake_world')
clock = n.create_publisher(Clock, '/clock', 10)
odom = n.create_publisher(Odometry, '/odom', 10)
scan = n.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
final = n.create_publisher(AckermannDriveStamped, '/final_ackermann_cmd', 10)
bridge = n.create_publisher(Twist, '/cmd_vel', 1)
br = TransformBroadcaster(n)
got = []
sel = []
n.create_subscription(Twist, '/cmd_vel', lambda m: got.append((time.time(), m.linear.x)), 50)
n.create_subscription(String, '/control_selector', lambda m: sel.append(m.data), 50)
t0 = time.time()
gone = None
last_scan = 0.0
route = []
with open('src/Raceline-Optimization/outputs/track_map3_pp.csv') as fh:
    for line in fh:
        route.append(tuple(float(c) for c in line.split(',')[:2]))
first_go = None
while time.time() - t0 < 60.0:
    el = time.time() - t0
    sim = 100.0 + el
    st = rclpy.time.Time(seconds=sim).to_msg()
    c = Clock()
    c.clock = st
    clock.publish(c)
    # Stands still until the start signal, then follows the route at 0.3 m/s.
    d = 0.3 * (time.time() - first_go) if first_go else 0.0
    i = int(d / 0.2) % len(route)
    x0, y0 = route[i]
    nx, ny = route[(i + 1) % len(route)]
    yaw = math.atan2(ny - y0, nx - x0)
    o = Odometry()
    o.header.stamp = st
    o.pose.pose.position.x = x0
    o.pose.pose.position.y = y0
    o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    o.twist.twist.linear.x = 0.3 if first_go else 0.0
    odom.publish(o)
    for parent, child, tx, ty, ang in (('map', 'odom', 0.0, 0.0, 0.0),
                                       ('odom', 'base_link', o.pose.pose.position.x,
                                        o.pose.pose.position.y, yaw)):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = st, parent, child
        t.transform.translation.x, t.transform.translation.y = tx, ty
        t.transform.rotation.z, t.transform.rotation.w = math.sin(ang / 2), math.cos(ang / 2)
        br.sendTransform(t)
    if el - last_scan > 0.125:
        last_scan = el
        s = LaserScan()
        s.header.stamp = st
        s.ranges = [3.0] * 10
        scan.publish(s)
    f = AckermannDriveStamped()
    f.drive.speed = 0.3
    final.publish(f)
    if first_go is None and 'pure_pursuit' in sel:
        first_go = time.time()
        print(f'start signal seen at {el:.1f} s', flush=True)
    if gone is not None and time.time() - gone > 20.0:
        break
    if gone is None and first_go is not None and time.time() - first_go > 6.0:
        n.destroy_publisher(bridge)
        gone = time.time()
        print(f'bridge publisher removed at {el:.1f} s; zeros before removal: {len(got)}', flush=True)
    rclpy.spin_once(n, timeout_sec=0.02)
    time.sleep(0.02)
after = [(t, v) for t, v in got if gone and t > gone + 0.5]
span = after[-1][0] - after[0][0] if len(after) > 1 else 0.0
print(f'zeros after bridge loss: {len(after)} over {span:.1f} s, all zero: '
      f'{all(v == 0.0 for _, v in after)}; stop selector seen: {"stop" in sel}')
print('PASS' if len(after) >= 20 and span > 5.0 and 'stop' in sel
      and all(v == 0.0 for _, v in after) else 'FAIL')
