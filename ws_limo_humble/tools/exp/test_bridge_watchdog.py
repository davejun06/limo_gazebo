#!/usr/bin/env python3
"""No-motion check of the bridge watchdog on private topics. Nothing reaches /cmd_vel.

usage: run <bridge> --ros-args -r __node:=t_bridge -r /pure_pursuit/ackermann_cmd:=/t_final
       -r /cmd_vel:=/t_cmd -p use_sim_time:=true, then this script.
"""
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter

rclpy.init()
n = Node('t_feed2', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
pub = n.create_publisher(AckermannDriveStamped, '/t_final', 1)
seen = []
n.create_subscription(Twist, '/t_cmd', lambda m: seen.append((time.time(), m.linear.x)), 50)
t0 = time.time()
while time.time() - t0 < 4.0:
    m = AckermannDriveStamped()
    m.drive.speed = 0.3
    pub.publish(m)
    rclpy.spin_once(n, timeout_sec=0.02)
stop = time.time()
fed = max((v for _, v in seen), default=None)
while time.time() - stop < 2.5:
    rclpy.spin_once(n, timeout_sec=0.02)
zero = [t - stop for t, v in seen if t > stop and v == 0.0]
print('fed max', fed, 'first zero after', round(zero[0], 2) if zero else None, 's, last', seen[-1][1])
print('PASS' if fed and zero and zero[0] < 1.0 and seen[-1][1] == 0.0 else 'FAIL')
