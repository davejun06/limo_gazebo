#!/usr/bin/env python3
"""No-motion check: when the Pure Pursuit command stops, does zero reach the bridge output?

Runs a private planner and bridge on remapped topics (nothing reaches /cmd_vel), feeds a
nonzero command, stops feeding it, and reports what the outputs last said.
Needs TF, /scan and /pp_path to be live. Start the two nodes first, see the usage line.
usage: planner: python3 src/frenet_planner_optimized.py --ros-args -r __node:=t_planner
         -p use_sim_time:=true -p pp_cmd_topic:=/t_pp -p out_topic:=/t_final
       bridge:  python3 src/ackermann_to_twist.py --ros-args -r __node:=t_bridge
         -r /pure_pursuit/ackermann_cmd:=/t_final -r /cmd_vel:=/t_cmd -p use_sim_time:=true
"""
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.parameter import Parameter

rclpy.init()
n = Node('t_feed', parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
pub = n.create_publisher(AckermannDriveStamped, '/t_pp', 1)
seen = dict(final=[], cmd=[])
n.create_subscription(AckermannDriveStamped, '/t_final',
                      lambda m: seen['final'].append(m.drive.speed), 10)
n.create_subscription(Twist, '/t_cmd', lambda m: seen['cmd'].append(m.linear.x), 10)


def run(seconds, feed):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if feed:
            m = AckermannDriveStamped()
            m.drive.speed = 0.3
            pub.publish(m)
        rclpy.spin_once(n, timeout_sec=0.02)


run(6.0, True)
fed_max = (max(seen['final'], default=None), max(seen['cmd'], default=None))
seen['final'].clear()
seen['cmd'].clear()
run(4.0, False)
print('while fed, max final/cmd:', fed_max)
print('after feed stopped, last final/cmd:', seen['final'][-3:], seen['cmd'][-3:])
print('PASS' if seen['cmd'] and seen['cmd'][-1] == 0.0 and fed_max[1] else 'FAIL')
