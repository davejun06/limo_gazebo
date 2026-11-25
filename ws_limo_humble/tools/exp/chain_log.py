#!/usr/bin/env python3
"""Read-only log of the command chain at 10 Hz: where does a command stop? Subscribes only.

usage: chain_log.py <out.csv> [seconds]
"""
import csv
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

out, limit = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 900.0
rclpy.init()
n = Node('chain_log', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)])
last = dict(final=(0.0, 0.0), cmd_vel=(0.0, 0.0), ctrl=(0.0, 0.0), odom=0.0, wheels=(0.0, 0.0), steer=(0.0, 0.0))
count = dict(final=0, cmd_vel=0, ctrl=0, odom=0, joints=0)


def on_final(m):
    last['final'] = (m.drive.speed, m.drive.steering_angle); count['final'] += 1


def on_cmd(m):
    last['cmd_vel'] = (m.linear.x, m.angular.z); count['cmd_vel'] += 1


def on_ctrl(m):
    last['ctrl'] = (m.linear.x, m.angular.z); count['ctrl'] += 1


def on_odom(m):
    last['odom'] = m.twist.twist.linear.x; count['odom'] += 1


def on_joints(m):
    count['joints'] += 1
    v = dict(zip(m.name, m.velocity))
    last['wheels'] = (v.get('rear_left_wheel_joint', 0.0), v.get('rear_right_wheel_joint', 0.0))
    q = dict(zip(m.name, m.position))
    last['steer'] = (q.get('front_left_steer_joint', 0.0), q.get('front_right_steer_joint', 0.0))


n.create_subscription(AckermannDriveStamped, '/final_ackermann_cmd', on_final, 10)
n.create_subscription(Twist, '/cmd_vel', on_cmd, 10)
n.create_subscription(Twist, '/ackermann_controller/cmd_vel', on_ctrl, 10)
n.create_subscription(Odometry, '/odom', on_odom, 10)
n.create_subscription(JointState, '/joint_states', on_joints, 10)
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow('wall sim final_v final_steer cmdvel_v cmdvel_w ctrl_v ctrl_w odom_v wheel_l wheel_r steer_l steer_r n_final n_cmdvel n_ctrl n_odom n_joints'.split())
    t0 = time.time()
    nxt = t0
    while rclpy.ok() and time.time() - t0 < limit:
        rclpy.spin_once(n, timeout_sec=0.01)
        if time.time() >= nxt:
            nxt += 0.1
            w.writerow([round(time.time(), 2), round(n.get_clock().now().nanoseconds * 1e-9, 2), *last['final'], *last['cmd_vel'],
                        *last['ctrl'], last['odom'], *last['wheels'], *last['steer'], *count.values()])
            f.flush()
