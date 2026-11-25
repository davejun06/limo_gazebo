#!/usr/bin/env python3
"""Does a bag hold what offline mapping needs? usage: bag_check.py <bag dir>. Exit 1 if not."""
import sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
n = {}; pairs = set(); static = set()
while r.has_next():
    t, data, _ = r.read_next()
    n[t] = n.get(t, 0) + 1
    if t in ('/tf', '/tf_static') and (t == '/tf_static' or n[t] < 20000):
        for x in deserialize_message(data, TFMessage).transforms:
            (static if t == '/tf_static' else pairs).add((x.header.frame_id, x.child_frame_id))
print('messages', n)
print('tf_static pairs', sorted(static)[:12], 'n', len(static))
print('tf pairs', sorted(pairs)[:10], 'n', len(pairs))
laser = [p for p in static | pairs if 'laser' in p[1] or 'lidar' in p[1] or 'scan' in p[1]]
ok = n.get('/scan', 0) > 500 and n.get('/odom', 0) > 1000 and ('odom', 'base_link') in pairs and ('map', 'odom') in static and bool(laser)
print('laser frames', laser, 'BAG OK' if ok else 'BAG NOT OK')
sys.exit(0 if ok else 1)
