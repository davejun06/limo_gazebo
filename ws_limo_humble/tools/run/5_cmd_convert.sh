#!/bin/bash
# Window 5: command conversion, /final_ackermann_cmd to /cmd_vel.
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
exec python3 src/ackermann_to_twist.py --ros-args -r /pure_pursuit/ackermann_cmd:=/final_ackermann_cmd -p use_sim_time:=true
