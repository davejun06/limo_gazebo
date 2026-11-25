#!/bin/bash
# Window 3: global path publisher and Pure Pursuit. Pure Pursuit's log goes to /tmp/pp.log too, for the check in window 6.
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
python3 src/limo_pp/pp_path_pub.py --ros-args -p csv_path:=$PWD/src/Raceline-Optimization/outputs/track_map3_line_pp.csv -p use_sim_time:=true &
PB=$!
trap 'kill -INT $PB 2>/dev/null' EXIT
ros2 run pure_pursuit pure_pursuit_node --ros-args --params-file src/pure_pursuit/config/limo_params.yaml -p use_sim_time:=true 2>&1 | tee /tmp/pp.log
