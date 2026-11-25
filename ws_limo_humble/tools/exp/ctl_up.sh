#!/bin/bash
# Start the control nodes with fixed lifetimes; they exit by themselves when those end.
# Pure Pursuit ends first, so the planner sees a stale command and sends zero while the
# bridge is still alive to forward it. SIGTERM is used because background jobs of a
# non-interactive shell start with SIGINT ignored.
# usage: ctl_up.sh <lifetime_s> <csv> [planner_param:=value ...]   (inside the container)
source /home/davejun06/LIMO_GAZEBO/ws_limo_humble/logs/rt/env.sh
T=$1; CSV=$2; shift 2
PLANNER_ARGS=""; for a in "$@"; do PLANNER_ARGS="$PLANNER_ARGS -p $a"; done
PPY=${PP_YAML:-src/pure_pursuit/config/limo_params.yaml}
nohup timeout "$T" ros2 run pure_pursuit pure_pursuit_node --ros-args --params-file "$PPY" -p use_sim_time:=true > /tmp/pp.log 2>&1 &
nohup timeout "$T" python3 src/limo_pp/pp_path_pub.py --ros-args -p csv_path:="$PWD/$CSV" -p use_sim_time:=true > /tmp/path.log 2>&1 &
nohup timeout "$((T + 6))" python3 "${PLANNER:-src/dwa_planner.py}" --ros-args -p use_sim_time:=true $PLANNER_ARGS > /tmp/planner.log 2>&1 &
nohup timeout "$((T + 10))" python3 src/ackermann_to_twist.py --ros-args -r /pure_pursuit/ackermann_cmd:=/final_ackermann_cmd -p use_sim_time:=true > /tmp/bridge.log 2>&1 &
