#!/bin/bash
# Window 4: the local planner, src/dwa_planner.py. Change values with p:=v, for example max_speed:=1.0
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
A=""; for a in "$@"; do A="$A -p $a"; done
exec python3 src/dwa_planner.py --ros-args -p use_sim_time:=true $A
