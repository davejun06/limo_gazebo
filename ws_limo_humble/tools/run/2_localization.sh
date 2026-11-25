#!/bin/bash
# Window 2: map server and AMCL, with update thresholds and the start pose from amcl_start.yaml.
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
MAP=${1:-track_map3}
ros2 launch nav2_bringup localization_launch.py map:=$PWD/src/Raceline-Optimization/maps/$MAP.yaml use_sim_time:=true params_file:=$PWD/tools/exp/amcl_start.yaml
