#!/bin/bash
# Offline: play a recorded bag into slam_toolbox on a private ROS domain, save a map, score it. No robot.
# usage (inside the container, from the workspace): tools/exp/slam_replay.sh <bag> <map name> [param:=value ...]
BAG=$1; NAME=$2; shift 2
source logs/rt/env.sh
export ROS_DOMAIN_ID=88
ARGS=""; for a in "$@"; do ARGS="$ARGS -p $a"; done
[ -e src/Raceline-Optimization/maps/$NAME.yaml ] && { echo "ABORT: map $NAME exists"; exit 2; }
# The bag's map->odom is left out: slam_toolbox must be the only parent of odom. base->laser is given here.
timeout -s INT 500 ros2 run tf2_ros static_transform_publisher --x 0.1236 --z -0.01 --frame-id base_link --child-frame-id laser_link --ros-args -p use_sim_time:=true > /tmp/replay_static.log 2>&1 &
L=$!
timeout -s INT 900 ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file tools/exp/slam_mapping.yaml $ARGS > /tmp/replay_$NAME.log 2>&1 &
S=$!
sleep 4
timeout -s INT 800 ros2 bag play $BAG --clock 50 --rate 1.0 --topics /scan /odom /tf > /tmp/play_$NAME.log 2>&1
sleep 4
timeout 40 ros2 run nav2_map_server map_saver_cli -f src/Raceline-Optimization/maps/$NAME --ros-args -p save_map_timeout:=20000.0 -p map_subscribe_transient_local:=true 2>&1 | tail -n 1 | cut -c1-90
kill -INT $S $L; wait $S 2>/dev/null
ls -la src/Raceline-Optimization/maps/$NAME.yaml
