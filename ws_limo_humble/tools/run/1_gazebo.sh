#!/bin/bash
# Window 1: Gazebo simulation, obstacle world, and RViz. Add gui:=false to leave the Gazebo window out.
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
exec ros2 launch limo_car roboracer_sim.launch.py \
  world:=${WORLD:-$PWD/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world} \
  rviz_config:=$PWD/tools/exp/limo_view.rviz "$@"
