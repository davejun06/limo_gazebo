#!/bin/bash
# Build a map on the obstacle-free world with slam_toolbox. Run from the host, inside systemd-inhibit.
# usage: tools/exp/map_drive.sh <exp> <map name>   Stops for any failed check, before the start signal.
EXP=$1; NAME=$2
[ -e src/Raceline-Optimization/maps/$NAME.yaml ] || [ -e src/Raceline-Optimization/maps/$NAME.pgm ] && { echo "ABORT: map $NAME exists"; exit 2; }
W=$(pwd); C=limo_gazebo_humble; CSV=src/Raceline-Optimization/outputs/track_map3_pp.csv
ENV="source $W/logs/rt/env.sh"
OLD='ros2 launch limo_car roboracer_sim|ros2 launch nav2_bringup localization|nav2_amcl/amcl|nav2_map_server/map_server|nav2_lifecycle_manager|gzserver|robot_state_publisher|ackermann_twist_adapter|lib/pure_pursuit/pure_pursuit_node|src/(frenet_planner_optimized|dwa_planner)\.py|src/ackermann_to_twist\.py|src/limo_pp/pp_path_pub\.py|tools/exp/chain_log\.py|async_slam_toolbox_node'
echo '--- running before stop'; docker exec $C bash -c "ps -eo pid,cmd | grep -E '$OLD' | grep -v grep | cut -c1-90"
docker exec $C pkill -INT -f "$OLD"
for i in $(seq 1 15); do n=$(docker exec $C bash -c "ps -eo cmd | grep -E '$OLD' | grep -v grep | wc -l"); [ "$n" = "0" ] && break; sleep 2; done
[ "$n" = "0" ] || { echo "ABORT: $n simulation processes remain"; docker exec $C bash -c "ps -eo pid,cmd | grep -E '$OLD' | grep -v grep | cut -c1-90"; exit 2; }
docker exec -d $C bash -c "$ENV; exec timeout -s INT 2400 ros2 launch limo_car roboracer_sim.launch.py world:=$W/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world rviz:=false gui:=false spawn_x:=7.4926918 spawn_y:=0.0260169 spawn_yaw:=0.6297 > /tmp/sim.log 2>&1"
for i in $(seq 1 15); do m=$(docker exec $C bash -c "$ENV; timeout 8 ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -cE '^/(gazebo|limo_ackermann_controller|ackermann_twist_adapter|robot_state_publisher)$'"); [ "$m" -ge 4 ] && break; done
[ "$m" -ge 4 ] || { echo "ABORT: simulator nodes $m"; exit 2; }
docker exec $C bash -c "ps -eo cmd | grep '^gzserver' | grep -q 'worlds/roboracer/ifac_roboracer.world'" || { echo 'ABORT: wrong world'; exit 2; }
base=$(docker exec $C bash -c "$ENV; timeout 8 gz model -m ifac_roboracer_map -p 2>&1 | grep -cE '^[-0-9]'"); [ "$base" = "1" ] || { echo 'ABORT: cannot read the world model, obstacle check not possible'; exit 2; }
for o in 1 2 3; do ob=$(docker exec $C bash -c "$ENV; timeout 8 gz model -m map1_obstacle_$o -p 2>&1 | grep -cE '^[-0-9]'"); [ "$ob" = "0" ] || { echo "ABORT: obstacle $o present"; exit 2; }; done
echo 'world ok: obstacle-free'
docker exec -d -w $W $C bash -c "$ENV; exec timeout -s INT 1200 ros2 run slam_toolbox async_slam_toolbox_node --ros-args --params-file tools/exp/slam_mapping.yaml > /tmp/slam.log 2>&1"
for i in $(seq 1 10); do docker exec $C bash -c "$ENV; timeout 8 ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -q '^/slam_toolbox$'" && break; done
docker exec -w $W $C bash -c "$ENV; bash tools/exp/ctl_up.sh 700 $CSV max_speed:=0.4"
sleep 12
OUT=logs/exp/$EXP
docker exec -w $W $C bash -c "$ENV; mkdir -p $OUT; timeout 100 python3 tools/exp/preflight.py --mapping 2>/dev/null | tee $OUT/preflight.txt | grep -E 'FAIL|fails' | cut -c1-120"
grep -q '"fails": \[\]' $OUT/preflight.txt || { echo 'ABORT: preflight failed, no start signal'; exit 3; }
docker exec -w $W $C bash -c "$ENV; python3 tools/exp/run_monitor.py --exp $EXP --path $CSV --out $OUT/run.csv --laps 2 --timeout 420 --near-stop -1 2> $OUT/monitor.err; echo monitor rc=\$?; grep -E 'STOP|lap |start signal' $OUT/monitor.err | cut -c1-110"
docker exec -w $W $C bash -c "$ENV; timeout 40 ros2 run nav2_map_server map_saver_cli -f src/Raceline-Optimization/maps/$NAME --ros-args -p save_map_timeout:=20000.0 -p map_subscribe_transient_local:=true 2>&1 | tail -n 2 | cut -c1-110; ls -la src/Raceline-Optimization/maps/$NAME.*"
docker exec $C bash -c "$ENV; timeout 5 ros2 topic echo /odom --once 2>/dev/null | grep -A1 'linear:' | sed -n 2p; timeout 5 ros2 topic echo /final_ackermann_cmd --once 2>/dev/null | grep -E '^  speed'"
PYTHONDONTWRITEBYTECODE=1 python3 tools/exp/map_vs_world.py src/Raceline-Optimization/maps/$NAME.yaml
