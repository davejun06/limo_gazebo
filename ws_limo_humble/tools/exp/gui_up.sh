#!/bin/bash
# Fresh simulator with windows, localization, path + Pure Pursuit, converter; checks the contact stream.
cd /home/davejun06/LIMO_GAZEBO/ws_limo_humble; C=limo_gazebo_humble; W=$PWD; ENV="source $W/logs/rt/env.sh"
OLD='ros2 launch limo_car roboracer_sim|ros2 launch nav2_bringup localization|nav2_amcl/amcl|nav2_map_server/map_server|nav2_lifecycle_manager|gzserver|gzclient|rviz2|robot_state_publisher|ackermann_twist_adapter|limo_ackermann|lib/pure_pursuit/pure_pursuit_node|src/dwa_planner\.py|src/ackermann_to_twist\.py|src/limo_pp/pp_path_pub\.py|tools/run/|tools/exp/run_monitor\.py|tools/exp/go\.py|ros2 bag record|gz topic -u -e'
docker exec $C bash -c "ps -eo cmd | grep -E '$OLD' | grep -v grep | wc -l"
docker exec $C pkill -INT -f "$OLD"
for i in $(seq 1 15); do n=$(docker exec $C bash -c "ps -eo cmd | grep -E '$OLD' | grep -v grep | wc -l"); [ "$n" = "0" ] && break; [ "$i" = "6" ] && docker exec $C pkill -TERM -f "$OLD"; [ "$i" = "11" ] && docker exec $C pkill -KILL -f 'nav2_lifecycle_manager/lifecycle_manager'; sleep 2; done; [ "$n" = "0" ] || { echo ABORT left $n; exit 2; }
docker exec -d -e DISPLAY=:0 -w $W $C bash -c "$ENV; exec ros2 launch limo_car roboracer_sim.launch.py > /tmp/w1.log 2>&1"
for i in $(seq 1 15); do m=$(docker exec $C bash -c "$ENV; timeout 8 ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -cE '^/(gazebo|limo_ackermann_controller|ackermann_twist_adapter|robot_state_publisher)$'"); [ "$m" -ge 4 ] && break; done; [ "$m" -ge 4 ] || { echo ABORT nodes $m; exit 2; }
# The lifecycle manager can hang on a lost service reply, leaving AMCL inactive: start localization again.
LOC='ros2 launch nav2_bringup localization|nav2_amcl/amcl|nav2_map_server/map_server|nav2_lifecycle_manager|tools/run/2_localization'
ok=0
for try in 1 2 3; do
  docker exec -d -w $W $C bash -c "exec bash tools/run/2_localization.sh > /tmp/w2.log 2>&1"
  for i in $(seq 1 25); do r=$(docker exec $C bash -c "$ENV; timeout 6 ros2 lifecycle get /amcl 2>/dev/null"); case "$r" in active*) ok=1; break;; esac; sleep 1; done
  echo "amcl try $try: $r"; [ "$ok" = "1" ] && break
  docker exec $C pkill -INT -f "$LOC"; sleep 6; docker exec $C pkill -TERM -f "$LOC"; sleep 2
done
[ "$ok" = "1" ] || { echo ABORT amcl not active; exit 4; }
docker exec -d -w $W $C bash -c "exec bash tools/run/3_path_pursuit.sh > /tmp/w3.log 2>&1"
docker exec -d -w $W $C bash -c "exec bash tools/run/4_dwa_planner.sh > /tmp/w4.log 2>&1"
docker exec -d -w $W $C bash -c "exec bash tools/run/5_cmd_convert.sh > /tmp/w5.log 2>&1"
sleep 20
b=$(docker exec $C bash -c "$ENV; timeout 6 gz topic -u -e /gazebo/ifac_roboracer/physics/contacts 2>/dev/null | wc -c"); echo contact-bytes $b; [ "$b" -gt 1000 ] || { echo ABORT contact stream silent; exit 3; }
xwininfo -root -tree 2>/dev/null | grep -cE 'limo_view.*RViz|"Gazebo"'
