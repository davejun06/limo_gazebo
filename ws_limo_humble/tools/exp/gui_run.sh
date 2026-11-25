#!/bin/bash
# One lap with the windows open. usage: LOC=0.30 gui_run.sh <exp> [planner p:=v ...]
# Restarts the planner, reads the values back, records a bag from the monitor's start, runs go.py, keeps the record, prints a summary.
EXP=$1; shift
cd /home/davejun06/LIMO_GAZEBO/ws_limo_humble; C=limo_gazebo_humble; W=$PWD; ENV="source $W/logs/rt/env.sh"; D=logs/exp/$EXP
[ -e "$D" ] && { echo "ABORT: $D exists"; exit 2; }
docker exec $C pkill -INT -f 'src/dwa_planner\.py'
for i in $(seq 1 15); do n=$(docker exec $C bash -c "ps -eo cmd | grep -E 'src/dwa_planner\.py' | grep -v grep | wc -l"); [ "$n" = "0" ] && break; [ "$i" = "5" ] && docker exec $C pkill -TERM -f 'src/dwa_planner\.py'; sleep 1; done; [ "$n" = "0" ] || { echo ABORT planner left; exit 2; }
docker exec -d -w $W $C bash -c "exec bash tools/run/4_dwa_planner.sh $* > /tmp/w4.log 2>&1"
sleep 12
docker exec -w $W $C bash -c "$ENV; echo amcl \$(timeout 8 ros2 lifecycle get /amcl); for k in pass_speed max_speed avoid_max_speed wall_speed free_speed dwa_keep_gap narrow_min_speed narrow_drift_time; do echo \$k \$(timeout 15 ros2 param get /frenet_local_planner \$k); done"
# Every value passed must be in force, or no run.
for a in "$@"; do k=${a%%:=*}; v=${a##*:=}; got=$(docker exec -w $W $C bash -c "$ENV; timeout 20 ros2 param get /frenet_local_planner $k" | awk '{print $NF}'); python3 -c "import sys; sys.exit(0 if abs(float('$got')-float('$v'))<1e-9 else 1)" 2>/dev/null || { echo "value not in force: $k wanted $v got $got, no run"; exit 1; }; echo "in force $k $got"; done
( for i in $(seq 1 200); do docker exec $C pgrep -f tools/exp/run_monitor.py > /dev/null && break; sleep 1; done; [ $i -lt 200 ] && docker exec -d -w $W $C bash -c "$ENV; exec timeout -s INT 400 ros2 bag record --use-sim-time -o /tmp/bag_${EXP}_$$ /scan /odom /tf /tf_static > /tmp/bag.log 2>&1" ) &
BW=$!
T0=$(date +%s)
docker exec -w $W $C bash -ic "python3 tools/exp/go.py 1 ${LOC:-0.30} ${NEAR:-}" 2>&1 | grep -vE 'bash: (cannot set|no job)' | cut -c1-125
kill $BW 2>/dev/null
docker exec $C pkill -INT -f 'ros2 bag record'; sleep 3
J=$(stat -c %Y logs/exp/my-run/run.json 2>/dev/null || echo 0); [ "$J" -ge "$T0" ] || { echo "ABORT: no new run.json, the car did not drive; nothing kept"; exit 5; }
docker exec -w $W $C bash -c "mkdir -p $D; cp logs/exp/my-run/run.json logs/exp/my-run/run.csv logs/exp/my-run/run.planner.log logs/exp/my-run/monitor.err logs/exp/my-run/preflight.txt $D/ 2>/dev/null; [ -d /tmp/bag_${EXP}_$$ ] && cp -r /tmp/bag_${EXP}_$$ $D/bag; ps -eo cmd | grep -E 'ros2 bag record|tools/exp/go.py|run_monitor' | grep -v grep | wc -l"
PYTHONDONTWRITEBYTECODE=1 python3 tools/exp/run_summary.py $EXP
grep -oE 'tick_ms=[0-9]+/[0-9]+' $D/run.planner.log | awk -F/ '{if($2>m)m=$2}END{print "tick max",m" ms"}'; grep -E 'lap |STOP' $D/monitor.err | cut -c1-110; docker exec -w $W $C bash -c "$ENV; ros2 bag info $D/bag 2>/dev/null | grep -E 'Duration|/scan' | cut -c1-80"
