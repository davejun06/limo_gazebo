#!/bin/bash
# One supervised run inside the container: start nodes (gate closed), place at A, preflight, drive.
# usage: run_once.sh <exp> <laps> <timeout_s> <lifetime_s> <csv> "<preflight asserts>" [planner p:=v ...]
source /home/davejun06/LIMO_GAZEBO/ws_limo_humble/logs/rt/env.sh
EXP=$1; LAPS=$2; TO=$3; LIFE=$4; CSV=$5; ASSERTS=$6; shift 6
OUT=logs/exp/$EXP; mkdir -p "$OUT"
if pgrep -f 'pure_pursuit_node|(frenet_planner_optimized|dwa_planner)|ackermann_to_twist|pp_path_pub' > /dev/null; then
  echo "ABORT: control nodes from an earlier run are still alive"; exit 2
fi
bash tools/exp/ctl_up.sh "$LIFE" "$CSV" "$@"
sleep 8
python3 tools/exp/place_start.py 2>/dev/null || { echo "ABORT: placement failed"; exit 2; }
sleep 3
python3 tools/exp/preflight.py $ASSERTS 2>/dev/null | tee "$OUT/preflight.txt" | grep -E 'FAIL|AMCL live|fails'
if ! grep -q '"fails": \[\]' "$OUT/preflight.txt"; then echo "ABORT: preflight failed, no start signal sent"; exit 3; fi
# Planner parameters this run depends on: abort unless the live node has them.
if [ -n "$REQUIRE_PLANNER" ]; then
  for kv in $REQUIRE_PLANNER; do
    k=${kv%%=*}; want=${kv#*=}
    got=$(timeout 15 ros2 param get /frenet_local_planner "$k" 2>/dev/null | awk '{print $NF}')
    echo "planner $k = $got (want $want)" | tee -a "$OUT/preflight.txt"
    if [ "$got" != "$want" ]; then echo "ABORT: planner parameter $k is '$got', not '$want'; no start signal sent"; exit 5; fi
  done
fi
python3 tools/exp/pp_ready.py 2>&1 | tee "$OUT/pp_ready.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "ABORT: Pure Pursuit not ready, no start signal sent"; exit 4; fi
python3 tools/exp/run_monitor.py --exp "$EXP" --path "$CSV" --out "$OUT/run.csv" --laps "$LAPS" --timeout "$TO" 2> "$OUT/monitor.err"
RC=$?
echo "monitor rc=$RC"; grep -E 'could not send|EMERG|STOP|lap |start signal|motion' "$OUT/monitor.err" | cut -c1-120
sleep 2
timeout 4 ros2 topic echo /final_ackermann_cmd --once 2>/dev/null | grep -E '^  speed' 
timeout 4 ros2 topic echo /odom --once 2>/dev/null | grep -A1 'linear:' | sed -n 2p
cp /tmp/pp.log /tmp/planner.log /tmp/bridge.log "$OUT/" 2>/dev/null
