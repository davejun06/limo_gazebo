#!/bin/bash
# Watching on screen only, not an experiment record: same as run_once.sh, but the one preflight
# failure allowed is the rule against the Gazebo window; no failure is fine too, as with RViz alone.
# Any other failure stops before the start signal. With both windows open the planner can pass 100 ms.
# usage (inside the container): run_gui.sh <laps> [planner p:=v ...]
source /home/davejun06/LIMO_GAZEBO/ws_limo_humble/logs/rt/env.sh
LAPS=$1; shift; OUT=logs/exp/my-run; mkdir -p "$OUT"; CSV=src/Raceline-Optimization/outputs/track_map3_line_pp.csv
if pgrep -f 'pure_pursuit_node|(frenet_planner_optimized|dwa_planner)|ackermann_to_twist|pp_path_pub' > /dev/null; then
  echo "ABORT: control nodes from an earlier run are still alive"; exit 2
fi
bash tools/exp/ctl_up.sh 720 "$CSV" "$@"
sleep 8
python3 tools/exp/place_start.py 2>/dev/null || { echo "ABORT: placement failed"; exit 2; }
sleep 3
python3 tools/exp/preflight.py alpha1=0.2 alpha2=0.2 alpha3=0.2 alpha4=0.2 update_min_d=0.1 update_min_a=0.1 2>/dev/null | tee "$OUT/preflight.txt" | grep -E 'FAIL|fails' | cut -c1-150
if ! grep -qE '^\{"fails": (\[\]|\["no gzclient during runs \(CPU\)"\])\}' "$OUT/preflight.txt"; then echo "ABORT: preflight, no start signal sent"; exit 3; fi
python3 tools/exp/pp_ready.py 2>&1 | tee "$OUT/pp_ready.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "ABORT: Pure Pursuit not ready, no start signal sent"; exit 4; fi
python3 tools/exp/run_monitor.py --exp my-run --path "$CSV" --out "$OUT/run.csv" --laps "$LAPS" --timeout 300 2> "$OUT/monitor.err"
echo "monitor rc=$?"; grep -E 'STOP|lap |start signal|motion' "$OUT/monitor.err" | cut -c1-120
sleep 2
timeout 4 ros2 topic echo /final_ackermann_cmd --once 2>/dev/null | grep -E '^  speed'
timeout 4 ros2 topic echo /odom --once 2>/dev/null | grep -A1 'linear:' | sed -n 2p
