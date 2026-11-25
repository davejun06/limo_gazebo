#!/bin/bash
# Window 6: checks, then puts the car behind the start line, sends the start signal, measures laps and stops on
# a safety rule. Ctrl-C or any exit closes the gate: a limited number of stop messages, then speed is read back.
# usage: 6_start.sh [laps], default 1. Watching only, not an experiment record: results go to logs/exp/my-run.
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
LAPS=${1:-1}; OUT=logs/exp/my-run; mkdir -p "$OUT"; STARTED=0
finish() {
  trap '' INT TERM
  if [ "$STARTED" = "1" ]; then
    echo "stopping: closing the gate"
    timeout 6 ros2 topic pub --times 6 --rate 5 /control_selector std_msgs/msg/String "{data: stop}" > /dev/null 2>&1
    sleep 1
    C=$(timeout 4 ros2 topic echo /final_ackermann_cmd --once 2>/dev/null | grep -E '^  speed')
    V=$(timeout 4 ros2 topic echo /odom --once 2>/dev/null | grep -A1 'linear:' | sed -n 2p)
    echo "final command:$C   odometry:$V"
    python3 -c "import sys;v=float('$V'.split(':')[-1] or 9);sys.exit(0 if abs(v)<0.02 else 1)" 2>/dev/null || echo "WARNING: the car may still be moving. Stop windows 3, 4 and 5 now (Ctrl-C)."
  fi
}
trap finish EXIT
trap 'exit 130' INT TERM
python3 tools/exp/place_start.py 2>/dev/null || { echo "ABORT: placement failed; are windows 1 and 2 up?"; exit 2; }
sleep 3
python3 tools/exp/preflight.py alpha1=0.2 alpha2=0.2 alpha3=0.2 alpha4=0.2 update_min_d=0.1 update_min_a=0.1 2>/dev/null | tee "$OUT/preflight.txt" | grep -E 'FAIL|fails' | cut -c1-150
if ! grep -qE '^\{"fails": (\[\]|\["no gzclient during runs \(CPU\)"\])\}' "$OUT/preflight.txt"; then echo "ABORT: preflight, no start signal sent. Often a slow query: run this window again."; exit 3; fi
python3 tools/exp/pp_ready.py || { echo "ABORT: Pure Pursuit not ready, no start signal sent; are windows 3, 4, 5 up?"; exit 4; }
STARTED=1
python3 tools/exp/run_monitor.py --exp my-run --path src/Raceline-Optimization/outputs/track_map3_line_pp.csv --out "$OUT/run.csv" --laps "$LAPS" --timeout $((LAPS * 300)) 2> >(tee "$OUT/monitor.err" | grep --line-buffered -E 'STOP|lap |start signal|motion')
