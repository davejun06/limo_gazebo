#!/bin/bash
# Host: run one offline replay in the container and score the map. usage: replay_eval.sh <tag> [param:=value ...]
T=$1; shift; W=$(pwd); N=slam_exp017_$T
docker exec -w "$W" limo_gazebo_humble bash tools/exp/slam_replay.sh logs/exp/exp-017/bag $N "$@"
docker exec limo_gazebo_humble bash -c "echo log-lines \$(wc -l < /tmp/replay_$N.log); grep -ciE 'error|fail' /tmp/replay_$N.log; grep -iE 'error|fail|extrapol' /tmp/replay_$N.log | head -n 3 | cut -c1-120; tail -n 1 /tmp/play_$N.log | cut -c1-100"
PYTHONDONTWRITEBYTECODE=1 python3 tools/exp/map_vs_world.py src/Raceline-Optimization/maps/$N.yaml | cut -c1-160
