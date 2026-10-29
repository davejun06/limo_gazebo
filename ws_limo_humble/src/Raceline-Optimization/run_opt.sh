#!/bin/bash
# 사용법: ./run_opt.sh <맵이름>
MAP=${1:-track_map2}
cd "$(dirname "$0")"
MPLBACKEND=Agg .venv/bin/python -u main_globaltraj_f110.py --map_name "$MAP" --export_path "outputs/${MAP}_pp.csv" 2>&1 | tee run.log
grep -E "EXIT|Laptime|Maximum|distance to boundaries" run.log
