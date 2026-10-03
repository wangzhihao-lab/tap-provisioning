#!/bin/bash
# Run one box's job list: phase 1 (tuning) with P1 parallel jobs, phase 2 (sweeps) with P2.
# Usage: run_box.sh <jobs_dir> <P1> <P2>.  Each phase file has one shell command per line.
set -uo pipefail
JOBS=$1; P1=${2:-3}; P2=${3:-8}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"
mkdir -p "$JOBS/logs"
run_phase() {  # $1 = phase file, $2 = parallelism
  local n=0
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    n=$((n+1))
    ( echo "[$(date '+%F %T')] START $cmd" >> "$JOBS/logs/phase.log"
      bash -c "$cmd" > "$JOBS/logs/job_$(basename "$1" .txt)_$n.log" 2>&1
      echo "[$(date '+%F %T')] EXIT $? $cmd" >> "$JOBS/logs/phase.log" ) &
    while [ "$(jobs -rp | wc -l)" -ge "$2" ]; do sleep 20; done
  done < "$1"
  wait
}
echo "[$(date '+%F %T')] phase1 start" >> "$JOBS/logs/phase.log"
[ -s "$JOBS/phase1.txt" ] && run_phase "$JOBS/phase1.txt" "$P1"
echo "[$(date '+%F %T')] phase2 start" >> "$JOBS/logs/phase.log"
[ -s "$JOBS/phase2.txt" ] && run_phase "$JOBS/phase2.txt" "$P2"
echo "[$(date '+%F %T')] ALL_DONE" >> "$JOBS/logs/phase.log"
touch "$JOBS/DONE"
