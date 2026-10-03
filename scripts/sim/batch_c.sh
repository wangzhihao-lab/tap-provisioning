#!/bin/bash
# Formal batch C (attempt 3, 2026-09-17 23:4x): calibrated 7B service model (batch B T3) and the
# workload defaults calibrated on the genuine AI-assistant exports (WorkloadConfig);
# baselines tuned per price ratio r on the whole institution (seeds 0,1 | held-out 2), every
# formal sweep on FRESH seeds 10-14 (test seeds independent of tuning);
# institution-wide adoption with one always-on base replica, 6 teaching weeks, per-week rows.
# Controls added: ``fixed`` (base tier only) and ``calendar_adapt`` (timetable + online size
# correction); an SLO-matched reactive/velocity pair (cheapest cell with slo_in >= 0.95) at r=30.
# Attempt 1: oracle DP dropped the base-tier floor.  Attempt 2: workload defaults still placeholder.
# Three lanes run concurrently (~6 GB RSS each).
set -uo pipefail
cd "$(dirname "$0")/../.."   # repository root
T3=scripts/q2/results/b1_remote/b1/T3_calibration.csv
MODEL=qwen2.5-7b
MODEL32=qwen2.5-32b-awq
C=0.0333; BETA=0.0333; MU=1067; MU32=288   # c = 2 USD/h nominal; beta/c = 1; mu from T3 at the workload's output length
OUT=outputs/batchC; mkdir -p $OUT
SEEDS5="10 11 12 13 14"; SEEDS3="10 11 12"
SCEN="--adoption 1.0 --min-replicas 1"   # institution-wide deployment: every enrolled student, one always-on base replica
POL="--policies fixed static reactive velocity forecast_ilp lcp calendar calendar_adapt bdinterp@0.1 offline informed stale_calendar"
EPSPOL="--policies calendar calendar_adapt bdinterp@0.1 bdinterp@0.5 bdinterp@1.0 lcp offline informed"
CAL="--calibration $T3 --model $MODEL --lead-stat median --c-run $C --beta-start $BETA"
CAL32="--calibration $T3 --model $MODEL32 --lead-stat median --c-run $C --beta-start $BETA"
LOG=$OUT/driver.log
pin() { python3 -c "print(round($1*$C/$2, 8))"; }
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() { stamp "RUN $*"; if ! uv run python "$@" >> "$LOG" 2>&1; then stamp "FAILED: $*"; return 1; fi; stamp "OK"; }
tune() {  # $1 = r, $2 = out dir, $3... = extra flags
  local R=$1 DIR=$2; shift 2
  stamp "=== r=$R: tuning baselines (whole institution, 1 week, seeds 0,1 | held-out 2) $*"
  run scripts/sim/tune_baselines.py --college ALL --weeks 1 --max-sections 0 --adoption 1.0 --min-replicas 1 \
      --lead-tier d_rep $CAL --price-ratio "$R" --out "$DIR" "$@"
}
block() {  # $1 = r : tuning, then the sigma and lead sweeps
  local R=$1 PIN; PIN=$(pin "$1" $MU)
  tune "$R" "$OUT/tuning_r$R" || return 1
  for SW in sigma lead; do
    run scripts/sim/run_sweep.py --sweep $SW --weeks 6 --seeds $SEEDS5 $SCEN $POL --lead-tier d_rep $CAL \
        --p-in "$PIN" --tuned "$OUT/tuning_r$R/best.json" --out "$OUT/r$R" || return 1
  done
}
lane_a() {
  LOG=$OUT/lane_A.log; stamp "lane A start: r=30 block, SLO-matched baselines, adoption/margin, 32B"
  block 30 || { stamp "lane A: r=30 block failed"; return 1; }
  local PIN TUNED; PIN=$(pin 30 $MU); TUNED="--tuned $OUT/tuning_r30/best.json"
  tune 30 "$OUT/tuning_r30_slo95" --slo-floor 0.95 && \
    run scripts/sim/run_sweep.py --sweep lead --weeks 6 --seeds $SEEDS5 $SCEN --policies reactive velocity \
        --lead-tier d_rep $CAL --p-in "$PIN" --tuned "$OUT/tuning_r30_slo95/best.json" --out "$OUT/r30_slo95"
  for SW in adoption margin; do
    run scripts/sim/run_sweep.py --sweep $SW --weeks 6 --seeds $SEEDS5 --min-replicas 1 $POL --lead-tier d_rep $CAL \
        --p-in "$PIN" $TUNED --out "$OUT/r30"
  done
  stamp "=== 32B-AWQ model sensitivity at r=30 (mu=$MU32)"
  local PIN32; PIN32=$(pin 30 $MU32)
  for SW in sigma lead; do
    run scripts/sim/run_sweep.py --sweep $SW --weeks 6 --seeds $SEEDS3 $SCEN $POL --lead-tier d_rep $CAL32 \
        --p-in "$PIN32" $TUNED --out "$OUT/r30_32b"
  done
  stamp "lane A done"
}
lane_b() {
  LOG=$OUT/lane_B.log; stamp "lane B start: r=6, 10, 100 blocks"
  for R in 6 10 100; do block $R || stamp "lane B: r=$R block failed"; done
  stamp "lane B done"
}
lane_c() {
  LOG=$OUT/lane_C.log; stamp "lane C: waiting for the r=30 tuning"
  while [ ! -s "$OUT/tuning_r30/best.json" ]; do sleep 30; done; sleep 10
  local PIN TUNED; PIN=$(pin 30 $MU); TUNED="--tuned $OUT/tuning_r30/best.json"
  stamp "=== epsilon two-axis (forecast scale x price ratio), r=30 tuned baselines, seeds 10-12, 6 weeks"
  run scripts/sim/run_sweep.py --sweep epsilon --weeks 6 --seeds $SEEDS3 $SCEN $EPSPOL --lead-tier d_rep $CAL \
      --price-ratios 6 10 30 100 $TUNED --out "$OUT/eps"
  stamp "=== trust (epsilon of BDINTERP-d) sweep at r=30"
  run scripts/sim/run_sweep.py --sweep trust --weeks 6 --seeds $SEEDS3 $SCEN --lead-tier d_rep $CAL \
      --p-in "$PIN" $TUNED --out "$OUT/r30"
  stamp "=== pool sweep (sign-in/ASSISTments vs genuine AI-assistant offset pools) at r=30"
  run scripts/sim/run_sweep.py --sweep pool --weeks 6 --seeds $SEEDS5 $SCEN $POL --lead-tier d_rep $CAL \
      --p-in "$PIN" $TUNED --out "$OUT/r30"
  stamp "lane C done"
}
stamp "BATCH_C attempt 3 start: calibrated workload + fresh seeds + controls; lanes A/B/C in parallel"
lane_a & PA=$!
lane_b & PB=$!
lane_c & PC=$!
wait $PA; stamp "lane A exit $?"
wait $PB; stamp "lane B exit $?"
wait $PC; stamp "lane C exit $?"
for R in 6 10 30 100; do run scripts/sim/make_figures.py --sweeps "$OUT/r$R" --out "$OUT/r$R/figures"; done
run scripts/sim/make_figures.py --sweeps "$OUT/r30_32b" --out "$OUT/r30_32b/figures"
run scripts/sim/make_figures.py --sweeps "$OUT/eps" --out "$OUT/eps/figures"
stamp "BATCH_C_DONE"
