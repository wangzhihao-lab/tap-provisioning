#!/usr/bin/env bash
# Launch the batch-B stage driver on the box, detached, and return immediately.
#
#   bash run_all.sh --root <dir on the box> [--cap-hours 14.5] [--dry-run] \
#                   [--max-model-len N] [--gpu-mem-util F] [--python /path/to/python]
#
# The driver (stages.py) owns the stages, the heartbeat, the budget watchdog and
# the shutdown; this file only exports the environment and nohup-launches it, so
# the run survives the ssh session that started it going away.
set -euo pipefail

ROOT="${Q2_ROOT:-}"
CAP_HOURS="14.5"
DRY=""
PY_OVERRIDE=""
MAX_MODEL_LEN=""
GPU_MEM_UTIL=""
RAMP_LO="60"
RAMP_HI="120"
ONLY=""
OUT_NAME=""
ON_DONE=""
ON_FAIL=""
ON_CAP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --out-name) OUT_NAME="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --cap-hours) CAP_HOURS="$2"; shift 2 ;;
    --python) PY_OVERRIDE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --ramp-band) RAMP_LO="$2"; RAMP_HI="$3"; shift 3 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    --on-done) ON_DONE="$2"; shift 2 ;;
    --on-fail) ON_FAIL="$2"; shift 2 ;;
    --on-cap) ON_CAP="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$ROOT" ]; then echo "run_all.sh: pass --root <dir> or set Q2_ROOT" >&2; exit 2; fi

HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS="$ROOT/q2_results"
LOGS="$RESULTS/logs"
PID_FILE="$RESULTS/run_all.pid"
mkdir -p "$LOGS"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
  echo "already running as pid $(cat "$PID_FILE"); not launching a second driver"
  exit 0
fi

if [ -n "$PY_OVERRIDE" ]; then
  PY="$PY_OVERRIDE"
elif [ -x "$ROOT/venv/bin/python" ]; then
  PY="$ROOT/venv/bin/python"
else
  PY="$(command -v python3)"
  echo "warning: $ROOT/venv/bin/python missing; falling back to $PY"
fi

export PYTHONUNBUFFERED=1
# FlashInfer is not installed on the Blackwell box (its JIT needs CUDA >= 12.9); every vllm serve,
# including the ones coldstart_vllm.py spawns itself, must skip the FlashInfer sampler probe.
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="$ROOT/hf"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# venv first so `vllm` resolves; the user-level bin directories are absent from a
# non-interactive remote shell PATH, so name them explicitly.
export PATH="$ROOT/venv/bin:$PATH:$HOME/.local/bin:$HOME/miniconda3/bin"

set -- run-all --root "$ROOT" --cap-hours "$CAP_HOURS" --ramp-band "$RAMP_LO" "$RAMP_HI"
if [ -n "$DRY" ]; then set -- "$@" "$DRY"; fi
if [ -z "$ON_DONE" ] || [ -z "$ON_FAIL" ] || [ -z "$ON_CAP" ]; then
  echo "run_all.sh: --on-done, --on-fail and --on-cap (halt|keep) are required" >&2; exit 2
fi
set -- "$@" --on-done "$ON_DONE" --on-fail "$ON_FAIL" --on-cap "$ON_CAP"
if [ -n "$MAX_MODEL_LEN" ]; then set -- "$@" --max-model-len "$MAX_MODEL_LEN"; fi
if [ -n "$GPU_MEM_UTIL" ]; then set -- "$@" --gpu-mem-util "$GPU_MEM_UTIL"; fi
if [ -n "$ONLY" ]; then set -- "$@" --only "$ONLY"; fi
if [ -n "$OUT_NAME" ]; then set -- "$@" --out-name "$OUT_NAME"; fi

# setsid where it exists (Linux box), plain nohup otherwise (macOS dry run).
LAUNCH="nohup"
if command -v setsid >/dev/null 2>&1; then LAUNCH="setsid nohup"; fi
echo "launching: $PY $HERE/stages.py $*"
# shellcheck disable=SC2086
$LAUNCH "$PY" "$HERE/stages.py" "$@" >> "$LOGS/run_all.log" 2>&1 < /dev/null &
sleep 1
echo "driver launched; log $LOGS/run_all.log, pid file $PID_FILE (written by stages.py)"
