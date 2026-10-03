#!/usr/bin/env bash
# Batch-B box bootstrap: venv, deps, three model checkpoints on local NVMe.
#
# Idempotent and safe to re-run on a box that has been set up by hand: an
# existing venv is reused and never recreated, nothing is installed with -U (the
# torch in that venv may be a deliberately chosen cu128 build), and a checkpoint
# is skipped only when every shard named by its safetensors index is on disk.
# Models come from ModelScope first and the HF mirror second.  Everything is
# logged to q2_results/logs/bootstrap.log.
#
#   bash bootstrap.sh --root <dir on the box> [--torch-index URL] [--dry-run]
#
# The box is an RTX PRO 6000D (Blackwell, sm_120, 84 GB), so the installed torch
# must carry CUDA 12.8+ kernels.  bootstrap installs the latest vllm (which pulls
# its own torch), then asserts cuda availability, sm_120 in the torch arch list
# and an NVIDIA driver >= 570; if that fails it reinstalls torch from the cu128
# index once and, if it still fails, stops with the fix printed.
#
# --dry-run only checks the layout (no uv, no pip, no downloads, arch check
# advisory); it exists so orchestrate.py --dry-run can exercise this script.
set -euo pipefail

ROOT="${Q2_ROOT:-}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
DRY=0
ONLY_MODELS="${Q2_MODELS:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --models) ONLY_MODELS="$2"; shift 2 ;;  # comma-separated slugs, e.g. qwen2.5-7b (default: all four)
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$ROOT" ]; then echo "bootstrap.sh: pass --root <dir> or set Q2_ROOT" >&2; exit 2; fi

# A non-interactive remote shell on the box has neither ~/.local/bin (uv) nor
# ~/miniconda3/bin (python3) on PATH.  Appended, not prepended, so the dry run
# on a laptop keeps its own interpreter.
export PATH="$PATH:$HOME/.local/bin:$HOME/miniconda3/bin"

RESULTS="$ROOT/q2_results"
LOGS="$RESULTS/logs"
VENV="$ROOT/venv"
MODELS="$ROOT/models"
mkdir -p "$LOGS" "$MODELS"
exec > >(tee -a "$LOGS/bootstrap.log") 2>&1
echo "=== bootstrap start $(date +%Y-%m-%dT%H:%M:%S%z) root=$ROOT dry_run=$DRY"
echo "torch index for the Blackwell fallback: $TORCH_INDEX"

# Optional provider network script; absent or broken is not fatal.
if [ -f /etc/network_turbo ]; then
  set +u
  # shellcheck disable=SC1091
  . /etc/network_turbo || echo "warning: /etc/network_turbo failed, continuing"
  set -u
  echo "sourced /etc/network_turbo"
fi

# True when every named module is importable in the venv (find_spec: no torch load).
have_modules () {
  [ -x "$VENV/bin/python" ] || return 1
  "$VENV/bin/python" - "$@" <<'PYEOF'
import importlib.util, sys
missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("missing from the venv: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ---------------------------------------------------------------- python env
if [ "$DRY" -eq 1 ]; then
  echo "--dry-run: skipping uv, venv and pip install"
  PY="$(command -v python3 || true)"
else
  if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.11 "$VENV"
  else
    echo "reusing the existing venv $VENV (never recreated)"
  fi
  # No -U anywhere: the torch in this venv may be a deliberately chosen cu128
  # build, and upgrading vllm could drag a different one in.  Install only what
  # is missing; gpu_probe below is what decides whether what is there works.
  if have_modules vllm requests numpy pandas matplotlib; then
    echo "venv already has vllm and the measurement deps; leaving torch alone"
  else
    # pinned to the version every calibration and replay of this paper ran on (B1-B5)
    uv pip install --python "$VENV/bin/python" \
      "vllm==${Q2_VLLM_VERSION:-0.29.0}" requests numpy pandas matplotlib aiohttp
  fi
  if ! have_modules modelscope huggingface_hub; then
    uv pip install --python "$VENV/bin/python" modelscope "huggingface_hub[cli]"
  fi
  PY="$VENV/bin/python"
fi

# ------------------------------------------------------ environment + GPU arch
DF_LINE="$(df -Pk "$ROOT" | awk 'NR==2 {print $4" "$6}')"
export Q2_ENV_ROOT="$ROOT"
export Q2_ENV_FREE_KB="${DF_LINE%% *}"
export Q2_ENV_MOUNT="${DF_LINE##* }"
export Q2_ENV_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
  | head -1 || true)"
export Q2_ENV_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
  | head -1 || true)"
export Q2_ENV_VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
  2>/dev/null | head -1 || true)"
export Q2_ENV_BOOT="$(uptime -s 2>/dev/null || true)"
export Q2_ENV_DRY="$DRY"
export Q2_MIN_DRIVER="${Q2_MIN_DRIVER:-570}"
export Q2_TORCH_INDEX="$TORCH_INDEX"
export Q2_VENV="$VENV"

# Writes env.json; exit 3 when this GPU's arch is not covered by the installed
# torch (an RTX PRO 6000D / Blackwell needs CUDA 12.8+ wheels with sm_120).
gpu_probe () {
  "$PY" - > "$RESULTS/env.json" <<'PYEOF'
import json, os, platform, sys, time

def version(module):
    try:
        return str(__import__(module).__version__)
    except Exception as exc:                       # probe only, never fatal here
        return f"unavailable: {type(exc).__name__}"

def driver_major(raw):
    head = (raw or "").split(".")[0].strip()
    return float(head) if head.isdigit() else None

free_kb = int(os.environ.get("Q2_ENV_FREE_KB") or 0)
raw_vram = (os.environ.get("Q2_ENV_VRAM_MIB") or "").strip()
vram_mib = int(raw_vram) if raw_vram.isdigit() else None
driver_raw = os.environ.get("Q2_ENV_DRIVER") or ""
min_driver = float(os.environ.get("Q2_MIN_DRIVER") or 570)
major = driver_major(driver_raw)

info = {
    "schema": "q2.env.v2",
    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "dry_run": os.environ.get("Q2_ENV_DRY") == "1",
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "vllm_version": version("vllm"),
    "gpu_name": os.environ.get("Q2_ENV_GPU_NAME") or "unknown",
    "gpu_memory_total_mib": vram_mib,
    "gpu_memory_total_gb": None if vram_mib is None else round(vram_mib / 1024, 1),
    "nvidia_driver_version": driver_raw or "unavailable",
    "nvidia_driver_min_required": min_driver,
    "root": os.environ.get("Q2_ENV_ROOT"),
    "root_mount_point": os.environ.get("Q2_ENV_MOUNT"),
    "root_free_gb": round(free_kb / 1048576, 1),
    "kernel_boot_time": os.environ.get("Q2_ENV_BOOT") or "unavailable",
}

problems = []
if major is None:
    problems.append("no nvidia-smi driver version")
elif major < min_driver:
    problems.append(f"driver {driver_raw} is older than the required {min_driver:g}")
info["driver_ok"] = major is not None and major >= min_driver

try:
    import torch
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    info["torch_arch_list"] = list(torch.cuda.get_arch_list())
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        cap = torch.cuda.get_device_capability(0)
        info["device_capability"] = list(cap)
        info["device_sm"] = f"sm_{cap[0]}{cap[1]}"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        if info["device_sm"] not in info["torch_arch_list"]:
            problems.append(
                f"torch {torch.__version__} (cuda {torch.version.cuda}) has no kernels for "
                f"{info['device_sm']}; arch list is {info['torch_arch_list']}"
            )
    else:
        problems.append("torch.cuda.is_available() is False")
except ImportError as exc:
    info["torch_version"] = f"unavailable: {type(exc).__name__}"
    info["cuda_available"] = False
    problems.append(f"torch is not importable ({exc})")

# The whole serving stack must import, not just torch: transformers pulls in torchaudio /
# torchvision, and a CUDA-version mismatch between them and torch only shows up at import time
# (seen on 2026-09-18: torch cu130 + torchaudio cu128 -> vllm serve died before loading anything).
try:
    import importlib
    importlib.import_module("vllm.entrypoints.cli.main")
    for extra in ("torchvision", "torchaudio"):
        try:
            mod = importlib.import_module(extra)
            info[f"{extra}_version"] = getattr(mod, "__version__", "?")
        except ImportError:
            info[f"{extra}_version"] = "absent"
    info["serving_stack_imports"] = True
except Exception as exc:  # noqa: BLE001 - any failure here means vllm serve cannot start
    info["serving_stack_imports"] = False
    problems.append(f"vllm serving stack does not import: {type(exc).__name__}: {str(exc)[:300]}")

info["arch_supported"] = not problems
info["arch_problems"] = problems
if problems:
    info["fix_hint"] = (
        f"uv pip install --python {os.environ.get('Q2_VENV')}/bin/python -U "
        f"--index-url {os.environ.get('Q2_TORCH_INDEX')} "
        "--extra-index-url https://pypi.org/simple --index-strategy unsafe-first-match torch"
        "   # then: uv pip install -U vllm, and re-run bootstrap.sh; for a torchaudio/torchvision "
        "CUDA mismatch: pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<torch cuda>"
    )
print(json.dumps(info, indent=2, ensure_ascii=False))
if problems:
    for problem in problems:
        print(f"GPU arch check: {problem}", file=sys.stderr)
    print(f"fix: {info['fix_hint']}", file=sys.stderr)
    sys.exit(3)
PYEOF
}

if [ -z "$PY" ]; then
  echo '{"schema": "q2.env.v2", "error": "no python3 available"}' > "$RESULTS/env.json"
  echo "FATAL: no python3 to probe the GPU with" >&2
  exit 1
fi

if [ "$DRY" -eq 1 ]; then
  gpu_probe || echo "--dry-run: ignoring the GPU arch check (no torch on a laptop)"
elif ! gpu_probe; then
  echo "arch check failed; reinstalling torch from $TORCH_INDEX and retrying once"
  uv pip install --python "$VENV/bin/python" -U \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-first-match torch
  if ! gpu_probe; then
    echo "FATAL: this GPU's architecture is still unsupported by the installed torch." >&2
    echo "       See arch_problems / fix_hint in $RESULTS/env.json." >&2
    echo "       Do not start the measurements: every stage would fail at full price." >&2
    exit 3
  fi
  echo "arch check passed after installing torch from $TORCH_INDEX"
fi
cat "$RESULTS/env.json"

# -------------------------------------------------------------------- models
HF_CLI="$VENV/bin/hf"
[ -x "$HF_CLI" ] || HF_CLI="$VENV/bin/huggingface-cli"
export HF_HOME="$ROOT/hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# The new hf CLI fails against hf-mirror with Xet 401s; ModelScope is the primary
# source and this switches the HF fallback back to the plain resolver protocol.
export HF_HUB_DISABLE_XET=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/modelscope}"

# A checkpoint counts as present only when every shard named by the safetensors
# index is on disk: a download interrupted half way leaves config.json plus some
# shards behind, and skipping that would fail later at full GPU price.
model_complete () {
  [ -n "$PY" ] || return 1
  "$PY" - "$1" <<'PYEOF'
import json, pathlib, sys
target = pathlib.Path(sys.argv[1])
if not (target / "config.json").is_file():
    sys.exit(1)
index = target / "model.safetensors.index.json"
if not index.is_file():
    sys.exit(0 if any(target.glob("*.safetensors")) else 1)
try:
    shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
except (OSError, ValueError, KeyError) as exc:
    print(f"unreadable safetensors index: {exc}", file=sys.stderr)
    sys.exit(1)
missing = [name for name in shards
           if not (target / name).is_file() or (target / name).stat().st_size == 0]
if missing:
    print(f"{len(missing)} of {len(shards)} shards missing, e.g. {missing[0]}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

fetch_model () {
  model_id="$1"; target="$2"
  if [ -x "$VENV/bin/modelscope" ]; then
    echo "  trying modelscope"
    if "$VENV/bin/modelscope" download --model "$model_id" --local_dir "$target"; then
      return 0
    fi
    echo "  modelscope failed for $model_id" >&2
  else
    echo "  modelscope not installed; skipping straight to the HF mirror" >&2
  fi
  echo "  trying $HF_CLI (endpoint $HF_ENDPOINT, HF_HUB_DISABLE_XET=1)"
  "$HF_CLI" download "$model_id" --local-dir "$target"
}

download_model () {
  model_id="$1"; target="$2"
  if model_complete "$target"; then
    echo "skip $model_id: complete checkpoint already in $target"
    return 0
  fi
  free_gb="$(df -Pk "$MODELS" | awk 'NR==2 {print int($4/1048576)}')"
  if [ "$free_gb" -lt 20 ]; then
    echo "FATAL: only ${free_gb} GB free on $MODELS; need >= 20 GB before $model_id" >&2
    exit 1
  fi
  attempt=1
  while [ "$attempt" -le 3 ]; do
    echo "download $model_id -> $target (attempt $attempt/3)"
    if fetch_model "$model_id" "$target"; then
      if model_complete "$target"; then
        echo "downloaded $model_id"
        return 0
      fi
      echo "warning: $model_id downloaded but is still incomplete" >&2
    fi
    echo "warning: attempt $attempt for $model_id failed" >&2
    attempt=$((attempt + 1))
    sleep 10
  done
  echo "FATAL: $model_id failed 3 times; stopping before the GPU bill grows" >&2
  exit 1
}

if [ "$DRY" -eq 1 ]; then
  echo "--dry-run: skipping model downloads, creating placeholder directories"
  for slug in qwen2.5-7b qwen2.5-14b qwen2.5-32b-awq qwen2.5-72b-awq; do mkdir -p "$MODELS/$slug"; done
else
  wanted () { [ -z "$ONLY_MODELS" ] || case ",$ONLY_MODELS," in *",$1,"*) true ;; *) false ;; esac; }
  wanted qwen2.5-7b      && download_model "Qwen/Qwen2.5-7B-Instruct"      "$MODELS/qwen2.5-7b"
  wanted qwen2.5-14b     && download_model "Qwen/Qwen2.5-14B-Instruct"     "$MODELS/qwen2.5-14b"
  wanted qwen2.5-32b-awq && download_model "Qwen/Qwen2.5-32B-Instruct-AWQ" "$MODELS/qwen2.5-32b-awq"
  wanted qwen2.5-72b-awq && download_model "Qwen/Qwen2.5-72B-Instruct-AWQ" "$MODELS/qwen2.5-72b-awq"
  du -sh "$MODELS"/* 2>/dev/null || true
  df -h "$ROOT"
fi

date +%Y-%m-%dT%H:%M:%S%z > "$RESULTS/bootstrap.done"
echo "=== bootstrap done; marker $RESULTS/bootstrap.done"
