#!/bin/bash
# MBPP/MBPP+ gated pipeline.
set -euo pipefail

TRAINLLM=~/git_home/trainllm
PY=~/.unsloth/studio/unsloth_studio/bin/python3
HF_HOME=$TRAINLLM/models/hf
DATA=$TRAINLLM/data/mbpp
RESULTS=$TRAINLLM/results/mbpp
TEST=$DATA/mbpp_plus_test.jsonl
DATA_VERSION=mbpp-function-tests-v1
MIN_BASE=0.10
MIN_DELTA=0.02
GREAT_7B=0.85

cd "$TRAINLLM"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

rate() {
  python3 - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["pass_rate"])
PY
}

gate_rate() {
  python3 - "$@" <<'PY'
import sys
value = float(sys.argv[1])
op = sys.argv[2]
threshold = float(sys.argv[3])
if op == "lt":
    sys.exit(0 if value < threshold else 1)
if op == "gt":
    sys.exit(0 if value > threshold else 1)
if op == "le":
    sys.exit(0 if value <= threshold else 1)
raise SystemExit(f"bad op {op}")
PY
}

current_version=$(python3 - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("$DATA/manifest.json")
print(json.load(p.open()).get("data_version", "")) if p.exists() else None
PY
)

if [ ! -f "$TEST" ] || [ "$current_version" != "$DATA_VERSION" ]; then
  log "Preparing MBPP train/holdout and MBPP+ eval data"
  "$PY" prepare_mbpp_data.py --output-dir "$DATA"
else
  log "MBPP data already prepared"
fi

mkdir -p "$RESULTS"

log "Smoke eval: base 0.5B on first 50 MBPP+ tasks"
HF_HOME=$HF_HOME "$PY" mbpp_eval.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --test "$TEST" \
  --output "$RESULTS/base-0.5b-smoke" \
  --limit 50
base05_smoke=$(rate "$RESULTS/base-0.5b-smoke/summary.json")
if gate_rate "$base05_smoke" lt "$MIN_BASE"; then
  log "STOP: base 0.5B smoke is broken (${base05_smoke} < ${MIN_BASE})"
  exit 20
fi

log "Full eval: base 0.5B"
HF_HOME=$HF_HOME "$PY" mbpp_eval.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --test "$TEST" \
  --output "$RESULTS/base-0.5b"
base05=$(rate "$RESULTS/base-0.5b/summary.json")

if [ -f "$TRAINLLM/lora/mbpp-0.5b/final/adapter_config.json" ]; then
  log "0.5B adapter already exists; skipping training"
else
  log "Training 0.5B on MBPP with holdout early stopping"
  TRAINLLM_CONFIG=$TRAINLLM/config.mbpp-0.5b.yaml \
  HF_HOME=$HF_HOME \
    "$PY" train.py --probe-data "$TEST" --probe-kind mbpp
fi

log "Full eval: trained 0.5B"
HF_HOME=$HF_HOME "$PY" mbpp_eval.py \
  --model "$TRAINLLM/lora/mbpp-0.5b/final" \
  --test "$TEST" \
  --output "$RESULTS/target-0.5b"
tuned05=$(rate "$RESULTS/target-0.5b/summary.json")

python3 - <<PY
base = float("$base05")
tuned = float("$tuned05")
delta = tuned - base
print(f"[gate] 0.5B base={base:.1%} tuned={tuned:.1%} delta={delta:+.1%}")
raise SystemExit(0 if delta >= float("$MIN_DELTA") else 30)
PY

log "Smoke eval: base 7B on first 50 MBPP+ tasks"
HF_HOME=$HF_HOME "$PY" mbpp_eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --test "$TEST" \
  --output "$RESULTS/base-7b-smoke" \
  --limit 50
base7_smoke=$(rate "$RESULTS/base-7b-smoke/summary.json")
if gate_rate "$base7_smoke" lt "$MIN_BASE"; then
  log "STOP: base 7B smoke is broken (${base7_smoke} < ${MIN_BASE})"
  exit 40
fi

log "Full eval: base 7B"
HF_HOME=$HF_HOME "$PY" mbpp_eval.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --test "$TEST" \
  --output "$RESULTS/base-7b"
base7=$(rate "$RESULTS/base-7b/summary.json")
if gate_rate "$base7" gt "$GREAT_7B"; then
  log "STOP: base 7B is already great (${base7} > ${GREAT_7B}); no useful quality headroom"
  exit 50
fi

if [ -f "$TRAINLLM/lora/mbpp-7b/final/adapter_config.json" ]; then
  log "7B adapter already exists; skipping training"
else
  log "Training 7B on MBPP with holdout early stopping"
  TRAINLLM_CONFIG=$TRAINLLM/config.mbpp-7b.yaml \
  HF_HOME=$HF_HOME \
    "$PY" train.py --probe-data "$TEST" --probe-kind mbpp
fi

log "Full eval: trained 7B"
HF_HOME=$HF_HOME "$PY" mbpp_eval.py \
  --model "$TRAINLLM/lora/mbpp-7b/final" \
  --test "$TEST" \
  --output "$RESULTS/target-7b"
tuned7=$(rate "$RESULTS/target-7b/summary.json")
python3 - <<PY
base = float("$base7")
tuned = float("$tuned7")
print(f"[gate] 7B base={base:.1%} tuned={tuned:.1%} delta={tuned-base:+.1%}")
PY
log "MBPP pipeline complete"
