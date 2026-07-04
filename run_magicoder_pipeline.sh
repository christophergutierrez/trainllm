#!/bin/bash
# Magicoder → HumanEval+ pipeline: prep data → train 0.5B → eval → train 7B → eval
# Run in tmux: tmux new-session -d -s magicoder "bash run_magicoder_pipeline.sh 2>&1 | tee /tmp/magicoder.log"
set -e

TRAINLLM=~/git_home/trainllm
PYTHON=~/.unsloth/studio/unsloth_studio/bin/python3
HF_HOME=$TRAINLLM/models/hf
DATA=$TRAINLLM/data/magicoder
RESULTS=$TRAINLLM/results/magicoder
TEST=$DATA/humaneval_plus_test.jsonl
DATA_VERSION=magicoder-python-clean-v2

cd $TRAINLLM

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Data prep ─────────────────────────────────────────────────────────
CURRENT_DATA_VERSION=$(python3 - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("$DATA/manifest.json")
print(json.load(p.open()).get("data_version", "")) if p.exists() else None
PY
)

if [ -f "$TEST" ] && [ "$CURRENT_DATA_VERSION" = "$DATA_VERSION" ]; then
    log "Data already prepared — skipping."
else
    log "Preparing Magicoder + HumanEval+ data..."
    python3 prepare_magicoder_data.py --output-dir "$DATA"
fi

# ── Step 2: Base evals (before training) ─────────────────────────────────────
if [ ! -f "$RESULTS/base-0.5b-plus-clean/summary.json" ]; then
    log "Running base 0.5B eval..."
    HF_HOME=$HF_HOME $PYTHON humaneval_eval.py \
        --backend hf \
        --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --test "$TEST" \
        --output "$RESULTS/base-0.5b-plus-clean"
    log "base-0.5b-plus-clean: $(python3 -c "import json; d=json.load(open('$RESULTS/base-0.5b-plus-clean/summary.json')); print(f\"{d['pass_rate']:.1%}\")")"
fi

if [ ! -f "$RESULTS/base-7b-plus-clean/summary.json" ]; then
    log "Running base 7B eval..."
    HF_HOME=$HF_HOME $PYTHON humaneval_eval.py \
        --backend hf \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --test "$TEST" \
        --output "$RESULTS/base-7b-plus-clean"
    log "base-7b-plus-clean: $(python3 -c "import json; d=json.load(open('$RESULTS/base-7b-plus-clean/summary.json')); print(f\"{d['pass_rate']:.1%}\")")"
fi

# ── Step 3: Train 0.5B ───────────────────────────────────────────────────────
if [ -f "$TRAINLLM/lora/magicoder-0.5b-py/final/adapter_config.json" ]; then
    log "0.5B draft adapter already exists — skipping training."
else
    log "Training 0.5B draft model..."
    TRAINLLM_CONFIG=$TRAINLLM/config.magicoder-0.5b.yaml \
    HF_HOME=$HF_HOME \
        $PYTHON train.py --probe-data "$TEST"
fi

log "Evaluating fine-tuned 0.5B..."
HF_HOME=$HF_HOME $PYTHON humaneval_eval.py \
    --backend hf \
    --model "$TRAINLLM/lora/magicoder-0.5b-py/final" \
    --test "$TEST" \
    --output "$RESULTS/target-0.5b-plus-clean"

RATE_05B=$(python3 -c "import json; d=json.load(open('$RESULTS/target-0.5b-plus-clean/summary.json')); print(f\"{d['pass_rate']:.1%}\")")
BASE_05B=$(python3 -c "import json; d=json.load(open('$RESULTS/base-0.5b-plus-clean/summary.json')); print(f\"{d['pass_rate']:.1%}\")")
log "0.5B: base=$BASE_05B → fine-tuned=$RATE_05B"

python3 - <<PY
import json
import sys
base = json.load(open("$RESULTS/base-0.5b-plus-clean/summary.json"))["pass_rate"]
tuned = json.load(open("$RESULTS/target-0.5b-plus-clean/summary.json"))["pass_rate"]
if tuned <= base:
    print(f"[gate] stopping before 7B: fine-tuned 0.5B regressed ({tuned:.1%} <= {base:.1%})")
    sys.exit(42)
print(f"[gate] 0.5B improved ({tuned:.1%} > {base:.1%}); continuing to 7B")
PY

# ── Step 4: Train 7B ─────────────────────────────────────────────────────────
if [ -f "$TRAINLLM/lora/magicoder-7b-py/final/adapter_config.json" ]; then
    log "7B target adapter already exists — skipping training."
else
    log "Training 7B target model..."
    TRAINLLM_CONFIG=$TRAINLLM/config.magicoder-7b.yaml \
    HF_HOME=$HF_HOME \
        $PYTHON train.py --probe-data "$TEST"
fi

log "Evaluating fine-tuned 7B..."
HF_HOME=$HF_HOME $PYTHON humaneval_eval.py \
    --backend hf \
    --model "$TRAINLLM/lora/magicoder-7b-py/final" \
    --test "$TEST" \
    --output "$RESULTS/target-7b-plus-clean"

RATE_7B=$(python3 -c "import json; d=json.load(open('$RESULTS/target-7b-plus-clean/summary.json')); print(f\"{d['pass_rate']:.1%}\")")
BASE_7B=$(python3 -c "import json; d=json.load(open('$RESULTS/base-7b-plus-clean/summary.json')); print(f\"{d['pass_rate']:.1%}\")")
log "7B: base=$BASE_7B → fine-tuned=$RATE_7B"

log "Pipeline done. See results/ for details."
log "Next: rsync results + adapters to Mac for speculative decoding eval."
